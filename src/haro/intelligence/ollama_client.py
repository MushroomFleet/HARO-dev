"""Ollama local LLM client for HARO.

Provides async interface to Ollama API for local LLM inference.
Used for status responses, simple questions, and routing to cloud.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Any

import httpx

from haro.utils.logging import get_logger

logger = get_logger(__name__)

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaError(Exception):
    """Exception for Ollama errors."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class OllamaResponse:
    """Response from Ollama API."""

    text: str
    model: str
    eval_count: int = 0
    eval_duration: float = 0.0
    latency: float = 0.0
    needs_cloud: bool = False  # Flag if local LLM suggests using cloud


class OllamaClient:
    """Ollama local LLM client with async support.

    Handles communication with local Ollama server for:
    - Status response generation
    - Simple question answering
    - Complexity assessment for routing to cloud
    """

    def __init__(self, config: Any) -> None:
        """Initialize Ollama client.

        Args:
            config: Ollama configuration (OllamaConfig dataclass).
        """
        self.config = config
        self.model = config.model
        self.base_url = config.base_url
        self.timeout = config.timeout

        self._initialized = False
        self._available = False

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._cloud_referrals = 0

        self.logger = logger.bind(component="OllamaClient")

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        return self._available

    async def initialize(self) -> None:
        """Initialize and check Ollama availability.

        Checks if Ollama server is running and if the configured model
        is available.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    self._available = True
                    self._initialized = True

                    # Check if model is available
                    data = response.json()
                    models = [m["name"] for m in data.get("models", [])]

                    # Check for exact match or with :latest suffix
                    model_found = (
                        self.model in models or
                        f"{self.model}:latest" in models or
                        any(m.startswith(f"{self.model}:") for m in models)
                    )

                    if not model_found:
                        self.logger.warning(
                            "ollama_model_not_found",
                            model=self.model,
                            available_models=models[:5],
                        )
                    else:
                        self.logger.info(
                            "ollama_model_found",
                            model=self.model,
                        )

                    self.logger.info(
                        "ollama_initialized",
                        model=self.model,
                        available_models=len(models),
                    )
                else:
                    self._available = False
                    self.logger.warning("ollama_not_available", status=response.status_code)

        except httpx.ConnectError:
            self._available = False
            self.logger.warning("ollama_not_running", hint="Start Ollama with 'ollama serve'")
        except Exception as e:
            self._available = False
            self.logger.warning("ollama_connection_failed", error=str(e))

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> OllamaResponse:
        """Send a completion request to Ollama.

        Args:
            prompt: The user's prompt.
            system_prompt: Optional system prompt.

        Returns:
            OllamaResponse with the model's response.

        Raises:
            OllamaError: If the request fails.
        """
        if not self._initialized:
            await self.initialize()

        if not self._available:
            raise OllamaError("Ollama not available", retryable=False)

        start_time = time.time()

        # Build request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        # LOG: Sending prompt to local LLM
        self.logger.info(
            "ollama_prompt_sent",
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            model=self.model,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )

                if response.status_code != 200:
                    raise OllamaError(f"Ollama error: {response.text}")

                data = response.json()

            latency = time.time() - start_time
            response_text = data.get("response", "")

            # Check if response suggests using cloud
            needs_cloud = self._check_needs_cloud(prompt, response_text)

            if needs_cloud:
                self._cloud_referrals += 1

            self._total_requests += 1
            self._total_tokens += data.get("eval_count", 0)

            # LOG: Response preview
            response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
            self.logger.info(
                "ollama_response_received",
                response_preview=response_preview,
                latency=f"{latency:.2f}s",
                tokens=data.get("eval_count", 0),
                needs_cloud=needs_cloud,
            )

            return OllamaResponse(
                text=response_text,
                model=data.get("model", self.model),
                eval_count=data.get("eval_count", 0),
                eval_duration=data.get("eval_duration", 0) / 1e9,  # Convert ns to s
                latency=latency,
                needs_cloud=needs_cloud,
            )

        except httpx.TimeoutException:
            raise OllamaError("Ollama timeout", retryable=True)
        except OllamaError:
            raise
        except Exception as e:
            raise OllamaError(f"Ollama error: {e}", retryable=False)

    def _check_needs_cloud(self, prompt: str, response: str) -> bool:
        """Check if request should be escalated to cloud LLM.

        Args:
            prompt: User's prompt.
            response: Local LLM response.

        Returns:
            True if cloud LLM should be used.
        """
        prompt_lower = prompt.lower()

        # Check for explicit cloud keywords from config
        for keyword in self.config.cloud_keywords:
            if keyword in prompt_lower:
                return True

        # Check for uncertainty markers in response
        uncertainty_markers = [
            "i'm not sure",
            "i don't know",
            "i cannot",
            "i am unable",
            "beyond my",
            "complex question",
            "need more context",
            "not certain",
            "difficult to say",
        ]

        response_lower = response.lower()
        for marker in uncertainty_markers:
            if marker in response_lower:
                return True

        return False

    async def generate_status_response(self, context: str) -> str:
        """Generate a dynamic status response.

        Args:
            context: Context about what HARO is doing.

        Returns:
            Natural language status response in 3rd person.
        """
        system = (
            "You are HARO, a helpful voice assistant. "
            "Generate brief, 3rd-person status updates. "
            "Always refer to yourself as 'HARO'. "
            "Keep responses under 10 words. "
            "End with 'HARO.' for single signoff."
        )

        prompt = f"Generate a brief status update for: {context}"

        try:
            response = await self.complete(prompt, system_prompt=system)
            return response.text.strip()
        except OllamaError:
            # Fallback to static response
            return f"HARO is {context}. HARO."

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "cloud_referrals": self._cloud_referrals,
            "available": self._available,
            "model": self.model,
        }
