"""Intelligence routing for HARO.

Routes requests between local Ollama and cloud Claude based on
complexity and explicit user requests.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Any

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RoutedResponse:
    """Response from routed request."""

    text: str
    source: str  # "local" or "cloud"
    latency: float = 0.0


class IntelligenceRouter:
    """Routes requests between local Ollama and cloud Claude.

    Decision logic:
    1. User says "ask Claude" or "use Claude" -> Cloud
    2. Local LLM flags response as complex/uncertain -> Cloud
    3. Simple questions -> Local
    4. Local failure/unavailable -> Cloud fallback
    """

    def __init__(
        self,
        ollama_client: Optional[Any] = None,
        claude_client: Optional[Any] = None,
        prefer_local: bool = True,
        local_timeout: float = 10.0,
        cloud_fallback: bool = True,
    ) -> None:
        """Initialize router.

        Args:
            ollama_client: Local Ollama client.
            claude_client: Cloud Claude client.
            prefer_local: Whether to try local LLM first.
            local_timeout: Timeout for local LLM requests.
            cloud_fallback: Whether to fall back to cloud on local failure.
        """
        self._ollama = ollama_client
        self._claude = claude_client
        self._prefer_local = prefer_local
        self._local_timeout = local_timeout
        self._cloud_fallback = cloud_fallback

        # Statistics
        self._local_requests = 0
        self._cloud_requests = 0
        self._escalations = 0
        self._fallbacks = 0

        self.logger = logger.bind(component="IntelligenceRouter")

    @property
    def has_local(self) -> bool:
        """Check if local LLM is available."""
        return self._ollama is not None and self._ollama.is_available

    @property
    def has_cloud(self) -> bool:
        """Check if cloud LLM is available."""
        return self._claude is not None

    async def route(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> RoutedResponse:
        """Route request to appropriate LLM.

        Args:
            prompt: User's prompt.
            system_prompt: Optional system prompt.

        Returns:
            RoutedResponse with text and source.

        Raises:
            RuntimeError: If no LLM is available.
        """
        prompt_lower = prompt.lower()

        # Check for explicit cloud request keywords
        # These trigger cloud LLM which has :online suffix for web search capability
        cloud_triggers = [
            "ask claude", "use claude", "ask the cloud", "use the api",
            "search the web", "search online", "look up online",
            "current news", "latest news", "what's happening",
        ]
        explicit_cloud = any(trigger in prompt_lower for trigger in cloud_triggers)

        if explicit_cloud:
            self.logger.info("routing_to_cloud", reason="explicit_request")
            return await self._call_cloud(prompt, system_prompt)

        # Try local first if available and configured
        if self._prefer_local and self._ollama and self._ollama.is_available:
            try:
                response = await asyncio.wait_for(
                    self._ollama.complete(prompt, system_prompt),
                    timeout=self._local_timeout,
                )

                # Check if local LLM flagged this as needing cloud
                if response.needs_cloud:
                    self._escalations += 1
                    self.logger.info("routing_to_cloud", reason="local_escalation")
                    return await self._call_cloud(prompt, system_prompt)

                self._local_requests += 1
                self.logger.info(
                    "routed_to_local",
                    latency=f"{response.latency:.2f}s",
                )

                return RoutedResponse(
                    text=response.text,
                    source="local",
                    latency=response.latency,
                )

            except asyncio.TimeoutError:
                self.logger.warning("local_timeout", timeout=self._local_timeout)
                if self._cloud_fallback:
                    self._fallbacks += 1
                    return await self._call_cloud(prompt, system_prompt)
                raise RuntimeError("Local LLM timed out and cloud fallback is disabled")

            except Exception as e:
                self.logger.warning("local_error", error=str(e))
                if self._cloud_fallback:
                    self._fallbacks += 1
                    return await self._call_cloud(prompt, system_prompt)
                raise

        # Default to cloud
        return await self._call_cloud(prompt, system_prompt)

    async def _call_cloud(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> RoutedResponse:
        """Call cloud Claude API.

        Args:
            prompt: User's prompt.
            system_prompt: Optional system prompt.

        Returns:
            RoutedResponse from cloud.

        Raises:
            RuntimeError: If no cloud client is configured.
        """
        if not self._claude:
            raise RuntimeError("No cloud client configured")

        self._cloud_requests += 1

        response = await self._claude.complete(
            user_input=prompt,
            system_prompt=system_prompt,
        )

        self.logger.info(
            "routed_to_cloud",
            latency=f"{response.latency:.2f}s",
        )

        return RoutedResponse(
            text=response.text,
            source="cloud",
            latency=response.latency,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "local_requests": self._local_requests,
            "cloud_requests": self._cloud_requests,
            "escalations": self._escalations,
            "fallbacks": self._fallbacks,
            "has_local": self.has_local,
            "has_cloud": self.has_cloud,
        }
