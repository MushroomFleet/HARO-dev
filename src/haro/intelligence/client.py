"""Claude API client for HARO.

Provides async interface to Claude API with retry logic, error handling,
and message history management.

Supports both direct Anthropic API and OpenRouter API (OpenAI-compatible).
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import httpx
from dotenv import load_dotenv

from haro.core.config import APIConfig
from haro.utils.logging import get_logger

logger = get_logger(__name__)

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_env_files() -> None:
    """Load environment variables from .env files.

    Searches for .env in:
    1. Executable directory (for PyInstaller frozen apps)
    2. Current working directory
    3. Project root (where pyproject.toml is)
    4. User's home config directory (~/.config/haro/)
    """
    import sys

    # Check if running as frozen executable (PyInstaller)
    if getattr(sys, 'frozen', False):
        # Look next to the executable
        exe_dir = Path(sys.executable).parent
        exe_env = exe_dir / ".env"
        if exe_env.exists():
            load_dotenv(exe_env)
            return

    # Try current directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)
        return

    # Try project root (look for pyproject.toml)
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            project_env = parent / ".env"
            if project_env.exists():
                load_dotenv(project_env)
                return

    # Try user config directory
    user_env = Path.home() / ".config" / "haro" / ".env"
    if user_env.exists():
        load_dotenv(user_env)


# Load .env files on module import
_load_env_files()


class APIError(Exception):
    """Exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class APIResponse:
    """Response from Claude API."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""
    latency: float = 0.0


@dataclass
class Message:
    """A conversation message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ClaudeClient:
    """Claude API client with async support and retry logic.

    Handles communication with the Anthropic Claude API including:
    - Async message completion
    - Automatic retry with exponential backoff
    - Rate limiting awareness
    - Message history management
    """

    def __init__(self, config: APIConfig) -> None:
        """Initialize Claude client.

        Args:
            config: API configuration.
        """
        self.config = config
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout

        # Message history
        self._messages: list[Message] = []
        self._max_history = 20  # Max messages to keep

        # Retry configuration
        self._max_retries = 3
        self._base_delay = 1.0

        # Client (lazy loaded)
        self._client = None
        self._initialized = False
        self._use_openrouter = False
        self._api_key = None
        self._base_url = None

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._errors = 0

        self.logger = logger.bind(component="ClaudeClient")

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @property
    def message_count(self) -> int:
        """Get number of messages in history."""
        return len(self._messages)

    async def initialize(self) -> None:
        """Initialize the API client.

        Supports both direct Anthropic API and OpenRouter API.
        - If OPENROUTER_API_KEY is set, uses OpenRouter (OpenAI-compatible)
        - If API key starts with 'sk-or', uses OpenRouter
        - Otherwise uses standard Anthropic API

        Raises:
            APIError: If initialization fails.
        """
        try:
            # Check for API keys
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

            # Determine which API to use
            api_key = openrouter_key or anthropic_key
            base_url = self.config.base_url

            # Auto-detect OpenRouter from key format
            if openrouter_key:
                self._use_openrouter = True
            elif api_key and api_key.startswith("sk-or"):
                self._use_openrouter = True

            if self._use_openrouter:
                # Use OpenRouter endpoint (OpenAI-compatible API)
                self._api_key = api_key
                self._base_url = base_url or os.environ.get(
                    "OPENROUTER_BASE_URL", OPENROUTER_BASE_URL
                )
                # No client needed for httpx-based requests
                self._client = None
                self.logger.info(
                    "client_initialized_openrouter",
                    model=self.model,
                    base_url=self._base_url,
                )
            else:
                # Use standard Anthropic API
                import anthropic
                if base_url:
                    self._client = anthropic.Anthropic(base_url=base_url)
                else:
                    self._client = anthropic.Anthropic()
                self.logger.info(
                    "client_initialized_anthropic",
                    model=self.model,
                )

            self._initialized = True

        except ImportError:
            raise APIError(
                "anthropic package not installed. Run: pip install anthropic",
                retryable=False,
            )
        except Exception as e:
            raise APIError(
                f"Failed to initialize client: {e}",
                retryable=False,
            )

    async def complete(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
    ) -> APIResponse:
        """Send a completion request to Claude.

        Args:
            user_input: The user's message.
            system_prompt: Optional system prompt to use.
            include_history: Whether to include message history.

        Returns:
            APIResponse with the model's response.

        Raises:
            APIError: If the request fails after retries.
        """
        if not self._initialized:
            await self.initialize()

        # Build messages
        messages = []
        if include_history:
            messages = self._get_history_messages()

        messages.append({"role": "user", "content": user_input})

        # LOG: Full prompt being sent to LLM
        self.logger.info(
            "llm_prompt_sent",
            user_input=user_input,
            history_messages=len(messages) - 1,
            model=self.model,
        )

        # Make request with retry
        start_time = time.time()
        response = await self._request_with_retry(
            system=system_prompt or "",
            messages=messages,
        )
        latency = time.time() - start_time

        # LOG: Response preview (first 50 chars)
        response_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
        self.logger.info(
            "llm_response_received",
            response_preview=response_preview,
            full_length=len(response.text),
            latency=f"{latency:.2f}s",
        )

        # Add to history
        self._add_message("user", user_input)
        self._add_message("assistant", response.text)

        # Update stats
        self._total_requests += 1
        self._total_tokens += response.usage.get("input_tokens", 0)
        self._total_tokens += response.usage.get("output_tokens", 0)

        response.latency = latency

        self.logger.info(
            "completion_success",
            model=response.model,
            latency=f"{latency:.2f}s",
            input_tokens=response.usage.get("input_tokens", 0),
            output_tokens=response.usage.get("output_tokens", 0),
        )

        return response

    async def _request_with_retry(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> APIResponse:
        """Make API request with retry logic.

        Args:
            system: System prompt.
            messages: List of messages.

        Returns:
            APIResponse from the model.

        Raises:
            APIError: If all retries fail.
        """
        last_error = None
        delay = self._base_delay

        for attempt in range(self._max_retries):
            try:
                response = await self._make_request(system, messages)
                return response

            except APIError as e:
                last_error = e
                self._errors += 1

                if not e.retryable:
                    raise

                self.logger.warning(
                    "request_failed_retrying",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )

                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        raise last_error or APIError("Request failed", retryable=False)

    async def _make_request(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> APIResponse:
        """Make a single API request.

        Args:
            system: System prompt.
            messages: List of messages.

        Returns:
            APIResponse from the model.

        Raises:
            APIError: If the request fails.
        """
        try:
            if self._use_openrouter:
                return await self._make_openrouter_request(system, messages)
            else:
                return await self._make_anthropic_request(system, messages)

        except asyncio.TimeoutError:
            raise APIError(
                f"Request timed out after {self.timeout}s",
                retryable=True,
            )
        except APIError:
            raise
        except Exception as e:
            error_str = str(e)

            # Check for rate limiting
            if "rate" in error_str.lower() or "429" in error_str:
                raise APIError(
                    f"Rate limited: {e}",
                    status_code=429,
                    retryable=True,
                )

            # Check for overloaded
            if "overloaded" in error_str.lower() or "529" in error_str:
                raise APIError(
                    f"API overloaded: {e}",
                    status_code=529,
                    retryable=True,
                )

            # Check for authentication errors
            if "auth" in error_str.lower() or "401" in error_str:
                raise APIError(
                    f"Authentication failed: {e}",
                    status_code=401,
                    retryable=False,
                )

            raise APIError(f"API error: {e}", retryable=False)

    async def _make_anthropic_request(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> APIResponse:
        """Make request to Anthropic API.

        Args:
            system: System prompt.
            messages: List of messages.

        Returns:
            APIResponse from the model.
        """
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self._client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system if system else None,
                messages=messages,
            ),
            timeout=self.timeout,
        )

        return APIResponse(
            text=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason or "",
        )

    async def _make_openrouter_request(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> APIResponse:
        """Make request to OpenRouter API (OpenAI-compatible).

        Args:
            system: System prompt.
            messages: List of messages.

        Returns:
            APIResponse from the model.
        """
        # Build OpenAI-compatible request
        openai_messages = []

        # Add system message if provided
        if system:
            openai_messages.append({"role": "system", "content": system})

        # Add user/assistant messages
        openai_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/user/haro",  # OpenRouter requires this
            "X-Title": "HARO Voice Assistant",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                error_text = response.text
                if response.status_code == 429:
                    raise APIError(f"Rate limited: {error_text}", status_code=429, retryable=True)
                elif response.status_code == 401:
                    raise APIError(f"Authentication failed: {error_text}", status_code=401, retryable=False)
                else:
                    raise APIError(f"API error ({response.status_code}): {error_text}", retryable=False)

            data = response.json()

        # Parse OpenAI-format response
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return APIResponse(
            text=choice["message"]["content"],
            model=data.get("model", self.model),
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            stop_reason=choice.get("finish_reason", ""),
        )

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to history.

        Args:
            role: Message role (user or assistant).
            content: Message content.
        """
        self._messages.append(Message(role=role, content=content))

        # Trim history if too long
        while len(self._messages) > self._max_history:
            self._messages.pop(0)

    def _get_history_messages(self) -> list[dict[str, str]]:
        """Get history as list of message dicts.

        Returns:
            List of message dictionaries.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._messages
        ]

    def clear_history(self) -> None:
        """Clear message history."""
        self._messages.clear()
        self.logger.info("history_cleared")

    def get_history(self, limit: Optional[int] = None) -> list[Message]:
        """Get message history.

        Args:
            limit: Optional limit on number of messages to return.

        Returns:
            List of messages.
        """
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "errors": self._errors,
            "message_count": len(self._messages),
        }
