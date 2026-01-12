"""Orchestrator for coordinating HARO workers.

Manages the interaction between ListenWorker, SpeechWorker, and API client
to provide responsive, parallel operation.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

from haro.core.listen_worker import ListenWorker
from haro.core.speech_worker import SpeechWorker, SpeechPriority
from haro.core.config import HaroConfig
from haro.intelligence.client import ClaudeClient
from haro.intelligence.prompts import PromptBuilder
from haro.context.manager import ContextManager
from haro.utils.logging import get_logger

logger = get_logger(__name__)


# Status phrases for HARO persona
STATUS_PHRASES = {
    "processing": [
        "Processing.",
        "Working on it.",
        "Acknowledged.",
        "Copy that.",
        "On it.",
    ],
    "thinking": [
        "Still working.",
        "One moment.",
        "Almost there.",
        "Thinking.",
    ],
    "ready": [
        "Ready.",
        "Listening.",
        "Standing by.",
    ],
    "error": [
        "Error encountered.",
        "Unable to process.",
        "Something went wrong.",
    ],
    "timeout": [
        "Request timed out.",
        "No response received.",
    ],
    "wake": [
        "Yes?",
        "I'm here.",
        "Listening.",
    ],
}


@dataclass
class OrchestratorStats:
    """Statistics for orchestrator."""

    requests: int = 0
    local_commands: int = 0
    api_calls: int = 0
    errors: int = 0
    total_latency: float = 0.0


class Orchestrator:
    """Coordinates HARO workers for parallel operation.

    Manages the flow:
    1. ListenWorker detects wake word → immediate status phrase
    2. ListenWorker provides transcript → immediate acknowledgment
    3. Check local command → handle immediately
    4. API call (parallel with status updates) → queue response
    """

    def __init__(
        self,
        listen_worker: ListenWorker,
        speech_worker: SpeechWorker,
        api_client: Optional[ClaudeClient],
        config: HaroConfig,
        context_manager: Optional[ContextManager] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            listen_worker: Listening worker instance.
            speech_worker: Speech output worker instance.
            api_client: Claude API client.
            config: HARO configuration.
            context_manager: Optional context manager.
            prompt_builder: Optional prompt builder for system prompts.
        """
        self._listen = listen_worker
        self._speech = speech_worker
        self._api = api_client
        self._config = config
        self._context = context_manager
        self._prompt_builder = prompt_builder

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thinking_delay = 2.5  # Seconds before playing thinking phrase
        self._api_timeout = config.api.timeout if config.api else 30.0

        self.stats = OrchestratorStats()
        self.logger = logger.bind(component="Orchestrator")

        # Register callbacks with listen worker
        self._listen.on_wake(self._on_wake)

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        self.logger.info("orchestrator_started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self.logger.info("orchestrator_stopped")

    async def _run(self) -> None:
        """Main orchestration loop."""
        self.logger.info("orchestrator_running")

        while self._running:
            try:
                # Wait for transcript from listen worker
                transcript = await self._listen.get_transcript(timeout=0.5)

                if transcript:
                    await self._handle_transcript(transcript)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("orchestrator_error", error=str(e))
                self.stats.errors += 1

        self.logger.info("orchestrator_loop_exited")

    async def _on_wake(self) -> None:
        """Handle wake word detection."""
        # Immediately speak wake confirmation
        phrase = random.choice(STATUS_PHRASES["wake"])
        await self._speech.speak_status(phrase)
        self.logger.debug("wake_acknowledged")

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle user transcript.

        Args:
            transcript: Transcribed user speech.
        """
        self.stats.requests += 1
        start_time = time.time()

        self.logger.info("handling_transcript", text=transcript[:50])

        # Pause listening during response
        self._listen.pause()

        try:
            # Check for local command first
            command = self._check_local_command(transcript)

            if command:
                self.stats.local_commands += 1
                response = await self._handle_local_command(command)

                if response:
                    # Speak with single signoff for local commands
                    await self._speech.speak(
                        response,
                        priority=SpeechPriority.STATUS,
                        signoff=self._config.wake.phrase,
                    )
            else:
                # API call with parallel status updates
                await self._handle_api_request(transcript)

            # Log turn to context
            if self._context:
                self._context.log_turn(
                    user_input=transcript,
                    response=self._speech._current_item.text if self._speech._current_item else "",
                )

        finally:
            # Resume listening after speech completes
            # Wait for speech queue to empty
            while self._speech.queue_size > 0 or self._speech.is_speaking:
                await asyncio.sleep(0.1)

            self._listen.resume()

            latency = time.time() - start_time
            self.stats.total_latency += latency
            self.logger.debug("transcript_handled", latency=f"{latency:.2f}s")

    async def _handle_api_request(self, transcript: str) -> None:
        """Handle API request with parallel status updates.

        Args:
            transcript: User transcript to send to API.
        """
        if not self._api:
            await self._speech.speak_status("API not configured.", urgent=True)
            return

        self.stats.api_calls += 1

        # Immediate acknowledgment
        phrase = random.choice(STATUS_PHRASES["processing"])
        await self._speech.speak_status(phrase)

        # Start API call
        api_task = asyncio.create_task(self._call_api(transcript))

        # Delayed thinking phrase if API is slow
        thinking_task = asyncio.create_task(self._delayed_thinking())

        try:
            response = await asyncio.wait_for(api_task, timeout=self._api_timeout)
            thinking_task.cancel()

            if response:
                # Queue response with double signoff
                await self._speech.speak_response(response, double_signoff=True)
            else:
                await self._speech.speak_status("No response received.", urgent=True)

        except asyncio.TimeoutError:
            thinking_task.cancel()
            phrase = random.choice(STATUS_PHRASES["timeout"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.warning("api_timeout")

        except Exception as e:
            thinking_task.cancel()
            phrase = random.choice(STATUS_PHRASES["error"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.error("api_error", error=str(e))

    async def _call_api(self, transcript: str) -> Optional[str]:
        """Call Claude API.

        Args:
            transcript: User transcript.

        Returns:
            Response text or None.
        """
        try:
            # Build system prompt using prompt builder
            system_prompt = None
            if self._prompt_builder:
                prompt = self._prompt_builder.build(user_input=transcript)
                system_prompt = prompt.content

            response = await self._api.complete(
                user_input=transcript,
                system_prompt=system_prompt,
                include_history=True,
            )

            return response.text

        except Exception as e:
            self.logger.error("api_call_failed", error=str(e))
            return None

    async def _delayed_thinking(self) -> None:
        """Play thinking phrase after delay."""
        try:
            await asyncio.sleep(self._thinking_delay)
            phrase = random.choice(STATUS_PHRASES["thinking"])
            await self._speech.speak_status(phrase)
        except asyncio.CancelledError:
            pass

    def _check_local_command(self, transcript: str) -> Optional[str]:
        """Check if transcript is a local command.

        Args:
            transcript: The transcribed text.

        Returns:
            Command name if detected, None otherwise.
        """
        text = transcript.lower().strip()
        wake_phrase = self._config.wake.phrase.lower()

        # Remove wake phrase prefix if present
        for prefix in [f"{wake_phrase}, ", f"{wake_phrase} "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        # Command patterns
        commands = [
            ("quieter", ["quieter", "volume down", "softer", "lower volume"]),
            ("louder", ["louder", "volume up", "speak up", "raise volume"]),
            ("mute", ["mute", "silence"]),
            ("unmute", ["unmute", "unsilence"]),
            ("stop", ["stop", "shut up", "be quiet", "cancel"]),
            ("repeat", ["repeat", "say that again", "repeat that", "what did you say"]),
            ("pause", ["pause", "hold on", "wait"]),
            ("sleep", ["go to sleep", "sleep mode", "sleep"]),
            ("goodbye", ["goodbye", "bye", "see you", "exit", "quit"]),
            ("new_conversation", ["new conversation", "start over", "reset", "clear history", "clear"]),
            ("time", ["what time is it", "what's the time", "current time"]),
            ("date", ["what's the date", "what day is it", "today's date"]),
            ("status", ["status", "how are you", "system status"]),
            ("help", ["help", "what can you do", "commands", "help me"]),
        ]

        for command, patterns in commands:
            for pattern in patterns:
                if text.startswith(pattern) or text == pattern:
                    return command

        return None

    async def _handle_local_command(self, command: str) -> Optional[str]:
        """Handle a local command.

        Args:
            command: The command name.

        Returns:
            Response text or None.
        """
        self.logger.info("handling_local_command", command=command)

        if command == "stop":
            await self._speech.interrupt()
            return None

        elif command == "repeat":
            # Would need to track last response
            return "I don't have anything to repeat."

        elif command == "louder":
            return "Volume increased."

        elif command == "quieter":
            return "Volume decreased."

        elif command == "mute":
            return "Muted."

        elif command == "unmute":
            return "Unmuted."

        elif command == "pause":
            return "I'm listening."

        elif command == "sleep":
            return "Going to sleep. Say the wake word to wake me up."

        elif command == "goodbye":
            return "Goodbye!"

        elif command == "new_conversation":
            if self._api:
                self._api.clear_history()
            return "Starting a new conversation."

        elif command == "time":
            from datetime import datetime
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')}."

        elif command == "date":
            from datetime import datetime
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        elif command == "status":
            return self._get_status_response()

        elif command == "help":
            return self._get_help_response()

        return None

    def _get_status_response(self) -> str:
        """Get status information."""
        return (
            f"HARO is operational. "
            f"Processed {self.stats.requests} requests. "
            f"API calls: {self.stats.api_calls}. "
            f"Errors: {self.stats.errors}."
        )

    def _get_help_response(self) -> str:
        """Get help information."""
        return (
            "Available commands: "
            "stop, repeat, louder, quieter, mute, unmute, "
            "time, date, status, new conversation, sleep, goodbye."
        )
