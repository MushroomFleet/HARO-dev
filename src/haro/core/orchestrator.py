"""Orchestrator for coordinating HARO workers.

Manages the interaction between ListenWorker, SpeechWorker, and API client
to provide responsive, parallel operation.
"""

import asyncio
import random
import re
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any

from haro.core.listen_worker import ListenWorker
from haro.core.speech_worker import SpeechWorker, SpeechPriority
from haro.core.config import HaroConfig
from haro.core.events import (
    EventBus,
    Event,
    EventType,
    StateChangeEvent,
    WakeWordEvent,
    TranscriptionEvent,
)
from haro.intelligence.client import ClaudeClient
from haro.intelligence.ollama_client import OllamaClient
from haro.intelligence.router import IntelligenceRouter
from haro.intelligence.prompts import PromptBuilder
from haro.context.manager import ContextManager
from haro.utils.logging import get_logger
from haro.utils.text_chunker import SentenceChunker

logger = get_logger(__name__)


def _extract_domain(url: str) -> str:
    """Extract domain.tld from a URL for TTS-friendly output.

    Examples:
        https://www.bbc.com/weather/2643743 -> bbc.com
        https://www.accuweather.com/en/gb/london -> accuweather.com
        https://docs.python.org/3/library -> python.org

    Args:
        url: Full URL string.

    Returns:
        Domain name with TLD (e.g., "bbc.com").
    """
    try:
        # Remove protocol
        domain = url.split("://")[-1]
        # Get hostname (before first /)
        domain = domain.split("/")[0]
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        # Remove port if present
        domain = domain.split(":")[0]
        return domain
    except Exception:
        return url


def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS output by removing markdown formatting.

    Removes characters that shouldn't be read aloud:
    - # (markdown headers)
    - * (bold/italic markers)
    - _ (underscores used for emphasis)
    - ` (code markers)
    - > (blockquotes)

    Also shortens URLs to just domain.tld for natural speech:
    - [bbc.com](https://www.bbc.com/weather/123) -> "bbc.com"
    - https://www.example.com/path -> "example.com"

    Args:
        text: Raw text that may contain markdown.

    Returns:
        Cleaned text suitable for TTS.
    """
    # Shorten markdown links [text](url) to just the domain
    # Pattern: [any text](url) -> domain.tld
    def replace_markdown_link(match):
        url = match.group(2)
        return _extract_domain(url)

    text = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', replace_markdown_link, text)

    # Shorten bare URLs to just the domain
    # Pattern: http(s)://... -> domain.tld
    def replace_bare_url(match):
        url = match.group(0)
        return _extract_domain(url)

    text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]+', replace_bare_url, text)

    # Remove markdown headers (# at start of lines)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers (* and _)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'_+', '', text)

    # Remove code markers
    text = re.sub(r'`+', '', text)

    # Remove blockquote markers
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Remove bullet points (- or * at start of lines)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)

    # Remove numbered list markers
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# Status phrases for HARO persona (3rd person with single HARO signoff)
STATUS_PHRASES = {
    "processing": [
        "HARO is processing.",
        "HARO is working on it.",
        "HARO acknowledges.",
        "HARO copies that.",
        "HARO is on it.",
    ],
    "transcribing": [
        "HARO is transcribing.",
        "HARO is processing speech.",
        "HARO heard that.",
    ],
    "thinking": [
        "HARO is still working.",
        "HARO needs a moment.",
        "HARO is almost there.",
        "HARO is thinking.",
    ],
    "ready": [
        "HARO is ready.",
        "HARO is listening.",
        "HARO is standing by.",
    ],
    "error": [
        "HARO encountered an error.",
        "HARO was unable to process.",
        "HARO ran into a problem.",
    ],
    "timeout": [
        "HARO's request timed out.",
        "HARO received no response.",
    ],
    "wake": [
        "Hello, HARO?",
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
        ollama_client: Optional[OllamaClient] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            listen_worker: Listening worker instance.
            speech_worker: Speech output worker instance.
            api_client: Claude API client (cloud).
            config: HARO configuration.
            context_manager: Optional context manager.
            prompt_builder: Optional prompt builder for system prompts.
            ollama_client: Optional Ollama client for local LLM.
            event_bus: Optional event bus for publishing UI events.
        """
        self._listen = listen_worker
        self._speech = speech_worker
        self._api = api_client
        self._ollama = ollama_client
        self._config = config
        self._context = context_manager
        self._prompt_builder = prompt_builder
        self._event_bus = event_bus

        # Track current state for UI
        self._current_state = "PASSIVE"
        self._last_llm_source = "cloud"  # Track which LLM was used for UI display

        # Create router if we have any LLM clients
        # Always prefer local LLM first - cloud is used on demand or escalation
        if ollama_client or api_client:
            self._router = IntelligenceRouter(
                ollama_client=ollama_client,
                claude_client=api_client,
                prefer_local=True,  # Always try local LLM first
                cloud_fallback=True,  # Fall back to cloud if local fails
            )
        else:
            self._router = None

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thinking_delay = 2.5  # Seconds before playing thinking phrase
        self._api_timeout = config.api.timeout if config.api else 30.0

        # Streaming configuration
        self._use_streaming = True  # Enable streaming by default for faster TTS
        self._sentence_chunker = SentenceChunker(
            min_chunk_length=15,  # Min chars before yielding
            max_buffer_length=300,  # Force yield if too long
        )

        self.stats = OrchestratorStats()
        self.logger = logger.bind(component="Orchestrator")

        # Register callbacks with listen worker
        self._listen.on_wake(self._on_wake)
        self._listen.on_interrupt(self._on_interrupt)

    async def _publish_state(self, new_state: str) -> None:
        """Publish state change event to UI.

        Args:
            new_state: The new state name.
        """
        if self._event_bus:
            old_state = self._current_state
            self._current_state = new_state
            await self._event_bus.publish(
                StateChangeEvent(
                    previous_state=old_state,
                    new_state=new_state,
                )
            )

    async def _publish_wake(self, confidence: float = 1.0) -> None:
        """Publish wake word event to UI."""
        if self._event_bus:
            await self._event_bus.publish(
                WakeWordEvent(confidence=confidence, phrase=self._config.wake.phrase)
            )

    async def _publish_transcription(self, text: str, confidence: float = 1.0) -> None:
        """Publish transcription event to UI."""
        if self._event_bus:
            await self._event_bus.publish(
                TranscriptionEvent(text=text, confidence=confidence)
            )

    async def _publish_api_start(self) -> None:
        """Publish API request start event."""
        if self._event_bus:
            await self._event_bus.publish(Event(type=EventType.API_REQUEST_START))

    async def _publish_api_response(self, response: str, source: str = "cloud") -> None:
        """Publish API response event with source info."""
        if self._event_bus:
            await self._event_bus.publish(
                Event(type=EventType.API_RESPONSE_RECEIVED, data={
                    "response": response,
                    "source": source,  # "local" or "cloud"
                })
            )

    async def _publish_api_error(self, error: str) -> None:
        """Publish API error event."""
        if self._event_bus:
            await self._event_bus.publish(
                Event(type=EventType.API_ERROR, data={"error": error})
            )

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
        """Handle wake word detection.

        Flow to prevent HARO from recording his own voice:
        1. Pause listening immediately
        2. Speak wake confirmation
        3. Wait for confirmation to finish
        4. Resume in active mode to record user speech
        """
        # Publish wake event for UI
        await self._publish_wake()
        await self._publish_state("ACTIVE")

        # Pause listening to prevent recording HARO's own voice
        self._listen.pause_full()

        # Speak wake confirmation
        phrase = random.choice(STATUS_PHRASES["wake"])
        await self._speech.speak_status(phrase)

        # Wait for wake confirmation to finish playing
        await self._speech.wait_for_speech(timeout=5.0)

        # Now resume in active mode - ready to record user speech
        self._listen.resume_active()
        self.logger.debug("wake_acknowledged_ready_for_speech")

    async def _on_interrupt(self) -> None:
        """Handle interrupt command detected during TTS playback."""
        self.logger.info("interrupt_command_received")

        # Stop current speech and clear queue
        await self._speech.interrupt()

        # Publish state change for UI
        await self._publish_state("PASSIVE")

        # Resume listening in passive mode (this also resets wake cooldown)
        self._listen.resume()

    def _get_prompt_preview(self, transcript: str, max_words: int = 5) -> str:
        """Get a short preview of the transcript for status TTS.

        Args:
            transcript: Full transcript text.
            max_words: Maximum words to include in preview.

        Returns:
            Truncated transcript with ellipsis if needed.
        """
        words = transcript.split()
        if len(words) <= max_words:
            return transcript
        return " ".join(words[:max_words]) + "..."

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle user transcript.

        Simplified flow:
        1. Single acknowledgment with prompt preview
        2. Send to LLM (local first, then cloud)
        3. Clean and speak response with signoff

        Supports ULTRATALK keyword for verbose responses.

        Args:
            transcript: Transcribed user speech.
        """
        self.stats.requests += 1
        start_time = time.time()

        self.logger.info("handling_transcript", text=transcript[:50])

        # Check for ULTRATALK mode (verbose responses)
        ultratalk_mode = "ultratalk" in transcript.lower()
        if ultratalk_mode:
            # Remove ULTRATALK from transcript before sending to LLM
            transcript = re.sub(r'\bultratalk\b', '', transcript, flags=re.IGNORECASE).strip()
            self.logger.info("ultratalk_mode_enabled")

        # Publish transcription event for UI
        await self._publish_transcription(transcript)

        # Pause listening during response
        self._listen.pause()

        try:
            # Check for local command first (no LLM needed)
            command = self._check_local_command(transcript)

            if command:
                self.stats.local_commands += 1
                response = await self._handle_local_command(command)

                if response:
                    # Publish state for speaking
                    await self._publish_state("SPEAKING")
                    # Speak with single signoff for local commands
                    await self._speech.speak(
                        response,
                        priority=SpeechPriority.STATUS,
                        signoff=self._config.wake.phrase,
                    )
            else:
                # Publish state for processing
                await self._publish_state("PROCESSING")

                # Single acknowledgment: brief status with prompt preview
                prompt_preview = self._get_prompt_preview(transcript)
                await self._speech.speak_status(f"HARO heard: {prompt_preview}")

                # Send to LLM (router will use local first, then cloud if needed)
                await self._handle_api_request(transcript, ultratalk=ultratalk_mode)

            # Log turn to context
            if self._context:
                self._context.log_turn(
                    user_input=transcript,
                    response=self._speech._current_item.text if self._speech._current_item else "",
                )

        finally:
            # Resume listening after speech completes
            # Wait for speech queue to empty
            await self._speech.wait_for_speech()

            # Return to passive state
            await self._publish_state("PASSIVE")

            self._listen.resume()

            latency = time.time() - start_time
            self.stats.total_latency += latency
            self.logger.debug("transcript_handled", latency=f"{latency:.2f}s")

    async def _handle_api_request(self, transcript: str, ultratalk: bool = False) -> None:
        """Handle LLM request with parallel status updates.

        Uses the intelligence router to decide between local (Ollama)
        and cloud (Claude) LLM based on query complexity.
        Local LLM is always tried first unless user explicitly asks for Claude.

        When streaming is enabled and using direct API (not router),
        sentences are sent to TTS as they arrive for faster response.

        Args:
            transcript: User transcript to send to LLM.
            ultratalk: If True, skip summarization for verbose output.
        """
        # Check if we have any LLM configured (router or direct API)
        if not self._router and not self._api:
            await self._speech.speak_status("HARO has no LLM configured. HARO.", urgent=True)
            return

        self.stats.api_calls += 1

        # Publish API start event for UI
        await self._publish_api_start()

        # Use streaming for direct API calls (not routed) when enabled
        use_router = self._router is not None
        use_streaming = self._use_streaming and not use_router and self._api is not None

        if use_streaming:
            # Streaming mode: sentences are spoken as they arrive
            await self._handle_streaming_api_request(transcript, ultratalk=ultratalk)
        else:
            # Non-streaming mode: wait for full response
            await self._handle_non_streaming_api_request(transcript, ultratalk=ultratalk, use_router=use_router)

    async def _handle_streaming_api_request(self, transcript: str, ultratalk: bool = False) -> None:
        """Handle LLM request with streaming for faster TTS.

        Streams the response and sends sentences to TTS as they complete,
        significantly reducing time-to-first-speech.

        Args:
            transcript: User transcript to send to LLM.
            ultratalk: If True, skip summarization for verbose output.
        """
        # Build system prompt
        system_prompt = None
        if self._prompt_builder:
            prompt = self._prompt_builder.build(user_input=transcript)
            system_prompt = prompt.content

        if not ultratalk:
            system_prompt = self._add_conciseness_instruction(system_prompt)
        else:
            system_prompt = self._add_verbose_instruction(system_prompt)

        self.logger.info("llm_streaming_start", transcript=transcript[:50])

        # Delayed thinking phrase if slow
        thinking_task = asyncio.create_task(self._delayed_thinking())
        first_chunk_received = False
        full_response = ""
        sentence_count = 0

        try:
            # Reset chunker for new stream
            self._sentence_chunker.reset()

            # Stream from API and chunk into sentences
            async for chunk in self._api.complete_streaming(
                user_input=transcript,
                system_prompt=system_prompt,
                include_history=True,
            ):
                # Cancel thinking phrase once we start receiving
                if not first_chunk_received:
                    first_chunk_received = True
                    thinking_task.cancel()
                    await self._publish_state("SPEAKING")

                full_response += chunk

                # Try to extract a complete sentence
                sentence = self._sentence_chunker.add_chunk(chunk)
                if sentence:
                    # Clean and speak this sentence immediately
                    clean_sentence = clean_text_for_tts(sentence.text)
                    if clean_sentence.strip():
                        sentence_count += 1
                        # First sentence gets no signoff, subsequent get none too
                        # Final signoff added after all sentences
                        await self._speech.speak(
                            clean_sentence,
                            priority=SpeechPriority.RESPONSE,
                            signoff=None,  # No signoff per-sentence
                        )
                        self.logger.debug(
                            "streaming_sentence_queued",
                            sentence_num=sentence_count,
                            text=clean_sentence[:30],
                        )

            # Flush any remaining content
            remaining = self._sentence_chunker.flush()
            if remaining and remaining.text.strip():
                clean_remaining = clean_text_for_tts(remaining.text)
                if clean_remaining.strip():
                    sentence_count += 1
                    # Last chunk gets the double signoff
                    await self._speech.speak(
                        clean_remaining,
                        priority=SpeechPriority.RESPONSE,
                        signoff=f"{self._config.wake.phrase} {self._config.wake.phrase}",
                    )
            elif sentence_count > 0:
                # Add signoff as separate utterance if we already queued sentences
                await self._speech.speak(
                    f"{self._config.wake.phrase} {self._config.wake.phrase}",
                    priority=SpeechPriority.RESPONSE,
                    signoff=None,
                )

            # Publish full response for UI
            clean_full = clean_text_for_tts(full_response)
            await self._publish_api_response(clean_full, source="cloud")

            self.logger.info(
                "llm_streaming_complete",
                sentences=sentence_count,
                total_length=len(full_response),
            )

        except asyncio.TimeoutError:
            thinking_task.cancel()
            await self._publish_api_error("Timeout")
            phrase = random.choice(STATUS_PHRASES["timeout"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.warning("llm_streaming_timeout")

        except Exception as e:
            thinking_task.cancel()
            await self._publish_api_error(str(e))
            phrase = random.choice(STATUS_PHRASES["error"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.error("llm_streaming_error", error=str(e))

    async def _handle_non_streaming_api_request(
        self,
        transcript: str,
        ultratalk: bool = False,
        use_router: bool = False,
    ) -> None:
        """Handle LLM request without streaming (original behavior).

        Args:
            transcript: User transcript to send to LLM.
            ultratalk: If True, skip summarization for verbose output.
            use_router: If True, use the intelligence router.
        """
        # Start LLM call (routed or direct)
        if use_router:
            llm_task = asyncio.create_task(self._call_routed(transcript, ultratalk=ultratalk))
        else:
            llm_task = asyncio.create_task(self._call_api(transcript, ultratalk=ultratalk))

        # Delayed thinking phrase if LLM is slow
        thinking_task = asyncio.create_task(self._delayed_thinking())

        try:
            result = await asyncio.wait_for(llm_task, timeout=self._api_timeout)
            thinking_task.cancel()

            if result:
                # Handle routed response (text, source) or direct response (text only)
                if use_router and isinstance(result, tuple):
                    response_text, source = result
                else:
                    response_text = result
                    source = "cloud"

                # Clean text for TTS (remove markdown formatting)
                response_text = clean_text_for_tts(response_text)

                # Publish API response and state change for UI
                await self._publish_api_response(response_text, source=source)
                await self._publish_state("SPEAKING")

                # Queue response with double signoff (HARO HARO) for LLM responses
                await self._speech.speak_response(response_text, double_signoff=True)
            else:
                await self._publish_api_error("No response received")
                await self._speech.speak_status("HARO received no response. HARO.", urgent=True)

        except asyncio.TimeoutError:
            thinking_task.cancel()
            await self._publish_api_error("Timeout")
            phrase = random.choice(STATUS_PHRASES["timeout"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.warning("llm_timeout")

        except Exception as e:
            thinking_task.cancel()
            await self._publish_api_error(str(e))
            phrase = random.choice(STATUS_PHRASES["error"])
            await self._speech.speak_status(phrase, urgent=True)
            self.logger.error("llm_error", error=str(e))

    async def _call_api(self, transcript: str, ultratalk: bool = False) -> Optional[str]:
        """Call Claude API.

        Args:
            transcript: User transcript.
            ultratalk: If True, request verbose responses.

        Returns:
            Response text or None.
        """
        try:
            # Build system prompt using prompt builder
            system_prompt = None
            if self._prompt_builder:
                prompt = self._prompt_builder.build(user_input=transcript)
                system_prompt = prompt.content

            # Add conciseness instruction unless ULTRATALK mode
            if not ultratalk:
                system_prompt = self._add_conciseness_instruction(system_prompt)
            else:
                # For ULTRATALK, add verbose instruction
                system_prompt = self._add_verbose_instruction(system_prompt)

            # LOG: Sending prompt to LLM
            self.logger.info(
                "llm_prompt_sending",
                transcript=transcript,
                ultratalk=ultratalk,
            )

            response = await self._api.complete(
                user_input=transcript,
                system_prompt=system_prompt,
                include_history=True,
            )

            # LOG: Response preview
            response_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
            self.logger.info(
                "llm_response_preview",
                preview=response_preview,
            )

            return response.text

        except Exception as e:
            self.logger.error("api_call_failed", error=str(e))
            return None

    async def _call_routed(self, transcript: str, ultratalk: bool = False) -> Optional[tuple[str, str]]:
        """Call LLM through the intelligence router.

        Routes between local (Ollama) and cloud (Claude) based on
        query complexity and explicit user triggers.

        Args:
            transcript: User transcript.
            ultratalk: If True, request verbose responses.

        Returns:
            Tuple of (response_text, source) or None.
            Source is "local" or "cloud".
        """
        try:
            # Build system prompt using prompt builder
            system_prompt = None
            if self._prompt_builder:
                prompt = self._prompt_builder.build(user_input=transcript)
                system_prompt = prompt.content

            # Add conciseness instruction unless ULTRATALK mode
            if not ultratalk:
                system_prompt = self._add_conciseness_instruction(system_prompt)
            else:
                # For ULTRATALK, add verbose instruction
                system_prompt = self._add_verbose_instruction(system_prompt)

            # LOG: Sending prompt through router
            self.logger.info(
                "llm_routing",
                transcript=transcript,
                ultratalk=ultratalk,
            )

            response = await self._router.route(
                prompt=transcript,
                system_prompt=system_prompt,
            )

            # LOG: Response received with source
            response_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
            self.logger.info(
                "llm_routed_response",
                source=response.source,
                preview=response_preview,
                latency=f"{response.latency:.2f}s",
            )

            # Track last source for UI display
            self._last_llm_source = response.source

            return (response.text, response.source)

        except Exception as e:
            self.logger.error("routed_call_failed", error=str(e))
            return None

    def _add_conciseness_instruction(self, system_prompt: Optional[str]) -> str:
        """Add instruction to keep responses concise for TTS.

        Args:
            system_prompt: Existing system prompt.

        Returns:
            System prompt with conciseness instruction added.
        """
        conciseness_instruction = """
IMPORTANT: Keep your response VERY SHORT - 1-2 sentences maximum.
This is a voice assistant - responses are spoken aloud via TTS.
Be direct and concise. Do not use bullet points, lists, or formatting.
Start with the most important information. Skip pleasantries.
"""
        if system_prompt:
            return system_prompt + "\n\n" + conciseness_instruction
        return conciseness_instruction

    def _add_verbose_instruction(self, system_prompt: Optional[str]) -> str:
        """Add instruction for verbose ULTRATALK mode responses.

        Args:
            system_prompt: Existing system prompt.

        Returns:
            System prompt with verbose instruction added.
        """
        verbose_instruction = """
ULTRATALK MODE ACTIVATED: The user has requested a detailed, verbose response.
Provide comprehensive information with full explanations.
You may use longer responses, but still speak naturally for TTS output.
Avoid bullet points and markdown - use flowing sentences instead.
"""
        if system_prompt:
            return system_prompt + "\n\n" + verbose_instruction
        return verbose_instruction

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
            return "HARO has nothing to repeat."

        elif command == "louder":
            return "HARO increased volume."

        elif command == "quieter":
            return "HARO decreased volume."

        elif command == "mute":
            return "HARO is muted."

        elif command == "unmute":
            return "HARO is unmuted."

        elif command == "pause":
            return "HARO is listening."

        elif command == "sleep":
            return "HARO is going to sleep. Say the wake word to wake HARO up."

        elif command == "goodbye":
            return "HARO says goodbye!"

        elif command == "new_conversation":
            if self._api:
                self._api.clear_history()
            return "HARO is starting a new conversation."

        elif command == "time":
            from datetime import datetime
            now = datetime.now()
            return f"HARO says the time is {now.strftime('%I:%M %p')}."

        elif command == "date":
            from datetime import datetime
            now = datetime.now()
            return f"HARO says today is {now.strftime('%A, %B %d, %Y')}."

        elif command == "status":
            return self._get_status_response()

        elif command == "help":
            return self._get_help_response()

        return None

    def _get_status_response(self) -> str:
        """Get status information."""
        return (
            f"HARO is operational. "
            f"HARO has processed {self.stats.requests} requests. "
            f"HARO has made {self.stats.api_calls} API calls. "
            f"HARO has encountered {self.stats.errors} errors."
        )

    def _get_help_response(self) -> str:
        """Get help information."""
        return (
            "HARO can respond to: "
            "stop, repeat, louder, quieter, mute, unmute, "
            "time, date, status, new conversation, sleep, and goodbye."
        )
