"""HARO agent state machine.

The core agent loop that manages state transitions between:
- PASSIVE: Listening for wake word
- ACTIVE: Recording user speech
- PROCESSING: API call in progress
- SPEAKING: TTS playback
- INTERRUPTED: Wake word detected during speech
- SLEEPING: Low-power mode
- ERROR: Error state with recovery
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Awaitable, Any

import numpy as np

from haro.core.config import HaroConfig
from haro.core.events import (
    EventBus,
    EventType,
    Event,
    StateChangeEvent,
    WakeWordEvent,
    TranscriptionEvent,
)
from haro.core.cache import ResponseCache, CacheConfig
from haro.audio.vad import VADState
from haro.utils.logging import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent operating states."""

    PASSIVE = auto()      # Listening for wake word
    ACTIVE = auto()       # Recording user speech
    PROCESSING = auto()   # API call in progress
    SPEAKING = auto()     # TTS playback
    INTERRUPTED = auto()  # Wake word during speech
    SLEEPING = auto()     # Low-power mode
    ERROR = auto()        # Error state


@dataclass
class StateContext:
    """Context passed between state handlers."""

    transcript: Optional[str] = None
    response: Optional[str] = None
    error: Optional[Exception] = None
    session_id: Optional[str] = None
    wake_confidence: float = 0.0
    speech_start_time: float = 0.0
    had_speech: bool = False  # Track if speech was detected during ACTIVE state
    last_activity_time: float = field(default_factory=time.time)
    is_local_command: bool = False  # Track if response is from local command vs LLM


@dataclass
class AgentStats:
    """Runtime statistics for the agent."""

    state_transitions: int = 0
    wake_detections: int = 0
    transcriptions: int = 0
    api_calls: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        """Get agent uptime in seconds."""
        return time.time() - self.start_time


# Type alias for state handler callbacks
StateHandler = Callable[[], Awaitable[Optional[AgentState]]]


class HaroAgent:
    """Main agent state machine for HARO.

    Manages the core voice assistant loop including:
    - Wake word detection in PASSIVE state
    - Speech recording in ACTIVE state
    - API calls in PROCESSING state
    - TTS playback in SPEAKING state
    - Interrupt handling when wake word detected during speech
    """

    def __init__(
        self,
        config: HaroConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize the agent.

        Args:
            config: HARO configuration.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.state = AgentState.PASSIVE
        self.context = StateContext()
        self.stats = AgentStats()

        # Component references (set during initialize)
        self._audio_capture = None
        self._audio_playback = None
        self._wake_detector = None
        self._continuous_wake = None
        self._vad = None
        self._stt = None
        self._tts = None
        self._feedback = None
        self._api_client = None
        self._context_manager = None
        self._prompt_builder = None
        self._response_parser = None

        # Response cache
        self._cache = ResponseCache(CacheConfig())
        self._last_response: Optional[str] = None
        self._last_transcript: Optional[str] = None

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._interrupt_event = asyncio.Event()

        # Timeouts
        self._active_timeout = config.vad.max_speech_duration
        self._silence_timeout = config.context.session_timeout
        self._processing_timeout = config.api.timeout

        # Thinking delay threshold (seconds before playing "thinking" phrase)
        self._thinking_delay_threshold = 2.5  # Default; can be configured

        # Wake word detection state
        self._passive_chunk_count = 0

        self.logger = logger.bind(component="HaroAgent")

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    @property
    def current_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    async def initialize(
        self,
        audio_capture=None,
        audio_playback=None,
        wake_detector=None,
        vad=None,
        stt=None,
        tts=None,
        feedback=None,
        api_client=None,
        context_manager=None,
        prompt_builder=None,
        response_parser=None,
    ) -> None:
        """Initialize the agent with component dependencies.

        Components are passed in to allow for dependency injection and testing.
        """
        self._audio_capture = audio_capture
        self._audio_playback = audio_playback
        self._wake_detector = wake_detector
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._feedback = feedback
        self._api_client = api_client
        self._context_manager = context_manager
        self._prompt_builder = prompt_builder
        self._response_parser = response_parser

        # Initialize wake detector with STT if both are provided
        if self._wake_detector and self._stt:
            await self._wake_detector.initialize(self._stt)

        # Initialize API client if provided
        if self._api_client and hasattr(self._api_client, 'initialize'):
            await self._api_client.initialize()

        self.logger.info(
            "agent_initialized",
            has_capture=audio_capture is not None,
            has_playback=audio_playback is not None,
            has_wake=wake_detector is not None,
            has_stt=stt is not None,
            has_tts=tts is not None,
            has_api=api_client is not None,
        )

    async def run(self) -> None:
        """Run the main agent loop.

        Continuously processes states until shutdown is requested.
        """
        self.logger.info("agent_run_called")

        if self._running:
            self.logger.warning("agent_already_running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats = AgentStats()

        self.logger.info("agent_starting")

        # Start audio capture if available
        if self._audio_capture:
            self.logger.info("starting_audio_capture")
            await self._audio_capture.start()
            self.logger.info("audio_capture_started")
        else:
            self.logger.warning("no_audio_capture_component")

        # Start audio playback if available
        if self._audio_playback:
            self.logger.info("starting_audio_playback")
            await self._audio_playback.start()
            self.logger.info("audio_playback_started")
        else:
            self.logger.warning("no_audio_playback_component")

        self.logger.info("entering_main_loop", running=self._running, shutdown_set=self._shutdown_event.is_set())

        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    await self._process_state()
                except asyncio.CancelledError:
                    self.logger.info("agent_cancelled")
                    break
                except Exception as e:
                    await self._handle_error(e)

        finally:
            self._running = False
            await self._cleanup()
            self.logger.info(
                "agent_stopped",
                uptime=self.stats.uptime,
                transitions=self.stats.state_transitions,
                wake_detections=self.stats.wake_detections,
            )

    async def stop(self) -> None:
        """Request graceful shutdown of the agent."""
        self.logger.info("agent_stop_requested")
        self._running = False
        self._shutdown_event.set()

    async def interrupt(self) -> None:
        """Signal an interrupt (e.g., wake word during speech)."""
        self._interrupt_event.set()

    async def _process_state(self) -> None:
        """Process current state and handle transitions."""
        handlers: dict[AgentState, StateHandler] = {
            AgentState.PASSIVE: self._handle_passive,
            AgentState.ACTIVE: self._handle_active,
            AgentState.PROCESSING: self._handle_processing,
            AgentState.SPEAKING: self._handle_speaking,
            AgentState.INTERRUPTED: self._handle_interrupted,
            AgentState.SLEEPING: self._handle_sleeping,
            AgentState.ERROR: self._handle_error_state,
        }

        handler = handlers.get(self.state)
        if handler:
            new_state = await handler()
            if new_state and new_state != self.state:
                await self._transition_to(new_state)

    async def _transition_to(self, new_state: AgentState) -> None:
        """Transition to a new state.

        Args:
            new_state: The state to transition to.
        """
        old_state = self.state
        self.state = new_state
        self.stats.state_transitions += 1

        self.logger.info(
            "state_transition",
            from_state=old_state.name,
            to_state=new_state.name,
        )

        # Publish state change event
        await self.event_bus.publish(
            StateChangeEvent(
                previous_state=old_state.name,
                new_state=new_state.name,
            )
        )

        # Update activity time
        self.context.last_activity_time = time.time()

    async def _handle_passive(self) -> Optional[AgentState]:
        """Handle PASSIVE state - listening for wake word.

        Returns:
            Next state or None to stay in current state.
        """
        if not self._audio_capture or not self._wake_detector:
            # Wait briefly if components not ready
            await asyncio.sleep(0.1)
            return None

        # Read audio chunk
        chunk = await self._audio_capture.read_chunk(timeout=0.1)
        if chunk is None:
            return None

        # Add to wake detector buffer
        self._wake_detector.add_audio(chunk)

        # Increment passive check counter
        self._passive_chunk_count += 1

        # Check for wake word periodically (every 8 chunks ~= 0.5 seconds at 1024 samples/16kHz)
        if self._passive_chunk_count >= 8:
            self._passive_chunk_count = 0

            energy = self._wake_detector.buffer.get_energy()
            self.logger.info(
                "checking_wake_word",
                buffer_energy=energy,
                energy_threshold=self._wake_detector._energy_threshold,
                is_initialized=self._wake_detector.is_initialized,
            )

            result = await self._wake_detector.detect()

            if result.detected:
                self.stats.wake_detections += 1
                self.context.wake_confidence = result.confidence

                self.logger.info(
                    "wake_word_detected",
                    confidence=result.confidence,
                    text=result.text,
                )

                # Clear buffer to prevent re-detecting same audio
                self._wake_detector.clear_buffer()

                # Play confirmation feedback
                if self._feedback:
                    await self._feedback.play_wake_confirmation()

                return AgentState.ACTIVE

        return None

    async def _handle_active(self) -> Optional[AgentState]:
        """Handle ACTIVE state - recording user speech.

        Returns:
            Next state or None to stay in current state.
        """
        if not self._audio_capture or not self._vad:
            self.logger.error("missing_components_for_active_state")
            return AgentState.ERROR

        # Initialize speech recording if just entered state
        if self.context.speech_start_time == 0:
            self.context.speech_start_time = time.time()
            self.context.had_speech = False
            self._vad.reset()

        # Read audio chunk
        chunk = await self._audio_capture.read_chunk(timeout=0.1)
        if chunk is None:
            return None

        # Also feed to wake detector for interrupt detection
        if self._wake_detector:
            self._wake_detector.add_audio(chunk)

        # Process through VAD
        result = await self._vad.process(chunk, time.time())

        # Log VAD state periodically (every ~0.5 seconds)
        if int(time.time() - self.context.speech_start_time) % 1 == 0:
            elapsed = time.time() - self.context.speech_start_time
            if int(elapsed * 10) % 5 == 0:  # Every 0.5 seconds
                self.logger.debug(
                    "active_vad_state",
                    elapsed=f"{elapsed:.1f}s",
                    energy=result.energy,
                    is_speech=result.is_speech,
                    vad_state=result.state.value,
                    had_speech=self.context.had_speech,
                )

        # Track if we've seen any speech
        if result.is_speech:
            self.context.had_speech = True

        # Check for interrupt (wake word during recording)
        if self._interrupt_event.is_set():
            self._interrupt_event.clear()
            return AgentState.INTERRUPTED

        # Check for end of speech: VAD returned to SILENCE after we had speech
        if result.state == VADState.SILENCE and self.context.had_speech:
            # Speech ended, time to process
            self.context.transcript = await self._transcribe_speech()
            self.context.speech_start_time = 0

            if self.context.transcript:
                self.stats.transcriptions += 1
                return AgentState.PROCESSING
            else:
                # No transcript, return to passive
                return AgentState.PASSIVE

        # Check for timeout
        elapsed = time.time() - self.context.speech_start_time
        if elapsed > self._active_timeout:
            self.logger.warning("speech_timeout", duration=elapsed, had_speech=self.context.had_speech)
            self.context.speech_start_time = 0
            # Play error feedback so user knows we didn't hear them
            if self._feedback:
                await self._feedback.play_error()
            return AgentState.PASSIVE

        return None

    async def _handle_processing(self) -> Optional[AgentState]:
        """Handle PROCESSING state - API call in progress.

        Pipeline:
        1. Play transcription confirmation ("Got that, HARO.")
        2. Check for local commands
        3. Start API call with delay-triggered thinking phrase
        4. Return response for speaking

        Returns:
            Next state or None to stay in current state.
        """
        if not self.context.transcript:
            self.logger.error("no_transcript_for_processing")
            return AgentState.PASSIVE

        # Play simple transcription confirmation with single HARO signoff
        if self._feedback:
            await self._feedback.play_transcription_confirmation(wait=True)

        # Check for local commands first
        command = self._check_local_command(self.context.transcript)
        if command:
            self.context.is_local_command = True  # Mark as local command
            response = await self._handle_local_command(command)
            if response:
                self.context.response = response
                return AgentState.SPEAKING
            else:
                return AgentState.PASSIVE

        # Make API call with delay-triggered thinking phrase
        self.context.is_local_command = False  # Mark as LLM response
        if self._api_client:
            try:
                self.stats.api_calls += 1

                # Create task for delayed thinking phrase
                thinking_task = asyncio.create_task(self._play_delayed_thinking())

                try:
                    response = await asyncio.wait_for(
                        self._call_api(self.context.transcript),
                        timeout=self._processing_timeout,
                    )
                finally:
                    # Cancel thinking task if still pending (API responded quickly)
                    thinking_task.cancel()
                    try:
                        await thinking_task
                    except asyncio.CancelledError:
                        pass

                self.context.response = response

                if response:
                    return AgentState.SPEAKING
                else:
                    return AgentState.PASSIVE

            except asyncio.TimeoutError:
                self.logger.error("api_timeout")
                self.context.error = Exception("API timeout")
                return AgentState.ERROR

            except Exception as e:
                self.logger.error("api_error", error=str(e))
                self.context.error = e
                return AgentState.ERROR
        else:
            # No API client, just echo for testing
            self.context.response = f"I heard: {self.context.transcript}"
            return AgentState.SPEAKING

    async def _handle_speaking(self) -> Optional[AgentState]:
        """Handle SPEAKING state - TTS playback.

        Uses double signoff (HARO HARO) for LLM responses,
        single signoff (HARO) for local commands.

        Returns:
            Next state or None to stay in current state.
        """
        if not self.context.response:
            return AgentState.PASSIVE

        # Determine signoff type based on response source
        # Local commands get single HARO, LLM responses get double HARO HARO
        use_double_signoff = not self.context.is_local_command

        # Add sign-off to response
        response_with_signoff = self.context.response
        if self._feedback:
            response_with_signoff = self._feedback.add_signoff(
                self.context.response,
                signoff=self.config.wake.phrase,  # Use wake phrase as sign-off
                double=use_double_signoff,
            )

        # Store for repeat command (with sign-off)
        self._last_response = response_with_signoff
        self._last_transcript = self.context.transcript

        # Log the turn to context manager
        if self._context_manager and self.context.transcript:
            self._context_manager.log_turn(
                user_input=self.context.transcript,
                response=response_with_signoff,
            )

        if self._tts and self._audio_playback:
            try:
                # Synthesize speech (with sign-off)
                result = await self._tts.synthesize(response_with_signoff)

                # Play with interrupt detection
                await self._play_with_interrupt(result.audio, result.sample_rate)

                # Check if interrupted
                if self._interrupt_event.is_set():
                    self._interrupt_event.clear()
                    return AgentState.INTERRUPTED

            except Exception as e:
                self.logger.error("tts_error", error=str(e))

        # Clear response and reset local command flag
        self.context.response = None
        self.context.transcript = None
        self.context.is_local_command = False

        return AgentState.PASSIVE

    async def _handle_interrupted(self) -> Optional[AgentState]:
        """Handle INTERRUPTED state - wake word during speech.

        Returns:
            Next state (ACTIVE) to record new input.
        """
        self.logger.info("handling_interrupt")

        # Stop any ongoing playback
        if self._audio_playback:
            await self._audio_playback.stop()
            await self._audio_playback.start()

        # Play acknowledgment
        if self._feedback:
            await self._feedback.play_wake_confirmation()

        # Clear context for new interaction
        self.context.response = None
        self.context.transcript = None
        self.context.speech_start_time = 0
        self.context.had_speech = False

        return AgentState.ACTIVE

    async def _handle_sleeping(self) -> Optional[AgentState]:
        """Handle SLEEPING state - low power mode.

        Returns:
            Next state or None to stay in current state.
        """
        # In sleep mode, only wake on explicit wake word
        if self._audio_capture and self._wake_detector:
            chunk = await self._audio_capture.read_chunk(timeout=0.5)
            if chunk is not None:
                self._wake_detector.add_audio(chunk)
                result = await self._wake_detector.detect()

                if result.detected:
                    self.logger.info("waking_from_sleep")
                    if self._feedback:
                        await self._feedback.play_ready()
                    return AgentState.PASSIVE

        return None

    async def _handle_error_state(self) -> Optional[AgentState]:
        """Handle ERROR state - recovery logic.

        Returns:
            Next state after recovery attempt.
        """
        self.stats.errors += 1

        # Log error
        if self.context.error:
            self.logger.error("in_error_state", error=str(self.context.error))

        # Play error feedback
        if self._feedback:
            await self._feedback.play_error()

        # Clear error context
        self.context.error = None
        self.context.response = None
        self.context.transcript = None
        self.context.had_speech = False

        # Wait briefly before recovery
        await asyncio.sleep(1.0)

        return AgentState.PASSIVE

    async def _handle_error(self, error: Exception) -> None:
        """Handle errors during state processing.

        Args:
            error: The exception that occurred.
        """
        self.logger.error("state_processing_error", error=str(error))
        self.context.error = error
        await self._transition_to(AgentState.ERROR)

    async def _transcribe_speech(self) -> Optional[str]:
        """Transcribe recorded speech.

        Returns:
            Transcribed text or None.
        """
        if not self._stt:
            return None

        # Get audio from VAD buffer
        if self._vad and hasattr(self._vad, 'get_speech_buffer'):
            audio = self._vad.get_speech_buffer()
        else:
            # Fallback: use last few seconds from capture
            audio = None

        if audio is None or len(audio) == 0:
            return None

        try:
            result = await self._stt.transcribe(
                audio,
                sample_rate=self.config.audio.sample_rate,
            )
            return result.text.strip()
        except Exception as e:
            self.logger.error("transcription_error", error=str(e))
            return None

    def _strip_wake_word(self, transcript: str) -> str:
        """Strip wake word from the beginning of transcript.

        Args:
            transcript: The transcribed text potentially starting with wake word.

        Returns:
            Transcript with wake word prefix removed.
        """
        text = transcript.strip()
        wake_phrase = self.config.wake.phrase.lower()

        # Try various common prefixes
        prefixes = [
            f"{wake_phrase}, ",
            f"{wake_phrase} ",
            f"{wake_phrase}. ",
            f"{wake_phrase}? ",
            f"{wake_phrase}! ",
            wake_phrase,
        ]

        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                # Preserve original case of remaining text
                return text[len(prefix):].strip()

        return text

    async def _call_api(self, transcript: str) -> Optional[str]:
        """Call the Claude API with the transcript.

        Uses caching to avoid repeated API calls for similar questions.

        Args:
            transcript: User's transcribed speech.

        Returns:
            API response text or None.
        """
        if not self._api_client:
            # No API client, return echo for testing
            return f"I heard you say: {transcript}"

        # Strip wake word from transcript before sending to API
        clean_transcript = self._strip_wake_word(transcript)
        self.logger.debug("stripped_wake_word", original=transcript, clean=clean_transcript)

        # Check cache first (use clean transcript for cache key)
        cached = self._cache.get(clean_transcript)
        if cached:
            self.logger.info("cache_hit", transcript=clean_transcript[:30])
            return cached

        # Note: Thinking phrase is now handled by delay-triggered task in _handle_processing()

        try:
            # Build system prompt if we have a prompt builder
            system_prompt = None
            if self._prompt_builder:
                prompt = self._prompt_builder.build(user_input=clean_transcript)
                system_prompt = prompt.content

            # Make API call
            response = await self._api_client.complete(
                user_input=clean_transcript,
                system_prompt=system_prompt,
            )

            # Parse response if we have a parser
            if self._response_parser:
                parsed = self._response_parser.parse(response.text)
                result = parsed.speech_text
            else:
                result = response.text

            # Cache the response
            if result:
                self._cache.put(clean_transcript, result)

            return result

        except Exception as e:
            self.logger.error("api_call_failed", error=str(e))
            return None

    async def _play_with_interrupt(
        self, audio: np.ndarray, sample_rate: int
    ) -> None:
        """Play audio with capture paused to avoid self-detection.

        Pauses audio capture during playback to prevent detecting our own
        speech as a wake word. Resumes capture after playback completes.

        Args:
            audio: Audio samples to play.
            sample_rate: Audio sample rate.
        """
        if not self._audio_playback:
            return

        # Pause capture during playback to avoid detecting our own speech
        if self._audio_capture:
            self._audio_capture.pause()

        try:
            # Play audio and wait for completion
            await self._audio_playback.play(audio, sample_rate=sample_rate, wait=True)
        finally:
            # Always resume capture after playback
            if self._audio_capture:
                self._audio_capture.resume()

    async def _play_delayed_thinking(self) -> None:
        """Play thinking phrase after delay threshold.

        This method waits for the configured delay threshold, then plays
        the thinking phrase if still in PROCESSING state. Should be run
        as a task that can be cancelled if API responds quickly.
        """
        try:
            await asyncio.sleep(self._thinking_delay_threshold)
            # Only play if we're still in PROCESSING state
            if self.state == AgentState.PROCESSING and self._feedback:
                await self._feedback.play_thinking(wait=True)
        except asyncio.CancelledError:
            # API completed before threshold - don't play thinking phrase
            pass

    def _check_local_command(self, transcript: str) -> Optional[str]:
        """Check if transcript is a local command.

        Args:
            transcript: The transcribed text.

        Returns:
            Command name if detected, None otherwise.
        """
        text = transcript.lower().strip()
        wake_phrase = self.config.wake.phrase.lower()

        # Remove wake phrase prefix if present
        for prefix in [f"{wake_phrase}, ", f"{wake_phrase} "]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        # Command patterns - order matters, check more specific patterns first
        commands = [
            # Volume controls
            ("quieter", ["quieter", "volume down", "softer", "lower volume"]),
            ("louder", ["louder", "volume up", "speak up", "raise volume"]),
            ("mute", ["mute", "silence"]),
            ("unmute", ["unmute", "unsilence"]),
            # Playback controls
            ("stop", ["stop", "shut up", "be quiet", "cancel"]),
            ("repeat", ["repeat", "say that again", "repeat that", "what did you say"]),
            ("pause", ["pause", "hold on", "wait"]),
            # Session controls
            ("sleep", ["go to sleep", "sleep mode", "sleep"]),
            ("goodbye", ["goodbye", "bye", "see you", "exit", "quit"]),
            ("new_conversation", ["new conversation", "start over", "reset", "clear history"]),
            # Information queries
            ("time", ["what time is it", "what's the time", "current time"]),
            ("date", ["what's the date", "what day is it", "today's date"]),
            ("status", ["status", "how are you", "system status"]),
            ("help", ["help", "what can you do", "commands", "help me"]),
            # Cache controls
            ("clear_cache", ["clear cache", "forget responses"]),
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
            if self._audio_playback:
                await self._audio_playback.stop()
            return None

        elif command == "repeat":
            # Return last response if available
            if self._last_response:
                return self._last_response
            return "I don't have anything to repeat."

        elif command == "louder":
            if self._audio_playback:
                self._audio_playback.volume = min(1.0, self._audio_playback.volume + 0.2)
            return "Volume increased."

        elif command == "quieter":
            if self._audio_playback:
                self._audio_playback.volume = max(0.1, self._audio_playback.volume - 0.2)
            return "Volume decreased."

        elif command == "mute":
            if self._audio_playback:
                self._audio_playback.volume = 0.0
            return "Muted."

        elif command == "unmute":
            if self._audio_playback:
                self._audio_playback.volume = 0.7
            return "Unmuted."

        elif command == "pause":
            # Brief pause before next command
            return "I'm listening."

        elif command == "sleep":
            await self._transition_to(AgentState.SLEEPING)
            return "Going to sleep. Say the wake word to wake me up."

        elif command == "goodbye":
            if self._feedback:
                await self._feedback.play_goodbye()
            return "Goodbye!"

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

        elif command == "new_conversation":
            self.context = StateContext()
            if self._api_client:
                self._api_client.clear_history()
            return "Starting a new conversation."

        elif command == "clear_cache":
            count = self._cache.invalidate()
            return f"Cleared {count} cached responses."

        return None

    def _get_status_response(self) -> str:
        """Get status information as spoken response.

        Returns:
            Status response text.
        """
        uptime_mins = int(self.stats.uptime / 60)

        parts = [
            "I'm running well.",
            f"I've been active for {uptime_mins} minutes." if uptime_mins > 0 else "Just started.",
            f"I've had {self.stats.wake_detections} conversations.",
        ]

        if self._cache.size > 0:
            parts.append(f"Cache has {self._cache.size} responses.")

        return " ".join(parts)

    def _get_help_response(self) -> str:
        """Get help information as spoken response.

        Returns:
            Help response text.
        """
        return (
            "I can help with questions, tasks, and conversation. "
            "Say stop to cancel, repeat to hear my last response, "
            "or goodbye when you're done. "
            "You can also ask me the time, date, or status."
        )

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self._audio_capture:
            await self._audio_capture.stop()

        if self._audio_playback:
            await self._audio_playback.stop()

    def go_to_sleep(self) -> None:
        """Put agent into sleep mode."""
        asyncio.create_task(self._transition_to(AgentState.SLEEPING))

    def wake_up(self) -> None:
        """Wake agent from sleep mode."""
        if self.state == AgentState.SLEEPING:
            asyncio.create_task(self._transition_to(AgentState.PASSIVE))
