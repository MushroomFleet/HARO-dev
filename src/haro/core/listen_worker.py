"""Continuous listening worker.

Handles audio capture, wake word detection, VAD, and transcription
as an independent async worker.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np

from haro.audio.capture import AudioCapture
from haro.audio.vad import VoiceActivityDetector, VADState
from haro.audio.wake import WakeWordDetector
from haro.speech.stt import WhisperSTT, TranscriptionResult
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ListenEvent:
    """Event from listen worker."""

    type: str  # "wake", "transcript", "error"
    data: Any = None


class ListenWorker:
    """Continuous listening worker.

    Runs independently, capturing audio and detecting:
    - Wake word ("HARO") in passive mode
    - Speech via VAD in active mode
    - Transcription via Whisper when speech ends
    """

    def __init__(
        self,
        capture: AudioCapture,
        vad: VoiceActivityDetector,
        wake_detector: WakeWordDetector,
        stt: WhisperSTT,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize listen worker.

        Args:
            capture: Audio capture instance.
            vad: Voice activity detector.
            wake_detector: Wake word detector.
            stt: Speech-to-text engine.
            sample_rate: Audio sample rate.
        """
        self._capture = capture
        self._vad = vad
        self._wake = wake_detector
        self._stt = stt
        self._sample_rate = sample_rate

        self._running = False
        self._passive = True  # True = listening for wake, False = recording speech
        self._had_speech = False
        self._paused = False

        # Output queue for transcripts
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()

        # Event callbacks
        self._on_wake_callback: Optional[Callable[[], Any]] = None
        self._on_transcript_callback: Optional[Callable[[str], Any]] = None

        # Chunk counter for periodic wake detection
        self._chunk_count = 0
        self._wake_check_interval = 8  # Check every 8 chunks (~0.5s)

        self._task: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="ListenWorker")

    async def start(self) -> None:
        """Start the listen worker."""
        if self._running:
            return

        self._running = True
        self._passive = True
        self._task = asyncio.create_task(self._run())
        self.logger.info("listen_worker_started")

    async def stop(self) -> None:
        """Stop the listen worker."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self.logger.info("listen_worker_stopped")

    def pause(self) -> None:
        """Pause listening (during TTS playback)."""
        self._paused = True
        self.logger.debug("listen_paused")

    def resume(self) -> None:
        """Resume listening."""
        self._paused = False
        # Reset to passive mode after pause
        self._passive = True
        self._had_speech = False
        self._vad.reset()
        self.logger.debug("listen_resumed")

    def set_active(self) -> None:
        """Set to active mode (recording speech)."""
        self._passive = False
        self._had_speech = False
        self._vad.reset()
        self.logger.debug("listen_active")

    def set_passive(self) -> None:
        """Set to passive mode (listening for wake word)."""
        self._passive = True
        self._had_speech = False
        self.logger.debug("listen_passive")

    def on_wake(self, callback: Callable[[], Any]) -> None:
        """Set callback for wake word detection."""
        self._on_wake_callback = callback

    def on_transcript(self, callback: Callable[[str], Any]) -> None:
        """Set callback for transcription complete."""
        self._on_transcript_callback = callback

    async def get_transcript(self, timeout: Optional[float] = None) -> Optional[str]:
        """Get next transcript from queue.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            Transcript string or None if timeout.
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._transcript_queue.get(), timeout=timeout
                )
            return await self._transcript_queue.get()
        except asyncio.TimeoutError:
            return None

    @property
    def is_passive(self) -> bool:
        """Check if in passive mode."""
        return self._passive

    @property
    def is_paused(self) -> bool:
        """Check if paused."""
        return self._paused

    async def _run(self) -> None:
        """Main worker loop."""
        self.logger.info("listen_worker_running")

        while self._running:
            try:
                # Read audio chunk
                chunk = await self._capture.read_chunk(timeout=0.1)
                if chunk is None:
                    continue

                # Skip processing if paused (TTS playing)
                if self._paused:
                    continue

                # Always feed to wake detector for interrupt detection
                self._wake.add_audio(chunk)

                if self._passive:
                    await self._handle_passive(chunk)
                else:
                    await self._handle_active(chunk)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("listen_error", error=str(e))

        self.logger.info("listen_worker_loop_exited")

    async def _handle_passive(self, chunk: np.ndarray) -> None:
        """Handle passive mode - listening for wake word."""
        self._chunk_count += 1

        # Check for wake word periodically
        if self._chunk_count >= self._wake_check_interval:
            self._chunk_count = 0

            result = await self._wake.detect()
            if result.detected:
                self.logger.info(
                    "wake_detected",
                    confidence=result.confidence,
                    text=result.text,
                )

                # Clear buffer to prevent re-detection
                self._wake.clear_buffer()

                # Switch to active mode
                self._passive = False
                self._had_speech = False
                self._vad.reset()

                # Fire callback
                if self._on_wake_callback:
                    try:
                        if asyncio.iscoroutinefunction(self._on_wake_callback):
                            await self._on_wake_callback()
                        else:
                            self._on_wake_callback()
                    except Exception as e:
                        self.logger.error("wake_callback_error", error=str(e))

    async def _handle_active(self, chunk: np.ndarray) -> None:
        """Handle active mode - recording user speech."""
        import time

        # Process through VAD
        result = await self._vad.process(chunk, time.time())

        # Track if we've heard speech
        if result.is_speech:
            self._had_speech = True

        # Check for end of speech
        if result.state == VADState.SILENCE and self._had_speech:
            # Get speech buffer and transcribe
            speech_audio = self._vad.get_speech_buffer()

            if speech_audio is not None and len(speech_audio) > 0:
                transcript = await self._transcribe(speech_audio)

                if transcript:
                    # Queue transcript
                    await self._transcript_queue.put(transcript)

                    # Fire callback
                    if self._on_transcript_callback:
                        try:
                            if asyncio.iscoroutinefunction(self._on_transcript_callback):
                                await self._on_transcript_callback(transcript)
                            else:
                                self._on_transcript_callback(transcript)
                        except Exception as e:
                            self.logger.error("transcript_callback_error", error=str(e))

            # Return to passive mode
            self._passive = True
            self._had_speech = False

    async def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio to text.

        Args:
            audio: Audio samples.

        Returns:
            Transcribed text or None.
        """
        try:
            result = await self._stt.transcribe(audio, sample_rate=self._sample_rate)

            if result.text and result.text.strip():
                self.logger.debug(
                    "transcription_complete",
                    text=result.text[:50],
                    confidence=result.confidence,
                )
                return result.text.strip()

        except Exception as e:
            self.logger.error("transcription_error", error=str(e))

        return None
