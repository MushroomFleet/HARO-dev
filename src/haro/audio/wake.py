"""Wake word detection for HARO.

Detects the wake word "HARO" using Whisper on a rolling audio buffer.
Uses a combination of energy detection and transcription to minimize
CPU usage while maintaining responsiveness.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable

import numpy as np

from haro.core.config import WakeConfig, AudioConfig
from haro.core.events import EventBus, EventType, WakeWordEvent
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WakeWordResult:
    """Result of wake word detection."""

    detected: bool
    confidence: float
    timestamp: float
    text: str = ""


class RollingBuffer:
    """Rolling audio buffer for continuous wake word detection.

    Maintains a fixed-duration buffer of audio samples that is
    continuously updated with new audio chunks.
    """

    def __init__(self, duration: float, sample_rate: int = 16000) -> None:
        """Initialize rolling buffer.

        Args:
            duration: Buffer duration in seconds.
            sample_rate: Audio sample rate.
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self._size = int(duration * sample_rate)
        self._buffer = np.zeros(self._size, dtype=np.float32)
        self._position = 0
        self._total_samples = 0

    def add(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer.

        Args:
            chunk: Audio samples to add.
        """
        chunk = chunk.astype(np.float32).flatten()
        chunk_len = len(chunk)

        if chunk_len >= self._size:
            # Chunk is larger than buffer, take last part
            self._buffer[:] = chunk[-self._size:]
            self._position = 0
        else:
            # Roll buffer and add chunk at end
            self._buffer = np.roll(self._buffer, -chunk_len)
            self._buffer[-chunk_len:] = chunk

        self._total_samples += chunk_len

    def get(self) -> np.ndarray:
        """Get current buffer contents.

        Returns:
            Buffer contents as numpy array.
        """
        return self._buffer.copy()

    def get_energy(self) -> float:
        """Get RMS energy of buffer.

        Returns:
            RMS energy value.
        """
        return float(np.sqrt(np.mean(self._buffer**2)))

    def clear(self) -> None:
        """Clear buffer contents."""
        self._buffer.fill(0)
        self._position = 0

    @property
    def is_filled(self) -> bool:
        """Check if buffer has been filled at least once."""
        return self._total_samples >= self._size


class WakeWordDetector:
    """Detect wake word using Whisper on rolling buffer.

    Continuously monitors audio for the wake phrase using a rolling
    buffer and periodic transcription. Uses energy detection to
    avoid unnecessary transcription during silence.
    """

    def __init__(
        self,
        config: WakeConfig,
        audio_config: Optional[AudioConfig] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize wake word detector.

        Args:
            config: Wake word configuration.
            audio_config: Optional audio configuration for sample rate.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.event_bus = event_bus

        self.wake_phrase = config.phrase.lower().strip()
        self.sensitivity = config.sensitivity

        # Buffer configuration
        sample_rate = audio_config.sample_rate if audio_config else 16000
        buffer_duration = (
            audio_config.buffer_duration if audio_config else 2.0
        )
        self.buffer = RollingBuffer(buffer_duration, sample_rate)
        self.sample_rate = sample_rate

        # STT reference (set via initialize)
        self._stt = None
        self._initialized = False

        # Detection state
        self._last_detection_time = 0.0
        self._cooldown = 2.0  # Seconds between detections
        self._energy_threshold = 0.01  # Minimum energy for detection
        self._check_interval = 0.5  # Seconds between checks

        # Statistics
        self._detection_count = 0
        self._false_positive_count = 0

        self.logger = logger.bind(component="WakeWordDetector")

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized with STT."""
        return self._initialized

    async def initialize(self, stt) -> None:
        """Initialize detector with STT instance.

        Args:
            stt: WhisperSTT instance to use for transcription.
        """
        self._stt = stt
        self._initialized = True
        self.logger.info(
            "wake_detector_initialized",
            wake_phrase=self.wake_phrase,
            sensitivity=self.sensitivity,
            buffer_duration=self.buffer.duration,
        )

    def add_audio(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer.

        Args:
            chunk: Audio samples to add.
        """
        self.buffer.add(chunk)

    async def detect(self) -> WakeWordResult:
        """Check buffer for wake word.

        Returns:
            WakeWordResult with detection status.
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self._last_detection_time < self._cooldown:
            self.logger.debug("detection_skipped_cooldown")
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=current_time,
            )

        # Quick energy check
        energy = self.buffer.get_energy()
        if energy < self._energy_threshold:
            self.logger.debug(
                "detection_skipped_low_energy",
                energy=energy,
                threshold=self._energy_threshold,
            )
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=current_time,
            )

        # Ensure initialized
        if not self._initialized or self._stt is None:
            self.logger.warning("detector_not_initialized")
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=current_time,
            )

        # Get buffer and transcribe
        audio = self.buffer.get()

        try:
            result = await self._stt.transcribe(
                audio,
                sample_rate=self.sample_rate,
                prompt=self.wake_phrase.upper(),  # Bias transcription
            )

            text = result.text.lower().strip()

            # Check for wake phrase
            if self._contains_wake_phrase(text):
                confidence = self._calculate_confidence(text, result.confidence)

                if confidence >= self.sensitivity:
                    self._last_detection_time = current_time
                    self._detection_count += 1

                    self.logger.info(
                        "wake_word_detected",
                        text=text,
                        confidence=confidence,
                        energy=energy,
                    )

                    # Publish event
                    if self.event_bus:
                        await self.event_bus.publish(
                            WakeWordEvent(
                                confidence=confidence,
                                phrase=self.wake_phrase,
                            )
                        )

                    return WakeWordResult(
                        detected=True,
                        confidence=confidence,
                        timestamp=current_time,
                        text=text,
                    )

            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=current_time,
                text=text,
            )

        except Exception as e:
            self.logger.error("detection_failed", error=str(e))
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                timestamp=current_time,
            )

    def _contains_wake_phrase(self, text: str) -> bool:
        """Check if text contains wake phrase.

        Handles variations like "haro", "hero", "harrow", etc.

        Args:
            text: Transcribed text to check.

        Returns:
            True if wake phrase is present.
        """
        # Direct match
        if self.wake_phrase in text:
            return True

        # Common Whisper mishearings for "HARO"
        variants = [
            "haro",
            "hero",
            "harrow",
            "harro",
            "hallow",
            "hello",  # Sometimes mishears as hello
            "hay row",
            "hey ro",
        ]

        for variant in variants:
            if variant in text:
                return True

        return False

    def _calculate_confidence(
        self, text: str, transcription_confidence: float
    ) -> float:
        """Calculate detection confidence.

        Args:
            text: Transcribed text.
            transcription_confidence: Confidence from transcription.

        Returns:
            Calculated confidence score (0.0 to 1.0).
        """
        base_conf = transcription_confidence

        # Exact match bonus
        if text == self.wake_phrase:
            return min(1.0, base_conf + 0.3)

        # Starts with wake phrase bonus
        if text.startswith(self.wake_phrase):
            return min(1.0, base_conf + 0.2)

        # Wake phrase at end (common pattern: "hey haro")
        if text.endswith(self.wake_phrase):
            return min(1.0, base_conf + 0.15)

        # Contains exact phrase
        if self.wake_phrase in text:
            return min(1.0, base_conf + 0.1)

        # Variant match (lower confidence)
        return max(0.0, base_conf - 0.1)

    def reset_cooldown(self) -> None:
        """Reset detection cooldown to allow immediate detection."""
        self._last_detection_time = 0.0

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self.buffer.clear()

    @property
    def detection_count(self) -> int:
        """Get total number of detections."""
        return self._detection_count

    @property
    def energy_threshold(self) -> float:
        """Get current energy threshold."""
        return self._energy_threshold

    @energy_threshold.setter
    def energy_threshold(self, value: float) -> None:
        """Set energy threshold."""
        self._energy_threshold = max(0.001, min(0.5, value))


class ContinuousWakeDetector:
    """Continuous wake word detection with automatic audio processing.

    Wraps WakeWordDetector with continuous monitoring loop that
    processes audio from capture and checks for wake word.
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        check_interval: float = 0.5,
    ) -> None:
        """Initialize continuous detector.

        Args:
            detector: WakeWordDetector instance.
            check_interval: Seconds between detection checks.
        """
        self.detector = detector
        self.check_interval = check_interval

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_wake: Optional[Callable[[WakeWordResult], Awaitable[None]]] = None

        self.logger = logger.bind(component="ContinuousWakeDetector")

    def set_wake_callback(
        self, callback: Callable[[WakeWordResult], Awaitable[None]]
    ) -> None:
        """Set callback for wake word detection.

        Args:
            callback: Async function called when wake word is detected.
        """
        self._on_wake = callback

    async def start(self) -> None:
        """Start continuous detection."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._detection_loop())
        self.logger.info("continuous_detection_started")

    async def stop(self) -> None:
        """Stop continuous detection."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self.logger.info("continuous_detection_stopped")

    async def _detection_loop(self) -> None:
        """Main detection loop."""
        while self._running:
            try:
                # Check for wake word
                result = await self.detector.detect()

                if result.detected and self._on_wake:
                    await self._on_wake(result)

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("detection_loop_error", error=str(e))
                await asyncio.sleep(1.0)  # Back off on error

    def add_audio(self, chunk: np.ndarray) -> None:
        """Add audio chunk to detector buffer.

        Args:
            chunk: Audio samples to add.
        """
        self.detector.add_audio(chunk)

    @property
    def is_running(self) -> bool:
        """Check if continuous detection is running."""
        return self._running
