"""Voice Activity Detection (VAD) for HARO.

Provides energy-based voice activity detection with adaptive noise floor
and configurable thresholds. Uses a state machine to track speech segments.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from haro.core.config import VADConfig
from haro.core.events import EventBus, EventType, VADEvent
from haro.utils.logging import get_logger

logger = get_logger(__name__)


class VADState(Enum):
    """Voice activity detection states."""

    SILENCE = "silence"
    SPEECH = "speech"
    TRAILING = "trailing"


@dataclass
class VADResult:
    """Result from VAD processing."""

    is_speech: bool
    energy: float
    duration: float
    state: VADState


class VoiceActivityDetector:
    """Energy-based voice activity detector.

    Uses RMS energy with adaptive noise floor to detect speech.
    Implements a state machine for tracking speech segments with
    configurable durations for speech detection and silence trailing.
    """

    def __init__(
        self,
        config: VADConfig,
        sample_rate: int = 16000,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize VAD.

        Args:
            config: VAD configuration.
            sample_rate: Audio sample rate.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.sample_rate = sample_rate
        self.event_bus = event_bus

        # State tracking
        self.state = VADState.SILENCE
        self._speech_start: Optional[float] = None
        self._last_speech: Optional[float] = None
        self._current_time: float = 0.0

        # Adaptive noise floor
        self._noise_floor: float = 0.01
        self._energy_history: list[float] = []
        self._history_size: int = 50  # ~1 second at 50 chunks/sec

        # Speech buffer for accumulating audio during speech segments
        self._speech_buffer: list[np.ndarray] = []
        self._max_buffer_chunks: int = int(config.max_speech_duration * sample_rate / 1024)  # Rough estimate

        self.logger = logger.bind(component="VAD")

    def reset(self) -> None:
        """Reset VAD state."""
        self.state = VADState.SILENCE
        self._speech_start = None
        self._last_speech = None
        self._current_time = 0.0
        self._noise_floor = 0.01
        self._energy_history.clear()
        self._speech_buffer.clear()
        self.logger.debug("vad_reset")

    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            chunk: Audio samples as numpy array.

        Returns:
            RMS energy value.
        """
        # Ensure float type for calculation
        audio = chunk.astype(np.float32)
        return float(np.sqrt(np.mean(audio**2)))

    def _update_noise_floor(self, energy: float, is_speech: bool) -> None:
        """Update adaptive noise floor during silence.

        Args:
            energy: Current energy level.
            is_speech: Whether current frame is speech.
        """
        if not is_speech:
            # Only update during silence
            adaptation_rate = self.config.noise_floor_adaptation
            self._noise_floor = (
                1.0 - adaptation_rate
            ) * self._noise_floor + adaptation_rate * energy

            # Keep noise floor bounded
            self._noise_floor = max(0.001, min(0.1, self._noise_floor))

    def _is_speech_energy(self, energy: float) -> bool:
        """Check if energy level indicates speech.

        Args:
            energy: Current energy level.

        Returns:
            True if energy indicates speech.
        """
        threshold = self._noise_floor * (1.0 + self.config.threshold * 10)
        return energy > threshold

    async def process(
        self, chunk: np.ndarray, timestamp: Optional[float] = None
    ) -> VADResult:
        """Process an audio chunk and return VAD result.

        Args:
            chunk: Audio samples as numpy array.
            timestamp: Optional timestamp. If not provided, accumulated from chunks.

        Returns:
            VADResult with speech status and timing.
        """
        # Calculate timestamp from chunk size if not provided
        if timestamp is not None:
            self._current_time = timestamp
        else:
            chunk_duration = len(chunk) / self.sample_rate
            self._current_time += chunk_duration

        # Calculate energy
        energy = self._calculate_energy(chunk)
        self._energy_history.append(energy)
        if len(self._energy_history) > self._history_size:
            self._energy_history.pop(0)

        # Check if energy indicates speech
        is_speech_energy = self._is_speech_energy(energy)

        # Log periodically for debugging (every 50 chunks ~ 1 second)
        if len(self._energy_history) % 50 == 0:
            threshold = self._noise_floor * (1.0 + self.config.threshold * 10)
            self.logger.debug(
                "vad_status",
                energy=energy,
                threshold=threshold,
                noise_floor=self._noise_floor,
                is_speech=is_speech_energy,
                state=self.state.value,
            )

        # Update noise floor
        self._update_noise_floor(energy, is_speech_energy)

        # State machine
        previous_state = self.state

        if self.state == VADState.SILENCE:
            if is_speech_energy:
                self.state = VADState.SPEECH
                self._speech_start = self._current_time
                self._last_speech = self._current_time
                self._speech_buffer.clear()  # Start fresh buffer for new speech
                self.logger.debug(
                    "speech_started",
                    timestamp=self._current_time,
                    energy=energy,
                )

        elif self.state == VADState.SPEECH:
            if is_speech_energy:
                self._last_speech = self._current_time
            else:
                self.state = VADState.TRAILING

        elif self.state == VADState.TRAILING:
            if is_speech_energy:
                self.state = VADState.SPEECH
                self._last_speech = self._current_time
            elif (
                self._last_speech is not None
                and self._current_time - self._last_speech > self.config.silence_duration
            ):
                # Speech segment ended
                duration = (
                    self._last_speech - self._speech_start
                    if self._speech_start is not None
                    else 0.0
                )
                self.logger.debug(
                    "speech_ended",
                    timestamp=self._current_time,
                    duration=duration,
                    buffer_chunks=len(self._speech_buffer),
                )
                self.state = VADState.SILENCE
                self._speech_start = None
                self._last_speech = None
                # Note: Don't clear buffer here - it will be retrieved by get_speech_buffer()

        # Check max speech duration
        if self._speech_start is not None:
            speech_duration = self._current_time - self._speech_start
            if speech_duration > self.config.max_speech_duration:
                self.logger.warning(
                    "max_speech_duration_exceeded",
                    duration=speech_duration,
                )
                self.state = VADState.SILENCE
                self._speech_start = None
                self._last_speech = None

        # Accumulate audio during speech and trailing states
        if self.state in (VADState.SPEECH, VADState.TRAILING):
            if len(self._speech_buffer) < self._max_buffer_chunks:
                self._speech_buffer.append(chunk.copy())

        # Calculate result
        is_speech = self.state in (VADState.SPEECH, VADState.TRAILING)
        duration = (
            self._current_time - self._speech_start if self._speech_start else 0.0
        )

        result = VADResult(
            is_speech=is_speech,
            energy=energy,
            duration=duration,
            state=self.state,
        )

        # Publish events on state transitions
        if self.event_bus and previous_state != self.state:
            if self.state == VADState.SPEECH and previous_state == VADState.SILENCE:
                await self.event_bus.publish(
                    VADEvent(
                        type=EventType.VAD_SPEECH_START,
                        is_speech=True,
                        energy=energy,
                        duration=0.0,
                    )
                )
            elif self.state == VADState.SILENCE and previous_state == VADState.TRAILING:
                await self.event_bus.publish(
                    VADEvent(
                        type=EventType.VAD_SPEECH_END,
                        is_speech=False,
                        energy=energy,
                        duration=duration,
                    )
                )

        return result

    def get_average_energy(self) -> float:
        """Get average energy over recent history.

        Returns:
            Average energy value.
        """
        if not self._energy_history:
            return 0.0
        return sum(self._energy_history) / len(self._energy_history)

    @property
    def noise_floor(self) -> float:
        """Get current noise floor estimate."""
        return self._noise_floor

    @property
    def speech_duration(self) -> float:
        """Get duration of current speech segment.

        Returns:
            Duration in seconds, or 0 if not in speech.
        """
        if self._speech_start is None:
            return 0.0
        return self._current_time - self._speech_start

    def get_speech_buffer(self) -> Optional[np.ndarray]:
        """Get accumulated speech audio buffer.

        Returns the audio collected during the last speech segment.
        The buffer is cleared after retrieval.

        Returns:
            Concatenated audio samples as numpy array, or None if empty.
        """
        if not self._speech_buffer:
            return None

        # Concatenate all chunks
        audio = np.concatenate(self._speech_buffer)

        # Clear buffer after retrieval
        self._speech_buffer.clear()

        self.logger.debug(
            "speech_buffer_retrieved",
            samples=len(audio),
            duration=len(audio) / self.sample_rate,
        )

        return audio
