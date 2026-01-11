"""Audio playback to speakers using sounddevice.

Provides async playback of audio data with queue-based operation,
volume control, and interrupt capability.
"""

import asyncio
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sounddevice as sd

from haro.core.config import AudioConfig
from haro.core.events import EventBus, EventType, Event
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PlaybackItem:
    """Item in the playback queue."""

    audio: np.ndarray
    sample_rate: int
    volume: float = 1.0
    completion_event: Optional[asyncio.Event] = None


class AudioPlayback:
    """Play audio to speakers.

    Supports async playback with queue-based operation, volume control,
    and the ability to interrupt current playback.
    """

    def __init__(
        self,
        config: AudioConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize audio playback.

        Args:
            config: Audio configuration.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.event_bus = event_bus

        self._queue: asyncio.Queue[PlaybackItem] = asyncio.Queue()
        self._running = False
        self._playing = False
        self._interrupted = False
        self._volume = config.output_device if hasattr(config, "volume") else 0.8
        self._current_playback: Optional[asyncio.Event] = None
        self._playback_task: Optional[asyncio.Task] = None

        self.logger = logger.bind(component="AudioPlayback")

    @staticmethod
    def list_output_devices() -> list:
        """List available audio output devices.

        Returns:
            List of available audio devices with output capabilities.
        """
        devices = []
        try:
            device_list = sd.query_devices()
            default_output = sd.default.device[1]

            for i, dev in enumerate(device_list):
                if dev["max_output_channels"] > 0:
                    devices.append(
                        {
                            "index": i,
                            "name": dev["name"],
                            "max_output_channels": dev["max_output_channels"],
                            "default_sample_rate": dev["default_samplerate"],
                            "is_default": i == default_output,
                        }
                    )
        except Exception as e:
            logger.error("failed_to_list_output_devices", error=str(e))

        return devices

    @property
    def volume(self) -> float:
        """Get current volume level (0.0 to 1.0)."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set volume level (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, value))
        self.logger.debug("volume_changed", volume=self._volume)

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing

    async def start(self) -> None:
        """Start the playback processor."""
        if self._running:
            return

        self._running = True
        self._playback_task = asyncio.create_task(self._playback_loop())
        self.logger.info("playback_started")

    async def stop(self) -> None:
        """Stop the playback processor."""
        if not self._running:
            return

        self._running = False
        self._interrupted = True

        # Cancel playback task
        if self._playback_task:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.logger.info("playback_stopped")

    async def _playback_loop(self) -> None:
        """Main playback loop processing queued audio."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                await self._play_audio(item)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("playback_loop_error", error=str(e))

    async def _play_audio(self, item: PlaybackItem) -> None:
        """Play a single audio item.

        Args:
            item: The PlaybackItem to play.
        """
        self._playing = True
        self._interrupted = False
        self._current_playback = asyncio.Event()

        try:
            # Apply volume
            audio = item.audio * item.volume * self._volume

            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Publish start event
            if self.event_bus:
                await self.event_bus.publish(
                    Event(type=EventType.TTS_PLAYBACK_START)
                )

            self.logger.debug(
                "playing_audio",
                samples=len(audio),
                sample_rate=item.sample_rate,
            )

            # Play audio using sounddevice
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sd.play(
                    audio,
                    samplerate=item.sample_rate,
                    device=self.config.output_device,
                    blocking=True,
                )
                if not self._interrupted
                else None,
            )

            # Check if we were interrupted
            if self._interrupted:
                sd.stop()
                if self.event_bus:
                    await self.event_bus.publish(
                        Event(type=EventType.TTS_PLAYBACK_INTERRUPTED)
                    )
            else:
                if self.event_bus:
                    await self.event_bus.publish(
                        Event(type=EventType.TTS_PLAYBACK_COMPLETE)
                    )

        except Exception as e:
            self.logger.error("playback_error", error=str(e))
        finally:
            self._playing = False
            if self._current_playback:
                self._current_playback.set()
            # Signal completion to caller waiting on this item
            if item.completion_event:
                item.completion_event.set()

    async def play(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        volume: float = 1.0,
        wait: bool = False,
    ) -> None:
        """Queue audio for playback.

        Args:
            audio: Audio data as numpy array.
            sample_rate: Sample rate (defaults to config).
            volume: Volume multiplier (0.0 to 1.0).
            wait: If True, wait for playback to complete.
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate

        item = PlaybackItem(
            audio=audio,
            sample_rate=sample_rate,
            volume=volume,
        )

        if wait:
            # Create completion event before queuing so we don't miss it
            completion_event = asyncio.Event()
            item.completion_event = completion_event
            await self._queue.put(item)
            await completion_event.wait()
        else:
            await self._queue.put(item)

    async def play_file(
        self,
        path: Union[str, Path],
        volume: float = 1.0,
        wait: bool = False,
    ) -> None:
        """Play a WAV file.

        Args:
            path: Path to WAV file.
            volume: Volume multiplier.
            wait: If True, wait for playback to complete.
        """
        path = Path(path)
        if not path.exists():
            self.logger.error("file_not_found", path=str(path))
            return

        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                n_frames = wav.getnframes()

                audio_bytes = wav.readframes(n_frames)

                # Convert to numpy array
                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.int8

                audio = np.frombuffer(audio_bytes, dtype=dtype)
                audio = audio.astype(np.float32) / np.iinfo(dtype).max

                # Convert to mono if stereo
                if n_channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)

            await self.play(audio, sample_rate=sample_rate, volume=volume, wait=wait)

        except Exception as e:
            self.logger.error("play_file_error", path=str(path), error=str(e))

    def interrupt(self) -> None:
        """Interrupt current playback."""
        self._interrupted = True
        sd.stop()
        self.logger.debug("playback_interrupted")

    def clear_queue(self) -> None:
        """Clear the playback queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.logger.debug("queue_cleared")

    async def play_tone(
        self,
        frequency: float = 440.0,
        duration: float = 0.5,
        volume: float = 0.5,
        wait: bool = False,
    ) -> None:
        """Play a simple sine wave tone.

        Useful for feedback sounds (wake word confirmation, etc.).

        Args:
            frequency: Tone frequency in Hz.
            duration: Duration in seconds.
            volume: Volume multiplier.
            wait: If True, wait for playback to complete.
        """
        sample_rate = self.config.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * frequency * t) * volume

        # Apply fade in/out to avoid clicks
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        if len(audio) > fade_samples * 2:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        await self.play(audio, sample_rate=sample_rate, volume=1.0, wait=wait)

    async def __aenter__(self) -> "AudioPlayback":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
