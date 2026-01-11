"""Audio capture from microphone using sounddevice.

Provides async streaming of audio chunks from the default or specified
microphone device. Uses sounddevice for cross-platform compatibility.
"""

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
import sounddevice as sd

from haro.core.config import AudioConfig
from haro.core.events import AudioChunkEvent, EventBus, EventType, Event
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDevice:
    """Information about an audio device."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


class AudioCapture:
    """Capture audio from microphone.

    Streams audio chunks as numpy arrays using sounddevice.
    Supports async iteration and event-based notification.
    """

    def __init__(
        self,
        config: AudioConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize audio capture.

        Args:
            config: Audio configuration.
            event_bus: Optional event bus for publishing audio chunks.
        """
        self.config = config
        self.event_bus = event_bus

        self._stream: Optional[sd.InputStream] = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._running = False
        self._paused = False

        self.logger = logger.bind(component="AudioCapture")

    @staticmethod
    def list_devices() -> list[AudioDevice]:
        """List available audio input devices.

        Returns:
            List of available audio devices with input capabilities.
        """
        devices = []
        try:
            device_list = sd.query_devices()
            default_input = sd.default.device[0]

            for i, dev in enumerate(device_list):
                if dev["max_input_channels"] > 0:
                    devices.append(
                        AudioDevice(
                            index=i,
                            name=dev["name"],
                            max_input_channels=dev["max_input_channels"],
                            max_output_channels=dev["max_output_channels"],
                            default_sample_rate=dev["default_samplerate"],
                            is_default_input=(i == default_input),
                        )
                    )
        except Exception as e:
            logger.error("failed_to_list_devices", error=str(e))

        return devices

    @staticmethod
    def get_default_device() -> Optional[AudioDevice]:
        """Get the default input device.

        Returns:
            The default input device, or None if not available.
        """
        devices = AudioCapture.list_devices()
        for dev in devices:
            if dev.is_default_input:
                return dev
        return devices[0] if devices else None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for sounddevice stream.

        Called from audio thread, so we need to be careful about thread safety.
        """
        if status:
            self.logger.warning("audio_callback_status", status=str(status))

        if not self._paused and self._running:
            # Copy data to avoid issues with buffer reuse
            chunk = indata.copy().flatten().astype(np.float32)
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Drop oldest chunk if queue is full
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(chunk)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self.logger.info(
            "starting_capture",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_size=self.config.chunk_size,
            device=self.config.input_device,
        )

        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                device=self.config.input_device,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self.logger.info("capture_started")
        except Exception as e:
            self.logger.error("capture_start_failed", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.logger.info("capture_stopped")

    def pause(self) -> None:
        """Pause audio capture (keeps stream running but discards data)."""
        self._paused = True
        self.logger.debug("capture_paused")

    def resume(self) -> None:
        """Resume audio capture."""
        self._paused = False
        self.logger.debug("capture_resumed")

    @property
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if capture is paused."""
        return self._paused

    async def read_chunk(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """Read a single audio chunk.

        Args:
            timeout: Timeout in seconds to wait for a chunk.

        Returns:
            Audio chunk as numpy array, or None if timeout.
        """
        if not self._running:
            return None

        try:
            chunk = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return chunk
        except asyncio.TimeoutError:
            return None

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Stream audio chunks as an async iterator.

        Yields:
            Audio chunks as numpy arrays.
        """
        if not self._running:
            await self.start()

        while self._running:
            chunk = await self.read_chunk()
            if chunk is not None:
                # Publish event if event bus is configured
                if self.event_bus:
                    event = AudioChunkEvent(
                        chunk=chunk,
                        sample_rate=self.config.sample_rate,
                        channels=self.config.channels,
                    )
                    await self.event_bus.publish(event)

                yield chunk

    async def __aenter__(self) -> "AudioCapture":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
