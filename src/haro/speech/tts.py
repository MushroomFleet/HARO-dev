"""Text-to-Speech using Piper TTS.

Provides speech synthesis using Piper, a fast, local neural TTS engine
optimized for CPU inference.
"""

import asyncio
import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from haro.core.config import TTSConfig
from haro.core.events import EventBus, EventType, Event
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SynthesisResult:
    """Result of speech synthesis."""

    audio: np.ndarray
    sample_rate: int
    duration: float


class PiperTTS:
    """Text-to-Speech using Piper.

    Wraps the piper-tts library for fast, local speech synthesis.
    """

    def __init__(
        self,
        config: TTSConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize Piper TTS.

        Args:
            config: TTS configuration.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.event_bus = event_bus
        self._voice = None
        self._voice_loaded = False
        self._sample_rate = 22050  # Piper default

        self.logger = logger.bind(component="PiperTTS")

    @property
    def is_loaded(self) -> bool:
        """Check if voice model is loaded."""
        return self._voice_loaded

    @property
    def sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        return self._sample_rate

    def _get_model_path(self) -> Path:
        """Get the path for model storage."""
        return Path(self.config.model_path).expanduser()

    async def load_voice(self) -> None:
        """Load the Piper voice model.

        This is done lazily on first synthesis, but can be called
        explicitly to preload the voice.
        """
        if self._voice_loaded:
            return

        self.logger.info(
            "loading_piper_voice",
            voice=self.config.voice,
        )

        try:
            # Import here to avoid slow startup if not using TTS
            from piper import PiperVoice

            model_path = self._get_model_path()
            model_path.mkdir(parents=True, exist_ok=True)

            # Try to find voice model
            voice_path = self._find_voice_model()

            if voice_path is None:
                # Download voice model
                await self._download_voice()
                voice_path = self._find_voice_model()

            if voice_path is None:
                raise FileNotFoundError(
                    f"Voice model not found: {self.config.voice}"
                )

            # Load voice in executor
            loop = asyncio.get_event_loop()
            self._voice = await loop.run_in_executor(
                None,
                lambda: PiperVoice.load(str(voice_path)),
            )

            # Get sample rate from voice config
            if hasattr(self._voice, "config") and hasattr(
                self._voice.config, "sample_rate"
            ):
                self._sample_rate = self._voice.config.sample_rate

            self._voice_loaded = True
            self.logger.info(
                "piper_voice_loaded",
                sample_rate=self._sample_rate,
            )

        except Exception as e:
            self.logger.error("failed_to_load_piper", error=str(e))
            raise

    def _find_voice_model(self) -> Optional[Path]:
        """Find the voice model file.

        Returns:
            Path to voice model, or None if not found.
        """
        model_path = self._get_model_path()
        voice_name = self.config.voice

        # Check common locations
        candidates = [
            model_path / f"{voice_name}.onnx",
            model_path / voice_name / f"{voice_name}.onnx",
            model_path / "voices" / f"{voice_name}.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / f"{voice_name}.onnx",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Search recursively
        for onnx_file in model_path.rglob("*.onnx"):
            if voice_name in onnx_file.stem:
                return onnx_file

        return None

    async def _download_voice(self) -> None:
        """Download voice model from Piper repository.

        Note: This is a placeholder. In production, implement proper
        model downloading from huggingface or piper releases.
        """
        self.logger.warning(
            "voice_download_not_implemented",
            voice=self.config.voice,
            hint="Download voice manually from https://github.com/rhasspy/piper/releases",
        )

    async def unload_voice(self) -> None:
        """Unload the voice model to free memory."""
        if self._voice is not None:
            del self._voice
            self._voice = None
            self._voice_loaded = False
            self.logger.info("piper_voice_unloaded")

    async def synthesize(
        self,
        text: str,
        speed: Optional[float] = None,
        volume: Optional[float] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            speed: Speech rate multiplier (default: from config).
            volume: Volume multiplier (default: from config).

        Returns:
            SynthesisResult with audio data and metadata.
        """
        if not self._voice_loaded:
            await self.load_voice()

        if not text.strip():
            return SynthesisResult(
                audio=np.array([], dtype=np.float32),
                sample_rate=self._sample_rate,
                duration=0.0,
            )

        speed = speed if speed is not None else self.config.speed
        volume = volume if volume is not None else self.config.volume

        self.logger.debug(
            "synthesizing",
            text=text[:50],
            speed=speed,
        )

        try:
            # Synthesize in executor
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None,
                lambda: self._synthesize_sync(text, speed),
            )

            # Convert to numpy
            audio = self._bytes_to_audio(audio_bytes)

            # Apply volume
            if volume != 1.0:
                audio = audio * volume
                audio = np.clip(audio, -1.0, 1.0)

            duration = len(audio) / self._sample_rate

            self.logger.debug(
                "synthesis_complete",
                duration=duration,
                samples=len(audio),
            )

            return SynthesisResult(
                audio=audio,
                sample_rate=self._sample_rate,
                duration=duration,
            )

        except Exception as e:
            self.logger.error("synthesis_failed", error=str(e))

            if self.event_bus:
                await self.event_bus.publish(
                    Event(
                        type=EventType.TTS_SYNTHESIS_ERROR,
                        data={"error": str(e)},
                    )
                )

            raise

    def _synthesize_sync(self, text: str, speed: float) -> bytes:
        """Synchronous synthesis (runs in executor).

        Args:
            text: Text to synthesize.
            speed: Speech rate multiplier.

        Returns:
            Raw audio bytes.
        """
        from piper.config import SynthesisConfig

        # Create synthesis config with length_scale for speed control
        # length_scale > 1.0 = slower, < 1.0 = faster
        syn_config = SynthesisConfig(
            length_scale=1.0 / speed if speed != 1.0 else None
        )

        # Synthesize returns iterable of AudioChunks
        audio_chunks = list(self._voice.synthesize(text, syn_config))

        # Combine all chunks into single audio array
        audio_int16 = np.concatenate([chunk.audio_int16_array for chunk in audio_chunks])

        # Create a bytes buffer for the WAV output
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    def _bytes_to_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to numpy array.

        Args:
            audio_bytes: Raw WAV file bytes.

        Returns:
            Audio as float32 numpy array.
        """
        buffer = io.BytesIO(audio_bytes)

        with wave.open(buffer, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        # Convert to numpy (16-bit signed)
        audio = np.frombuffer(audio_data, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0

    async def synthesize_to_file(
        self,
        text: str,
        path: Union[str, Path],
        speed: Optional[float] = None,
    ) -> float:
        """Synthesize speech and save to file.

        Args:
            text: Text to synthesize.
            path: Output file path.
            speed: Speech rate multiplier.

        Returns:
            Duration of generated audio in seconds.
        """
        result = await self.synthesize(text, speed=speed, volume=1.0)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as WAV
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(result.sample_rate)

            # Convert back to int16
            audio_int16 = (result.audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        return result.duration

    @staticmethod
    def get_available_voices() -> list[str]:
        """Get list of commonly available Piper voices.

        Returns:
            List of voice names.
        """
        return [
            "en_US-lessac-medium",
            "en_US-lessac-high",
            "en_US-amy-medium",
            "en_US-ryan-medium",
            "en_US-ryan-high",
            "en_GB-alan-medium",
            "en_GB-southern_english_female-medium",
        ]

    @staticmethod
    def get_voice_info(voice_name: str) -> dict:
        """Get information about a Piper voice.

        Args:
            voice_name: Name of the voice.

        Returns:
            Dictionary with voice information.
        """
        voice_info = {
            "en_US-lessac-medium": {
                "language": "en_US",
                "quality": "medium",
                "size_mb": 60,
                "gender": "female",
            },
            "en_US-lessac-high": {
                "language": "en_US",
                "quality": "high",
                "size_mb": 100,
                "gender": "female",
            },
            "en_US-amy-medium": {
                "language": "en_US",
                "quality": "medium",
                "size_mb": 60,
                "gender": "female",
            },
            "en_US-ryan-medium": {
                "language": "en_US",
                "quality": "medium",
                "size_mb": 60,
                "gender": "male",
            },
            "en_US-ryan-high": {
                "language": "en_US",
                "quality": "high",
                "size_mb": 100,
                "gender": "male",
            },
            "en_GB-alan-medium": {
                "language": "en_GB",
                "quality": "medium",
                "size_mb": 60,
                "gender": "male",
            },
        }
        return voice_info.get(voice_name, {})
