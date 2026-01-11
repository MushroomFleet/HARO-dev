"""Speech-to-Text using faster-whisper.

Provides transcription of audio using the Whisper model via faster-whisper
(CTranslate2 backend) for efficient CPU inference.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from haro.core.config import STTConfig
from haro.core.events import EventBus, EventType, TranscriptionEvent, Event
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""

    text: str
    language: str
    confidence: float
    duration: float
    segments: list[dict]


class WhisperSTT:
    """Speech-to-Text using faster-whisper.

    Wraps the faster-whisper library for efficient CPU-based transcription
    using the Whisper model.
    """

    def __init__(
        self,
        config: STTConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize Whisper STT.

        Args:
            config: STT configuration.
            event_bus: Optional event bus for publishing events.
        """
        self.config = config
        self.event_bus = event_bus
        self._model = None
        self._model_loaded = False

        self.logger = logger.bind(component="WhisperSTT")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def _get_model_path(self) -> Path:
        """Get the path for model storage."""
        return Path(self.config.model_path).expanduser()

    async def load_model(self) -> None:
        """Load the Whisper model.

        This is done lazily on first transcription, but can be called
        explicitly to preload the model.
        """
        if self._model_loaded:
            return

        self.logger.info(
            "loading_whisper_model",
            model=self.config.model,
            compute_type=self.config.compute_type,
        )

        try:
            # Import here to avoid slow startup if not using STT
            from faster_whisper import WhisperModel

            # Load model in executor to not block
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.config.model,
                    device="cpu",
                    compute_type=self.config.compute_type,
                    download_root=str(self._get_model_path()),
                ),
            )
            self._model_loaded = True
            self.logger.info("whisper_model_loaded")

        except Exception as e:
            self.logger.error("failed_to_load_whisper", error=str(e))
            raise

    async def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            self.logger.info("whisper_model_unloaded")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32, mono).
            sample_rate: Sample rate of the audio.
            prompt: Optional prompt to bias transcription.
            language: Optional language code (default: from config).

        Returns:
            TranscriptionResult with transcription and metadata.
        """
        if not self._model_loaded:
            await self.load_model()

        # Ensure audio is float32 and mono
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Normalize audio if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Resample if not 16kHz (Whisper expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)

        duration = len(audio) / 16000

        self.logger.debug(
            "transcribing",
            duration=duration,
            samples=len(audio),
        )

        # Publish start event
        if self.event_bus:
            await self.event_bus.publish(
                Event(type=EventType.STT_TRANSCRIPTION_START)
            )

        try:
            # Run transcription in executor
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio,
                    language=language or self.config.language,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                    initial_prompt=prompt,
                ),
            )

            # Collect segments
            segment_list = []
            full_text_parts = []

            for segment in segments:
                segment_list.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": segment.avg_logprob,
                    }
                )
                full_text_parts.append(segment.text)

            full_text = "".join(full_text_parts).strip()

            # Calculate average confidence from log probabilities
            if segment_list:
                avg_logprob = sum(s["avg_logprob"] for s in segment_list) / len(
                    segment_list
                )
                # Convert log probability to confidence (0-1)
                confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 2))
            else:
                confidence = 0.0

            result = TranscriptionResult(
                text=full_text,
                language=info.language,
                confidence=confidence,
                duration=duration,
                segments=segment_list,
            )

            self.logger.debug(
                "transcription_complete",
                text=full_text[:100],
                language=info.language,
                confidence=confidence,
            )

            # Publish completion event
            if self.event_bus:
                await self.event_bus.publish(
                    TranscriptionEvent(
                        text=full_text,
                        confidence=confidence,
                        language=info.language,
                    )
                )

            return result

        except Exception as e:
            self.logger.error("transcription_failed", error=str(e))

            if self.event_bus:
                await self.event_bus.publish(
                    Event(
                        type=EventType.STT_TRANSCRIPTION_ERROR,
                        data={"error": str(e)},
                    )
                )

            raise

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.

        Simple linear interpolation resampling. For production use,
        consider scipy.signal.resample or librosa.

        Args:
            audio: Input audio samples.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio.
        """
        if orig_sr == target_sr:
            return audio

        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def transcribe_file(
        self,
        path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio from a file.

        Args:
            path: Path to audio file (WAV, MP3, etc.).
            prompt: Optional prompt to bias transcription.

        Returns:
            TranscriptionResult with transcription and metadata.
        """
        import wave

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Load WAV file
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()

            audio_bytes = wav.readframes(n_frames)

            # Convert to numpy
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.int8

            audio = np.frombuffer(audio_bytes, dtype=dtype)
            audio = audio.astype(np.float32) / np.iinfo(dtype).max

            # Convert to mono
            if n_channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

        return await self.transcribe(audio, sample_rate=sample_rate, prompt=prompt)

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available Whisper models.

        Returns:
            List of model names that can be used.
        """
        return [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v2",
            "large-v3",
        ]

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get information about a Whisper model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with model information.
        """
        model_info = {
            "tiny": {"parameters": "39M", "size_mb": 75, "english_only": False},
            "tiny.en": {"parameters": "39M", "size_mb": 75, "english_only": True},
            "base": {"parameters": "74M", "size_mb": 140, "english_only": False},
            "base.en": {"parameters": "74M", "size_mb": 140, "english_only": True},
            "small": {"parameters": "244M", "size_mb": 460, "english_only": False},
            "small.en": {"parameters": "244M", "size_mb": 460, "english_only": True},
            "medium": {"parameters": "769M", "size_mb": 1500, "english_only": False},
            "medium.en": {"parameters": "769M", "size_mb": 1500, "english_only": True},
            "large-v2": {"parameters": "1550M", "size_mb": 3000, "english_only": False},
            "large-v3": {"parameters": "1550M", "size_mb": 3000, "english_only": False},
        }
        return model_info.get(model_name, {})
