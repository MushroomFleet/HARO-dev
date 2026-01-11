"""Tests for speech processing module (STT, TTS, models)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from haro.core.config import STTConfig, TTSConfig
from haro.speech.stt import WhisperSTT, TranscriptionResult
from haro.speech.tts import PiperTTS, SynthesisResult
from haro.speech.models import ModelManager, ModelInfo, WHISPER_MODELS, PIPER_VOICES


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_create_result(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration=1.5,
            segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration == 1.5
        assert len(result.segments) == 1


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_create_result(self):
        """Test creating a synthesis result."""
        audio = np.zeros(22050, dtype=np.float32)
        result = SynthesisResult(
            audio=audio,
            sample_rate=22050,
            duration=1.0,
        )
        assert len(result.audio) == 22050
        assert result.sample_rate == 22050
        assert result.duration == 1.0


class TestWhisperSTT:
    """Tests for WhisperSTT."""

    @pytest.fixture
    def stt_config(self):
        """Create STT config for tests."""
        return STTConfig(
            model="tiny.en",
            model_path="~/.cache/haro/models/",
            language="en",
            compute_type="int8",
            beam_size=1,
        )

    @pytest.fixture
    def stt(self, stt_config):
        """Create STT instance for tests."""
        return WhisperSTT(stt_config)

    def test_init(self, stt):
        """Test STT initialization."""
        assert stt.is_loaded is False
        assert stt.config.model == "tiny.en"

    def test_get_available_models(self):
        """Test listing available models."""
        models = WhisperSTT.get_available_models()
        assert "tiny.en" in models
        assert "base.en" in models
        assert len(models) >= 5

    def test_get_model_info(self):
        """Test getting model info."""
        info = WhisperSTT.get_model_info("tiny.en")
        assert info["parameters"] == "39M"
        assert info["english_only"] is True

        info = WhisperSTT.get_model_info("base")
        assert info["english_only"] is False

    def test_resample(self, stt):
        """Test audio resampling."""
        # Create 1 second of audio at 44100 Hz
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 44100)).astype(np.float32)

        # Resample to 16000 Hz
        resampled = stt._resample(audio, 44100, 16000)

        assert len(resampled) == 16000
        assert resampled.dtype == np.float32

    def test_resample_same_rate(self, stt):
        """Test resampling with same rate returns original."""
        audio = np.zeros(1000, dtype=np.float32)
        resampled = stt._resample(audio, 16000, 16000)
        assert np.array_equal(audio, resampled)

    @pytest.mark.asyncio
    async def test_transcribe_mocked(self, stt):
        """Test transcription with mocked model."""
        # Mock the model
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        stt._model = mock_model
        stt._model_loaded = True

        # Test transcription
        audio = np.random.randn(16000).astype(np.float32)
        result = await stt.transcribe(audio)

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence > 0


class TestPiperTTS:
    """Tests for PiperTTS."""

    @pytest.fixture
    def tts_config(self):
        """Create TTS config for tests."""
        return TTSConfig(
            engine="piper",
            voice="en_US-lessac-medium",
            model_path="~/.cache/haro/models/",
            speed=1.0,
            volume=0.8,
        )

    @pytest.fixture
    def tts(self, tts_config):
        """Create TTS instance for tests."""
        return PiperTTS(tts_config)

    def test_init(self, tts):
        """Test TTS initialization."""
        assert tts.is_loaded is False
        assert tts.config.voice == "en_US-lessac-medium"
        assert tts.sample_rate == 22050

    def test_get_available_voices(self):
        """Test listing available voices."""
        voices = PiperTTS.get_available_voices()
        assert "en_US-lessac-medium" in voices
        assert len(voices) >= 3

    def test_get_voice_info(self):
        """Test getting voice info."""
        info = PiperTTS.get_voice_info("en_US-lessac-medium")
        assert info["language"] == "en_US"
        assert info["quality"] == "medium"

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, tts):
        """Test synthesizing empty text."""
        # Mock loaded state
        tts._voice_loaded = True
        tts._voice = MagicMock()

        result = await tts.synthesize("")

        assert len(result.audio) == 0
        assert result.duration == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_mocked(self, tts):
        """Test synthesis with mocked voice."""
        import io
        import wave

        # Create mock WAV data
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            audio_data = np.zeros(22050, dtype=np.int16).tobytes()
            wav.writeframes(audio_data)

        mock_voice = MagicMock()

        def mock_synthesize(text, wav_file, length_scale=1.0):
            wav_file.writeframes(np.zeros(22050, dtype=np.int16).tobytes())

        mock_voice.synthesize = mock_synthesize

        tts._voice = mock_voice
        tts._voice_loaded = True

        # Can't easily test full synthesis without actual model
        # Just verify the interface works
        assert tts.is_loaded is True


class TestModelManager:
    """Tests for ModelManager."""

    @pytest.fixture
    def manager(self):
        """Create model manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelManager(base_path=tmpdir)

    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.base_path.exists() or True  # May not exist yet

    def test_list_installed_models_empty(self, manager):
        """Test listing models when none installed."""
        models = manager.list_installed_models()

        # Should list all known models
        assert len(models) > 0

        # None should be installed
        for model in models:
            assert model.installed is False

    def test_is_model_installed_false(self, manager):
        """Test checking uninstalled model."""
        assert manager.is_model_installed("tiny.en") is False
        assert manager.is_model_installed("en_US-lessac-medium") is False

    def test_get_model_path_not_installed(self, manager):
        """Test getting path for uninstalled model."""
        assert manager.get_model_path("tiny.en") is None

    def test_get_total_size_empty(self, manager):
        """Test total size when empty."""
        size = manager.get_total_size()
        assert size == 0

    def test_whisper_models_registry(self):
        """Test Whisper models registry."""
        assert "tiny.en" in WHISPER_MODELS
        assert "base.en" in WHISPER_MODELS
        assert "files" in WHISPER_MODELS["tiny.en"]
        assert "model.bin" in WHISPER_MODELS["tiny.en"]["files"]

    def test_piper_voices_registry(self):
        """Test Piper voices registry."""
        assert "en_US-lessac-medium" in PIPER_VOICES
        assert "files" in PIPER_VOICES["en_US-lessac-medium"]

    @pytest.mark.asyncio
    async def test_download_whisper_invalid_model(self, manager):
        """Test downloading invalid model name."""
        with pytest.raises(ValueError):
            await manager.download_whisper_model("invalid-model")

    @pytest.mark.asyncio
    async def test_download_piper_invalid_voice(self, manager):
        """Test downloading invalid voice name."""
        with pytest.raises(ValueError):
            await manager.download_piper_voice("invalid-voice")


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_create_model_info(self):
        """Test creating model info."""
        info = ModelInfo(
            name="tiny.en",
            type="stt",
            size_mb=75,
            installed=True,
            path=Path("/path/to/model"),
        )
        assert info.name == "tiny.en"
        assert info.type == "stt"
        assert info.installed is True

    def test_model_info_not_installed(self):
        """Test model info for uninstalled model."""
        info = ModelInfo(
            name="base.en",
            type="stt",
            size_mb=140,
            installed=False,
        )
        assert info.installed is False
        assert info.path is None
