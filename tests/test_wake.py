"""Tests for wake word detection module."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from haro.core.config import WakeConfig, AudioConfig
from haro.core.events import EventBus, EventType
from haro.audio.wake import (
    WakeWordDetector,
    WakeWordResult,
    RollingBuffer,
    ContinuousWakeDetector,
)
from haro.audio.feedback import AudioFeedback, FeedbackConfig


class TestWakeWordResult:
    """Tests for WakeWordResult dataclass."""

    def test_create_result(self):
        """Test creating a wake word result."""
        result = WakeWordResult(
            detected=True,
            confidence=0.9,
            timestamp=1234567890.0,
            text="haro",
        )
        assert result.detected is True
        assert result.confidence == 0.9
        assert result.text == "haro"

    def test_default_text(self):
        """Test default empty text."""
        result = WakeWordResult(
            detected=False,
            confidence=0.0,
            timestamp=0.0,
        )
        assert result.text == ""


class TestRollingBuffer:
    """Tests for RollingBuffer."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = RollingBuffer(duration=2.0, sample_rate=16000)
        assert buffer.duration == 2.0
        assert buffer.sample_rate == 16000
        assert len(buffer.get()) == 32000  # 2 seconds at 16kHz

    def test_add_chunk(self):
        """Test adding audio chunks."""
        buffer = RollingBuffer(duration=1.0, sample_rate=16000)

        # Add a chunk
        chunk = np.ones(1000, dtype=np.float32)
        buffer.add(chunk)

        # Last 1000 samples should be ones
        data = buffer.get()
        assert np.all(data[-1000:] == 1.0)

    def test_rolling_behavior(self):
        """Test that buffer rolls correctly."""
        buffer = RollingBuffer(duration=0.1, sample_rate=16000)  # 1600 samples

        # Fill with known values
        buffer.add(np.ones(800, dtype=np.float32) * 1.0)
        buffer.add(np.ones(800, dtype=np.float32) * 2.0)

        data = buffer.get()
        # First half should be 1.0, second half should be 2.0
        assert np.all(data[:800] == 1.0)
        assert np.all(data[800:] == 2.0)

    def test_large_chunk(self):
        """Test adding chunk larger than buffer."""
        buffer = RollingBuffer(duration=0.1, sample_rate=16000)  # 1600 samples

        # Add chunk larger than buffer
        chunk = np.arange(3200, dtype=np.float32)
        buffer.add(chunk)

        data = buffer.get()
        # Should contain last 1600 samples of chunk
        assert len(data) == 1600
        assert np.array_equal(data, chunk[-1600:])

    def test_get_energy(self):
        """Test energy calculation."""
        buffer = RollingBuffer(duration=1.0, sample_rate=16000)

        # Zero buffer should have zero energy
        assert buffer.get_energy() == 0.0

        # Add audio with known energy
        buffer.add(np.ones(16000, dtype=np.float32))
        assert buffer.get_energy() == 1.0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = RollingBuffer(duration=1.0, sample_rate=16000)
        buffer.add(np.ones(16000, dtype=np.float32))
        buffer.clear()

        assert buffer.get_energy() == 0.0
        assert np.all(buffer.get() == 0.0)

    def test_is_filled(self):
        """Test is_filled property."""
        buffer = RollingBuffer(duration=1.0, sample_rate=16000)

        assert buffer.is_filled is False

        # Add enough samples
        buffer.add(np.ones(16000, dtype=np.float32))
        assert buffer.is_filled is True


class TestWakeWordDetector:
    """Tests for WakeWordDetector."""

    @pytest.fixture
    def wake_config(self):
        """Create wake config for tests."""
        return WakeConfig(
            phrase="haro",
            sensitivity=0.5,
            confirmation_sound=True,
        )

    @pytest.fixture
    def audio_config(self):
        """Create audio config for tests."""
        return AudioConfig(
            sample_rate=16000,
            buffer_duration=2.0,
        )

    @pytest.fixture
    def detector(self, wake_config, audio_config):
        """Create detector instance."""
        return WakeWordDetector(wake_config, audio_config)

    def test_init(self, detector):
        """Test detector initialization."""
        assert detector.wake_phrase == "haro"
        assert detector.sensitivity == 0.5
        assert detector.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, detector):
        """Test detector initialization with STT."""
        mock_stt = MagicMock()
        await detector.initialize(mock_stt)

        assert detector.is_initialized is True

    def test_add_audio(self, detector):
        """Test adding audio to detector."""
        chunk = np.random.randn(1024).astype(np.float32)
        detector.add_audio(chunk)

        # Buffer should have received audio
        assert detector.buffer.get_energy() > 0

    @pytest.mark.asyncio
    async def test_detect_not_initialized(self, detector):
        """Test detection when not initialized."""
        detector.add_audio(np.ones(1024, dtype=np.float32))
        result = await detector.detect()

        assert result.detected is False

    @pytest.mark.asyncio
    async def test_detect_silence(self, detector):
        """Test detection with silent audio."""
        mock_stt = MagicMock()
        await detector.initialize(mock_stt)

        # Buffer is silent (zeros)
        result = await detector.detect()

        assert result.detected is False
        # STT should not be called for silent audio
        mock_stt.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_wake_word(self, detector):
        """Test successful wake word detection."""
        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "haro"
        mock_result.confidence = 0.9
        mock_stt.transcribe = AsyncMock(return_value=mock_result)

        await detector.initialize(mock_stt)

        # Add audio with energy
        detector.add_audio(np.random.randn(32000).astype(np.float32) * 0.5)

        result = await detector.detect()

        assert result.detected is True
        assert result.confidence >= detector.sensitivity

    @pytest.mark.asyncio
    async def test_detect_cooldown(self, detector):
        """Test detection cooldown prevents rapid re-detection."""
        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "haro"
        mock_result.confidence = 0.9
        mock_stt.transcribe = AsyncMock(return_value=mock_result)

        await detector.initialize(mock_stt)
        detector.add_audio(np.random.randn(32000).astype(np.float32) * 0.5)

        # First detection should succeed
        result1 = await detector.detect()
        assert result1.detected is True

        # Immediate second detection should fail (cooldown)
        result2 = await detector.detect()
        assert result2.detected is False

    @pytest.mark.asyncio
    async def test_reset_cooldown(self, detector):
        """Test resetting detection cooldown."""
        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "haro"
        mock_result.confidence = 0.9
        mock_stt.transcribe = AsyncMock(return_value=mock_result)

        await detector.initialize(mock_stt)
        detector.add_audio(np.random.randn(32000).astype(np.float32) * 0.5)

        # First detection
        await detector.detect()

        # Reset cooldown
        detector.reset_cooldown()

        # Should be able to detect again
        result = await detector.detect()
        assert result.detected is True

    def test_contains_wake_phrase(self, detector):
        """Test wake phrase detection in text."""
        assert detector._contains_wake_phrase("haro") is True
        assert detector._contains_wake_phrase("hey haro") is True
        assert detector._contains_wake_phrase("haro help me") is True
        assert detector._contains_wake_phrase("hello") is True  # Variant
        assert detector._contains_wake_phrase("hero") is True  # Variant
        assert detector._contains_wake_phrase("goodbye") is False

    def test_calculate_confidence(self, detector):
        """Test confidence calculation."""
        # Exact match gets bonus
        conf1 = detector._calculate_confidence("haro", 0.8)
        conf2 = detector._calculate_confidence("hey haro", 0.8)

        assert conf1 > conf2  # Exact match should be higher

    def test_energy_threshold_property(self, detector):
        """Test energy threshold getter/setter."""
        detector.energy_threshold = 0.05
        assert detector.energy_threshold == 0.05

        # Should be clamped
        detector.energy_threshold = 1.0
        assert detector.energy_threshold == 0.5

        detector.energy_threshold = -1.0
        assert detector.energy_threshold == 0.001

    @pytest.mark.asyncio
    async def test_event_publishing(self, wake_config, audio_config):
        """Test that wake events are published."""
        event_bus = EventBus()
        detector = WakeWordDetector(wake_config, audio_config, event_bus)

        events_received = []

        async def handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.WAKE_DETECTED, handler)

        mock_stt = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "haro"
        mock_result.confidence = 0.9
        mock_stt.transcribe = AsyncMock(return_value=mock_result)

        await detector.initialize(mock_stt)
        detector.add_audio(np.random.randn(32000).astype(np.float32) * 0.5)

        await detector.detect()

        assert len(events_received) == 1
        assert events_received[0].type == EventType.WAKE_DETECTED


class TestContinuousWakeDetector:
    """Tests for ContinuousWakeDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector with mock wake detector."""
        wake_config = WakeConfig(phrase="haro", sensitivity=0.5)
        wake_detector = WakeWordDetector(wake_config)
        return ContinuousWakeDetector(wake_detector, check_interval=0.1)

    def test_init(self, detector):
        """Test continuous detector initialization."""
        assert detector.is_running is False

    @pytest.mark.asyncio
    async def test_start_stop(self, detector):
        """Test starting and stopping continuous detection."""
        # Need to initialize the inner detector
        mock_stt = MagicMock()
        mock_stt.transcribe = AsyncMock(
            return_value=MagicMock(text="", confidence=0.0)
        )
        await detector.detector.initialize(mock_stt)

        await detector.start()
        assert detector.is_running is True

        await asyncio.sleep(0.05)

        await detector.stop()
        assert detector.is_running is False

    def test_add_audio(self, detector):
        """Test adding audio to continuous detector."""
        chunk = np.random.randn(1024).astype(np.float32)
        detector.add_audio(chunk)

        # Should pass through to inner detector
        assert detector.detector.buffer.get_energy() > 0


class TestFeedbackConfig:
    """Tests for FeedbackConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = FeedbackConfig()
        assert config.confirmation_sound is True
        assert len(config.confirmation_phrases) > 0
        assert config.chime_frequency == 880.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = FeedbackConfig(
            confirmation_sound=False,
            chime_frequency=440.0,
        )
        assert config.confirmation_sound is False
        assert config.chime_frequency == 440.0


class TestAudioFeedback:
    """Tests for AudioFeedback."""

    @pytest.fixture
    def mock_playback(self):
        """Create mock playback."""
        playback = MagicMock()
        playback.play = AsyncMock()
        return playback

    @pytest.fixture
    def feedback(self, mock_playback):
        """Create feedback instance."""
        wake_config = WakeConfig(
            phrase="haro",
            confirmation_sound=True,
            confirmation_phrases=["Yes?", "I'm here."],
        )
        return AudioFeedback(mock_playback, wake_config)

    def test_init(self, feedback):
        """Test feedback initialization."""
        assert feedback.config.confirmation_sound is True
        assert len(feedback.config.confirmation_phrases) == 2

    def test_generate_chime(self, feedback):
        """Test chime generation."""
        chime = feedback._chime
        assert isinstance(chime, np.ndarray)
        assert chime.dtype == np.float32
        assert len(chime) > 0

    @pytest.mark.asyncio
    async def test_play_wake_confirmation_chime(self, feedback, mock_playback):
        """Test playing wake confirmation chime."""
        await feedback.play_wake_confirmation(use_verbal=False)

        mock_playback.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_error(self, feedback, mock_playback):
        """Test playing error sound."""
        await feedback.play_error()

        mock_playback.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_ready(self, feedback, mock_playback):
        """Test playing ready sound."""
        await feedback.play_ready()

        mock_playback.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_goodbye(self, feedback, mock_playback):
        """Test playing goodbye sound."""
        await feedback.play_goodbye()

        mock_playback.play.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirmation_disabled(self, mock_playback):
        """Test that confirmation can be disabled."""
        wake_config = WakeConfig(
            phrase="haro",
            confirmation_sound=False,
        )
        feedback = AudioFeedback(mock_playback, wake_config)

        await feedback.play_wake_confirmation()

        mock_playback.play.assert_not_called()
