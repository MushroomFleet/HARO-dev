"""Tests for audio module (capture, playback, VAD)."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from haro.core.config import AudioConfig, VADConfig
from haro.core.events import EventBus, EventType
from haro.audio.vad import VoiceActivityDetector, VADState, VADResult


class TestVADResult:
    """Tests for VADResult dataclass."""

    def test_create_result(self):
        """Test creating a VAD result."""
        result = VADResult(
            is_speech=True,
            energy=0.5,
            duration=1.0,
            state=VADState.SPEECH,
        )
        assert result.is_speech is True
        assert result.energy == 0.5
        assert result.duration == 1.0
        assert result.state == VADState.SPEECH


class TestVADState:
    """Tests for VADState enum."""

    def test_states_exist(self):
        """Test all VAD states exist."""
        assert VADState.SILENCE.value == "silence"
        assert VADState.SPEECH.value == "speech"
        assert VADState.TRAILING.value == "trailing"


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector."""

    @pytest.fixture
    def vad_config(self):
        """Create VAD config for tests."""
        return VADConfig(
            threshold=0.5,
            min_speech_duration=0.5,
            max_speech_duration=30.0,
            silence_duration=1.5,
            noise_floor_adaptation=0.05,
        )

    @pytest.fixture
    def vad(self, vad_config):
        """Create VAD instance for tests."""
        return VoiceActivityDetector(
            config=vad_config,
            sample_rate=16000,
        )

    def test_init(self, vad):
        """Test VAD initialization."""
        assert vad.state == VADState.SILENCE
        assert vad.sample_rate == 16000

    def test_reset(self, vad):
        """Test VAD reset."""
        vad.state = VADState.SPEECH
        vad._speech_start = 1.0
        vad.reset()
        assert vad.state == VADState.SILENCE
        assert vad._speech_start is None

    @pytest.mark.asyncio
    async def test_process_silence(self, vad):
        """Test processing silent audio."""
        silence = np.zeros(1024, dtype=np.float32)
        result = await vad.process(silence, timestamp=0.0)

        assert result.is_speech is False
        assert result.state == VADState.SILENCE
        assert result.energy < 0.01

    @pytest.mark.asyncio
    async def test_process_speech(self, vad):
        """Test processing speech-like audio."""
        # Create noise that should trigger speech detection
        speech = np.random.randn(1024).astype(np.float32) * 0.5

        # Process multiple chunks to trigger detection
        for i in range(10):
            result = await vad.process(speech, timestamp=i * 0.1)

        assert result.is_speech is True
        assert result.state in (VADState.SPEECH, VADState.TRAILING)

    @pytest.mark.asyncio
    async def test_speech_to_silence_transition(self, vad):
        """Test transition from speech back to silence."""
        speech = np.random.randn(1024).astype(np.float32) * 0.5
        silence = np.zeros(1024, dtype=np.float32)

        # Start with speech
        for i in range(5):
            await vad.process(speech, timestamp=i * 0.1)

        # Transition to silence
        # Need enough silence to exceed silence_duration
        for i in range(50):  # 5 seconds of silence
            result = await vad.process(silence, timestamp=0.5 + i * 0.1)

        assert result.state == VADState.SILENCE

    @pytest.mark.asyncio
    async def test_adaptive_noise_floor(self, vad):
        """Test adaptive noise floor updates."""
        initial_noise_floor = vad._noise_floor

        # Process quiet audio (should lower noise floor)
        quiet = np.ones(1024, dtype=np.float32) * 0.001
        for _ in range(100):
            await vad.process(quiet)

        # Noise floor should have adapted
        assert vad._noise_floor != initial_noise_floor

    @pytest.mark.asyncio
    async def test_max_speech_duration(self, vad):
        """Test max speech duration cutoff."""
        vad.config.max_speech_duration = 1.0  # 1 second max

        speech = np.random.randn(1024).astype(np.float32) * 0.5

        # Track if we see speech reset at the max duration
        speech_duration_at_cutoff = None

        # Process speech for longer than max duration
        for i in range(20):  # 2 seconds
            result = await vad.process(speech, timestamp=i * 0.1)
            # Check if duration was reset after exceeding max
            if result.duration < 0.5 and i > 12:  # After 1.2 seconds
                speech_duration_at_cutoff = i * 0.1
                break

        # VAD should have reset (duration went back down) when max was exceeded
        # The exact state depends on whether new speech was detected
        assert speech_duration_at_cutoff is not None or vad.speech_duration < 1.5

    def test_noise_floor_property(self, vad):
        """Test noise floor property."""
        assert vad.noise_floor == vad._noise_floor

    def test_speech_duration_property(self, vad):
        """Test speech duration property."""
        assert vad.speech_duration == 0.0

        vad._speech_start = 0.0
        vad._current_time = 1.5
        assert vad.speech_duration == 1.5

    @pytest.mark.asyncio
    async def test_event_publishing(self, vad_config):
        """Test VAD publishes events on state changes."""
        event_bus = EventBus()
        vad = VoiceActivityDetector(
            config=vad_config,
            sample_rate=16000,
            event_bus=event_bus,
        )

        events_received = []

        async def handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.VAD_SPEECH_START, handler)
        event_bus.subscribe(EventType.VAD_SPEECH_END, handler)

        # Trigger speech start
        speech = np.random.randn(1024).astype(np.float32) * 0.5
        for i in range(5):
            await vad.process(speech, timestamp=i * 0.1)

        # Should have received speech start event
        speech_starts = [e for e in events_received if e.type == EventType.VAD_SPEECH_START]
        assert len(speech_starts) >= 1


class TestAudioCapture:
    """Tests for AudioCapture (mocked)."""

    @pytest.fixture
    def audio_config(self):
        """Create audio config for tests."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
        )

    def test_list_devices_with_mock(self, audio_config):
        """Test listing devices with mocked sounddevice."""
        from haro.audio.capture import AudioCapture

        with patch("haro.audio.capture.sd") as mock_sd:
            mock_sd.query_devices.return_value = [
                {
                    "name": "Test Microphone",
                    "max_input_channels": 2,
                    "max_output_channels": 0,
                    "default_samplerate": 44100.0,
                },
                {
                    "name": "Test Speaker",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 44100.0,
                },
            ]
            mock_sd.default.device = [0, 1]

            devices = AudioCapture.list_devices()

            assert len(devices) == 1
            assert devices[0].name == "Test Microphone"
            assert devices[0].is_default_input is True


class TestAudioPlayback:
    """Tests for AudioPlayback (mocked)."""

    @pytest.fixture
    def audio_config(self):
        """Create audio config for tests."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
        )

    def test_volume_property(self, audio_config):
        """Test volume getter and setter."""
        from haro.audio.playback import AudioPlayback

        playback = AudioPlayback(audio_config)

        # Test setter clamps values
        playback.volume = 1.5
        assert playback.volume == 1.0

        playback.volume = -0.5
        assert playback.volume == 0.0

        playback.volume = 0.7
        assert playback.volume == 0.7

    @pytest.mark.asyncio
    async def test_play_tone_generation(self, audio_config):
        """Test tone generation produces valid audio."""
        from haro.audio.playback import AudioPlayback

        playback = AudioPlayback(audio_config)

        with patch.object(playback, "play", new_callable=AsyncMock) as mock_play:
            await playback.play_tone(frequency=440.0, duration=0.5, wait=True)

            # Should have called play with audio data
            mock_play.assert_called_once()
            audio = mock_play.call_args[0][0]

            # Check audio properties
            expected_samples = int(16000 * 0.5)
            assert len(audio) == expected_samples
            assert audio.dtype == np.float32
