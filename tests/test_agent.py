"""Tests for agent state machine module."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from haro.core.config import (
    HaroConfig,
    AudioConfig,
    VADConfig,
    WakeConfig,
    STTConfig,
    TTSConfig,
    APIConfig,
    ContextConfig,
    LoggingConfig,
    CommandsConfig,
)
from haro.core.events import EventBus, EventType, StateChangeEvent
from haro.core.agent import (
    AgentState,
    StateContext,
    AgentStats,
    HaroAgent,
)


class TestAgentState:
    """Tests for AgentState enum."""

    def test_states_exist(self):
        """Test that all required states exist."""
        assert AgentState.PASSIVE
        assert AgentState.ACTIVE
        assert AgentState.PROCESSING
        assert AgentState.SPEAKING
        assert AgentState.INTERRUPTED
        assert AgentState.SLEEPING
        assert AgentState.ERROR

    def test_state_values_are_unique(self):
        """Test that state values are unique."""
        values = [s.value for s in AgentState]
        assert len(values) == len(set(values))


class TestStateContext:
    """Tests for StateContext dataclass."""

    def test_create_context(self):
        """Test creating a state context."""
        ctx = StateContext()
        assert ctx.transcript is None
        assert ctx.response is None
        assert ctx.error is None
        assert ctx.session_id is None
        assert ctx.wake_confidence == 0.0
        assert ctx.speech_start_time == 0.0
        assert ctx.last_activity_time > 0

    def test_context_with_values(self):
        """Test context with values."""
        ctx = StateContext(
            transcript="hello",
            response="hi there",
            wake_confidence=0.9,
        )
        assert ctx.transcript == "hello"
        assert ctx.response == "hi there"
        assert ctx.wake_confidence == 0.9


class TestAgentStats:
    """Tests for AgentStats dataclass."""

    def test_create_stats(self):
        """Test creating stats."""
        stats = AgentStats()
        assert stats.state_transitions == 0
        assert stats.wake_detections == 0
        assert stats.transcriptions == 0
        assert stats.api_calls == 0
        assert stats.errors == 0
        assert stats.start_time > 0

    def test_uptime(self):
        """Test uptime calculation."""
        stats = AgentStats()
        assert stats.uptime >= 0
        # Uptime should increase
        import time
        time.sleep(0.01)
        assert stats.uptime > 0


class TestHaroAgent:
    """Tests for HaroAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HaroConfig(
            device_id="test-device",
            audio=AudioConfig(sample_rate=16000, chunk_size=1024),
            vad=VADConfig(threshold=0.5, max_speech_duration=10),
            wake=WakeConfig(phrase="haro", sensitivity=0.5),
            stt=STTConfig(model="tiny.en"),
            tts=TTSConfig(engine="piper", voice="en_US-lessac-medium"),
            api=APIConfig(timeout=10),
            context=ContextConfig(session_timeout=60),
            logging=LoggingConfig(),
            commands=CommandsConfig(),
        )

    @pytest.fixture
    def agent(self, config):
        """Create agent instance."""
        return HaroAgent(config)

    def test_init(self, agent):
        """Test agent initialization."""
        assert agent.state == AgentState.PASSIVE
        assert agent.is_running is False
        assert agent.stats.state_transitions == 0

    def test_current_state(self, agent):
        """Test current state property."""
        assert agent.current_state == AgentState.PASSIVE
        agent.state = AgentState.ACTIVE
        assert agent.current_state == AgentState.ACTIVE

    @pytest.mark.asyncio
    async def test_initialize(self, agent):
        """Test agent initialization with components."""
        mock_capture = MagicMock()
        mock_stt = MagicMock()
        mock_wake = MagicMock()
        mock_wake.initialize = AsyncMock()

        await agent.initialize(
            audio_capture=mock_capture,
            stt=mock_stt,
            wake_detector=mock_wake,
        )

        assert agent._audio_capture == mock_capture
        assert agent._stt == mock_stt
        assert agent._wake_detector == mock_wake
        mock_wake.initialize.assert_called_once_with(mock_stt)

    @pytest.mark.asyncio
    async def test_stop(self, agent):
        """Test stop request."""
        await agent.stop()
        assert agent._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_interrupt(self, agent):
        """Test interrupt signal."""
        await agent.interrupt()
        assert agent._interrupt_event.is_set()

    @pytest.mark.asyncio
    async def test_transition_to(self, agent):
        """Test state transition."""
        event_bus = EventBus()
        agent.event_bus = event_bus

        events_received = []

        async def handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_CHANGED, handler)

        await agent._transition_to(AgentState.ACTIVE)

        assert agent.state == AgentState.ACTIVE
        assert agent.stats.state_transitions == 1
        assert len(events_received) == 1
        assert events_received[0].previous_state == "PASSIVE"
        assert events_received[0].new_state == "ACTIVE"

    @pytest.mark.asyncio
    async def test_handle_passive_no_components(self, agent):
        """Test passive handler with no components."""
        result = await agent._handle_passive()
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_passive_with_wake_detection(self, agent):
        """Test passive handler with wake word detection."""
        mock_capture = MagicMock()
        mock_capture.read_chunk = AsyncMock(
            return_value=np.zeros(1024, dtype=np.float32)
        )

        mock_wake = MagicMock()
        mock_wake.add_audio = MagicMock()
        mock_result = MagicMock()
        mock_result.detected = True
        mock_result.confidence = 0.9
        mock_result.text = "haro"
        mock_wake.detect = AsyncMock(return_value=mock_result)

        mock_feedback = MagicMock()
        mock_feedback.play_wake_confirmation = AsyncMock()

        agent._audio_capture = mock_capture
        agent._wake_detector = mock_wake
        agent._feedback = mock_feedback

        result = await agent._handle_passive()

        assert result == AgentState.ACTIVE
        assert agent.stats.wake_detections == 1
        mock_feedback.play_wake_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_active_missing_components(self, agent):
        """Test active handler with missing components."""
        result = await agent._handle_active()
        assert result == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_handle_processing_no_transcript(self, agent):
        """Test processing handler with no transcript."""
        agent.context.transcript = None
        result = await agent._handle_processing()
        assert result == AgentState.PASSIVE

    @pytest.mark.asyncio
    async def test_handle_processing_with_transcript(self, agent):
        """Test processing handler with transcript."""
        agent.context.transcript = "hello world"
        # No API client, should return echo response
        result = await agent._handle_processing()
        assert result == AgentState.SPEAKING
        assert "hello world" in agent.context.response

    @pytest.mark.asyncio
    async def test_handle_speaking_no_response(self, agent):
        """Test speaking handler with no response."""
        agent.context.response = None
        result = await agent._handle_speaking()
        assert result == AgentState.PASSIVE

    @pytest.mark.asyncio
    async def test_handle_interrupted(self, agent):
        """Test interrupted handler."""
        mock_playback = MagicMock()
        mock_playback.stop = AsyncMock()
        mock_playback.start = AsyncMock()

        mock_feedback = MagicMock()
        mock_feedback.play_wake_confirmation = AsyncMock()

        agent._audio_playback = mock_playback
        agent._feedback = mock_feedback

        result = await agent._handle_interrupted()

        assert result == AgentState.ACTIVE
        mock_playback.stop.assert_called_once()
        mock_playback.start.assert_called_once()
        mock_feedback.play_wake_confirmation.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_sleeping(self, agent):
        """Test sleeping handler without wake detection."""
        result = await agent._handle_sleeping()
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_sleeping_with_wake(self, agent):
        """Test sleeping handler with wake detection."""
        mock_capture = MagicMock()
        mock_capture.read_chunk = AsyncMock(
            return_value=np.zeros(1024, dtype=np.float32)
        )

        mock_wake = MagicMock()
        mock_wake.add_audio = MagicMock()
        mock_result = MagicMock()
        mock_result.detected = True
        mock_wake.detect = AsyncMock(return_value=mock_result)

        mock_feedback = MagicMock()
        mock_feedback.play_ready = AsyncMock()

        agent._audio_capture = mock_capture
        agent._wake_detector = mock_wake
        agent._feedback = mock_feedback

        result = await agent._handle_sleeping()

        assert result == AgentState.PASSIVE
        mock_feedback.play_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_error_state(self, agent):
        """Test error state handler."""
        mock_feedback = MagicMock()
        mock_feedback.play_error = AsyncMock()

        agent._feedback = mock_feedback
        agent.context.error = Exception("test error")

        result = await agent._handle_error_state()

        assert result == AgentState.PASSIVE
        assert agent.stats.errors == 1
        assert agent.context.error is None
        mock_feedback.play_error.assert_called_once()

    def test_check_local_command_stop(self, agent):
        """Test local command detection - stop."""
        result = agent._check_local_command("haro stop")
        assert result == "stop"

        result = agent._check_local_command("haro, stop")
        assert result == "stop"

    def test_check_local_command_repeat(self, agent):
        """Test local command detection - repeat."""
        result = agent._check_local_command("haro repeat that")
        assert result == "repeat"

    def test_check_local_command_volume(self, agent):
        """Test local command detection - volume."""
        result = agent._check_local_command("haro louder")
        assert result == "louder"

        result = agent._check_local_command("haro quieter")
        assert result == "quieter"

    def test_check_local_command_time(self, agent):
        """Test local command detection - time."""
        result = agent._check_local_command("haro what time is it")
        assert result == "time"

    def test_check_local_command_none(self, agent):
        """Test no local command detected."""
        result = agent._check_local_command("tell me a joke")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_local_command_stop(self, agent):
        """Test handling stop command."""
        mock_playback = MagicMock()
        mock_playback.stop = AsyncMock()
        agent._audio_playback = mock_playback

        result = await agent._handle_local_command("stop")
        assert result is None
        mock_playback.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_local_command_louder(self, agent):
        """Test handling louder command."""
        mock_playback = MagicMock()
        mock_playback.volume = 0.5
        agent._audio_playback = mock_playback

        result = await agent._handle_local_command("louder")
        assert result == "Volume increased."
        assert mock_playback.volume == 0.7

    @pytest.mark.asyncio
    async def test_handle_local_command_quieter(self, agent):
        """Test handling quieter command."""
        mock_playback = MagicMock()
        mock_playback.volume = 0.5
        agent._audio_playback = mock_playback

        result = await agent._handle_local_command("quieter")
        assert result == "Volume decreased."
        assert mock_playback.volume == 0.3

    @pytest.mark.asyncio
    async def test_handle_local_command_time(self, agent):
        """Test handling time command."""
        result = await agent._handle_local_command("time")
        assert "It's" in result
        assert "M" in result  # AM or PM

    @pytest.mark.asyncio
    async def test_handle_local_command_goodbye(self, agent):
        """Test handling goodbye command."""
        mock_feedback = MagicMock()
        mock_feedback.play_goodbye = AsyncMock()
        agent._feedback = mock_feedback

        result = await agent._handle_local_command("goodbye")
        assert result == "Goodbye!"
        mock_feedback.play_goodbye.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_local_command_new_conversation(self, agent):
        """Test handling new conversation command."""
        agent.context.transcript = "old text"
        agent.context.response = "old response"

        result = await agent._handle_local_command("new_conversation")
        assert result == "Starting a new conversation."
        assert agent.context.transcript is None
        assert agent.context.response is None

    @pytest.mark.asyncio
    async def test_cleanup(self, agent):
        """Test cleanup on shutdown."""
        mock_capture = MagicMock()
        mock_capture.stop = AsyncMock()
        mock_playback = MagicMock()
        mock_playback.stop = AsyncMock()

        agent._audio_capture = mock_capture
        agent._audio_playback = mock_playback

        await agent._cleanup()

        mock_capture.stop.assert_called_once()
        mock_playback.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_and_stop(self, agent):
        """Test running and stopping the agent."""
        mock_capture = MagicMock()
        mock_capture.start = AsyncMock()
        mock_capture.stop = AsyncMock()
        mock_capture.read_chunk = AsyncMock(return_value=None)

        mock_playback = MagicMock()
        mock_playback.start = AsyncMock()
        mock_playback.stop = AsyncMock()

        agent._audio_capture = mock_capture
        agent._audio_playback = mock_playback

        # Start agent in background
        run_task = asyncio.create_task(agent.run())

        # Let it run briefly
        await asyncio.sleep(0.1)
        assert agent.is_running

        # Stop agent
        await agent.stop()
        await asyncio.sleep(0.1)

        # Check that it stopped
        assert not agent.is_running

        # Cancel the task if still running
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass


class TestAgentIntegration:
    """Integration tests for agent with mock components."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HaroConfig(
            device_id="test-device",
            audio=AudioConfig(sample_rate=16000, chunk_size=1024),
            vad=VADConfig(threshold=0.5, max_speech_duration=10),
            wake=WakeConfig(phrase="haro", sensitivity=0.5),
            stt=STTConfig(model="tiny.en"),
            tts=TTSConfig(engine="piper", voice="en_US-lessac-medium"),
            api=APIConfig(timeout=10),
            context=ContextConfig(session_timeout=60),
            logging=LoggingConfig(),
            commands=CommandsConfig(),
        )

    @pytest.mark.asyncio
    async def test_full_wake_to_response_cycle(self, config):
        """Test a complete cycle from wake to response."""
        agent = HaroAgent(config)

        # Mock capture - returns audio chunks
        mock_capture = MagicMock()
        chunk = np.random.randn(1024).astype(np.float32) * 0.5
        mock_capture.read_chunk = AsyncMock(return_value=chunk)
        mock_capture.start = AsyncMock()
        mock_capture.stop = AsyncMock()

        # Mock playback
        mock_playback = MagicMock()
        mock_playback.start = AsyncMock()
        mock_playback.stop = AsyncMock()
        mock_playback.play = AsyncMock()
        mock_playback.is_playing = False
        mock_playback.volume = 0.8

        # Mock wake detector - detects on first call
        mock_wake = MagicMock()
        mock_wake.add_audio = MagicMock()
        mock_wake.initialize = AsyncMock()

        wake_result = MagicMock()
        wake_result.detected = True
        wake_result.confidence = 0.9
        wake_result.text = "haro"
        mock_wake.detect = AsyncMock(return_value=wake_result)

        # Mock VAD
        mock_vad = MagicMock()
        mock_vad.reset = MagicMock()
        mock_vad.speech_duration = 1.0
        vad_result = MagicMock()
        vad_result.is_speech = False  # End of speech
        mock_vad.process = MagicMock(return_value=vad_result)

        # Mock STT
        mock_stt = MagicMock()
        stt_result = MagicMock()
        stt_result.text = "hello world"
        mock_stt.transcribe = AsyncMock(return_value=stt_result)

        # Mock TTS
        mock_tts = MagicMock()
        tts_result = MagicMock()
        tts_result.audio = np.zeros(8000, dtype=np.float32)
        tts_result.sample_rate = 16000
        mock_tts.synthesize = AsyncMock(return_value=tts_result)

        # Mock feedback
        mock_feedback = MagicMock()
        mock_feedback.play_wake_confirmation = AsyncMock()

        await agent.initialize(
            audio_capture=mock_capture,
            audio_playback=mock_playback,
            wake_detector=mock_wake,
            vad=mock_vad,
            stt=mock_stt,
            tts=mock_tts,
            feedback=mock_feedback,
        )

        # Test PASSIVE → ACTIVE transition
        assert agent.state == AgentState.PASSIVE
        result = await agent._handle_passive()
        assert result == AgentState.ACTIVE
        await agent._transition_to(AgentState.ACTIVE)
        assert agent.stats.wake_detections == 1

        # Test ACTIVE → PROCESSING transition
        # Configure VAD to simulate end of speech with buffer
        mock_vad.get_speech_buffer = MagicMock(
            return_value=np.random.randn(16000).astype(np.float32)
        )
        agent.context.speech_start_time = 1.0  # Set start time

        result = await agent._handle_active()
        # Should transcribe and move to PROCESSING
        assert result == AgentState.PROCESSING
        await agent._transition_to(AgentState.PROCESSING)
        assert agent.stats.transcriptions == 1

        # Test PROCESSING → SPEAKING transition
        result = await agent._handle_processing()
        assert result == AgentState.SPEAKING
        await agent._transition_to(AgentState.SPEAKING)

        # Test SPEAKING → PASSIVE transition
        result = await agent._handle_speaking()
        assert result == AgentState.PASSIVE

        # Verify the complete cycle
        assert agent.stats.state_transitions == 3
