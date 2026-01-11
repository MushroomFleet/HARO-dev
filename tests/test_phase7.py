"""Tests for HARO Phase 7 features.

Tests for:
- Response caching
- Enhanced local commands
- Startup/shutdown lifecycle
- Performance profiling
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from haro.core.cache import CacheConfig, CacheEntry, ResponseCache
from haro.core.lifecycle import (
    LifecyclePhase,
    StartupResult,
    LifecycleManager,
    StartupChecker,
)
from haro.core.agent import HaroAgent, AgentState
from haro.core.config import HaroConfig, load_config
from haro.utils.profiling import (
    Profiler,
    TimingStats,
    LatencyTracker,
    MemoryTracker,
    timed,
    async_timed,
    get_profiler,
)


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            response="Hello!",
            created_at=time.time(),
        )
        assert entry.response == "Hello!"
        assert entry.hits == 0

    def test_entry_age(self):
        """Test age calculation."""
        entry = CacheEntry(
            response="test",
            created_at=time.time() - 10,
        )
        assert 9 <= entry.age <= 11


# =============================================================================
# CacheConfig Tests
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.max_entries == 100
        assert config.ttl_seconds == 3600.0
        assert config.similarity_threshold == 0.85

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            enabled=False,
            max_entries=50,
            ttl_seconds=1800.0,
        )
        assert config.enabled is False
        assert config.max_entries == 50
        assert config.ttl_seconds == 1800.0


# =============================================================================
# ResponseCache Tests
# =============================================================================


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_create_cache(self):
        """Test creating a cache."""
        cache = ResponseCache()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = ResponseCache()

        cache.put("What time is it?", "It's 3 PM.")
        result = cache.get("What time is it?")

        assert result == "It's 3 PM."
        assert cache.size == 1

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResponseCache()

        result = cache.get("unknown query")

        assert result is None

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = ResponseCache()

        cache.put("query1", "response1")
        cache.get("query1")  # hit
        cache.get("query1")  # hit
        cache.get("unknown")  # miss

        assert cache.hit_rate == 2 / 3

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        config = CacheConfig(ttl_seconds=0.1)
        cache = ResponseCache(config)

        cache.put("query", "response")
        assert cache.get("query") == "response"

        time.sleep(0.15)
        assert cache.get("query") is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        config = CacheConfig(max_entries=3)
        cache = ResponseCache(config)

        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.put("q3", "r3")
        cache.put("q4", "r4")  # Should evict q1

        assert cache.get("q1") is None
        assert cache.get("q2") == "r2"
        assert cache.size == 3

    def test_invalidate_specific(self):
        """Test invalidating a specific entry."""
        cache = ResponseCache()

        cache.put("q1", "r1")
        cache.put("q2", "r2")

        count = cache.invalidate("q1")

        assert count == 1
        assert cache.get("q1") is None
        assert cache.get("q2") == "r2"

    def test_invalidate_all(self):
        """Test invalidating all entries."""
        cache = ResponseCache()

        cache.put("q1", "r1")
        cache.put("q2", "r2")

        count = cache.invalidate()

        assert count == 2
        assert cache.size == 0

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        config = CacheConfig(ttl_seconds=0.1)
        cache = ResponseCache(config)

        cache.put("q1", "r1")
        time.sleep(0.15)
        cache.put("q2", "r2")

        removed = cache.cleanup_expired()

        assert removed == 1
        assert cache.get("q2") == "r2"

    def test_disabled_cache(self):
        """Test that disabled cache doesn't store."""
        config = CacheConfig(enabled=False)
        cache = ResponseCache(config)

        cache.put("query", "response")
        result = cache.get("query")

        assert result is None
        assert cache.size == 0

    def test_similar_query_matching(self):
        """Test fuzzy matching for similar queries."""
        config = CacheConfig(similarity_threshold=0.5)
        cache = ResponseCache(config)

        cache.put("what is the weather today", "It's sunny.")

        # Similar query should match
        result = cache.get("what is weather today")
        assert result == "It's sunny."

    def test_normalize_key(self):
        """Test key normalization."""
        cache = ResponseCache()

        cache.put("please tell me the time", "3 PM")
        result = cache.get("please tell me the time")

        # Should match same query
        assert result == "3 PM"

        # Test that filler words are removed in normalization
        cache.put("can you tell me something", "Sure!")
        result2 = cache.get("tell me something")
        # Should match due to "can you" being removed
        assert result2 == "Sure!"

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = ResponseCache()

        cache.put("q1", "r1")
        cache.get("q1")  # hit
        cache.get("q2")  # miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# =============================================================================
# Profiler Tests
# =============================================================================


class TestTimingStats:
    """Tests for TimingStats dataclass."""

    def test_create_stats(self):
        """Test creating timing stats."""
        stats = TimingStats(name="test_op")
        assert stats.name == "test_op"
        assert stats.call_count == 0
        assert stats.avg_time == 0.0

    def test_record_timing(self):
        """Test recording a timing."""
        stats = TimingStats(name="test")

        stats.record(0.1)
        stats.record(0.2)
        stats.record(0.3)

        assert stats.call_count == 3
        assert abs(stats.total_time - 0.6) < 0.0001  # Float comparison
        assert abs(stats.avg_time - 0.2) < 0.0001
        assert stats.min_time == 0.1
        assert stats.max_time == 0.3

    def test_to_dict(self):
        """Test converting to dictionary."""
        stats = TimingStats(name="test")
        stats.record(0.1)

        d = stats.to_dict()

        assert d["name"] == "test"
        assert d["call_count"] == 1
        assert d["total_time_ms"] == 100.0


class TestProfiler:
    """Tests for Profiler class."""

    def test_singleton(self):
        """Test profiler is a singleton."""
        p1 = Profiler.get_instance()
        p2 = Profiler.get_instance()
        assert p1 is p2

    def test_record(self):
        """Test recording a timing."""
        profiler = Profiler()

        profiler.record("test_op", 0.5)

        stats = profiler.get_stats("test_op")
        assert stats["call_count"] == 1

    def test_time_block(self):
        """Test timing a code block."""
        profiler = Profiler()

        with profiler.time_block("test_block"):
            time.sleep(0.05)

        stats = profiler.get_stats("test_block")
        assert stats["call_count"] == 1
        assert stats["total_time_ms"] >= 40  # At least 40ms

    def test_disabled_profiler(self):
        """Test disabled profiler doesn't record."""
        profiler = Profiler()
        profiler.enabled = False

        profiler.record("test", 0.1)

        stats = profiler.get_stats("test")
        assert stats == {}

    def test_get_summary(self):
        """Test getting summary."""
        profiler = Profiler()
        profiler.reset()

        profiler.record("op1", 0.1)
        profiler.record("op2", 0.2)

        summary = profiler.get_summary()

        assert "op1" in summary
        assert "op2" in summary

    def test_reset(self):
        """Test resetting statistics."""
        profiler = Profiler()
        profiler.record("test", 0.1)

        profiler.reset()

        stats = profiler.get_stats()
        assert len(stats["operations"]) == 0


class TestTimedDecorators:
    """Tests for timing decorators."""

    def test_timed_sync(self):
        """Test @timed decorator."""
        profiler = Profiler.get_instance()
        profiler.reset()

        @timed(name="sync_func")
        def my_func():
            time.sleep(0.01)
            return "result"

        result = my_func()

        assert result == "result"
        stats = profiler.get_stats("sync_func")
        assert stats["call_count"] == 1

    @pytest.mark.asyncio
    async def test_async_timed(self):
        """Test @async_timed decorator."""
        profiler = Profiler.get_instance()
        profiler.reset()

        @async_timed(name="async_func")
        async def my_async_func():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await my_async_func()

        assert result == "async_result"
        stats = profiler.get_stats("async_func")
        assert stats["call_count"] == 1


class TestLatencyTracker:
    """Tests for LatencyTracker."""

    def test_basic_tracking(self):
        """Test basic latency tracking."""
        tracker = LatencyTracker()

        tracker.start()
        time.sleep(0.05)
        tracker.checkpoint("step1")
        time.sleep(0.05)
        total = tracker.end()

        assert total >= 0.08
        assert "step1" in tracker.checkpoints

    def test_breakdown(self):
        """Test latency breakdown."""
        tracker = LatencyTracker()

        tracker.start()
        tracker.checkpoint("a")
        time.sleep(0.01)
        tracker.checkpoint("b")

        breakdown = tracker.get_breakdown()

        assert "a" in breakdown
        assert "b" in breakdown
        assert breakdown["b"]["elapsed_ms"] > breakdown["a"]["elapsed_ms"]


class TestMemoryTracker:
    """Tests for MemoryTracker."""

    def test_availability(self):
        """Test memory tracker availability check."""
        tracker = MemoryTracker()
        # Just check it doesn't crash
        _ = tracker.available
        _ = tracker.get_summary()

    def test_get_current_mb(self):
        """Test getting current memory."""
        tracker = MemoryTracker()

        if tracker.available:
            mb = tracker.get_current_mb()
            assert mb > 0
        else:
            assert tracker.get_current_mb() == 0


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestLifecyclePhase:
    """Tests for LifecyclePhase enum."""

    def test_phases(self):
        """Test lifecycle phases exist."""
        assert LifecyclePhase.NOT_STARTED
        assert LifecyclePhase.INITIALIZING
        assert LifecyclePhase.RUNNING
        assert LifecyclePhase.STOPPING
        assert LifecyclePhase.STOPPED


class TestStartupResult:
    """Tests for StartupResult dataclass."""

    def test_default_success(self):
        """Test default result is success."""
        result = StartupResult(
            success=True,
            phase=LifecyclePhase.RUNNING,
        )
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []

    def test_with_errors(self):
        """Test result with errors."""
        result = StartupResult(
            success=False,
            phase=LifecyclePhase.ERROR,
            errors=["Config invalid"],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestLifecycleManager:
    """Tests for LifecycleManager class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return HaroConfig()

    def test_create_manager(self, config):
        """Test creating lifecycle manager."""
        manager = LifecycleManager(config)

        assert manager.phase == LifecyclePhase.NOT_STARTED
        assert manager.is_running is False

    def test_register_component(self, config):
        """Test registering a component."""
        manager = LifecycleManager(config)
        mock_component = MagicMock()

        manager.register_component("test", mock_component)

        assert "test" in manager._components

    def test_register_shutdown_handler(self, config):
        """Test registering shutdown handler."""
        manager = LifecycleManager(config)
        handler = AsyncMock()

        manager.register_shutdown_handler(handler)

        assert handler in manager._shutdown_handlers

    @pytest.mark.asyncio
    async def test_startup(self, config):
        """Test startup sequence."""
        manager = LifecycleManager(config)

        result = await manager.startup()

        assert result.success is True
        assert "config" in result.initialized_components
        assert manager.phase == LifecyclePhase.STARTING

    @pytest.mark.asyncio
    async def test_shutdown(self, config):
        """Test shutdown sequence."""
        manager = LifecycleManager(config)
        await manager.startup()

        await manager.shutdown()

        assert manager.phase == LifecyclePhase.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_handlers_called(self, config):
        """Test shutdown handlers are called."""
        manager = LifecycleManager(config)
        handler = AsyncMock()
        manager.register_shutdown_handler(handler)

        await manager.startup()
        await manager.shutdown()

        handler.assert_called_once()


class TestStartupChecker:
    """Tests for StartupChecker class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return HaroConfig()

    @pytest.mark.asyncio
    async def test_run_checks(self, config):
        """Test running startup checks."""
        checker = StartupChecker(config)

        results = await checker.run_checks()

        assert "checks" in results
        assert "python_version" in results["checks"]
        assert "dependencies" in results["checks"]

    def test_check_python(self, config):
        """Test Python version check."""
        checker = StartupChecker(config)

        result = checker._check_python()

        assert result["passed"] is True
        assert "Python" in result["message"]


# =============================================================================
# Agent Local Commands Tests
# =============================================================================


class TestAgentLocalCommands:
    """Tests for enhanced local commands."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        config = HaroConfig()
        return HaroAgent(config)

    def test_check_mute(self, agent):
        """Test mute command detection."""
        assert agent._check_local_command("mute") == "mute"
        assert agent._check_local_command("silence") == "mute"

    def test_check_unmute(self, agent):
        """Test unmute command detection."""
        assert agent._check_local_command("unmute") == "unmute"

    def test_check_pause(self, agent):
        """Test pause command detection."""
        assert agent._check_local_command("pause") == "pause"
        assert agent._check_local_command("hold on") == "pause"
        assert agent._check_local_command("wait") == "pause"

    def test_check_date(self, agent):
        """Test date command detection."""
        assert agent._check_local_command("what's the date") == "date"
        assert agent._check_local_command("what day is it") == "date"
        assert agent._check_local_command("today's date") == "date"

    def test_check_status(self, agent):
        """Test status command detection."""
        assert agent._check_local_command("status") == "status"
        assert agent._check_local_command("how are you") == "status"
        assert agent._check_local_command("system status") == "status"

    def test_check_help(self, agent):
        """Test help command detection."""
        assert agent._check_local_command("help") == "help"
        assert agent._check_local_command("what can you do") == "help"
        assert agent._check_local_command("commands") == "help"

    def test_check_clear_cache(self, agent):
        """Test clear cache command detection."""
        assert agent._check_local_command("clear cache") == "clear_cache"
        assert agent._check_local_command("forget responses") == "clear_cache"

    def test_check_exit_commands(self, agent):
        """Test exit/quit commands."""
        assert agent._check_local_command("exit") == "goodbye"
        assert agent._check_local_command("quit") == "goodbye"

    def test_check_cancel(self, agent):
        """Test cancel command."""
        assert agent._check_local_command("cancel") == "stop"

    @pytest.mark.asyncio
    async def test_handle_mute(self, agent):
        """Test handling mute command."""
        agent._audio_playback = MagicMock()
        agent._audio_playback.volume = 0.7

        response = await agent._handle_local_command("mute")

        assert response == "Muted."
        assert agent._audio_playback.volume == 0.0

    @pytest.mark.asyncio
    async def test_handle_unmute(self, agent):
        """Test handling unmute command."""
        agent._audio_playback = MagicMock()
        agent._audio_playback.volume = 0.0

        response = await agent._handle_local_command("unmute")

        assert response == "Unmuted."
        assert agent._audio_playback.volume == 0.7

    @pytest.mark.asyncio
    async def test_handle_pause(self, agent):
        """Test handling pause command."""
        response = await agent._handle_local_command("pause")
        assert response == "I'm listening."

    @pytest.mark.asyncio
    async def test_handle_date(self, agent):
        """Test handling date command."""
        response = await agent._handle_local_command("date")
        assert "Today is" in response

    @pytest.mark.asyncio
    async def test_handle_status(self, agent):
        """Test handling status command."""
        response = await agent._handle_local_command("status")
        assert "running" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_help(self, agent):
        """Test handling help command."""
        response = await agent._handle_local_command("help")
        assert "stop" in response.lower()
        assert "repeat" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_clear_cache(self, agent):
        """Test handling clear cache command."""
        # Add something to cache first
        agent._cache.put("test", "response")

        response = await agent._handle_local_command("clear_cache")

        assert "Cleared" in response
        assert agent._cache.size == 0

    @pytest.mark.asyncio
    async def test_handle_repeat_no_previous(self, agent):
        """Test repeat when no previous response."""
        agent._last_response = None

        response = await agent._handle_local_command("repeat")

        assert "don't have anything" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_repeat_with_previous(self, agent):
        """Test repeat with previous response."""
        agent._last_response = "The weather is sunny."

        response = await agent._handle_local_command("repeat")

        assert response == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_handle_new_conversation_clears_history(self, agent):
        """Test new conversation clears API history."""
        agent._api_client = MagicMock()

        response = await agent._handle_local_command("new_conversation")

        agent._api_client.clear_history.assert_called_once()
        assert "new conversation" in response.lower()


class TestAgentCaching:
    """Tests for agent response caching."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        config = HaroConfig()
        return HaroAgent(config)

    @pytest.mark.asyncio
    async def test_cache_stores_response(self, agent):
        """Test that responses are cached."""
        # Set up mock API client
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value=MagicMock(text="Test response")
        )
        agent._api_client = mock_client
        await agent.initialize(api_client=mock_client)

        # First call - should hit API
        response1 = await agent._call_api("test query")
        assert response1 == "Test response"

        # Second call - should hit cache
        response2 = await agent._call_api("test query")
        assert response2 == "Test response"

        # API should only be called once
        assert mock_client.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_last_response_stored(self, agent):
        """Test that last response is stored for repeat."""
        agent.context.response = "Hello there!"
        agent.context.transcript = "Hi"

        await agent._handle_speaking()

        assert agent._last_response == "Hello there!"
        assert agent._last_transcript == "Hi"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase7Integration:
    """Integration tests for Phase 7 features."""

    @pytest.mark.asyncio
    async def test_cache_with_profiling(self):
        """Test cache with profiling enabled."""
        cache = ResponseCache()
        profiler = get_profiler()
        profiler.reset()

        with profiler.time_block("cache_operations"):
            cache.put("q1", "r1")
            cache.put("q2", "r2")
            cache.get("q1")
            cache.get("q2")
            cache.get("q3")

        stats = profiler.get_stats("cache_operations")
        assert stats["call_count"] == 1

        cache_stats = cache.get_stats()
        assert cache_stats["hits"] == 2
        assert cache_stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_lifecycle_with_shutdown_cleanup(self):
        """Test lifecycle cleanup on shutdown."""
        config = HaroConfig()
        manager = LifecycleManager(config)

        cleanup_called = False

        async def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        manager.register_shutdown_handler(cleanup)

        await manager.startup()
        await manager.shutdown()

        assert cleanup_called is True
        assert manager.phase == LifecyclePhase.STOPPED
