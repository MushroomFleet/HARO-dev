"""Tests for HARO context module.

Tests session management, knowledge base, and context manager.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from haro.context.session import Turn, Session, SessionManager
from haro.context.knowledge import KnowledgeBase, KnowledgeEntry
from haro.context.manager import ContextManager
from haro.core.config import ContextConfig, APIConfig


# =============================================================================
# Turn Tests
# =============================================================================


class TestTurn:
    """Tests for Turn dataclass."""

    def test_turn_creation(self):
        """Test creating a turn."""
        turn = Turn(
            user_input="Hello",
            response="Hi there!",
        )
        assert turn.user_input == "Hello"
        assert turn.response == "Hi there!"
        assert isinstance(turn.timestamp, datetime)
        assert turn.metadata == {}

    def test_turn_with_metadata(self):
        """Test turn with custom metadata."""
        turn = Turn(
            user_input="What time is it?",
            response="It's 3 PM.",
            metadata={"confidence": 0.95, "local_command": False},
        )
        assert turn.metadata["confidence"] == 0.95
        assert turn.metadata["local_command"] is False

    def test_turn_with_custom_timestamp(self):
        """Test turn with custom timestamp."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0)
        turn = Turn(
            user_input="Test",
            response="Response",
            timestamp=custom_time,
        )
        assert turn.timestamp == custom_time


# =============================================================================
# Session Tests
# =============================================================================


class TestSession:
    """Tests for Session class."""

    def test_session_creation(self, tmp_path):
        """Test creating a session."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
            device_id="test-device",
            model="claude-3-sonnet",
        )
        assert session.id == "test-session"
        assert session.device_id == "test-device"
        assert session.model == "claude-3-sonnet"
        assert session.turn_count == 0
        assert session.turns == []

    def test_session_add_turn(self, tmp_path):
        """Test adding turns to a session."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        turn = session.add_turn("Hello", "Hi there!")

        assert session.turn_count == 1
        assert turn.user_input == "Hello"
        assert turn.response == "Hi there!"
        assert session.turns[0] == turn

    def test_session_add_turn_with_metadata(self, tmp_path):
        """Test adding turn with metadata."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        turn = session.add_turn(
            "What's the weather?",
            "I don't know.",
            confidence=0.5,
            source="api",
        )

        assert turn.metadata["confidence"] == 0.5
        assert turn.metadata["source"] == "api"

    def test_session_turn_count(self, tmp_path):
        """Test turn count property."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        assert session.turn_count == 0

        session.add_turn("One", "Response 1")
        assert session.turn_count == 1

        session.add_turn("Two", "Response 2")
        session.add_turn("Three", "Response 3")
        assert session.turn_count == 3

    def test_session_duration(self, tmp_path):
        """Test duration calculation."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
            started=datetime.now() - timedelta(seconds=60),
        )

        # No turns = 0 duration
        assert session.duration == 0.0

        # Add a turn with timestamp 30 seconds after start
        turn = session.add_turn("Hello", "Hi")
        turn.timestamp = session.started + timedelta(seconds=30)

        assert 29 <= session.duration <= 31

    def test_session_is_active(self, tmp_path):
        """Test is_active property."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        assert not session.is_active  # File doesn't exist yet

        session.add_turn("Hello", "Hi")
        session.save()

        assert session.is_active  # File exists now

    def test_session_get_history(self, tmp_path):
        """Test getting formatted history."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        session.add_turn("Hello", "Hi there!")
        session.add_turn("How are you?", "I'm doing well.")

        history = session.get_history()

        assert "User: Hello" in history
        assert "HARO: Hi there!" in history
        assert "User: How are you?" in history
        assert "HARO: I'm doing well." in history

    def test_session_get_history_with_limit(self, tmp_path):
        """Test history with limit."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        session.add_turn("One", "R1")
        session.add_turn("Two", "R2")
        session.add_turn("Three", "R3")

        history = session.get_history(limit=2)

        assert "User: One" not in history
        assert "User: Two" in history
        assert "User: Three" in history

    def test_session_get_messages(self, tmp_path):
        """Test getting messages for API."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        session.add_turn("Hello", "Hi!")
        session.add_turn("Goodbye", "Bye!")

        messages = session.get_messages()

        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}
        assert messages[2] == {"role": "user", "content": "Goodbye"}
        assert messages[3] == {"role": "assistant", "content": "Bye!"}

    def test_session_get_messages_with_limit(self, tmp_path):
        """Test messages with limit."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
        )

        session.add_turn("One", "R1")
        session.add_turn("Two", "R2")
        session.add_turn("Three", "R3")

        messages = session.get_messages(limit=1)

        assert len(messages) == 2
        assert messages[0]["content"] == "Three"

    def test_session_save(self, tmp_path):
        """Test saving session to file."""
        session = Session(
            id="test-session",
            path=tmp_path / "sessions" / "test-session.md",
            device_id="steam-deck",
            model="claude-3-sonnet",
        )

        session.add_turn("What's my name?", "I don't know your name yet.")
        session.save()

        assert session.path.exists()
        content = session.path.read_text()

        assert "# Session test-session" in content
        assert "**Device**: steam-deck" in content
        assert "**Model**: claude-3-sonnet" in content
        assert "**User**: What's my name?" in content
        assert "**HARO**: I don't know your name yet." in content

    def test_session_save_with_metadata(self, tmp_path):
        """Test saving session with custom metadata."""
        session = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
            metadata={"version": "1.0", "test_mode": True},
        )

        session.save()
        content = session.path.read_text()

        assert "**version**: 1.0" in content
        assert "**test_mode**: True" in content

    def test_session_load(self, tmp_path):
        """Test loading session from file."""
        # Create and save a session
        original = Session(
            id="test-session",
            path=tmp_path / "test-session.md",
            device_id="device-123",
            model="claude-3-opus",
        )
        original.add_turn("Hello", "Hi!")
        original.save()

        # Load it back
        loaded = Session.load(original.path)

        assert loaded is not None
        assert loaded.id == "test-session"
        assert loaded.device_id == "device-123"
        assert loaded.model == "claude-3-opus"
        assert loaded.turn_count == 1
        assert loaded.turns[0].user_input == "Hello"
        assert loaded.turns[0].response == "Hi!"

    def test_session_load_nonexistent(self, tmp_path):
        """Test loading non-existent session."""
        result = Session.load(tmp_path / "does-not-exist.md")
        assert result is None


# =============================================================================
# SessionManager Tests
# =============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_session_manager_creation(self, tmp_path):
        """Test creating session manager."""
        manager = SessionManager(tmp_path / "sessions")

        assert manager.sessions_path == tmp_path / "sessions"
        assert manager.current_session is None

    def test_start_session(self, tmp_path):
        """Test starting a new session."""
        manager = SessionManager(tmp_path / "sessions")

        session = manager.start_session(
            device_id="test-device",
            model="claude-3-sonnet",
        )

        assert session is not None
        assert manager.current_session == session
        assert session.device_id == "test-device"
        assert session.model == "claude-3-sonnet"
        assert len(session.id) > 0

    def test_start_session_with_metadata(self, tmp_path):
        """Test starting session with extra metadata."""
        manager = SessionManager(tmp_path / "sessions")

        session = manager.start_session(
            device_id="test",
            model="test",
            custom_field="custom_value",
        )

        assert session.metadata["custom_field"] == "custom_value"

    def test_end_session(self, tmp_path):
        """Test ending a session."""
        manager = SessionManager(tmp_path / "sessions")

        session = manager.start_session()
        session.add_turn("Hello", "Hi")

        ended = manager.end_session()

        assert ended == session
        assert manager.current_session is None
        assert ended.path.exists()  # Should be saved

    def test_end_session_no_save_if_empty(self, tmp_path):
        """Test that empty sessions aren't saved."""
        manager = SessionManager(tmp_path / "sessions")

        session = manager.start_session()
        manager.end_session()

        # Empty session shouldn't be saved
        assert not session.path.exists()

    def test_end_session_when_none(self, tmp_path):
        """Test ending when no session exists."""
        manager = SessionManager(tmp_path / "sessions")

        result = manager.end_session()
        assert result is None

    def test_log_turn(self, tmp_path):
        """Test logging a turn."""
        manager = SessionManager(tmp_path / "sessions")
        manager.start_session()

        turn = manager.log_turn("Hello", "Hi!")

        assert turn is not None
        assert turn.user_input == "Hello"
        assert turn.response == "Hi!"
        assert manager.current_session.turn_count == 1

    def test_log_turn_auto_starts_session(self, tmp_path):
        """Test that log_turn starts a session if needed."""
        manager = SessionManager(tmp_path / "sessions")

        assert manager.current_session is None

        turn = manager.log_turn("Hello", "Hi!")

        assert turn is not None
        assert manager.current_session is not None
        assert manager.current_session.turn_count == 1

    def test_log_turn_auto_save(self, tmp_path):
        """Test auto-save on turn logging."""
        manager = SessionManager(tmp_path / "sessions")
        manager.start_session()

        manager.log_turn("Test", "Response", auto_save=True)

        assert manager.current_session.path.exists()

    def test_log_turn_no_auto_save(self, tmp_path):
        """Test disabling auto-save."""
        manager = SessionManager(tmp_path / "sessions")
        manager.start_session()

        manager.log_turn("Test", "Response", auto_save=False)

        assert not manager.current_session.path.exists()

    def test_list_sessions(self, tmp_path):
        """Test listing sessions."""
        manager = SessionManager(tmp_path / "sessions")

        # Create sessions with unique IDs by modifying their paths
        for i in range(3):
            session = manager.start_session()
            # Give each session a unique ID to avoid timestamp collision
            session.id = f"session_{i}"
            session.path = tmp_path / "sessions" / f"session_{i}.md"
            session.add_turn(f"Turn {i}", f"Response {i}")
            manager.end_session()

        sessions = manager.list_sessions()

        assert len(sessions) == 3

    def test_list_sessions_with_limit(self, tmp_path):
        """Test listing sessions with limit."""
        manager = SessionManager(tmp_path / "sessions")

        # Create sessions with unique IDs
        for i in range(5):
            session = manager.start_session()
            session.id = f"session_{i}"
            session.path = tmp_path / "sessions" / f"session_{i}.md"
            session.add_turn("Test", "Response")
            manager.end_session()

        sessions = manager.list_sessions(limit=2)

        assert len(sessions) == 2

    def test_list_sessions_empty(self, tmp_path):
        """Test listing sessions when none exist."""
        manager = SessionManager(tmp_path / "sessions")

        sessions = manager.list_sessions()

        assert sessions == []

    def test_load_session(self, tmp_path):
        """Test loading a session by ID."""
        manager = SessionManager(tmp_path / "sessions")

        # Create and save a session
        original = manager.start_session()
        original.add_turn("Hello", "Hi")
        session_id = original.id
        manager.end_session()

        # Load it back
        loaded = manager.load_session(session_id)

        assert loaded is not None
        assert loaded.id == session_id
        assert loaded.turn_count == 1

    def test_load_session_nonexistent(self, tmp_path):
        """Test loading non-existent session."""
        manager = SessionManager(tmp_path / "sessions")

        result = manager.load_session("does-not-exist")

        assert result is None

    def test_get_recent_turns(self, tmp_path):
        """Test getting recent turns across sessions."""
        manager = SessionManager(tmp_path / "sessions")

        # Create sessions with turns
        for i in range(3):
            session = manager.start_session()
            session.add_turn(f"Q{i}", f"A{i}")
            manager.end_session()

        turns = manager.get_recent_turns(limit=2)

        assert len(turns) <= 2


# =============================================================================
# KnowledgeEntry Tests
# =============================================================================


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry dataclass."""

    def test_knowledge_entry_creation(self):
        """Test creating a knowledge entry."""
        entry = KnowledgeEntry(
            key="location",
            value="Seattle, WA",
        )
        assert entry.key == "location"
        assert entry.value == "Seattle, WA"
        assert entry.category == "facts"
        assert entry.source == ""

    def test_knowledge_entry_with_all_fields(self):
        """Test entry with all fields."""
        entry = KnowledgeEntry(
            key="name",
            value="John",
            category="preferences",
            source="conversation",
        )
        assert entry.category == "preferences"
        assert entry.source == "conversation"
        assert isinstance(entry.timestamp, datetime)


# =============================================================================
# KnowledgeBase Tests
# =============================================================================


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""

    def test_knowledge_base_creation(self, tmp_path):
        """Test creating a knowledge base."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        assert kb.knowledge_path == tmp_path / "knowledge"

    def test_ensure_structure(self, tmp_path):
        """Test ensuring directory structure."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        assert not kb.knowledge_path.exists()

        kb.ensure_structure()

        assert kb.knowledge_path.exists()

    def test_add_preference(self, tmp_path):
        """Test adding a preference."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_preference("name", "Alice", source="user")

        # Check file was created
        pref_file = kb.knowledge_path / "user_preferences.md"
        assert pref_file.exists()

        content = pref_file.read_text()
        assert "## name" in content
        assert "Alice" in content
        assert "*Source: user*" in content

    def test_add_fact(self, tmp_path):
        """Test adding a fact."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_fact("capital", "Paris is the capital of France")

        facts_file = kb.knowledge_path / "facts.md"
        assert facts_file.exists()

        content = facts_file.read_text()
        assert "## capital" in content
        assert "Paris is the capital of France" in content

    def test_add_correction(self, tmp_path):
        """Test adding a correction."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_correction(
            original="The sky is green",
            corrected="The sky is blue",
            context="Weather discussion",
        )

        corrections_file = kb.knowledge_path / "corrections.md"
        assert corrections_file.exists()

        content = corrections_file.read_text()
        assert "**Original**: The sky is green" in content
        assert "**Corrected**: The sky is blue" in content
        assert "**Context**: Weather discussion" in content

    def test_get_preference(self, tmp_path):
        """Test getting a preference."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_preference("favorite_color", "blue")

        result = kb.get_preference("favorite_color")

        assert result is not None
        assert "blue" in result

    def test_get_preference_not_found(self, tmp_path):
        """Test getting non-existent preference."""
        kb = KnowledgeBase(tmp_path / "knowledge")
        kb.ensure_structure()

        result = kb.get_preference("does_not_exist")

        assert result is None

    def test_get_fact(self, tmp_path):
        """Test getting a fact."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_fact("pi", "The value of pi is approximately 3.14159")

        result = kb.get_fact("pi")

        assert result is not None
        assert "3.14159" in result

    def test_search(self, tmp_path):
        """Test searching knowledge."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_preference("location", "Seattle, Washington")
        kb.add_fact("weather", "Seattle is known for rainy weather")
        kb.add_fact("space", "The moon orbits Earth")

        results = kb.search("Seattle")

        assert len(results) >= 1
        # Both Seattle entries should match
        keys = [r.key for r in results]
        assert "location" in keys or "weather" in keys

    def test_search_no_results(self, tmp_path):
        """Test search with no matches."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_fact("test", "Some content")

        results = kb.search("xyznonexistent")

        assert len(results) == 0

    def test_search_with_limit(self, tmp_path):
        """Test search with result limit."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        # Add many facts with common keyword
        for i in range(10):
            kb.add_fact(f"fact{i}", f"Python is great {i}")

        results = kb.search("Python", limit=3)

        assert len(results) <= 3

    def test_search_removes_stop_words(self, tmp_path):
        """Test that stop words don't affect search."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_fact("coffee", "I like coffee in the morning")

        # Search with stop words
        results = kb.search("the coffee is what")

        # Should still find coffee entry
        assert len(results) >= 1

    def test_get_all_preferences(self, tmp_path):
        """Test getting all preferences."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_preference("name", "Alice")
        kb.add_preference("location", "Seattle")

        prefs = kb.get_all_preferences()

        assert len(prefs) == 2
        keys = [p.key for p in prefs]
        assert "name" in keys
        assert "location" in keys

    def test_get_all_preferences_empty(self, tmp_path):
        """Test getting preferences when none exist."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        prefs = kb.get_all_preferences()

        assert prefs == []

    def test_get_all_facts(self, tmp_path):
        """Test getting all facts."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_fact("fact1", "Value 1")
        kb.add_fact("fact2", "Value 2")

        facts = kb.get_all_facts()

        assert len(facts) == 2

    def test_get_summary(self, tmp_path):
        """Test getting knowledge base summary."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        kb.add_preference("name", "Alice")
        kb.add_fact("fact1", "Value")
        kb.add_correction("wrong", "right")

        summary = kb.get_summary()

        assert summary["preferences_count"] == 1
        assert summary["facts_count"] == 1
        assert summary["corrections_count"] == 1
        assert summary["total_files"] == 3
        assert summary["total_size_kb"] > 0

    def test_get_summary_empty(self, tmp_path):
        """Test summary when empty."""
        kb = KnowledgeBase(tmp_path / "knowledge")

        summary = kb.get_summary()

        assert summary["preferences_count"] == 0
        assert summary["facts_count"] == 0
        assert summary["total_files"] == 0


# =============================================================================
# ContextManager Tests
# =============================================================================


class TestContextManager:
    """Tests for ContextManager class."""

    @pytest.fixture
    def context_config(self, tmp_path):
        """Create a test context config."""
        return ContextConfig(
            path=str(tmp_path / ".context"),
            history_turns=10,
            session_timeout=3600,
            auto_save=True,
        )

    @pytest.fixture
    def api_config(self):
        """Create a test API config."""
        return APIConfig(
            model="claude-3-sonnet",
            max_tokens=1000,
            temperature=0.7,
            timeout=30,
        )

    def test_context_manager_creation(self, context_config, api_config):
        """Test creating context manager."""
        cm = ContextManager(context_config, api_config)

        assert cm.config == context_config
        assert cm.api_config == api_config
        assert cm.history_turns == 10
        assert cm.current_session is None

    def test_ensure_structure(self, context_config):
        """Test ensuring directory structure."""
        cm = ContextManager(context_config)

        cm.ensure_structure()

        assert cm.base_path.exists()
        assert cm.sessions_path.exists()
        assert cm.knowledge_path.exists()
        assert cm.config_path.exists()

    def test_start_session(self, context_config, api_config):
        """Test starting a session."""
        cm = ContextManager(context_config, api_config)
        cm.ensure_structure()

        session = cm.start_session(device_id="test-device")

        assert session is not None
        assert cm.current_session == session
        assert session.device_id == "test-device"
        assert session.model == "claude-3-sonnet"

    def test_start_session_without_api_config(self, context_config):
        """Test starting session without API config."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        session = cm.start_session()

        assert session.model == "unknown"

    def test_end_session(self, context_config):
        """Test ending a session."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        session = cm.start_session()
        session.add_turn("Hello", "Hi")

        ended = cm.end_session()

        assert ended == session
        assert cm.current_session is None

    def test_log_turn(self, context_config):
        """Test logging a turn."""
        cm = ContextManager(context_config)
        cm.ensure_structure()
        cm.start_session()

        turn = cm.log_turn("What time is it?", "I don't know.")

        assert turn is not None
        assert cm.current_session.turn_count == 1

    def test_get_conversation_history(self, context_config):
        """Test getting conversation history."""
        cm = ContextManager(context_config)
        cm.ensure_structure()
        cm.start_session()

        cm.log_turn("Hello", "Hi!")
        cm.log_turn("How are you?", "Great!")

        history = cm.get_conversation_history()

        assert "User: Hello" in history
        assert "HARO: Hi!" in history

    def test_get_conversation_history_no_session(self, context_config):
        """Test getting history when no session."""
        cm = ContextManager(context_config)

        history = cm.get_conversation_history()

        assert history == ""

    def test_get_messages_for_api(self, context_config):
        """Test getting messages for API."""
        cm = ContextManager(context_config)
        cm.ensure_structure()
        cm.start_session()

        cm.log_turn("Hello", "Hi!")

        messages = cm.get_messages_for_api()

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_get_messages_for_api_no_session(self, context_config):
        """Test getting messages when no session."""
        cm = ContextManager(context_config)

        messages = cm.get_messages_for_api()

        assert messages == []

    def test_process_knowledge_updates(self, context_config):
        """Test processing knowledge updates."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        updates = [
            KnowledgeEntry(key="name", value="Bob", category="preferences"),
            KnowledgeEntry(key="fact1", value="Test fact", category="facts"),
        ]

        cm.process_knowledge_updates(updates)

        # Verify they were added
        assert cm.knowledge.get_preference("name") is not None
        assert cm.knowledge.get_fact("fact1") is not None

    def test_search_knowledge(self, context_config):
        """Test searching knowledge."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        cm.knowledge.add_preference("city", "Seattle")

        results = cm.search_knowledge("Seattle")

        assert len(results) >= 1

    def test_get_relevant_context(self, context_config):
        """Test getting relevant context."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        cm.knowledge.add_fact("weather", "Seattle has rainy weather")

        context = cm.get_relevant_context("What's the weather like?")

        assert "## Relevant Knowledge" in context
        assert "weather" in context.lower()

    def test_get_relevant_context_empty(self, context_config):
        """Test relevant context when nothing matches."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        context = cm.get_relevant_context("random query xyz")

        assert context == ""

    def test_get_user_preferences_summary(self, context_config):
        """Test getting preferences summary."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        cm.knowledge.add_preference("name", "Alice")
        cm.knowledge.add_preference("location", "NYC")

        summary = cm.get_user_preferences_summary()

        assert "## User Preferences" in summary
        assert "**name**" in summary
        assert "**location**" in summary

    def test_get_user_preferences_summary_empty(self, context_config):
        """Test preferences summary when empty."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        summary = cm.get_user_preferences_summary()

        assert summary == ""

    def test_load_context_file(self, context_config, tmp_path):
        """Test loading a context file."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        # Create a test file
        test_file = cm.base_path / "test.md"
        test_file.write_text("# Test Content")

        content = cm.load_context_file("test.md")

        assert content == "# Test Content"

    def test_load_context_file_not_found(self, context_config):
        """Test loading non-existent file."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        content = cm.load_context_file("nonexistent.md")

        assert content is None

    def test_get_stats(self, context_config):
        """Test getting statistics."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        # Create some data
        cm.start_session()
        cm.log_turn("Hello", "Hi")
        cm.knowledge.add_preference("test", "value")

        stats = cm.get_stats()

        assert stats["current_session_turns"] == 1
        assert stats["preferences_count"] == 1
        assert stats["knowledge_files"] >= 1

    def test_get_stats_no_session(self, context_config):
        """Test stats without session."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        stats = cm.get_stats()

        assert stats["current_session_turns"] == 0
        assert stats["session_count"] == 0

    def test_cleanup_old_sessions(self, context_config):
        """Test cleaning up old sessions."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        # Create multiple sessions with unique IDs to avoid timestamp collision
        for i in range(5):
            session = cm.start_session()
            session.id = f"session_{i}"
            session.path = cm.sessions_path / f"session_{i}.md"
            session.add_turn("Test", "Response")
            cm.end_session()

        assert len(cm.sessions.list_sessions()) == 5

        deleted = cm.cleanup_old_sessions(keep_count=2)

        assert deleted == 3
        assert len(cm.sessions.list_sessions()) == 2

    def test_cleanup_old_sessions_none_to_delete(self, context_config):
        """Test cleanup when nothing to delete."""
        cm = ContextManager(context_config)
        cm.ensure_structure()

        # Create fewer sessions than keep_count
        session = cm.start_session()
        session.add_turn("Test", "Response")
        cm.end_session()

        deleted = cm.cleanup_old_sessions(keep_count=10)

        assert deleted == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestContextIntegration:
    """Integration tests for context module."""

    def test_full_session_workflow(self, tmp_path):
        """Test complete session workflow."""
        config = ContextConfig(
            path=str(tmp_path / ".context"),
            history_turns=5,
            auto_save=True,
        )
        api_config = APIConfig(model="claude-3-sonnet")

        cm = ContextManager(config, api_config)
        cm.ensure_structure()

        # Start session
        session = cm.start_session(device_id="test")
        session_id = session.id

        # Log some turns
        cm.log_turn("Hello HARO", "Hello! How can I help?")
        cm.log_turn("What's the weather?", "I don't have weather data.")
        cm.log_turn("Thanks", "You're welcome!")

        # End session
        cm.end_session()

        # Verify session was saved
        session_path = cm.sessions_path / f"{session_id}.md"
        assert session_path.exists()

        # Load and verify
        loaded = Session.load(session_path)
        assert loaded.turn_count == 3
        assert loaded.turns[0].user_input == "Hello HARO"

    def test_knowledge_persistence(self, tmp_path):
        """Test that knowledge persists across manager instances."""
        config = ContextConfig(path=str(tmp_path / ".context"))

        # First manager adds knowledge
        cm1 = ContextManager(config)
        cm1.ensure_structure()
        cm1.knowledge.add_preference("name", "Alice")
        cm1.knowledge.add_fact("birthday", "January 1st")

        # Second manager should see it
        cm2 = ContextManager(config)

        name = cm2.knowledge.get_preference("name")
        birthday = cm2.knowledge.get_fact("birthday")

        assert name is not None
        assert "Alice" in name
        assert birthday is not None
        assert "January" in birthday

    def test_context_assembly_for_api(self, tmp_path):
        """Test assembling context for API request."""
        config = ContextConfig(path=str(tmp_path / ".context"))

        cm = ContextManager(config)
        cm.ensure_structure()

        # Add some knowledge
        cm.knowledge.add_preference("name", "Bob")
        cm.knowledge.add_fact("location", "Bob lives in Seattle")

        # Start session with history
        cm.start_session()
        cm.log_turn("Hello", "Hi Bob!")
        cm.log_turn("Where do I live?", "You live in Seattle.")

        # Get context
        history = cm.get_conversation_history()
        prefs = cm.get_user_preferences_summary()
        relevant = cm.get_relevant_context("Seattle")

        assert "User: Hello" in history
        assert "## User Preferences" in prefs
        assert "name" in prefs.lower()
        # Knowledge should be searchable
        assert len(cm.search_knowledge("Seattle")) >= 1
