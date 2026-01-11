"""Tests for intelligence module."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile

import pytest

from haro.core.config import APIConfig, ContextConfig, WakeConfig
from haro.intelligence.client import ClaudeClient, APIResponse, APIError, Message
from haro.intelligence.prompts import PromptBuilder, SystemPrompt
from haro.intelligence.parser import ResponseParser, ParsedResponse, KnowledgeUpdate


class TestAPIError:
    """Tests for APIError exception."""

    def test_create_error(self):
        """Test creating an API error."""
        error = APIError("Test error", status_code=500, retryable=True)
        assert str(error) == "Test error"
        assert error.status_code == 500
        assert error.retryable is True

    def test_default_values(self):
        """Test default error values."""
        error = APIError("Test error")
        assert error.status_code is None
        assert error.retryable is False


class TestAPIResponse:
    """Tests for APIResponse dataclass."""

    def test_create_response(self):
        """Test creating an API response."""
        response = APIResponse(
            text="Hello!",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason="end_turn",
            latency=0.5,
        )
        assert response.text == "Hello!"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage["input_tokens"] == 10
        assert response.stop_reason == "end_turn"
        assert response.latency == 0.5

    def test_default_values(self):
        """Test default response values."""
        response = APIResponse(text="Hi", model="test")
        assert response.usage == {}
        assert response.stop_reason == ""
        assert response.latency == 0.0


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_message_with_timestamp(self):
        """Test message with explicit timestamp."""
        msg = Message(role="assistant", content="Hi", timestamp=12345.0)
        assert msg.timestamp == 12345.0


class TestClaudeClient:
    """Tests for ClaudeClient class."""

    @pytest.fixture
    def config(self):
        """Create test API configuration."""
        return APIConfig(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.7,
            timeout=30,
        )

    @pytest.fixture
    def client(self, config):
        """Create client instance."""
        return ClaudeClient(config)

    def test_init(self, client, config):
        """Test client initialization."""
        assert client.model == config.model
        assert client.max_tokens == config.max_tokens
        assert client.temperature == config.temperature
        assert client.timeout == config.timeout
        assert client.is_initialized is False
        assert client.message_count == 0

    def test_message_count(self, client):
        """Test message count tracking."""
        assert client.message_count == 0
        client._add_message("user", "Hello")
        assert client.message_count == 1
        client._add_message("assistant", "Hi")
        assert client.message_count == 2

    def test_clear_history(self, client):
        """Test clearing message history."""
        client._add_message("user", "Hello")
        client._add_message("assistant", "Hi")
        assert client.message_count == 2

        client.clear_history()
        assert client.message_count == 0

    def test_get_history(self, client):
        """Test getting message history."""
        client._add_message("user", "Hello")
        client._add_message("assistant", "Hi")
        client._add_message("user", "How are you?")

        history = client.get_history()
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[0].content == "Hello"

        limited = client.get_history(limit=2)
        assert len(limited) == 2
        assert limited[0].content == "Hi"

    def test_get_history_messages(self, client):
        """Test getting history as message dicts."""
        client._add_message("user", "Hello")
        client._add_message("assistant", "Hi")

        messages = client._get_history_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

    def test_history_trimming(self, client):
        """Test that history is trimmed when too long."""
        client._max_history = 5

        for i in range(10):
            client._add_message("user", f"Message {i}")

        assert client.message_count == 5
        # Should have the last 5 messages
        history = client.get_history()
        assert history[0].content == "Message 5"

    def test_get_stats(self, client):
        """Test getting client statistics."""
        stats = client.get_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["errors"] == 0
        assert stats["message_count"] == 0

    @pytest.mark.asyncio
    async def test_initialize_missing_package(self, client):
        """Test initialization with missing anthropic package."""
        with patch.dict("sys.modules", {"anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(APIError) as exc:
                    await client.initialize()
                assert "not installed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_complete_not_initialized(self, client):
        """Test complete auto-initializes."""
        # Mock the anthropic module
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello!")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        # Clear env vars to ensure we use mocked Anthropic (not OpenRouter from .env)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
                with patch("haro.intelligence.client.asyncio.to_thread") as mock_thread:
                    mock_thread.return_value = mock_response
                    response = await client.complete("Hello")

                    assert response.text == "Hello!"
                    assert client.is_initialized


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.fixture
    def temp_context(self):
        """Create temporary context directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            context_path.mkdir()
            (context_path / "knowledge").mkdir()
            (context_path / "config").mkdir()
            yield context_path

    @pytest.fixture
    def config(self, temp_context):
        """Create test context configuration."""
        return ContextConfig(
            path=str(temp_context),
            history_turns=10,
        )

    @pytest.fixture
    def wake_config(self):
        """Create test wake configuration."""
        return WakeConfig(phrase="haro", sensitivity=0.5)

    @pytest.fixture
    def builder(self, config, wake_config):
        """Create prompt builder instance."""
        return PromptBuilder(config, wake_config)

    def test_init(self, builder, config):
        """Test builder initialization."""
        assert builder.context_path == Path(config.path).expanduser()
        assert builder.history_turns == 10
        assert builder.wake_phrase == "haro"

    def test_build_minimal(self, builder):
        """Test building minimal prompt."""
        prompt = builder.build_minimal()
        assert isinstance(prompt, SystemPrompt)
        assert "haro" in prompt.content.lower()
        assert prompt.token_estimate > 0
        assert len(prompt.source_files) == 0

    def test_build_with_defaults(self, builder):
        """Test building prompt with default content."""
        prompt = builder.build()
        assert isinstance(prompt, SystemPrompt)
        assert "HARO" in prompt.content
        assert prompt.token_estimate > 0
        # Should use defaults since no files exist
        assert len(prompt.source_files) > 0

    def test_build_with_context_files(self, builder, temp_context):
        """Test building prompt with actual context files."""
        # Create substrate file
        (temp_context / "substrate.md").write_text("# Custom Substrate\nTest content")

        prompt = builder.build()
        assert "Custom Substrate" in prompt.content
        assert "substrate.md" in prompt.source_files

    def test_build_with_knowledge(self, builder, temp_context):
        """Test building prompt with knowledge files."""
        # Create knowledge file
        (temp_context / "knowledge" / "test.md").write_text("# Test Knowledge\nPython is great")

        prompt = builder.build(user_input="Tell me about Python")
        assert "Test Knowledge" in prompt.content

    def test_load_file_missing(self, builder):
        """Test loading missing file returns default."""
        content = builder._load_file("nonexistent.md", "default content")
        assert content == "default content"

    def test_load_file_exists(self, builder, temp_context):
        """Test loading existing file."""
        (temp_context / "test.md").write_text("Test content")
        content = builder._load_file("test.md")
        assert content == "Test content"

    def test_find_relevant_knowledge(self, builder, temp_context):
        """Test finding relevant knowledge files."""
        (temp_context / "knowledge" / "python.md").write_text("Python programming language")
        (temp_context / "knowledge" / "java.md").write_text("Java programming language")

        relevant = builder._find_relevant_knowledge("Tell me about Python")
        assert len(relevant) >= 1
        # Python file should be found
        found_python = any("python" in name.lower() for name, _ in relevant)
        assert found_python

    def test_get_context_size(self, builder, temp_context):
        """Test getting context file sizes."""
        (temp_context / "test.md").write_text("Test content here")
        sizes = builder.get_context_size()
        assert "test.md" in sizes
        assert sizes["test.md"] > 0


class TestResponseParser:
    """Tests for ResponseParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return ResponseParser()

    def test_parse_simple_response(self, parser):
        """Test parsing a simple response."""
        response = parser.parse("Hello! How can I help you today?")
        assert response.speech_text == "Hello! How can I help you today?"
        assert response.is_question is True
        assert response.is_acknowledgment is False

    def test_parse_acknowledgment(self, parser):
        """Test parsing an acknowledgment."""
        response = parser.parse("Okay.")
        assert response.is_acknowledgment is True

        response = parser.parse("Got it")
        assert response.is_acknowledgment is True

    def test_parse_knowledge_update(self, parser):
        """Test extracting knowledge updates."""
        text = "I'll remember that. [REMEMBER: user_name = John]"
        response = parser.parse(text)

        assert len(response.knowledge_updates) == 1
        assert response.knowledge_updates[0].key == "user_name"
        assert response.knowledge_updates[0].value == "John"
        assert response.knowledge_updates[0].category == "facts"

        # Tag should be removed from speech text
        assert "[REMEMBER:" not in response.speech_text

    def test_parse_preference_update(self, parser):
        """Test extracting preference updates."""
        text = "Noted! [PREFERENCE: temperature = celsius]"
        response = parser.parse(text)

        assert len(response.knowledge_updates) == 1
        assert response.knowledge_updates[0].key == "temperature"
        assert response.knowledge_updates[0].value == "celsius"
        assert response.knowledge_updates[0].category == "preferences"

    def test_parse_followup(self, parser):
        """Test extracting follow-up suggestions."""
        text = "Here's the info. [FOLLOWUP: Would you like more details?]"
        response = parser.parse(text)

        assert len(response.suggested_followups) == 1
        assert "more details" in response.suggested_followups[0]
        assert "[FOLLOWUP:" not in response.speech_text

    def test_clean_markdown(self, parser):
        """Test cleaning markdown from response."""
        text = "Here's **bold** and *italic* text."
        response = parser.parse(text)
        assert response.speech_text == "Here's bold and italic text."

    def test_clean_code_blocks(self, parser):
        """Test removing code blocks."""
        text = "Here's code:\n```python\nprint('hello')\n```\nThat's it."
        response = parser.parse(text)
        assert "```" not in response.speech_text
        assert "print" not in response.speech_text

    def test_clean_links(self, parser):
        """Test cleaning links."""
        text = "Check out [this link](https://example.com)."
        response = parser.parse(text)
        assert response.speech_text == "Check out this link."

    def test_expand_abbreviations(self, parser):
        """Test expanding abbreviations."""
        text = "The API uses JSON format."
        response = parser.parse(text)
        assert "A P I" in response.speech_text
        assert "J SON" in response.speech_text

    def test_requires_action(self, parser):
        """Test detecting action-requiring responses."""
        text = "I'll remember that for next time."
        response = parser.parse(text)
        assert response.requires_action is True

        text = "The weather is nice today."
        response = parser.parse(text)
        assert response.requires_action is False

    def test_get_speech_summary(self, parser):
        """Test getting speech summary."""
        response = ParsedResponse(
            speech_text="Hello world!",
            knowledge_updates=[KnowledgeUpdate("key", "value")],
            suggested_followups=["Follow up?"],
            is_question=True,
        )

        summary = parser.get_speech_summary(response)
        assert summary["text_length"] == 12
        assert summary["word_count"] == 2
        assert summary["knowledge_updates"] == 1
        assert summary["followups"] == 1
        assert summary["is_question"] is True


class TestKnowledgeUpdate:
    """Tests for KnowledgeUpdate dataclass."""

    def test_create_update(self):
        """Test creating a knowledge update."""
        update = KnowledgeUpdate(
            key="user_location",
            value="Bristol, UK",
            category="preferences",
        )
        assert update.key == "user_location"
        assert update.value == "Bristol, UK"
        assert update.category == "preferences"

    def test_default_category(self):
        """Test default category."""
        update = KnowledgeUpdate(key="fact", value="test")
        assert update.category == "facts"


class TestParsedResponse:
    """Tests for ParsedResponse dataclass."""

    def test_create_response(self):
        """Test creating a parsed response."""
        response = ParsedResponse(
            speech_text="Hello!",
            is_question=True,
            confidence=0.95,
        )
        assert response.speech_text == "Hello!"
        assert response.is_question is True
        assert response.confidence == 0.95
        assert len(response.knowledge_updates) == 0
        assert len(response.suggested_followups) == 0

    def test_default_values(self):
        """Test default values."""
        response = ParsedResponse(speech_text="Test")
        assert response.confidence == 1.0
        assert response.is_question is False
        assert response.is_acknowledgment is False
        assert response.requires_action is False
