"""Tests for context initialization."""

import tempfile
from pathlib import Path

import pytest

from haro.context.init import init_context, ContextInitResult


class TestContextInitResult:
    """Tests for ContextInitResult dataclass."""

    def test_success_when_no_errors(self):
        """Test success property when no errors."""
        result = ContextInitResult(
            path=Path(".context"),
            created_dirs=["sessions"],
            created_files=["substrate.md"],
        )
        assert result.success is True

    def test_failure_when_errors(self):
        """Test success property when errors exist."""
        result = ContextInitResult(
            path=Path(".context"),
            errors=["Failed to create file"],
        )
        assert result.success is False


class TestInitContext:
    """Tests for init_context function."""

    def test_creates_directories(self):
        """Test that directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            result = init_context(str(context_path))

            assert result.success
            assert context_path.exists()
            assert (context_path / "sessions").exists()
            assert (context_path / "knowledge").exists()
            assert (context_path / "config").exists()

    def test_creates_default_files(self):
        """Test that default files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            result = init_context(str(context_path))

            assert result.success
            assert (context_path / "substrate.md").exists()
            assert (context_path / "guidelines.md").exists()
            assert (context_path / "config" / "personality.md").exists()
            assert (context_path / "knowledge" / "user_preferences.md").exists()
            assert (context_path / "knowledge" / "facts.md").exists()

    def test_does_not_overwrite_existing_files(self):
        """Test that existing files are not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            context_path.mkdir()

            # Create existing substrate file
            substrate_path = context_path / "substrate.md"
            substrate_path.write_text("# Custom Content")

            result = init_context(str(context_path), overwrite=False)

            assert result.success
            assert "substrate.md" in result.skipped_files
            assert substrate_path.read_text() == "# Custom Content"

    def test_overwrites_when_flag_set(self):
        """Test that files are overwritten when flag is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            context_path.mkdir()

            # Create existing substrate file
            substrate_path = context_path / "substrate.md"
            substrate_path.write_text("# Custom Content")

            result = init_context(str(context_path), overwrite=True)

            assert result.success
            assert "substrate.md" in result.created_files
            assert substrate_path.read_text() != "# Custom Content"

    def test_default_path(self):
        """Test using default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = init_context()

                assert result.success
                assert result.path == Path(tmpdir) / ".context"
            finally:
                os.chdir(original_cwd)

    def test_substrate_contains_expected_content(self):
        """Test that substrate file has expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            init_context(str(context_path))

            content = (context_path / "substrate.md").read_text()

            assert "HARO" in content
            assert "Helpful Autonomous Responsive Operator" in content
            assert "Voice Interaction" in content

    def test_personality_contains_expected_content(self):
        """Test that personality file has expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            init_context(str(context_path))

            content = (context_path / "config" / "personality.md").read_text()

            assert "Personality" in content
            assert "Tone" in content
            assert "Friendly" in content

    def test_guidelines_contains_expected_content(self):
        """Test that guidelines file has expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            init_context(str(context_path))

            content = (context_path / "guidelines.md").read_text()

            assert "Guidelines" in content
            assert "Privacy" in content

    def test_result_lists(self):
        """Test that result contains correct lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = Path(tmpdir) / ".context"
            result = init_context(str(context_path))

            assert len(result.created_dirs) == 3  # sessions, knowledge, config
            assert len(result.created_files) == 5
            assert len(result.skipped_files) == 0
            assert len(result.errors) == 0
