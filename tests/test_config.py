"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
import yaml

from haro.core.config import (
    AudioConfig,
    VADConfig,
    HaroConfig,
    load_config,
    get_default_config,
    _expand_path,
    _merge_dicts,
)


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.input_device is None
        assert config.output_device is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=512,
        )
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_size == 512


class TestVADConfig:
    """Tests for VADConfig dataclass."""

    def test_default_values(self):
        """Test default VAD configuration."""
        config = VADConfig()
        assert config.threshold == 0.5
        assert config.min_speech_duration == 0.5
        assert config.max_speech_duration == 30.0
        assert config.silence_duration == 1.5

    def test_custom_values(self):
        """Test custom VAD configuration."""
        config = VADConfig(
            threshold=0.7,
            silence_duration=2.0,
        )
        assert config.threshold == 0.7
        assert config.silence_duration == 2.0


class TestHaroConfig:
    """Tests for HaroConfig dataclass."""

    def test_default_values(self):
        """Test default root configuration."""
        config = HaroConfig()
        assert config.device_id == "haro-default"
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.vad, VADConfig)

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = HaroConfig()
        assert config.audio.sample_rate == 16000
        assert config.vad.threshold == 0.5
        assert config.api.model == "claude-sonnet-4-20250514"


class TestExpandPath:
    """Tests for path expansion."""

    def test_expand_home(self):
        """Test home directory expansion."""
        path = _expand_path("~/test")
        assert "~" not in path
        assert "test" in path

    def test_expand_env_var(self):
        """Test environment variable expansion."""
        import os

        os.environ["TEST_VAR"] = "test_value"
        path = _expand_path("$TEST_VAR/file")
        assert "test_value" in path
        del os.environ["TEST_VAR"]

    def test_plain_path(self):
        """Test plain path without expansion."""
        path = _expand_path("/usr/local/bin")
        assert path == "/usr/local/bin"


class TestMergeDicts:
    """Tests for dictionary merging."""

    def test_simple_merge(self):
        """Test simple key override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3, "z": 4}}
        result = _merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_deep_nested_merge(self):
        """Test deeply nested merge."""
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = _merge_dicts(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, HaroConfig)
        assert config.device_id == "haro-default"

    def test_get_default_config(self):
        """Test getting default config without file loading."""
        config = get_default_config()
        assert isinstance(config, HaroConfig)
        assert config.audio.sample_rate == 16000

    def test_load_from_file(self):
        """Test loading configuration from a file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(
                {
                    "haro": {
                        "device_id": "test-device",
                        "audio": {"sample_rate": 44100},
                    }
                },
                f,
            )
            f.flush()

            config = load_config(f.name)
            assert config.device_id == "test-device"
            assert config.audio.sample_rate == 44100
            # Other values should be defaults
            assert config.audio.channels == 1

        Path(f.name).unlink()

    def test_load_missing_file(self):
        """Test loading with non-existent explicit path."""
        config = load_config("/nonexistent/path/config.yaml")
        # Should return default config
        assert isinstance(config, HaroConfig)
        assert config.device_id == "haro-default"
