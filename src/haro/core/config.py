"""Configuration management for HARO.

Loads configuration from YAML files with support for:
- Default configuration (bundled with package)
- User configuration (~/.config/haro/config.yaml)
- Environment variable overrides
- Path expansion (~, environment variables)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class AudioConfig:
    """Audio input/output configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    buffer_duration: float = 2.0


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    threshold: float = 0.5
    min_speech_duration: float = 0.5
    max_speech_duration: float = 30.0
    silence_duration: float = 3.0  # Stop recording after 3s of silence
    noise_floor_adaptation: float = 0.05


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""

    model: str = "base.en"
    model_path: str = "~/.cache/haro/models/"
    language: str = "en"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = True


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""

    engine: str = "piper"
    voice: str = "en_US-lessac-medium"
    model_path: str = "~/.cache/haro/models/"
    speed: float = 1.0
    volume: float = 0.8


@dataclass
class APIConfig:
    """Claude API configuration."""

    provider: str = "anthropic"
    # Default model for OpenRouter - can be overridden via OPENROUTER_MODEL env var
    # The :online suffix enables OpenRouter web search for up-to-date information
    model: str = "google/gemini-3-flash-preview:online"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    base_url: Optional[str] = None  # For OpenRouter or custom endpoints


@dataclass
class OllamaConfig:
    """Ollama local LLM configuration."""

    enabled: bool = True
    model: str = "ministral:3b"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 512
    # Phrases that trigger cloud LLM (with web search via :online suffix)
    cloud_keywords: list[str] = field(default_factory=lambda: [
        "ask claude", "use claude",
        "search the web", "search online", "look up online",
        "current news", "latest news", "what's happening",
        "complex", "detailed", "explain in depth"
    ])


@dataclass
class ContextConfig:
    """Context management configuration."""

    path: str = ".context/"
    history_turns: int = 10
    session_timeout: int = 300
    auto_save: bool = True
    max_context_tokens: int = 4000


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: str = "~/.local/share/haro/logs/haro.log"
    console: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class WakeConfig:
    """Wake word detection configuration."""

    phrase: str = "haro"
    sensitivity: float = 0.5
    confirmation_sound: bool = True
    confirmation_phrases: list[str] = field(
        default_factory=lambda: ["Hello, HARO?"]
    )


@dataclass
class CommandsConfig:
    """Local voice commands configuration."""

    stop: list[str] = field(default_factory=lambda: ["stop", "be quiet", "shut up"])
    repeat: list[str] = field(
        default_factory=lambda: ["repeat that", "say that again", "what did you say"]
    )
    new_session: list[str] = field(
        default_factory=lambda: ["new conversation", "start over", "forget that"]
    )
    sleep: list[str] = field(default_factory=lambda: ["go to sleep", "sleep mode"])
    goodbye: list[str] = field(default_factory=lambda: ["goodbye", "bye", "see you"])
    time: list[str] = field(
        default_factory=lambda: ["what time is it", "what's the time"]
    )
    volume_up: list[str] = field(
        default_factory=lambda: ["louder", "volume up", "speak up"]
    )
    volume_down: list[str] = field(
        default_factory=lambda: ["quieter", "volume down", "speak softer"]
    )


@dataclass
class HaroConfig:
    """Root configuration for HARO."""

    device_id: str = "haro-default"
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)


def _expand_path(path: str) -> str:
    """Expand ~ and environment variables in path."""
    return os.path.expandvars(os.path.expanduser(path))


def _dict_to_config(data: dict[str, Any], config_class: type) -> Any:
    """Convert a dictionary to a dataclass config, handling nested configs."""
    if not data:
        return config_class()

    # Get field names and types from the dataclass
    field_types = {f.name: f.type for f in config_class.__dataclass_fields__.values()}

    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Handle nested dataclass configs
        if isinstance(value, dict):
            # Check if it's a config dataclass
            if hasattr(field_type, "__dataclass_fields__"):
                value = _dict_to_config(value, field_type)

        # Expand paths for string fields that look like paths
        if isinstance(value, str) and ("/" in value or "~" in value or "$" in value):
            value = _expand_path(value)

        kwargs[key] = value

    return config_class(**kwargs)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _get_package_dir() -> Path:
    """Get package directory, handling frozen apps (PyInstaller)."""
    import sys
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle: config is next to executable
        return Path(sys.executable).parent
    else:
        # Normal Python: relative to this file
        return Path(__file__).parent.parent.parent.parent


def _find_config_files() -> list[Path]:
    """Find all configuration files in order of precedence."""
    config_files = []

    # 1. Default config bundled with package (or next to frozen exe)
    package_dir = _get_package_dir()
    default_config = package_dir / "config" / "default.yaml"
    if default_config.exists():
        config_files.append(default_config)

    # 2. System-wide config (Linux)
    system_config = Path("/etc/haro/config.yaml")
    if system_config.exists():
        config_files.append(system_config)

    # 3. User config
    user_config = Path.home() / ".config" / "haro" / "config.yaml"
    if user_config.exists():
        config_files.append(user_config)

    # 4. Local config (current directory)
    local_config = Path.cwd() / "haro.yaml"
    if local_config.exists():
        config_files.append(local_config)

    return config_files


def load_config(config_path: Optional[str] = None) -> HaroConfig:
    """Load HARO configuration from files.

    Configuration is loaded in this order (later overrides earlier):
    1. Default config (bundled)
    2. System config (/etc/haro/config.yaml)
    3. User config (~/.config/haro/config.yaml)
    4. Local config (./haro.yaml)
    5. Explicit config_path if provided

    Args:
        config_path: Optional explicit path to a config file.

    Returns:
        Loaded HaroConfig instance.
    """
    merged_config: dict[str, Any] = {}

    # Load and merge all config files
    config_files = _find_config_files()

    if config_path:
        explicit_path = Path(_expand_path(config_path))
        if explicit_path.exists():
            config_files.append(explicit_path)

    for config_file in config_files:
        try:
            with open(config_file) as f:
                data = yaml.safe_load(f)
                if data and "haro" in data:
                    merged_config = _merge_dicts(merged_config, data["haro"])
        except Exception:
            # Skip files that can't be loaded
            pass

    # Convert to HaroConfig dataclass
    return _dict_to_config(merged_config, HaroConfig)


def get_default_config() -> HaroConfig:
    """Get the default configuration without loading from files.

    Returns:
        Default HaroConfig instance.
    """
    return HaroConfig()
