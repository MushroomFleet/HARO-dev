"""HARO core module.

Contains configuration management, event system, and core agent logic.
"""

from haro.core.config import (
    AudioConfig,
    VADConfig,
    STTConfig,
    TTSConfig,
    APIConfig,
    ContextConfig,
    LoggingConfig,
    WakeConfig,
    CommandsConfig,
    HaroConfig,
    load_config,
)
from haro.core.events import EventBus, EventType, Event, StateChangeEvent
from haro.core.agent import AgentState, StateContext, AgentStats, HaroAgent
from haro.core.cache import CacheConfig, CacheEntry, ResponseCache
from haro.core.lifecycle import (
    LifecyclePhase,
    StartupResult,
    LifecycleManager,
    StartupChecker,
)

__all__ = [
    "AudioConfig",
    "VADConfig",
    "STTConfig",
    "TTSConfig",
    "APIConfig",
    "ContextConfig",
    "LoggingConfig",
    "WakeConfig",
    "CommandsConfig",
    "HaroConfig",
    "load_config",
    "EventBus",
    "EventType",
    "Event",
    "StateChangeEvent",
    "AgentState",
    "StateContext",
    "AgentStats",
    "HaroAgent",
    "CacheConfig",
    "CacheEntry",
    "ResponseCache",
    "LifecyclePhase",
    "StartupResult",
    "LifecycleManager",
    "StartupChecker",
]
