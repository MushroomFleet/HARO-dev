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
from haro.core.parallel_agent import ParallelAgent
from haro.core.speech_worker import SpeechWorker, SpeechPriority, SpeechItem
from haro.core.listen_worker import ListenWorker
from haro.core.orchestrator import Orchestrator
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
    "ParallelAgent",
    "SpeechWorker",
    "SpeechPriority",
    "SpeechItem",
    "ListenWorker",
    "Orchestrator",
    "CacheConfig",
    "CacheEntry",
    "ResponseCache",
    "LifecyclePhase",
    "StartupResult",
    "LifecycleManager",
    "StartupChecker",
]
