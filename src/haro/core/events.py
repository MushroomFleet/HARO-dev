"""Event system for HARO component communication.

Provides an async pub/sub event bus for loose coupling between components.
Events are typed using dataclasses for type safety and IDE support.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional

import numpy as np


class EventType(Enum):
    """Types of events in the HARO system."""

    # Audio events
    AUDIO_CHUNK = auto()
    AUDIO_DEVICE_CONNECTED = auto()
    AUDIO_DEVICE_DISCONNECTED = auto()
    AUDIO_ERROR = auto()

    # VAD events
    VAD_SPEECH_START = auto()
    VAD_SPEECH_END = auto()
    VAD_ENERGY_UPDATE = auto()

    # Wake word events
    WAKE_DETECTED = auto()
    WAKE_TIMEOUT = auto()

    # STT events
    STT_TRANSCRIPTION_START = auto()
    STT_TRANSCRIPTION_COMPLETE = auto()
    STT_TRANSCRIPTION_ERROR = auto()

    # TTS events
    TTS_PLAYBACK_START = auto()
    TTS_PLAYBACK_COMPLETE = auto()
    TTS_PLAYBACK_INTERRUPTED = auto()
    TTS_SYNTHESIS_ERROR = auto()

    # Agent state events
    STATE_CHANGED = auto()
    COMMAND_DETECTED = auto()

    # API events
    API_REQUEST_START = auto()
    API_RESPONSE_RECEIVED = auto()
    API_ERROR = auto()

    # Session events
    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    TURN_LOGGED = auto()

    # System events
    SYSTEM_SHUTDOWN = auto()
    SYSTEM_ERROR = auto()


@dataclass
class Event:
    """Base event class with common metadata."""

    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioChunkEvent(Event):
    """Event containing an audio chunk."""

    type: EventType = EventType.AUDIO_CHUNK
    chunk: Optional[np.ndarray] = None
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class VADEvent(Event):
    """Event from voice activity detection."""

    type: EventType = EventType.VAD_SPEECH_START
    is_speech: bool = False
    energy: float = 0.0
    duration: float = 0.0


@dataclass
class WakeWordEvent(Event):
    """Event for wake word detection."""

    type: EventType = EventType.WAKE_DETECTED
    confidence: float = 0.0
    phrase: str = ""


@dataclass
class TranscriptionEvent(Event):
    """Event containing transcribed text."""

    type: EventType = EventType.STT_TRANSCRIPTION_COMPLETE
    text: str = ""
    confidence: float = 0.0
    language: str = "en"


@dataclass
class StateChangeEvent(Event):
    """Event for agent state changes."""

    type: EventType = EventType.STATE_CHANGED
    previous_state: str = ""
    new_state: str = ""


@dataclass
class CommandEvent(Event):
    """Event for detected local commands."""

    type: EventType = EventType.COMMAND_DETECTED
    command: str = ""
    args: list[str] = field(default_factory=list)


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async event bus for component communication.

    Allows components to subscribe to events by type and publish events
    to all subscribers. Supports multiple handlers per event type.
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: The type of event to subscribe to.
            handler: Async function to call when event is published.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type.

        Args:
            event_type: The type of event to unsubscribe from.
            handler: The handler to remove.
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: The event to publish.
        """
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return

        # Run all handlers concurrently
        tasks = [asyncio.create_task(handler(event)) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_and_wait(
        self, event: Event, timeout: Optional[float] = None
    ) -> list[Any]:
        """Publish an event and wait for all handlers to complete.

        Args:
            event: The event to publish.
            timeout: Optional timeout in seconds.

        Returns:
            List of results from handlers (or exceptions).
        """
        handlers = self._handlers.get(event.type, [])
        if not handlers:
            return []

        tasks = [asyncio.create_task(handler(event)) for handler in handlers]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            return list(results)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            raise

    def clear(self, event_type: Optional[EventType] = None) -> None:
        """Clear handlers for an event type or all handlers.

        Args:
            event_type: If provided, clear only this type. Otherwise clear all.
        """
        if event_type:
            self._handlers.pop(event_type, None)
        else:
            self._handlers.clear()

    def has_handlers(self, event_type: EventType) -> bool:
        """Check if an event type has any handlers.

        Args:
            event_type: The event type to check.

        Returns:
            True if there are handlers registered for this event type.
        """
        return bool(self._handlers.get(event_type))


# Global event bus instance (can be overridden for testing)
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.

    Returns:
        The global EventBus instance.
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus instance.

    Args:
        bus: The EventBus to use as the global instance.
    """
    global _global_event_bus
    _global_event_bus = bus
