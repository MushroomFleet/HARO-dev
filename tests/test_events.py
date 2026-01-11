"""Tests for event system."""

import asyncio
from datetime import datetime

import pytest

from haro.core.events import (
    Event,
    EventBus,
    EventType,
    AudioChunkEvent,
    VADEvent,
    WakeWordEvent,
    get_event_bus,
    set_event_bus,
)


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self):
        """Test creating a basic event."""
        event = Event(type=EventType.AUDIO_CHUNK)
        assert event.type == EventType.AUDIO_CHUNK
        assert isinstance(event.timestamp, datetime)
        assert event.data == {}

    def test_event_with_data(self):
        """Test creating event with data."""
        event = Event(
            type=EventType.STATE_CHANGED,
            data={"previous": "passive", "new": "active"},
        )
        assert event.data["previous"] == "passive"
        assert event.data["new"] == "active"


class TestAudioChunkEvent:
    """Tests for AudioChunkEvent."""

    def test_create_audio_event(self):
        """Test creating audio chunk event."""
        import numpy as np

        chunk = np.zeros(1024, dtype=np.float32)
        event = AudioChunkEvent(chunk=chunk, sample_rate=16000)
        assert event.type == EventType.AUDIO_CHUNK
        assert event.chunk is not None
        assert len(event.chunk) == 1024
        assert event.sample_rate == 16000


class TestVADEvent:
    """Tests for VADEvent."""

    def test_create_vad_event(self):
        """Test creating VAD event."""
        event = VADEvent(
            type=EventType.VAD_SPEECH_START,
            is_speech=True,
            energy=0.5,
            duration=1.5,
        )
        assert event.type == EventType.VAD_SPEECH_START
        assert event.is_speech is True
        assert event.energy == 0.5
        assert event.duration == 1.5


class TestWakeWordEvent:
    """Tests for WakeWordEvent."""

    def test_create_wake_event(self):
        """Test creating wake word event."""
        event = WakeWordEvent(
            confidence=0.9,
            phrase="haro",
        )
        assert event.type == EventType.WAKE_DETECTED
        assert event.confidence == 0.9
        assert event.phrase == "haro"


class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribing to and publishing events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler)
        event = Event(type=EventType.AUDIO_CHUNK)
        await event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0] == event

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        results = []

        async def handler1(event):
            results.append("handler1")

        async def handler2(event):
            results.append("handler2")

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler1)
        event_bus.subscribe(EventType.AUDIO_CHUNK, handler2)

        await event_bus.publish(Event(type=EventType.AUDIO_CHUNK))

        assert len(results) == 2
        assert "handler1" in results
        assert "handler2" in results

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler)
        event_bus.unsubscribe(EventType.AUDIO_CHUNK, handler)

        await event_bus.publish(Event(type=EventType.AUDIO_CHUNK))

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_different_event_types(self, event_bus):
        """Test handlers only receive their event types."""
        audio_events = []
        vad_events = []

        async def audio_handler(event):
            audio_events.append(event)

        async def vad_handler(event):
            vad_events.append(event)

        event_bus.subscribe(EventType.AUDIO_CHUNK, audio_handler)
        event_bus.subscribe(EventType.VAD_SPEECH_START, vad_handler)

        await event_bus.publish(Event(type=EventType.AUDIO_CHUNK))
        await event_bus.publish(Event(type=EventType.VAD_SPEECH_START))

        assert len(audio_events) == 1
        assert len(vad_events) == 1

    @pytest.mark.asyncio
    async def test_publish_no_handlers(self, event_bus):
        """Test publishing event with no handlers doesn't error."""
        await event_bus.publish(Event(type=EventType.AUDIO_CHUNK))
        # Should not raise

    def test_has_handlers(self, event_bus):
        """Test has_handlers method."""
        async def handler(event):
            pass

        assert not event_bus.has_handlers(EventType.AUDIO_CHUNK)

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler)
        assert event_bus.has_handlers(EventType.AUDIO_CHUNK)
        assert not event_bus.has_handlers(EventType.VAD_SPEECH_START)

    def test_clear_handlers(self, event_bus):
        """Test clearing handlers."""
        async def handler(event):
            pass

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler)
        event_bus.subscribe(EventType.VAD_SPEECH_START, handler)

        event_bus.clear(EventType.AUDIO_CHUNK)
        assert not event_bus.has_handlers(EventType.AUDIO_CHUNK)
        assert event_bus.has_handlers(EventType.VAD_SPEECH_START)

        event_bus.clear()
        assert not event_bus.has_handlers(EventType.VAD_SPEECH_START)

    @pytest.mark.asyncio
    async def test_publish_and_wait(self, event_bus):
        """Test publish_and_wait returns results."""
        async def handler(event):
            return "result"

        event_bus.subscribe(EventType.AUDIO_CHUNK, handler)
        results = await event_bus.publish_and_wait(
            Event(type=EventType.AUDIO_CHUNK)
        )

        assert len(results) == 1
        assert results[0] == "result"


class TestGlobalEventBus:
    """Tests for global event bus functions."""

    def test_get_event_bus(self):
        """Test getting global event bus."""
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_set_event_bus(self):
        """Test setting global event bus."""
        custom_bus = EventBus()
        set_event_bus(custom_bus)
        assert get_event_bus() is custom_bus

        # Reset
        set_event_bus(EventBus())
