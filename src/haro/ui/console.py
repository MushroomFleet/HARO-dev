"""Rich console display for HARO.

Provides a real-time dashboard showing:
- Current agent state with visual indicators
- Conversation history
- Audio levels and VAD status
- System statistics
- Recent events

Inspired by Claude Code's terminal interface.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from haro.core.events import (
    Event,
    EventBus,
    EventType,
    StateChangeEvent,
    TranscriptionEvent,
    WakeWordEvent,
)
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DisplayConfig:
    """Configuration for the console display."""

    refresh_rate: float = 4.0  # Updates per second
    max_conversation_turns: int = 5  # Number of turns to show
    max_events: int = 8  # Number of recent events to show
    show_audio_levels: bool = True
    show_stats: bool = True
    show_events: bool = True
    compact_mode: bool = False  # Single-panel compact view


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    source: str = ""  # "local" or "cloud" for assistant responses
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DisplayState:
    """Current state for the display."""

    agent_state: str = "PASSIVE"
    previous_state: str = ""
    conversation: list[ConversationTurn] = field(default_factory=list)
    events: list[tuple[datetime, str, str]] = field(default_factory=list)  # (time, type, detail)
    audio_energy: float = 0.0
    is_speech: bool = False
    wake_confidence: float = 0.0
    last_transcript: str = ""
    uptime_seconds: float = 0.0
    wake_count: int = 0
    api_calls: int = 0
    local_calls: int = 0  # Track local LLM usage
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    api_in_progress: bool = False
    tts_in_progress: bool = False
    last_llm_source: str = ""  # "local" or "cloud"


class ConsoleDisplay:
    """Rich console display for HARO voice assistant.

    Provides a real-time dashboard that updates based on events
    from the agent's event bus.

    Usage:
        display = ConsoleDisplay(event_bus, config)
        await display.start()
        # ... agent runs ...
        await display.stop()
    """

    # State colors and indicators
    STATE_STYLES = {
        "PASSIVE": ("dim cyan", "..."),
        "ACTIVE": ("bold green", "REC"),
        "PROCESSING": ("bold yellow", "API"),
        "SPEAKING": ("bold blue", "TTS"),
        "INTERRUPTED": ("bold magenta", "INT"),
        "SLEEPING": ("dim", "ZZZ"),
        "ERROR": ("bold red", "ERR"),
    }

    # Animated HARO character ASCII art for each state
    HARO_STATES = {
        "idle": [
            "╭─────╮",
            "│ ◡ ◡ │",
            "│  ○  │",
            "│ ═══ │",
            "╰─────╯",
        ],
        "listening": [
            "╭─────╮",
            "│ ◉ ◉ │",
            "│ ))) │",
            "│ ═══ │",
            "╰─────╯",
        ],
        "thinking": [
            "╭─────╮",
            "│ ◐ ◐ │",
            "│  ?  │",
            "│ ─── │",
            "╰─────╯",
        ],
        "speaking": [
            "╭─────╮",
            "│ ◠ ◠ │",
            "│ ≋≋≋ │",
            "│ ═══ │",
            "╰─────╯",
        ],
        "error": [
            "╭─────╮",
            "│ ✖ ✖ │",
            "│  !  │",
            "│ ─── │",
            "╰─────╯",
        ],
        "processing": [
            "╭─────╮",
            "│ ◑ ◑ │",
            "│ ••• │",
            "│ ─── │",
            "╰─────╯",
        ],
    }

    # Map agent states to HARO character states
    STATE_TO_HARO = {
        "PASSIVE": "idle",
        "ACTIVE": "listening",
        "PROCESSING": "thinking",
        "SPEAKING": "speaking",
        "INTERRUPTED": "error",
        "SLEEPING": "idle",
        "ERROR": "error",
    }

    # Colors for each HARO state
    HARO_STATE_COLORS = {
        "idle": "cyan",
        "listening": "green",
        "thinking": "yellow",
        "speaking": "blue",
        "error": "red",
        "processing": "magenta",
    }

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[DisplayConfig] = None,
        console: Optional[Console] = None,
    ) -> None:
        """Initialize the console display.

        Args:
            event_bus: Event bus to subscribe to for updates.
            config: Display configuration options.
            console: Optional Rich console instance.
        """
        self.event_bus = event_bus
        self.config = config or DisplayConfig()
        self.console = console or Console()
        self.state = DisplayState()
        self._live: Optional[Live] = None
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self.logger = logger.bind(component="ConsoleDisplay")

    async def start(self) -> None:
        """Start the live display."""
        if self._running:
            return

        self._running = True
        self._subscribe_to_events()
        self.state.start_time = datetime.now()

        # Start live display
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=self.config.refresh_rate,
            screen=False,  # Don't clear screen, allow scrollback
        )
        self._live.start()

        # Start background update task for uptime
        self._update_task = asyncio.create_task(self._uptime_updater())

        self.logger.info("console_display_started")

    async def stop(self) -> None:
        """Stop the live display."""
        if not self._running:
            return

        self._running = False
        self._unsubscribe_from_events()

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        if self._live:
            self._live.stop()
            self._live = None

        self.logger.info("console_display_stopped")

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events from the event bus."""
        self.event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_change)
        self.event_bus.subscribe(EventType.WAKE_DETECTED, self._on_wake_detected)
        self.event_bus.subscribe(EventType.VAD_SPEECH_START, self._on_vad_event)
        self.event_bus.subscribe(EventType.VAD_SPEECH_END, self._on_vad_event)
        self.event_bus.subscribe(EventType.VAD_ENERGY_UPDATE, self._on_energy_update)
        self.event_bus.subscribe(EventType.STT_TRANSCRIPTION_COMPLETE, self._on_transcription)
        self.event_bus.subscribe(EventType.API_REQUEST_START, self._on_api_start)
        self.event_bus.subscribe(EventType.API_RESPONSE_RECEIVED, self._on_api_response)
        self.event_bus.subscribe(EventType.API_ERROR, self._on_api_error)
        self.event_bus.subscribe(EventType.TTS_PLAYBACK_START, self._on_tts_start)
        self.event_bus.subscribe(EventType.TTS_PLAYBACK_COMPLETE, self._on_tts_complete)
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._on_system_error)

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        self.event_bus.unsubscribe(EventType.STATE_CHANGED, self._on_state_change)
        self.event_bus.unsubscribe(EventType.WAKE_DETECTED, self._on_wake_detected)
        self.event_bus.unsubscribe(EventType.VAD_SPEECH_START, self._on_vad_event)
        self.event_bus.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_event)
        self.event_bus.unsubscribe(EventType.VAD_ENERGY_UPDATE, self._on_energy_update)
        self.event_bus.unsubscribe(EventType.STT_TRANSCRIPTION_COMPLETE, self._on_transcription)
        self.event_bus.unsubscribe(EventType.API_REQUEST_START, self._on_api_start)
        self.event_bus.unsubscribe(EventType.API_RESPONSE_RECEIVED, self._on_api_response)
        self.event_bus.unsubscribe(EventType.API_ERROR, self._on_api_error)
        self.event_bus.unsubscribe(EventType.TTS_PLAYBACK_START, self._on_tts_start)
        self.event_bus.unsubscribe(EventType.TTS_PLAYBACK_COMPLETE, self._on_tts_complete)
        self.event_bus.unsubscribe(EventType.SYSTEM_ERROR, self._on_system_error)

    # Event handlers
    async def _on_state_change(self, event: Event) -> None:
        """Handle state change events."""
        if isinstance(event, StateChangeEvent):
            self.state.previous_state = event.previous_state
            self.state.agent_state = event.new_state
            self._add_event("STATE", f"{event.previous_state} -> {event.new_state}")
            self._refresh()

    async def _on_wake_detected(self, event: Event) -> None:
        """Handle wake word detection events."""
        if isinstance(event, WakeWordEvent):
            self.state.wake_confidence = event.confidence
            self.state.wake_count += 1
            self._add_event("WAKE", f"confidence: {event.confidence:.2f}")
            self._refresh()

    async def _on_vad_event(self, event: Event) -> None:
        """Handle VAD events."""
        self.state.is_speech = event.type == EventType.VAD_SPEECH_START
        if event.type == EventType.VAD_SPEECH_START:
            self._add_event("VAD", "speech started")
        else:
            self._add_event("VAD", "speech ended")
        self._refresh()

    async def _on_energy_update(self, event: Event) -> None:
        """Handle audio energy updates."""
        self.state.audio_energy = event.data.get("energy", 0.0)
        self.state.is_speech = event.data.get("is_speech", False)
        self._refresh()

    async def _on_transcription(self, event: Event) -> None:
        """Handle transcription events."""
        if isinstance(event, TranscriptionEvent):
            self.state.last_transcript = event.text
            self._add_conversation("user", event.text)
            self._add_event("STT", f'"{event.text[:40]}..."' if len(event.text) > 40 else f'"{event.text}"')
            self._refresh()

    async def _on_api_start(self, event: Event) -> None:
        """Handle API request start."""
        self.state.api_in_progress = True
        self.state.api_calls += 1
        self._add_event("API", "request started")
        self._refresh()

    async def _on_api_response(self, event: Event) -> None:
        """Handle API response."""
        self.state.api_in_progress = False
        response = event.data.get("response", "")
        source = event.data.get("source", "cloud")  # "local" or "cloud"

        # Track source for display
        self.state.last_llm_source = source
        if source == "local":
            self.state.local_calls += 1

        if response:
            self._add_conversation("assistant", response, source=source)

        # Show source in event log
        source_label = "(local)" if source == "local" else "(cloud)"
        self._add_event("API", f"response {source_label}")
        self._refresh()

    async def _on_api_error(self, event: Event) -> None:
        """Handle API errors."""
        self.state.api_in_progress = False
        self.state.errors += 1
        error = event.data.get("error", "unknown")
        self._add_event("ERROR", f"API: {error}")
        self._refresh()

    async def _on_tts_start(self, event: Event) -> None:
        """Handle TTS playback start."""
        self.state.tts_in_progress = True
        self._add_event("TTS", "playback started")
        self._refresh()

    async def _on_tts_complete(self, event: Event) -> None:
        """Handle TTS playback complete."""
        self.state.tts_in_progress = False
        self._add_event("TTS", "playback complete")
        self._refresh()

    async def _on_system_error(self, event: Event) -> None:
        """Handle system errors."""
        self.state.errors += 1
        error = event.data.get("error", "unknown")
        self._add_event("ERROR", str(error))
        self._refresh()

    def _add_event(self, event_type: str, detail: str) -> None:
        """Add an event to the recent events list."""
        self.state.events.insert(0, (datetime.now(), event_type, detail))
        # Trim to max events
        if len(self.state.events) > self.config.max_events:
            self.state.events = self.state.events[: self.config.max_events]

    def _add_conversation(self, role: str, content: str, source: str = "") -> None:
        """Add a turn to the conversation history.

        Args:
            role: The speaker role ("user" or "assistant").
            content: The message content.
            source: For assistant responses, "local" or "cloud" to indicate LLM source.
        """
        self.state.conversation.append(ConversationTurn(role=role, content=content, source=source))
        # Trim to max turns
        if len(self.state.conversation) > self.config.max_conversation_turns:
            self.state.conversation = self.state.conversation[-self.config.max_conversation_turns :]

    async def _uptime_updater(self) -> None:
        """Background task to update uptime."""
        while self._running:
            self.state.uptime_seconds = (datetime.now() - self.state.start_time).total_seconds()
            self._refresh()
            await asyncio.sleep(1.0)

    def _refresh(self) -> None:
        """Refresh the display with current state."""
        if self._live:
            self._live.update(self._build_layout())

    def _build_layout(self) -> Panel:
        """Build the complete layout for the display."""
        if self.config.compact_mode:
            return self._build_compact_layout()
        return self._build_full_layout()

    def _build_compact_layout(self) -> Panel:
        """Build a compact single-panel layout with HARO character."""
        # Get HARO state and color
        haro_state = self.STATE_TO_HARO.get(self.state.agent_state, "idle")
        if self.state.api_in_progress:
            haro_state = "processing"
        color = self.HARO_STATE_COLORS.get(haro_state, "cyan")
        char_lines = self.HARO_STATES.get(haro_state, self.HARO_STATES["idle"])

        # Create grid with HARO character on left, status on right
        content = Table.grid(expand=True, padding=(0, 1))
        content.add_column(width=9)  # HARO character
        content.add_column(ratio=1)  # Status info

        # Build character text
        char_text = Text()
        for line in char_lines:
            char_text.append(line + "\n", style=f"bold {color}")

        # Build status info
        status_text = Text()
        style, indicator = self.STATE_STYLES.get(self.state.agent_state, ("white", "???"))
        status_text.append(f"[{indicator}] ", style=style)
        status_text.append(self.state.agent_state, style=f"bold {style}")

        if self.state.is_speech:
            status_text.append("  SPEECH", style="green")
        if self.state.api_in_progress:
            status_text.append("  API...", style="yellow")
        if self.state.tts_in_progress:
            status_text.append("  TTS...", style="blue")

        # Add last transcript if present
        if self.state.last_transcript:
            status_text.append(f"\n> {self.state.last_transcript}", style="cyan")

        content.add_row(char_text, status_text)

        return Panel(content, title="HARO", border_style=color)

    # ASCII art for HARO logo (compact, 3 lines)
    HARO_ASCII = """╦ ╦╔═╗╦═╗╔═╗
╠═╣╠═╣╠╦╝║ ║
╩ ╩╩ ╩╩╚═╚═╝"""

    def _build_full_layout(self) -> Panel:
        """Build the full multi-panel layout."""
        # Create main layout grid
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=7),  # Increased for HARO character
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Split header: HARO character (left), status (middle), logo (right)
        layout["header"].split_row(
            Layout(name="haro_char", size=11),  # HARO character art
            Layout(name="status", ratio=3),
            Layout(name="logo", size=16),
        )

        # Split main into left and right
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Populate sections
        layout["haro_char"].update(self._build_haro_character_panel())
        layout["status"].update(self._build_header())
        layout["logo"].update(self._build_logo_panel())
        layout["left"].update(self._build_conversation_panel())
        layout["right"].split_column(
            Layout(self._build_audio_panel(), name="audio", size=5),
            Layout(self._build_events_panel(), name="events"),
        )
        layout["footer"].update(self._build_stats_bar())

        return Panel(layout, title="HARO Voice Assistant", border_style="blue")

    def _build_logo_panel(self) -> Panel:
        """Build the ASCII logo panel."""
        logo_text = Text()
        for line in self.HARO_ASCII.strip().split("\n"):
            logo_text.append(line + "\n", style="bold cyan")
        return Panel(logo_text, border_style="dim")

    def _build_haro_character_panel(self) -> Panel:
        """Build the animated HARO character panel based on current state."""
        # Get the HARO character state from agent state
        haro_state = self.STATE_TO_HARO.get(self.state.agent_state, "idle")

        # Handle api_in_progress as "processing" state
        if self.state.api_in_progress:
            haro_state = "processing"

        # Get the character art and color for this state
        char_lines = self.HARO_STATES.get(haro_state, self.HARO_STATES["idle"])
        color = self.HARO_STATE_COLORS.get(haro_state, "cyan")

        # Build the character text with color
        char_text = Text()
        for line in char_lines:
            char_text.append(line + "\n", style=f"bold {color}")

        return Panel(char_text, border_style=color)

    def _build_header(self) -> Panel:
        """Build the header with state indicator."""
        style, indicator = self.STATE_STYLES.get(self.state.agent_state, ("white", "???"))

        # Create state display
        state_text = Text()
        state_text.append(f" [{indicator}] ", style=f"bold {style}")
        state_text.append(self.state.agent_state, style=f"bold {style}")

        # Add activity indicators
        indicators = []
        if self.state.is_speech:
            indicators.append(("[SPEECH]", "green"))
        if self.state.api_in_progress:
            indicators.append(("[API]", "yellow"))
        if self.state.tts_in_progress:
            indicators.append(("[TTS]", "blue"))

        if indicators:
            state_text.append("  ")
            for text, ind_style in indicators:
                state_text.append(text, style=ind_style)
                state_text.append(" ")

        return Panel(state_text, border_style=style)

    def _build_conversation_panel(self) -> Panel:
        """Build the conversation history panel."""
        conv_text = Text()

        if not self.state.conversation:
            conv_text.append("Waiting for conversation...", style="dim")
        else:
            for turn in self.state.conversation:
                # Role indicator
                if turn.role == "user":
                    conv_text.append("YOU: ", style="bold cyan")
                else:
                    conv_text.append("HARO", style="bold green")
                    # Show source indicator for assistant responses
                    if turn.source == "local":
                        conv_text.append(" (local)", style="dim yellow")
                    elif turn.source == "cloud":
                        conv_text.append(" (cloud)", style="dim blue")
                    conv_text.append(": ", style="bold green")

                # Content (truncate if too long)
                content = turn.content
                if len(content) > 200:
                    content = content[:197] + "..."
                conv_text.append(f"{content}\n\n")

        return Panel(conv_text, title="Conversation", border_style="cyan")

    def _build_audio_panel(self) -> Panel:
        """Build the audio level indicator panel."""
        # Energy bar
        energy_pct = min(self.state.audio_energy * 100, 100)
        bar_width = 20
        filled = int(energy_pct / 100 * bar_width)

        bar_text = Text()
        bar_text.append("Level: ")

        # Color based on speech detection
        bar_color = "green" if self.state.is_speech else "blue"
        bar_text.append("[" + "#" * filled, style=bar_color)
        bar_text.append("-" * (bar_width - filled) + "]", style="dim")
        bar_text.append(f" {energy_pct:.0f}%")

        if self.state.is_speech:
            bar_text.append(" SPEECH", style="bold green")

        return Panel(bar_text, title="Audio", border_style="blue")

    def _build_events_panel(self) -> Panel:
        """Build the recent events panel."""
        events_text = Text()

        if not self.state.events:
            events_text.append("No recent events", style="dim")
        else:
            for timestamp, event_type, detail in self.state.events[:self.config.max_events]:
                time_str = timestamp.strftime("%H:%M:%S")
                events_text.append(f"{time_str} ", style="dim")

                # Color by event type
                type_styles = {
                    "STATE": "magenta",
                    "WAKE": "green",
                    "VAD": "cyan",
                    "STT": "yellow",
                    "API": "blue",
                    "TTS": "blue",
                    "ERROR": "red",
                }
                type_style = type_styles.get(event_type, "white")
                events_text.append(f"[{event_type}] ", style=type_style)
                events_text.append(f"{detail}\n")

        return Panel(events_text, title="Events", border_style="dim")

    def _build_stats_bar(self) -> Panel:
        """Build the statistics footer bar."""
        # Format uptime
        uptime = self.state.uptime_seconds
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)

        if hours > 0:
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            uptime_str = f"{minutes}m {seconds}s"
        else:
            uptime_str = f"{seconds}s"

        stats_text = Text()
        stats_text.append(f"Uptime: {uptime_str}", style="dim")
        stats_text.append("  |  ")
        stats_text.append(f"Wake: {self.state.wake_count}", style="green")
        stats_text.append("  |  ")
        # Show cloud and local API calls separately
        cloud_calls = self.state.api_calls - self.state.local_calls
        stats_text.append(f"Cloud: {cloud_calls}", style="blue")
        stats_text.append("  |  ")
        stats_text.append(f"Local: {self.state.local_calls}", style="yellow")
        stats_text.append("  |  ")
        if self.state.errors > 0:
            stats_text.append(f"Errors: {self.state.errors}", style="red")
        else:
            stats_text.append("Errors: 0", style="dim")

        return Panel(stats_text, border_style="dim")

    # Public methods for manual updates (useful when not using events)
    def set_state(self, state: str) -> None:
        """Manually set the agent state."""
        self.state.agent_state = state
        self._refresh()

    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation."""
        self._add_conversation("user", message)
        self._refresh()

    def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the conversation."""
        self._add_conversation("assistant", message)
        self._refresh()

    def set_audio_energy(self, energy: float, is_speech: bool = False) -> None:
        """Update audio energy level."""
        self.state.audio_energy = energy
        self.state.is_speech = is_speech
        self._refresh()

    def increment_wake_count(self) -> None:
        """Increment the wake word detection count."""
        self.state.wake_count += 1
        self._refresh()

    def increment_api_calls(self) -> None:
        """Increment the API call count."""
        self.state.api_calls += 1
        self._refresh()

    def increment_errors(self) -> None:
        """Increment the error count."""
        self.state.errors += 1
        self._refresh()
