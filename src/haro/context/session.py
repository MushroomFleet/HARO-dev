"""Session management for HARO.

Handles conversation session creation, logging, and retrieval.
Sessions are stored as markdown files in .context/sessions/.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    """A single conversation turn."""

    user_input: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """A conversation session with logging capabilities.

    Tracks conversation turns and persists them to a markdown file.
    """

    id: str
    path: Path
    started: datetime = field(default_factory=datetime.now)
    device_id: str = "unknown"
    model: str = "unknown"
    turns: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Get number of turns in session."""
        return len(self.turns)

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        if not self.turns:
            return 0.0
        last_turn = self.turns[-1].timestamp
        return (last_turn - self.started).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if session file exists."""
        return self.path.exists()

    def add_turn(self, user_input: str, response: str, **metadata) -> Turn:
        """Add a conversation turn.

        Args:
            user_input: User's transcribed speech.
            response: HARO's response.
            **metadata: Additional metadata for the turn.

        Returns:
            The created Turn.
        """
        turn = Turn(
            user_input=user_input,
            response=response,
            timestamp=datetime.now(),
            metadata=metadata,
        )
        self.turns.append(turn)
        return turn

    def get_history(self, limit: Optional[int] = None) -> str:
        """Get conversation history as formatted text.

        Args:
            limit: Maximum number of turns to include.

        Returns:
            Formatted conversation history.
        """
        turns_to_include = self.turns[-limit:] if limit else self.turns
        lines = []

        for turn in turns_to_include:
            lines.append(f"User: {turn.user_input}")
            lines.append(f"HARO: {turn.response}")
            lines.append("")

        return "\n".join(lines)

    def get_messages(self, limit: Optional[int] = None) -> list[dict[str, str]]:
        """Get conversation history as message list for API.

        Args:
            limit: Maximum number of turns to include.

        Returns:
            List of message dictionaries.
        """
        turns_to_include = self.turns[-limit:] if limit else self.turns
        messages = []

        for turn in turns_to_include:
            messages.append({"role": "user", "content": turn.user_input})
            messages.append({"role": "assistant", "content": turn.response})

        return messages

    def save(self) -> None:
        """Save session to markdown file."""
        content = self._to_markdown()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content, encoding="utf-8")

    def _to_markdown(self) -> str:
        """Convert session to markdown format.

        Returns:
            Markdown-formatted session content.
        """
        lines = [
            f"# Session {self.id}",
            "",
            "## Metadata",
            f"- **Started**: {self.started.isoformat()}",
            f"- **Device**: {self.device_id}",
            f"- **Model**: {self.model}",
            f"- **Turns**: {self.turn_count}",
            "",
        ]

        if self.metadata:
            for key, value in self.metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.extend(["## Conversation", ""])

        for i, turn in enumerate(self.turns, 1):
            timestamp = turn.timestamp.strftime("%H:%M:%S")
            lines.extend([
                f"### Turn {i} [{timestamp}]",
                f"**User**: {turn.user_input}",
                "",
                f"**HARO**: {turn.response}",
                "",
            ])

            if turn.metadata:
                lines.append("*Metadata:*")
                for key, value in turn.metadata.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

        # Session stats
        lines.extend([
            "## Session Stats",
            f"- **Total Turns**: {self.turn_count}",
            f"- **Duration**: {self.duration:.1f} seconds",
            "",
        ])

        return "\n".join(lines)

    @classmethod
    def load(cls, path: Path) -> Optional["Session"]:
        """Load a session from a markdown file.

        Args:
            path: Path to the session file.

        Returns:
            Session object or None if loading fails.
        """
        if not path.exists():
            return None

        try:
            content = path.read_text(encoding="utf-8")
            return cls._from_markdown(content, path)
        except Exception as e:
            logger.error("session_load_failed", path=str(path), error=str(e))
            return None

    @classmethod
    def _from_markdown(cls, content: str, path: Path) -> "Session":
        """Parse session from markdown content.

        Args:
            content: Markdown content.
            path: Path to the session file.

        Returns:
            Parsed Session object.
        """
        import re

        # Extract session ID from filename
        session_id = path.stem

        # Parse metadata section
        started = datetime.now()
        device_id = "unknown"
        model = "unknown"

        started_match = re.search(r"\*\*Started\*\*:\s*(.+)", content)
        if started_match:
            try:
                started = datetime.fromisoformat(started_match.group(1).strip())
            except ValueError:
                pass

        device_match = re.search(r"\*\*Device\*\*:\s*(.+)", content)
        if device_match:
            device_id = device_match.group(1).strip()

        model_match = re.search(r"\*\*Model\*\*:\s*(.+)", content)
        if model_match:
            model = model_match.group(1).strip()

        # Parse turns
        turns = []
        turn_pattern = re.compile(
            r"### Turn \d+ \[(\d{2}:\d{2}:\d{2})\]\n"
            r"\*\*User\*\*:\s*(.+?)\n\n"
            r"\*\*HARO\*\*:\s*(.+?)(?=\n\n###|\n\n## |$)",
            re.DOTALL
        )

        for match in turn_pattern.finditer(content):
            time_str = match.group(1)
            user_input = match.group(2).strip()
            response = match.group(3).strip()

            # Construct timestamp using session date and turn time
            try:
                turn_time = datetime.strptime(time_str, "%H:%M:%S").time()
                timestamp = datetime.combine(started.date(), turn_time)
            except ValueError:
                timestamp = datetime.now()

            turns.append(Turn(
                user_input=user_input,
                response=response,
                timestamp=timestamp,
            ))

        return cls(
            id=session_id,
            path=path,
            started=started,
            device_id=device_id,
            model=model,
            turns=turns,
        )


class SessionManager:
    """Manage conversation sessions.

    Handles session creation, retrieval, and persistence.
    """

    def __init__(self, sessions_path: Path) -> None:
        """Initialize session manager.

        Args:
            sessions_path: Path to sessions directory.
        """
        self.sessions_path = sessions_path
        self._current_session: Optional[Session] = None
        self.logger = logger.bind(component="SessionManager")

    @property
    def current_session(self) -> Optional[Session]:
        """Get current active session."""
        return self._current_session

    def start_session(
        self,
        device_id: str = "unknown",
        model: str = "unknown",
        **metadata,
    ) -> Session:
        """Start a new conversation session.

        Args:
            device_id: Device identifier.
            model: Model being used.
            **metadata: Additional session metadata.

        Returns:
            The new Session.
        """
        # Generate session ID
        now = datetime.now()
        session_id = now.strftime("%Y-%m-%d_%H%M%S")

        # Create session
        session = Session(
            id=session_id,
            path=self.sessions_path / f"{session_id}.md",
            started=now,
            device_id=device_id,
            model=model,
            metadata=metadata,
        )

        self._current_session = session
        self.logger.info(
            "session_started",
            session_id=session_id,
            device_id=device_id,
        )

        return session

    def end_session(self, save: bool = True) -> Optional[Session]:
        """End the current session.

        Args:
            save: Whether to save the session before ending.

        Returns:
            The ended session or None.
        """
        if not self._current_session:
            return None

        session = self._current_session

        if save and session.turn_count > 0:
            session.save()
            self.logger.info(
                "session_ended",
                session_id=session.id,
                turns=session.turn_count,
                duration=session.duration,
            )

        self._current_session = None
        return session

    def log_turn(
        self,
        user_input: str,
        response: str,
        auto_save: bool = True,
        **metadata,
    ) -> Optional[Turn]:
        """Log a conversation turn.

        Args:
            user_input: User's input.
            response: HARO's response.
            auto_save: Whether to auto-save after logging.
            **metadata: Additional turn metadata.

        Returns:
            The logged Turn or None if no active session.
        """
        if not self._current_session:
            self._current_session = self.start_session()

        turn = self._current_session.add_turn(user_input, response, **metadata)

        if auto_save:
            self._current_session.save()

        self.logger.debug(
            "turn_logged",
            session_id=self._current_session.id,
            turn_number=self._current_session.turn_count,
        )

        return turn

    def list_sessions(self, limit: Optional[int] = None) -> list[Path]:
        """List available session files.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session file paths, newest first.
        """
        if not self.sessions_path.exists():
            return []

        sessions = sorted(
            self.sessions_path.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if limit:
            sessions = sessions[:limit]

        return sessions

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID.

        Args:
            session_id: The session ID to load.

        Returns:
            The loaded Session or None.
        """
        path = self.sessions_path / f"{session_id}.md"
        return Session.load(path)

    def get_recent_turns(self, limit: int = 10) -> list[Turn]:
        """Get recent turns from recent sessions.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            List of recent turns across sessions.
        """
        all_turns = []

        for session_path in self.list_sessions(limit=5):
            session = Session.load(session_path)
            if session:
                all_turns.extend(session.turns)

        # Sort by timestamp and return most recent
        all_turns.sort(key=lambda t: t.timestamp, reverse=True)
        return all_turns[:limit]
