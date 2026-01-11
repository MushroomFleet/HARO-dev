"""Context manager for HARO.

Coordinates session logging, knowledge management, and context
assembly for API requests.
"""

from pathlib import Path
from typing import Optional

from haro.core.config import ContextConfig, APIConfig
from haro.context.session import SessionManager, Session, Turn
from haro.context.knowledge import KnowledgeBase, KnowledgeEntry
from haro.utils.logging import get_logger

logger = get_logger(__name__)


class ContextManager:
    """Manage the .context directory and all context operations.

    Coordinates:
    - Session management (conversation logging)
    - Knowledge base (preferences, facts)
    - Context assembly for API requests
    """

    def __init__(
        self,
        config: ContextConfig,
        api_config: Optional[APIConfig] = None,
    ) -> None:
        """Initialize context manager.

        Args:
            config: Context configuration.
            api_config: Optional API configuration for model info.
        """
        self.config = config
        self.api_config = api_config

        # Set up paths
        self.base_path = Path(config.path).expanduser().resolve()
        self.sessions_path = self.base_path / "sessions"
        self.knowledge_path = self.base_path / "knowledge"
        self.config_path = self.base_path / "config"

        # Initialize components
        self.sessions = SessionManager(self.sessions_path)
        self.knowledge = KnowledgeBase(self.knowledge_path)

        # Configuration
        self.history_turns = config.history_turns
        self.session_timeout = config.session_timeout
        self.auto_save = config.auto_save

        self.logger = logger.bind(component="ContextManager")

    @property
    def current_session(self) -> Optional[Session]:
        """Get current active session."""
        return self.sessions.current_session

    def ensure_structure(self) -> None:
        """Ensure the .context directory structure exists."""
        for path in [self.base_path, self.sessions_path, self.knowledge_path, self.config_path]:
            path.mkdir(parents=True, exist_ok=True)

    def start_session(self, device_id: str = "unknown") -> Session:
        """Start a new conversation session.

        Args:
            device_id: Device identifier.

        Returns:
            The new Session.
        """
        model = self.api_config.model if self.api_config else "unknown"
        return self.sessions.start_session(
            device_id=device_id,
            model=model,
        )

    def end_session(self) -> Optional[Session]:
        """End the current session.

        Returns:
            The ended session or None.
        """
        return self.sessions.end_session(save=self.auto_save)

    def log_turn(
        self,
        user_input: str,
        response: str,
        **metadata,
    ) -> Optional[Turn]:
        """Log a conversation turn.

        Args:
            user_input: User's input.
            response: HARO's response.
            **metadata: Additional turn metadata.

        Returns:
            The logged Turn or None.
        """
        return self.sessions.log_turn(
            user_input=user_input,
            response=response,
            auto_save=self.auto_save,
            **metadata,
        )

    def get_conversation_history(self, limit: Optional[int] = None) -> str:
        """Get current session's conversation history.

        Args:
            limit: Maximum turns to include. Defaults to config.history_turns.

        Returns:
            Formatted conversation history.
        """
        if not self.current_session:
            return ""

        max_turns = limit or self.history_turns
        return self.current_session.get_history(max_turns)

    def get_messages_for_api(self, limit: Optional[int] = None) -> list[dict[str, str]]:
        """Get conversation history as API message format.

        Args:
            limit: Maximum turns to include.

        Returns:
            List of message dictionaries.
        """
        if not self.current_session:
            return []

        max_turns = limit or self.history_turns
        return self.current_session.get_messages(max_turns)

    def process_knowledge_updates(
        self,
        updates: list[KnowledgeEntry],
    ) -> None:
        """Process knowledge updates from a response.

        Args:
            updates: List of knowledge entries to add.
        """
        for update in updates:
            if update.category == "preferences":
                self.knowledge.add_preference(
                    key=update.key,
                    value=update.value,
                    source="conversation",
                )
            elif update.category == "facts":
                self.knowledge.add_fact(
                    key=update.key,
                    value=update.value,
                    source="conversation",
                )
            else:
                # Default to facts
                self.knowledge.add_fact(
                    key=update.key,
                    value=update.value,
                    source="conversation",
                )

    def search_knowledge(self, query: str, limit: int = 5) -> list[KnowledgeEntry]:
        """Search the knowledge base.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            List of matching entries.
        """
        return self.knowledge.search(query, limit)

    def get_relevant_context(self, user_input: str) -> str:
        """Get relevant context for a user input.

        Searches knowledge base for entries relevant to the input.

        Args:
            user_input: The user's input.

        Returns:
            Formatted relevant context.
        """
        relevant = self.knowledge.search(user_input, limit=3)
        if not relevant:
            return ""

        lines = ["## Relevant Knowledge", ""]
        for entry in relevant:
            lines.append(f"### {entry.key}")
            lines.append(entry.value)
            lines.append("")

        return "\n".join(lines)

    def get_user_preferences_summary(self) -> str:
        """Get a summary of user preferences for context.

        Returns:
            Formatted preferences summary.
        """
        preferences = self.knowledge.get_all_preferences()
        if not preferences:
            return ""

        lines = ["## User Preferences", ""]
        for pref in preferences:
            lines.append(f"- **{pref.key}**: {pref.value}")
        lines.append("")

        return "\n".join(lines)

    def load_context_file(self, relative_path: str) -> Optional[str]:
        """Load a context file.

        Args:
            relative_path: Path relative to .context directory.

        Returns:
            File content or None.
        """
        file_path = self.base_path / relative_path
        if file_path.exists():
            try:
                return file_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.warning(
                    "context_file_load_failed",
                    path=relative_path,
                    error=str(e),
                )
        return None

    def get_stats(self) -> dict:
        """Get context manager statistics.

        Returns:
            Dictionary of statistics.
        """
        knowledge_summary = self.knowledge.get_summary()
        session_count = len(self.sessions.list_sessions())

        return {
            "session_count": session_count,
            "current_session_turns": (
                self.current_session.turn_count
                if self.current_session else 0
            ),
            "preferences_count": knowledge_summary.get("preferences_count", 0),
            "facts_count": knowledge_summary.get("facts_count", 0),
            "knowledge_files": knowledge_summary.get("total_files", 0),
            "knowledge_size_kb": knowledge_summary.get("total_size_kb", 0),
        }

    def cleanup_old_sessions(self, keep_count: int = 100) -> int:
        """Clean up old session files.

        Args:
            keep_count: Number of recent sessions to keep.

        Returns:
            Number of sessions deleted.
        """
        sessions = self.sessions.list_sessions()
        to_delete = sessions[keep_count:]

        deleted = 0
        for session_path in to_delete:
            try:
                session_path.unlink()
                deleted += 1
            except Exception as e:
                self.logger.warning(
                    "session_delete_failed",
                    path=str(session_path),
                    error=str(e),
                )

        if deleted > 0:
            self.logger.info("sessions_cleaned_up", count=deleted)

        return deleted
