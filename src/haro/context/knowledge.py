"""Knowledge management for HARO.

Handles reading, writing, and searching knowledge files
in the .context/knowledge/ directory.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeEntry:
    """A single knowledge entry."""

    key: str
    value: str
    category: str = "facts"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


class KnowledgeBase:
    """Manage the knowledge base in .context/knowledge/.

    Provides operations for storing, retrieving, and searching
    user preferences and learned facts.
    """

    def __init__(self, knowledge_path: Path) -> None:
        """Initialize knowledge base.

        Args:
            knowledge_path: Path to the knowledge directory.
        """
        self.knowledge_path = knowledge_path
        self.logger = logger.bind(component="KnowledgeBase")

        # Default files
        self._preferences_file = "user_preferences.md"
        self._facts_file = "facts.md"
        self._corrections_file = "corrections.md"

    def ensure_structure(self) -> None:
        """Ensure the knowledge directory structure exists."""
        self.knowledge_path.mkdir(parents=True, exist_ok=True)

    def add_preference(
        self,
        key: str,
        value: str,
        source: str = "conversation",
    ) -> None:
        """Add or update a user preference.

        Args:
            key: Preference key (e.g., "location", "name").
            value: Preference value.
            source: Where this preference came from.
        """
        self._add_entry(
            file_name=self._preferences_file,
            key=key,
            value=value,
            source=source,
        )
        self.logger.info(
            "preference_added",
            key=key,
            value=value[:50],
        )

    def add_fact(
        self,
        key: str,
        value: str,
        source: str = "conversation",
    ) -> None:
        """Add a learned fact.

        Args:
            key: Fact key or topic.
            value: The fact content.
            source: Where this fact came from.
        """
        self._add_entry(
            file_name=self._facts_file,
            key=key,
            value=value,
            source=source,
        )
        self.logger.info(
            "fact_added",
            key=key,
            value=value[:50],
        )

    def add_correction(
        self,
        original: str,
        corrected: str,
        context: str = "",
    ) -> None:
        """Add a user correction.

        Args:
            original: What HARO said incorrectly.
            corrected: The correct information.
            context: Context about when this occurred.
        """
        entry = f"**Original**: {original}\n**Corrected**: {corrected}"
        if context:
            entry += f"\n**Context**: {context}"

        self._add_entry(
            file_name=self._corrections_file,
            key="Correction",
            value=entry,
            source="user_correction",
        )
        self.logger.info("correction_added")

    def _add_entry(
        self,
        file_name: str,
        key: str,
        value: str,
        source: str = "",
    ) -> None:
        """Add an entry to a knowledge file.

        Args:
            file_name: Name of the file to update.
            key: Entry key/heading.
            value: Entry content.
            source: Source of this entry.
        """
        self.ensure_structure()
        file_path = self.knowledge_path / file_name

        # Ensure file exists with header
        if not file_path.exists():
            header = f"# {file_name.replace('.md', '').replace('_', ' ').title()}\n\n"
            file_path.write_text(header, encoding="utf-8")

        # Format entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_lines = [
            "",
            f"## {key}",
            f"*Added: {timestamp}*",
            "",
            value,
            "",
        ]
        if source:
            entry_lines.insert(-1, f"*Source: {source}*")

        # Append to file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(entry_lines))

    def get_preference(self, key: str) -> Optional[str]:
        """Get a user preference by key.

        Args:
            key: Preference key to look for.

        Returns:
            Preference value or None if not found.
        """
        return self._search_file(self._preferences_file, key)

    def get_fact(self, key: str) -> Optional[str]:
        """Get a fact by key.

        Args:
            key: Fact key to look for.

        Returns:
            Fact value or None if not found.
        """
        return self._search_file(self._facts_file, key)

    def _search_file(self, file_name: str, key: str) -> Optional[str]:
        """Search a knowledge file for a specific key.

        Args:
            file_name: File to search.
            key: Key to look for.

        Returns:
            Value if found, None otherwise.
        """
        file_path = self.knowledge_path / file_name
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8")

        # Look for section with matching key
        pattern = rf"## {re.escape(key)}\n.*?\n\n(.+?)(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if match:
            # Extract value, removing metadata lines
            value = match.group(1)
            # Remove lines starting with *
            lines = [
                line for line in value.split("\n")
                if not line.strip().startswith("*")
            ]
            return "\n".join(lines).strip()

        return None

    def search(self, query: str, limit: int = 5) -> list[KnowledgeEntry]:
        """Search all knowledge files for relevant entries.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching KnowledgeEntry objects.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "what", "who", "how", "why", "when", "where", "do", "does", "i", "you", "me", "my"}
        query_words = query_words - stop_words

        if not query_words:
            return results

        for file_path in self.knowledge_path.glob("*.md"):
            entries = self._parse_file(file_path)
            for entry in entries:
                # Score based on keyword matches
                content_lower = (entry.key + " " + entry.value).lower()
                matches = sum(1 for word in query_words if word in content_lower)
                if matches > 0:
                    results.append((matches, entry))

        # Sort by match count and return top results
        results.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in results[:limit]]

    def _parse_file(self, file_path: Path) -> list[KnowledgeEntry]:
        """Parse a knowledge file into entries.

        Args:
            file_path: Path to the file.

        Returns:
            List of KnowledgeEntry objects.
        """
        entries = []
        content = file_path.read_text(encoding="utf-8")

        # Determine category from filename
        category = file_path.stem.replace("_", " ")

        # Parse sections
        section_pattern = re.compile(
            r"## (.+?)\n(.*?)(?=\n## |\Z)",
            re.DOTALL
        )

        for match in section_pattern.finditer(content):
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Extract timestamp if present
            timestamp = datetime.now()
            timestamp_match = re.search(r"\*Added: (\d{4}-\d{2}-\d{2} \d{2}:\d{2})\*", value)
            if timestamp_match:
                try:
                    timestamp = datetime.strptime(
                        timestamp_match.group(1),
                        "%Y-%m-%d %H:%M"
                    )
                except ValueError:
                    pass

            # Extract source if present
            source = ""
            source_match = re.search(r"\*Source: (.+?)\*", value)
            if source_match:
                source = source_match.group(1)

            # Clean value
            clean_value = re.sub(r"\*Added:.*?\*\n?", "", value)
            clean_value = re.sub(r"\*Source:.*?\*\n?", "", clean_value)
            clean_value = clean_value.strip()

            if clean_value:
                entries.append(KnowledgeEntry(
                    key=key,
                    value=clean_value,
                    category=category,
                    timestamp=timestamp,
                    source=source,
                ))

        return entries

    def get_all_preferences(self) -> list[KnowledgeEntry]:
        """Get all user preferences.

        Returns:
            List of preference entries.
        """
        file_path = self.knowledge_path / self._preferences_file
        if not file_path.exists():
            return []
        return self._parse_file(file_path)

    def get_all_facts(self) -> list[KnowledgeEntry]:
        """Get all learned facts.

        Returns:
            List of fact entries.
        """
        file_path = self.knowledge_path / self._facts_file
        if not file_path.exists():
            return []
        return self._parse_file(file_path)

    def get_summary(self) -> dict:
        """Get a summary of the knowledge base.

        Returns:
            Dictionary with knowledge base statistics.
        """
        summary = {
            "preferences_count": 0,
            "facts_count": 0,
            "corrections_count": 0,
            "total_files": 0,
            "total_size_kb": 0,
        }

        if not self.knowledge_path.exists():
            return summary

        for file_path in self.knowledge_path.glob("*.md"):
            summary["total_files"] += 1
            summary["total_size_kb"] += file_path.stat().st_size / 1024

            entries = self._parse_file(file_path)
            if self._preferences_file in file_path.name:
                summary["preferences_count"] = len(entries)
            elif self._facts_file in file_path.name:
                summary["facts_count"] = len(entries)
            elif self._corrections_file in file_path.name:
                summary["corrections_count"] = len(entries)

        return summary
