"""Response parsing for HARO.

Parses Claude API responses to extract structured information
like knowledge updates, follow-up suggestions, and speech text.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeUpdate:
    """An extracted knowledge update."""

    key: str
    value: str
    category: str = "facts"  # facts, preferences, corrections


@dataclass
class ParsedResponse:
    """A parsed response with extracted components."""

    # Main response text (for TTS)
    speech_text: str

    # Optional structured data
    knowledge_updates: list[KnowledgeUpdate] = field(default_factory=list)
    suggested_followups: list[str] = field(default_factory=list)
    confidence: float = 1.0

    # Metadata
    is_question: bool = False
    is_acknowledgment: bool = False
    requires_action: bool = False


class ResponseParser:
    """Parse Claude API responses for HARO.

    Extracts structured information from responses including:
    - Clean speech text (for TTS)
    - Knowledge updates (facts to remember)
    - Follow-up suggestions
    - Response metadata
    """

    def __init__(self) -> None:
        """Initialize response parser."""
        self.logger = logger.bind(component="ResponseParser")

        # Patterns for extracting structured data
        self._knowledge_pattern = re.compile(
            r"\[REMEMBER:\s*(.+?)\s*=\s*(.+?)\]",
            re.IGNORECASE | re.DOTALL,
        )
        self._preference_pattern = re.compile(
            r"\[PREFERENCE:\s*(.+?)\s*=\s*(.+?)\]",
            re.IGNORECASE | re.DOTALL,
        )
        self._followup_pattern = re.compile(
            r"\[FOLLOWUP:\s*(.+?)\]",
            re.IGNORECASE,
        )

        # Patterns for cleaning speech text
        self._markdown_patterns = [
            (re.compile(r"```.*?```", re.DOTALL), ""),  # Code blocks (must be first)
            (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # Bold
            (re.compile(r"\*(.+?)\*"), r"\1"),  # Italic
            (re.compile(r"`(.+?)`"), r"\1"),  # Inline code
            (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),  # Links
            (re.compile(r"#+\s+(.+)"), r"\1"),  # Headers
            (re.compile(r"^\s*[-*]\s+", re.MULTILINE), ""),  # List bullets
            (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),  # Numbered lists
        ]

    def parse(self, response_text: str) -> ParsedResponse:
        """Parse a response into structured components.

        Args:
            response_text: The raw response text from Claude.

        Returns:
            ParsedResponse with extracted components.
        """
        # Extract knowledge updates
        knowledge_updates = self._extract_knowledge(response_text)

        # Extract follow-up suggestions
        followups = self._extract_followups(response_text)

        # Clean text for speech
        speech_text = self._clean_for_speech(response_text)

        # Analyze response type
        is_question = speech_text.rstrip().endswith("?")
        is_acknowledgment = self._is_acknowledgment(speech_text)
        requires_action = self._requires_action(speech_text)

        return ParsedResponse(
            speech_text=speech_text,
            knowledge_updates=knowledge_updates,
            suggested_followups=followups,
            is_question=is_question,
            is_acknowledgment=is_acknowledgment,
            requires_action=requires_action,
        )

    def _extract_knowledge(self, text: str) -> list[KnowledgeUpdate]:
        """Extract knowledge updates from response.

        Args:
            text: Response text.

        Returns:
            List of knowledge updates.
        """
        updates = []

        # Extract REMEMBER tags
        for match in self._knowledge_pattern.finditer(text):
            updates.append(
                KnowledgeUpdate(
                    key=match.group(1).strip(),
                    value=match.group(2).strip(),
                    category="facts",
                )
            )

        # Extract PREFERENCE tags
        for match in self._preference_pattern.finditer(text):
            updates.append(
                KnowledgeUpdate(
                    key=match.group(1).strip(),
                    value=match.group(2).strip(),
                    category="preferences",
                )
            )

        return updates

    def _extract_followups(self, text: str) -> list[str]:
        """Extract follow-up suggestions from response.

        Args:
            text: Response text.

        Returns:
            List of suggested follow-up questions/actions.
        """
        followups = []
        for match in self._followup_pattern.finditer(text):
            followups.append(match.group(1).strip())
        return followups

    def _clean_for_speech(self, text: str) -> str:
        """Clean text for text-to-speech output.

        Args:
            text: Raw response text.

        Returns:
            Cleaned text suitable for TTS.
        """
        result = text

        # Remove structured tags
        result = self._knowledge_pattern.sub("", result)
        result = self._preference_pattern.sub("", result)
        result = self._followup_pattern.sub("", result)

        # Remove markdown formatting
        for pattern, replacement in self._markdown_patterns:
            result = pattern.sub(replacement, result)

        # Clean up whitespace
        result = re.sub(r"\n{3,}", "\n\n", result)  # Multiple newlines
        result = re.sub(r" {2,}", " ", result)  # Multiple spaces
        result = result.strip()

        # Handle common abbreviations for better TTS
        result = self._expand_abbreviations(result)

        return result

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations for clearer TTS.

        Args:
            text: Text with abbreviations.

        Returns:
            Text with expanded abbreviations.
        """
        expansions = {
            r"\bAPI\b": "A P I",
            r"\bAPIs\b": "A P Is",
            r"\bUI\b": "U I",
            r"\bURL\b": "U R L",
            r"\bURLs\b": "U R Ls",
            r"\bHTML\b": "H T M L",
            r"\bCSS\b": "C S S",
            r"\bJSON\b": "J SON",
            r"\bXML\b": "X M L",
            r"\bSQL\b": "S Q L",
            r"\bCPU\b": "C P U",
            r"\bRAM\b": "RAM",
            r"\bGB\b": "gigabytes",
            r"\bMB\b": "megabytes",
            r"\bKB\b": "kilobytes",
            r"\betc\.\b": "et cetera",
            r"\be\.g\.\b": "for example",
            r"\bi\.e\.\b": "that is",
        }

        result = text
        for pattern, replacement in expansions.items():
            result = re.sub(pattern, replacement, result)

        return result

    def _is_acknowledgment(self, text: str) -> bool:
        """Check if response is a simple acknowledgment.

        Args:
            text: Response text.

        Returns:
            True if response is an acknowledgment.
        """
        short_phrases = [
            "okay",
            "ok",
            "sure",
            "done",
            "got it",
            "understood",
            "noted",
            "i understand",
            "i see",
            "alright",
            "all right",
            "yes",
            "no problem",
            "of course",
        ]

        text_lower = text.lower().strip().rstrip("!.,")
        return text_lower in short_phrases

    def _requires_action(self, text: str) -> bool:
        """Check if response suggests action is needed.

        Args:
            text: Response text.

        Returns:
            True if response indicates action needed.
        """
        action_phrases = [
            "i'll remember",
            "i've noted",
            "i've saved",
            "let me",
            "i can",
            "shall i",
            "would you like me to",
            "do you want me to",
        ]

        text_lower = text.lower()
        return any(phrase in text_lower for phrase in action_phrases)

    def get_speech_summary(self, response: ParsedResponse) -> dict:
        """Get a summary of the response for logging.

        Args:
            response: Parsed response.

        Returns:
            Dictionary with summary information.
        """
        return {
            "text_length": len(response.speech_text),
            "word_count": len(response.speech_text.split()),
            "knowledge_updates": len(response.knowledge_updates),
            "followups": len(response.suggested_followups),
            "is_question": response.is_question,
            "is_acknowledgment": response.is_acknowledgment,
        }
