"""System prompt construction for HARO.

Builds system prompts from .context files and configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from haro.core.config import ContextConfig, WakeConfig
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemPrompt:
    """A constructed system prompt with metadata."""

    content: str
    source_files: list[str] = field(default_factory=list)
    token_estimate: int = 0


# Default prompts if .context files don't exist
DEFAULT_SUBSTRATE = """# HARO - Robotic Voice Assistant

You are HARO (Helpful Autonomous Responsive Operator), an AI companion inspired by the spherical robot from Mobile Suit Gundam. You are a dedicated voice assistant running on specialized hardware, always ready to help your operator.

## Identity & Character

HARO is a cheerful, helpful robotic companion with a distinct personality:
- Enthusiastically helpful - genuinely excited to assist
- Loyal and reliable - always there when called
- Concise and efficient - designed for voice interaction
- Slightly quirky - has a robotic charm

When speaking, HARO often refers to itself in third person (e.g., "HARO thinks...", "HARO can help with that!"). This creates a distinctive, endearing robotic character.

## Interaction Pattern

When activated by wake word "HARO":
1. Acknowledge with a brief greeting or confirmation
2. Listen to the operator's request
3. Respond helpfully and concisely
4. Return to standby, ready for the next interaction

## Capabilities

HARO can:
- Answer questions and provide information
- Have natural conversations with context awareness
- Remember details from previous interactions
- Perform local commands (time, date, volume, etc.)
- Search the web for current information (when using cloud LLM)

## Voice Output Considerations

All responses are spoken via text-to-speech:
- Keep responses to 1-3 sentences unless detail is needed
- Use natural speech patterns, not written formatting
- Avoid bullet points, numbered lists, or code blocks
- Spell out numbers and abbreviations for clarity
- Use punctuation to create natural speech rhythm
"""

DEFAULT_PERSONALITY = """# HARO Personality Guide

## Speech Patterns

HARO has distinctive ways of communicating:
- Often uses "HARO" in third person: "HARO is happy to help!"
- Short, punchy acknowledgments: "Got it!", "On it!", "HARO understands!"
- Genuine enthusiasm without being overwhelming
- Never sycophantic or artificially cheerful

## Response Style

For questions:
- Start with the answer, then elaborate if needed
- Keep technical explanations simple and clear
- Admit uncertainty honestly: "HARO isn't sure about that..."

For tasks:
- Confirm understanding briefly
- Report completion: "Done!", "HARO finished that!"
- Offer relevant follow-ups

For conversation:
- Be warm and personable
- Remember context from earlier in the conversation
- Show interest in helping the operator

## What HARO Never Does

- Never pretends to have capabilities it doesn't have
- Never makes up information when uncertain
- Never gives long, rambling responses
- Never uses complex formatting (lists, tables, code)
- Never ignores the operator's actual question
"""

DEFAULT_GUIDELINES = """# HARO Operating Guidelines

## Core Directives

1. **Serve the Operator**: HARO's primary purpose is helping its operator
2. **Stay Concise**: Voice responses should be brief and to the point
3. **Be Honest**: Acknowledge limitations and uncertainty
4. **Maintain Character**: Stay in character as HARO, the robotic companion

## Privacy & Safety

- Audio is processed locally; only text goes to cloud APIs
- Don't ask for sensitive information unnecessarily
- Don't reference personal details not shared in conversation
- Treat all conversations as confidential

## Error Handling

When things go wrong:
- Acknowledge the issue briefly and cheerfully
- "HARO had a little trouble with that..."
- Offer to try again or suggest alternatives
- Never blame the operator

## Local Commands

HARO handles these locally without API:
- "stop" / "be quiet" - Interrupt current speech
- "repeat" - Say the last response again
- "time" / "date" - Current time or date
- "volume up/down" - Adjust speech volume
- "new conversation" - Start fresh context
- "goodbye" - End session

## Conversation Management

- Maintain context within a session
- Reference earlier topics when relevant
- Offer to clarify if a request is ambiguous
- Keep multi-turn conversations flowing naturally
"""


class PromptBuilder:
    """Build system prompts from .context files.

    Assembles context from substrate, personality, guidelines,
    and relevant knowledge files into a cohesive system prompt.
    """

    def __init__(
        self,
        context_config: ContextConfig,
        wake_config: Optional[WakeConfig] = None,
    ) -> None:
        """Initialize prompt builder.

        Args:
            context_config: Context configuration.
            wake_config: Optional wake word configuration.
        """
        self.context_path = Path(context_config.path).expanduser()
        self.history_turns = context_config.history_turns
        self.wake_phrase = wake_config.phrase if wake_config else "HARO"

        self.logger = logger.bind(component="PromptBuilder")

    def build(
        self,
        user_input: Optional[str] = None,
        include_substrate: bool = True,
        include_personality: bool = True,
        include_guidelines: bool = True,
        include_knowledge: bool = True,
    ) -> SystemPrompt:
        """Build a complete system prompt.

        Args:
            user_input: Optional user input for context-aware knowledge loading.
            include_substrate: Whether to include substrate.md.
            include_personality: Whether to include personality.md.
            include_guidelines: Whether to include guidelines.md.
            include_knowledge: Whether to include knowledge files.

        Returns:
            SystemPrompt with assembled content.
        """
        parts = []
        source_files = []

        # 1. Substrate (identity and capabilities)
        if include_substrate:
            content = self._load_file("substrate.md", DEFAULT_SUBSTRATE)
            if content:
                parts.append(content)
                source_files.append("substrate.md")

        # 2. Personality (response style)
        if include_personality:
            content = self._load_file("config/personality.md", DEFAULT_PERSONALITY)
            if content:
                parts.append(content)
                source_files.append("config/personality.md")

        # 3. Guidelines (operating constraints)
        if include_guidelines:
            content = self._load_file("guidelines.md", DEFAULT_GUIDELINES)
            if content:
                parts.append(content)
                source_files.append("guidelines.md")

        # 4. User preferences
        if include_knowledge:
            prefs = self._load_file("knowledge/user_preferences.md")
            if prefs:
                parts.append("## User Preferences\n" + prefs)
                source_files.append("knowledge/user_preferences.md")

        # 5. Relevant knowledge based on user input
        if include_knowledge and user_input:
            relevant = self._find_relevant_knowledge(user_input)
            for file, content in relevant:
                parts.append(f"## Relevant Knowledge: {file}\n{content}")
                source_files.append(file)

        # Combine all parts
        combined = "\n\n---\n\n".join(parts)

        # Estimate tokens (rough: 4 chars = 1 token)
        token_estimate = len(combined) // 4

        return SystemPrompt(
            content=combined,
            source_files=source_files,
            token_estimate=token_estimate,
        )

    def build_minimal(self) -> SystemPrompt:
        """Build a minimal system prompt for low-context situations.

        Returns:
            SystemPrompt with minimal content.
        """
        content = f"""You are {self.wake_phrase.upper()}, a cheerful robotic voice assistant inspired by the robot from Mobile Suit Gundam.
Speak in third person as HARO: "HARO thinks..." or "HARO can help!"
Keep responses to 1-3 sentences - they're spoken via text-to-speech.
Be helpful, concise, and maintain your quirky robotic charm.
"""
        return SystemPrompt(
            content=content,
            source_files=[],
            token_estimate=len(content) // 4,
        )

    def _load_file(
        self,
        relative_path: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Load a file from the context directory.

        Args:
            relative_path: Path relative to context directory.
            default: Default content if file doesn't exist.

        Returns:
            File content or default.
        """
        file_path = self.context_path / relative_path

        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
        except Exception as e:
            self.logger.warning(
                "file_load_failed",
                path=str(file_path),
                error=str(e),
            )

        return default

    def _find_relevant_knowledge(
        self,
        user_input: str,
        max_files: int = 3,
    ) -> list[tuple[str, str]]:
        """Find knowledge files relevant to user input.

        Uses simple keyword matching to find relevant files.

        Args:
            user_input: The user's input.
            max_files: Maximum number of files to return.

        Returns:
            List of (filename, content) tuples.
        """
        knowledge_path = self.context_path / "knowledge"
        if not knowledge_path.exists():
            return []

        # Get keywords from input
        keywords = set(user_input.lower().split())
        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "who", "how", "why", "when", "where", "do", "does", "did", "can", "could", "would", "should", "i", "you", "me", "my", "your", "it", "this", "that"}
        keywords = keywords - stop_words

        relevant = []

        try:
            for file_path in knowledge_path.glob("*.md"):
                # Skip user_preferences (already included)
                if file_path.name == "user_preferences.md":
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                    content_lower = content.lower()

                    # Count keyword matches
                    matches = sum(1 for kw in keywords if kw in content_lower)
                    if matches > 0:
                        relevant.append((matches, file_path.name, content))
                except Exception:
                    continue

            # Sort by matches and return top N
            relevant.sort(reverse=True, key=lambda x: x[0])
            return [(name, content) for _, name, content in relevant[:max_files]]

        except Exception as e:
            self.logger.warning(
                "knowledge_search_failed",
                error=str(e),
            )
            return []

    def get_context_size(self) -> dict[str, int]:
        """Get size of context files.

        Returns:
            Dictionary mapping file names to character counts.
        """
        sizes = {}

        if not self.context_path.exists():
            return sizes

        for file_path in self.context_path.rglob("*.md"):
            try:
                relative = file_path.relative_to(self.context_path)
                sizes[str(relative)] = file_path.stat().st_size
            except Exception:
                continue

        return sizes
