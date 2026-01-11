"""HARO context module.

Manages the .context directory for persistent memory and knowledge.
"""

from haro.context.init import init_context, ContextInitResult
from haro.context.session import Session, Turn, SessionManager
from haro.context.knowledge import KnowledgeBase, KnowledgeEntry
from haro.context.manager import ContextManager

__all__ = [
    "init_context",
    "ContextInitResult",
    "Session",
    "Turn",
    "SessionManager",
    "KnowledgeBase",
    "KnowledgeEntry",
    "ContextManager",
]
