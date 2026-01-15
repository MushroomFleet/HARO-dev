"""HARO intelligence module.

Provides Claude API integration for intelligent responses,
with optional local LLM (Ollama) support for simple queries.
"""

from haro.intelligence.client import ClaudeClient, APIResponse, APIError
from haro.intelligence.prompts import PromptBuilder, SystemPrompt
from haro.intelligence.parser import ResponseParser, ParsedResponse
from haro.intelligence.ollama_client import OllamaClient, OllamaResponse, OllamaError
from haro.intelligence.router import IntelligenceRouter, RoutedResponse

__all__ = [
    "ClaudeClient",
    "APIResponse",
    "APIError",
    "PromptBuilder",
    "SystemPrompt",
    "ResponseParser",
    "ParsedResponse",
    "OllamaClient",
    "OllamaResponse",
    "OllamaError",
    "IntelligenceRouter",
    "RoutedResponse",
]
