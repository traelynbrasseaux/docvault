"""Generation sub-package: prompt construction, LLM client, and response parsing."""

from docvault.generation.client import LLMClient
from docvault.generation.prompt import build_prompt, build_context_block, SYSTEM_MESSAGE
from docvault.generation.response import CitedSource, Response, parse_response

__all__ = [
    "build_context_block",
    "build_prompt",
    "CitedSource",
    "LLMClient",
    "parse_response",
    "Response",
    "SYSTEM_MESSAGE",
]
