"""Prompt construction: system message + numbered context block + user question."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docvault.retrieval.search import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions based only on the provided "
    "context. If the context does not contain enough information to answer, say so "
    "clearly. Always cite which document and page number your answer comes from by "
    "referencing the source numbers in square brackets, e.g. [1] or [2]."
)

# Rough character-to-token ratio for estimation (conservative for English prose)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of *text* using a character heuristic.

    Args:
        text: Input string.

    Returns:
        Estimated token count (``len(text) // 4``).
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


def build_context_block(
    results: "list[SearchResult]",
    max_context_tokens: int = 6000,
) -> tuple[str, list["SearchResult"]]:
    """Assemble the numbered context block from retrieval results.

    Chunks are added in score-descending order until the running token budget
    is exhausted.  The returned list contains only the chunks that fit, in the
    same order they appear in the context block (so citation indices match).

    Args:
        results: Retrieval results, assumed to be sorted by score descending.
        max_context_tokens: Approximate token budget for the entire context
            section (excludes system message and user question).

    Returns:
        A tuple of ``(context_text, included_results)`` where *context_text*
        is the formatted multi-line string and *included_results* is the
        subset of *results* that was included (preserving the ``[N]`` numbering
        used in the text).
    """
    lines: list[str] = []
    included: list["SearchResult"] = []
    budget = max_context_tokens

    for i, result in enumerate(results, start=1):
        header = (
            f"[{i}] From '{result.chunk.metadata.doc_name}', "
            f"page {result.chunk.metadata.page_number} "
            f"(relevance: {result.score:.3f}):"
        )
        entry = f"{header}\n{result.chunk.text}"
        cost = _estimate_tokens(entry) + 2  # +2 for surrounding newlines

        if budget - cost < 0 and included:
            logger.debug(
                "Context budget exhausted after %d/%d chunk(s) — "
                "dropping remaining results.",
                len(included),
                len(results),
            )
            break

        lines.append(entry)
        included.append(result)
        budget -= cost

    context_text = "\n\n".join(lines)
    logger.debug(
        "Built context block: %d chunk(s), ~%d tokens.",
        len(included),
        _estimate_tokens(context_text),
    )
    return context_text, included


def build_prompt(
    query: str,
    results: "list[SearchResult]",
    max_context_tokens: int = 6000,
) -> tuple[str, str, list["SearchResult"]]:
    """Construct the system message and user message for the LLM.

    Args:
        query: The user's natural-language question.
        results: Retrieval results from
            :class:`~docvault.retrieval.search.Searcher`.
        max_context_tokens: Approximate token budget for retrieved context.

    Returns:
        A three-tuple ``(system_message, user_message, included_results)``
        where *included_results* carries the citation mapping used by
        :func:`~docvault.generation.response.parse_response`.
    """
    context_block, included = build_context_block(results, max_context_tokens)

    user_message = (
        f"Context:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    return SYSTEM_MESSAGE, user_message, included
