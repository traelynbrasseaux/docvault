"""Response parsing: citation extraction and structured output."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docvault.retrieval.search import SearchResult

logger = logging.getLogger(__name__)

# Matches [1], [2], [12], etc. — citation markers in the LLM response text
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class CitedSource:
    """A single source cited by the LLM in its answer.

    Attributes:
        citation_number: The ``[N]`` number as it appears in the answer text.
        doc_name: Filename of the source PDF.
        page_number: 1-based page number within the source document.
        chunk_text: The exact chunk text the model drew from.
        score: Relevance score for this chunk (cosine sim or cross-encoder).
    """

    citation_number: int
    doc_name: str
    page_number: int
    chunk_text: str
    score: float


@dataclass
class Response:
    """Structured output returned by :func:`parse_response`.

    Attributes:
        query: The original user question.
        answer: The full answer text as returned by the LLM, including
            inline ``[N]`` citation markers.
        cited_sources: Ordered list of sources actually referenced in the
            answer, keyed by their citation number.
        all_results: All retrieval results that were passed into the prompt
            (including any not cited by the LLM).
    """

    query: str
    answer: str
    cited_sources: list[CitedSource] = field(default_factory=list)
    all_results: list = field(default_factory=list)  # list[SearchResult]

    @property
    def has_citations(self) -> bool:
        """``True`` if the answer contains at least one citation marker."""
        return bool(self.cited_sources)

    def format_citations(self) -> str:
        """Return a human-readable citation block for CLI or log output.

        Returns:
            Multi-line string listing each cited source, or an empty string
            if no citations were found.
        """
        if not self.cited_sources:
            return ""
        lines = ["Sources:"]
        for src in self.cited_sources:
            lines.append(
                f"  [{src.citation_number}] {src.doc_name}, "
                f"page {src.page_number} (score: {src.score:.3f})"
            )
        return "\n".join(lines)


def parse_response(
    query: str,
    raw_text: str,
    included_results: "list[SearchResult]",
) -> Response:
    """Parse the LLM's raw reply into a structured :class:`Response`.

    Scans *raw_text* for ``[N]`` citation markers, maps each to the
    corresponding entry in *included_results* (which are 1-indexed in the
    prompt), and builds a deduplicated, ordered list of
    :class:`CitedSource` objects.

    Citations that reference an out-of-range index (e.g. the model
    hallucinated ``[99]``) are logged as warnings and omitted.

    Args:
        query: The original user question.
        raw_text: Full response text from the LLM.
        included_results: The ordered list of
            :class:`~docvault.retrieval.search.SearchResult` objects that
            were included in the prompt (as returned by
            :func:`~docvault.generation.prompt.build_prompt`).

    Returns:
        A :class:`Response` with the answer text and resolved citations.
    """
    # Extract unique citation numbers in order of first appearance
    seen: set[int] = set()
    ordered_numbers: list[int] = []
    for match in _CITATION_RE.finditer(raw_text):
        num = int(match.group(1))
        if num not in seen:
            seen.add(num)
            ordered_numbers.append(num)

    cited_sources: list[CitedSource] = []
    for num in ordered_numbers:
        idx = num - 1  # prompt uses 1-based numbering
        if idx < 0 or idx >= len(included_results):
            logger.warning(
                "LLM cited [%d] but only %d source(s) were in the prompt — "
                "ignoring out-of-range citation.",
                num,
                len(included_results),
            )
            continue
        result = included_results[idx]
        cited_sources.append(
            CitedSource(
                citation_number=num,
                doc_name=result.chunk.metadata.doc_name,
                page_number=result.chunk.metadata.page_number,
                chunk_text=result.chunk.text,
                score=result.score,
            )
        )

    logger.info(
        "Parsed response: %d citation(s) found, %d resolved.",
        len(ordered_numbers),
        len(cited_sources),
    )

    return Response(
        query=query,
        answer=raw_text,
        cited_sources=cited_sources,
        all_results=list(included_results),
    )
