"""CLI script: query the docvault index and print a grounded answer.

Usage
-----
Windows (PowerShell / CMD)::

    python -m scripts.query "What are the key findings?" --top-k 5 --rerank

Unix / macOS::

    python -m scripts.query "What are the key findings?" --top-k 5 --rerank

Options
-------
question  The question to answer (positional, required).
--top-k   Number of chunks to retrieve (default from .env / 5).
--rerank  Enable cross-encoder reranking (flag, default: off).
--index-dir  Override the index directory (default from .env).
--verbose    Enable DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

# Support running as `python scripts/query.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvault.config import Config
from docvault.pipeline import Pipeline

# Terminal width for wrapping answer text
_WRAP_WIDTH = 88

# ANSI colour codes — disabled automatically on Windows when not in a colour terminal
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    """Return True if the terminal is likely to support ANSI colour codes."""
    import os
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query",
        description="Query the docvault index and display a grounded answer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        type=str,
        help="The question to answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Number of chunks to retrieve (default from config).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking of retrieved chunks.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override the index directory (default from INDEX_DIR env var).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _print_response(question: str, response, use_color: bool) -> None:
    """Pretty-print the answer and citation block to stdout."""
    sep = "─" * min(_WRAP_WIDTH, 72)

    print()
    print(_c(_BOLD, "Question", use_color))
    print(sep)
    print(textwrap.fill(question, width=_WRAP_WIDTH))
    print()

    print(_c(_BOLD, "Answer", use_color))
    print(sep)
    # Wrap each paragraph independently
    for para in response.answer.split("\n"):
        if para.strip():
            print(textwrap.fill(para, width=_WRAP_WIDTH))
        else:
            print()
    print()

    if response.cited_sources:
        print(_c(_BOLD, "Sources", use_color))
        print(sep)
        for src in response.cited_sources:
            header = (
                f"[{src.citation_number}]  "
                f"{_c(_CYAN, src.doc_name, use_color)} — "
                f"page {src.page_number}  "
                f"{_c(_DIM, f'(score: {src.score:.3f})', use_color)}"
            )
            print(header)
            # Indent and wrap the chunk text
            wrapped = textwrap.fill(
                src.chunk_text,
                width=_WRAP_WIDTH - 6,
                initial_indent="      ",
                subsequent_indent="      ",
            )
            print(_c(_DIM, wrapped, use_color))
            print()
    else:
        print(
            _c(_YELLOW, "No citations found in response.", use_color)
        )
        print()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the query CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    use_color = _supports_color()

    # ── Build config ───────────────────────────────────────────────────────
    config = Config()
    if args.index_dir is not None:
        config.index_dir = args.index_dir.resolve()
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.rerank:
        config.rerank = True

    # ── Validate API key early for a clean error message ──────────────────
    try:
        config.require_api_key()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ── Run query ──────────────────────────────────────────────────────────
    pipeline = Pipeline(config=config)

    try:
        response = pipeline.query(
            question=args.question,
            top_k=args.top_k,
            rerank=args.rerank if args.rerank else None,
        )
    except RuntimeError as exc:
        # Empty index or similar operational error
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    _print_response(args.question, response, use_color)
    return 0


if __name__ == "__main__":
    sys.exit(main())
