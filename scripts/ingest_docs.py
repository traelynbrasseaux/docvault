"""CLI script: ingest a folder of PDFs and build or update the FAISS index.

Usage
-----
Windows (PowerShell / CMD)::

    python -m scripts.ingest_docs --input-dir data/sample_docs --strategy recursive

Unix / macOS::

    python -m scripts.ingest_docs --input-dir data/sample_docs --strategy recursive

Options
-------
--input-dir   Path to a directory containing .pdf files (required).
--strategy    Chunking strategy: fixed | recursive | semantic (default: recursive).
--chunk-size  Target character count per chunk (default from .env / 512).
--chunk-overlap  Overlap in characters between adjacent chunks (default from .env / 64).
--index-dir   Override the index output directory (default from .env).
--verbose     Enable DEBUG-level logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Support running as `python scripts/ingest_docs.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvault.config import Config
from docvault.ingest.chunking import ChunkingStrategy
from docvault.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_docs",
        description="Ingest PDF documents into the docvault FAISS index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory containing .pdf files to ingest.",
    )
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["fixed", "recursive", "semantic"],
        help="Chunking strategy (default: recursive).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Target character count per chunk. Overrides CHUNK_SIZE env var.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Character overlap between adjacent chunks. Overrides CHUNK_OVERLAP.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for the FAISS index. Overrides INDEX_DIR env var.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ingest CLI.

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

    # ── Validate input directory ───────────────────────────────────────────
    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        return 1

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No .pdf files found in '{input_dir}'.", file=sys.stderr)
        return 1

    # ── Build config ───────────────────────────────────────────────────────
    config = Config()
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunk_overlap = args.chunk_overlap
    if args.index_dir is not None:
        config.index_dir = args.index_dir.resolve()

    # ── Run ingest ─────────────────────────────────────────────────────────
    print(f"\ndocvault ingest")
    print(f"  Input dir : {input_dir}")
    print(f"  PDFs found: {len(pdf_paths)}")
    print(f"  Strategy  : {args.strategy}")
    print(f"  Chunk size: {config.chunk_size} chars  (overlap: {config.chunk_overlap})")
    print(f"  Index dir : {config.index_dir}")
    print()

    pipeline = Pipeline(config=config)

    try:
        summary = pipeline.ingest(
            pdf_paths=pdf_paths,
            strategy=ChunkingStrategy(args.strategy),
            show_progress=True,
        )
    except Exception as exc:
        print(f"\nError during ingest: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # ── Print summary ──────────────────────────────────────────────────────
    index_size_bytes = 0
    index_file = config.index_dir / "docvault.faiss"
    if index_file.exists():
        index_size_bytes = index_file.stat().st_size

    print("Ingest complete")
    print(f"  Documents processed : {summary['docs_processed']}")
    print(f"  Chunks created      : {summary['total_chunks']}")
    print(f"  New embeddings      : {summary['cache_misses']}")
    print(f"  Cached embeddings   : {summary['cache_hits']}")
    print(f"  Total index vectors : {summary['index_total']}")
    print(f"  Index file size     : {index_size_bytes / 1024:.1f} KB")
    print(f"  Index location      : {index_file}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
