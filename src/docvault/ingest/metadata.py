"""Chunk metadata dataclass and the Chunk container used throughout docvault."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ChunkMetadata:
    """Metadata attached to every text chunk produced by the ingest pipeline.

    Attributes:
        doc_name: Filename of the source PDF (e.g. ``report.pdf``).
        page_number: 1-based page number within the source document.
        chunk_index: 0-based position of this chunk within the document.
        strategy_used: Name of the chunking strategy (``fixed``, ``recursive``,
            or ``semantic``).
        char_count: Number of characters in the chunk text.
    """

    doc_name: str
    page_number: int
    chunk_index: int
    strategy_used: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON storage.

        Returns:
            Dictionary with all metadata fields.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        """Deserialize from a plain dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A :class:`ChunkMetadata` instance.
        """
        return cls(
            doc_name=data["doc_name"],
            page_number=data["page_number"],
            chunk_index=data["chunk_index"],
            strategy_used=data["strategy_used"],
            char_count=data["char_count"],
        )


@dataclass
class Chunk:
    """A single text chunk with its associated metadata.

    Attributes:
        text: The raw chunk text.
        metadata: Metadata describing the chunk's origin and properties.
    """

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            Dictionary with ``text`` and ``metadata`` keys.
        """
        return {"text": self.text, "metadata": self.metadata.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Deserialize from a plain dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A :class:`Chunk` instance.
        """
        return cls(
            text=data["text"],
            metadata=ChunkMetadata.from_dict(data["metadata"]),
        )


def save_chunks_json(chunks: list[Chunk], path: Path) -> None:
    """Persist a list of chunks to a JSON file.

    Args:
        chunks: Chunks to serialize.
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in chunks], fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write chunks to {path}: {exc}") from exc


def load_chunks_json(path: Path) -> list[Chunk]:
    """Load a list of chunks from a JSON file.

    Args:
        path: Path to a JSON file produced by :func:`save_chunks_json`.

    Returns:
        Deserialized list of :class:`Chunk` objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
        OSError: If the file cannot be read or parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Chunk metadata file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [Chunk.from_dict(item) for item in raw]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise OSError(f"Failed to load chunks from {path}: {exc}") from exc
