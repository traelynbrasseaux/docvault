"""Ingest sub-package: PDF extraction, chunking, and metadata."""

from docvault.ingest.chunking import ChunkingStrategy, chunk_pages, chunk_text
from docvault.ingest.extract import extract_pdf, extract_pdfs
from docvault.ingest.metadata import Chunk, ChunkMetadata, load_chunks_json, save_chunks_json

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkingStrategy",
    "chunk_pages",
    "chunk_text",
    "extract_pdf",
    "extract_pdfs",
    "load_chunks_json",
    "save_chunks_json",
]
