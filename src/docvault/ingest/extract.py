"""PDF text extraction using PyMuPDF (fitz)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from a PDF file, returning one record per non-empty page.

    Each record contains:

    - ``page_number`` (int): 1-based page index.
    - ``text`` (str): Extracted plain text for the page.
    - ``doc_name`` (str): Filename of the source PDF.

    Encrypted PDFs are skipped with a warning.  Pages with no extractable
    text are also skipped.  If a page contains images but no text the caller
    is warned that OCR is not supported.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of page-level dictionaries.  Empty list if no text could be
        extracted (e.g. encrypted or fully image-based document).

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        OSError: If the file cannot be opened by PyMuPDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = pdf_path.name
    pages: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise OSError(f"Could not open {pdf_path}: {exc}") from exc

    try:
        # Handle encrypted PDFs
        if doc.is_encrypted:
            logger.warning(
                "Skipping encrypted PDF '%s'. "
                "Provide a decrypted copy to index this document.",
                doc_name,
            )
            return []

        total_pages = len(doc)
        image_only_pages = 0

        for page_index in range(total_pages):
            page = doc[page_index]
            page_number = page_index + 1

            text = page.get_text()
            text_stripped = text.strip()

            if not text_stripped:
                # Check whether the page has images (likely scanned)
                images = page.get_images(full=False)
                if images:
                    image_only_pages += 1
                    logger.debug(
                        "'%s' page %d: no text found but %d image(s) detected — "
                        "OCR is not supported in this version; page skipped.",
                        doc_name,
                        page_number,
                        len(images),
                    )
                else:
                    logger.debug(
                        "'%s' page %d: empty page — skipped.", doc_name, page_number
                    )
                continue

            pages.append(
                {
                    "page_number": page_number,
                    "text": text_stripped,
                    "doc_name": doc_name,
                }
            )

        if image_only_pages > 0:
            logger.warning(
                "'%s': %d/%d page(s) contained only images and were skipped. "
                "OCR is not supported in this version of docvault.",
                doc_name,
                image_only_pages,
                total_pages,
            )

        if not pages:
            logger.warning(
                "'%s': no extractable text found across %d page(s). "
                "The document may be fully image-based or empty.",
                doc_name,
                total_pages,
            )

        logger.info(
            "Extracted %d page(s) from '%s'.", len(pages), doc_name
        )

    finally:
        doc.close()

    return pages


def extract_pdfs(pdf_paths: list[Path]) -> list[dict[str, Any]]:
    """Extract text from multiple PDF files.

    Processes each file in order; errors on individual files are logged and
    skipped so that one bad file does not halt the entire batch.

    Args:
        pdf_paths: Paths to PDF files.

    Returns:
        Concatenated list of page-level dictionaries from all documents.
    """
    results: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        try:
            pages = extract_pdf(pdf_path)
            results.extend(pages)
        except (FileNotFoundError, OSError) as exc:
            logger.error("Skipping '%s': %s", pdf_path, exc)
    return results
