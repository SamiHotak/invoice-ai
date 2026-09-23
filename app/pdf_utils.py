"""Convert PDF pages to PIL images with PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class PdfError(ValueError):
    """Raised when a PDF cannot be opened or has no pages."""


def _import_pymupdf():
    """Import PyMuPDF. New versions use "pymupdf", old versions only "fitz"."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # type: ignore[no-redef]
    return pymupdf


def pdf_page_count(path: Union[str, Path]) -> int:
    """Return the number of pages in a PDF."""
    pymupdf = _import_pymupdf()
    try:
        with pymupdf.open(str(path)) as doc:
            return doc.page_count
    except Exception as exc:  # PyMuPDF raises several error types
        raise PdfError(f"Could not open PDF: {Path(path).name}") from exc


def pdf_to_images(
    path: Union[str, Path],
    dpi: int = 200,
    max_pages: Optional[int] = None,
) -> list[Image.Image]:
    """Render the pages of a PDF as RGB images.

    Args:
        path: Path to the PDF file.
        dpi: Render resolution. 200 is sharp enough for small receipt text.
        max_pages: Only render the first N pages. None renders all pages.

    Returns:
        One PIL image per page, in page order.

    Raises:
        PdfError: If the file is missing, broken, password protected or empty.
    """
    path = Path(path)
    if not path.is_file():
        raise PdfError(f"File not found: {path}")

    pymupdf = _import_pymupdf()
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # PyMuPDF raises several error types
        raise PdfError(f"Could not open PDF: {path.name}") from exc

    with doc:
        if doc.needs_pass:
            raise PdfError(f"PDF is password protected: {path.name}")
        if doc.page_count == 0:
            raise PdfError(f"PDF has no pages: {path.name}")

        page_limit = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
        if page_limit < doc.page_count:
            logger.warning(
                "%s has %d pages. Only the first %d are used.",
                path.name,
                doc.page_count,
                page_limit,
            )

        images: list[Image.Image] = []
        for page_index in range(page_limit):
            pixmap = doc.load_page(page_index).get_pixmap(dpi=dpi, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            images.append(image)

    logger.info("Rendered %d page(s) from %s at %d DPI.", len(images), path.name, dpi)
    return images
