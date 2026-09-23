"""Full flow: file -> pages -> OCR + VLM extraction -> matching -> result.

Usage:
    from app.pipeline import process_file
    doc = process_file("samples/receipt.jpg")
    print(doc.result.model_dump_json(indent=2))
    doc.draw(page=0).show()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from app.config import Settings, settings as default_settings
from app.extractor import InvoiceExtractor, get_extractor, prepare_image
from app.matcher import FieldMatcher
from app.ocr import OcrEngine, OcrError, OcrLine, get_ocr_engine
from app.pdf_utils import pdf_page_count, pdf_to_images
from app.schema import DocumentResult
from app.visualize import draw_fields

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
SUPPORTED_EXTENSIONS: frozenset[str] = IMAGE_EXTENSIONS | PDF_EXTENSIONS


class UnsupportedFileError(ValueError):
    """Raised for file types the pipeline cannot handle."""


@dataclass
class ProcessedDocument:
    """Pipeline output: the result plus the page images (for drawing)."""

    result: DocumentResult
    pages: list[Image.Image]
    ocr_lines: list[OcrLine] = field(default_factory=list)

    def draw(self, page: int = 0, show_line_items: bool = True) -> Image.Image:
        """Page image with colored boxes around the found fields."""
        if not 0 <= page < len(self.pages):
            raise IndexError(f"Page {page} does not exist (document has {len(self.pages)}).")
        return draw_fields(self.pages[page], self.result, page, show_line_items)


class InvoicePipeline:
    """Connects all parts. Models are loaded once and reused."""

    def __init__(
        self,
        extractor: Optional[InvoiceExtractor] = None,
        ocr: Optional[OcrEngine] = None,
        matcher: Optional[FieldMatcher] = None,
        config: Settings = default_settings,
    ) -> None:
        self.config = config
        self.extractor = extractor or get_extractor()
        self.ocr = ocr or get_ocr_engine()
        self.matcher = matcher or FieldMatcher(config)

    def load(self) -> None:
        """Load all models now (otherwise they load on the first file)."""
        self.extractor.load()
        self.ocr.load()

    def load_pages(self, path: Union[str, Path]) -> list[Image.Image]:
        """Open a JPG/PNG/PDF file as a list of resized RGB page images.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnsupportedFileError: If the file type is not supported.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            raw_pages = [path]
        elif suffix in PDF_EXTENSIONS:
            raw_pages = pdf_to_images(path, dpi=self.config.pdf_dpi, max_pages=self.config.max_pdf_pages)
        else:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise UnsupportedFileError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

        # Same size for OCR, model and drawing, so the boxes line up.
        return [prepare_image(page, self.config.max_image_side) for page in raw_pages]

    def _run_ocr(self, pages: list[Image.Image], warnings: list[str]) -> list[OcrLine]:
        """OCR all pages. If OCR fails, continue without it (all fields become low)."""
        lines: list[OcrLine] = []
        try:
            for index, page in enumerate(pages):
                lines.extend(self.ocr.read(page, page=index))
        except OcrError as exc:
            logger.error("OCR failed: %s", exc)
            warnings.append(f"OCR failed, confidence could not be checked: {exc}")
            return []
        return lines

    def process_file(self, path: Union[str, Path]) -> ProcessedDocument:
        """Process one invoice or receipt file.

        Raises:
            FileNotFoundError, UnsupportedFileError, PdfError: For bad input files.
            ExtractionError: If the model gives no usable answer.
        """
        start = time.perf_counter()
        path = Path(path)
        warnings: list[str] = []

        pages = self.load_pages(path)
        if path.suffix.lower() in PDF_EXTENSIONS:
            total_pages = pdf_page_count(path)
            if total_pages > len(pages):
                warnings.append(f"PDF has {total_pages} pages. Only the first {len(pages)} were used.")

        ocr_lines = self._run_ocr(pages, warnings)
        invoice = self.extractor.extract(pages)
        fields, line_items = self.matcher.match_invoice(invoice, ocr_lines)

        low_fields = [name for name, f in fields.items() if f.confidence == "low"]
        if low_fields and ocr_lines:
            warnings.append(f"Not found in OCR text, please check: {', '.join(low_fields)}")

        result = DocumentResult(
            file_name=path.name,
            num_pages=len(pages),
            invoice=invoice,
            fields=fields,
            line_items=line_items,
            ocr_line_count=len(ocr_lines),
            processing_seconds=round(time.perf_counter() - start, 2),
            warnings=warnings,
        )
        logger.info("Processed %s in %.1f s.", path.name, result.processing_seconds)
        return ProcessedDocument(result=result, pages=pages, ocr_lines=ocr_lines)


_default_pipeline: Optional[InvoicePipeline] = None


def get_pipeline() -> InvoicePipeline:
    """Return the shared pipeline (created once per process)."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = InvoicePipeline()
    return _default_pipeline


def process_file(path: Union[str, Path]) -> ProcessedDocument:
    """Process one file with the shared pipeline."""
    return get_pipeline().process_file(path)
