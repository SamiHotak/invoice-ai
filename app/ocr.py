"""OCR wrapper: returns text lines with boxes and confidence.

We use RapidOCR, which runs the PaddleOCR (PP-OCR) models with ONNX Runtime.
Why not the "paddleocr" package itself? PaddlePaddle crashes in the current
Google Colab environment, and it often clashes with PyTorch. RapidOCR uses
the same models, runs on CPU, and has no conflict with PyTorch.
The models are included in the pip package, so nothing extra is downloaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from app.config import Settings, settings as default_settings
from app.schema import BoundingBox

logger = logging.getLogger(__name__)


class OcrError(RuntimeError):
    """Raised when OCR cannot be loaded or fails on an image."""


@dataclass(frozen=True)
class OcrLine:
    """One line of text found by OCR."""

    text: str
    box: BoundingBox
    confidence: float

    @property
    def page(self) -> int:
        """Page index of this line."""
        return self.box.page


def _box_from_points(points: Iterable[Iterable[float]], page: int) -> BoundingBox:
    """Turn a polygon (list of x, y points) into a rectangle."""
    array = np.asarray(list(points), dtype=float).reshape(-1, 2)
    return BoundingBox(
        x0=float(array[:, 0].min()),
        y0=float(array[:, 1].min()),
        x1=float(array[:, 0].max()),
        y1=float(array[:, 1].max()),
        page=page,
    )


class OcrEngine:
    """Loads RapidOCR once and reads text lines from page images."""

    def __init__(self, config: Settings = default_settings) -> None:
        self.config = config
        self._engine: Any = None

    @property
    def is_loaded(self) -> bool:
        """True if the OCR engine is already in memory."""
        return self._engine is not None

    def load(self) -> None:
        """Load the OCR engine. Safe to call many times."""
        if self.is_loaded:
            return
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise OcrError("RapidOCR is not installed. Run: pip install -r requirements.txt") from exc

        logger.info("Loading RapidOCR (PP-OCR models, ONNX Runtime, CPU).")
        try:
            self._engine = RapidOCR()
        except Exception as exc:
            raise OcrError(f"Could not load RapidOCR: {exc}") from exc
        logger.info("RapidOCR loaded.")

    def read(self, image: Image.Image, page: int = 0) -> list[OcrLine]:
        """Find all text lines on one page image.

        Args:
            image: The page as a PIL image.
            page: Page index, stored in each box.

        Returns:
            Text lines in reading order (top to bottom), low-confidence lines removed.

        Raises:
            OcrError: If OCR fails.
        """
        self.load()
        # RapidOCR expects a BGR numpy array (OpenCV style).
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])

        try:
            output = self._engine(bgr, use_cls=False)  # pages are upright, skip rotation check
        except Exception as exc:
            raise OcrError(f"OCR failed on page {page}: {exc}") from exc

        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if boxes is None or texts is None or scores is None:
            logger.warning("OCR page %d: no text found.", page)
            return []

        lines: list[OcrLine] = []
        for points, text, score in zip(boxes, texts, scores):
            text = str(text).strip()
            if text:
                lines.append(OcrLine(text, _box_from_points(points, page), float(score)))

        kept = [line for line in lines if line.confidence >= self.config.ocr_min_confidence]
        logger.info("OCR page %d: %d lines (%d kept).", page, len(lines), len(kept))
        return kept


_default_engine: Optional[OcrEngine] = None


def get_ocr_engine() -> OcrEngine:
    """Return the shared OCR engine (created once per process)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = OcrEngine()
    return _default_engine
