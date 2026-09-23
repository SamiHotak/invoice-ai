"""PaddleOCR wrapper: returns text lines with boxes and confidence.

Works with PaddleOCR 3.x (new "predict" API) and falls back to the
old 2.x "ocr" API if an older version is installed.

OCR runs on CPU by default. This is on purpose: PaddlePaddle GPU and
PyTorch GPU in the same Colab session often clash (CUDA versions).
"""

from __future__ import annotations

import logging
import os
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
    """Loads PaddleOCR once and reads text lines from page images."""

    def __init__(self, config: Settings = default_settings) -> None:
        self.config = config
        self._ocr: Any = None
        self._major_version: int = 3

    @property
    def is_loaded(self) -> bool:
        """True if PaddleOCR is already in memory."""
        return self._ocr is not None

    def load(self) -> None:
        """Load PaddleOCR. Safe to call many times."""
        if self.is_loaded:
            return

        # Skip a slow online check of model sources on every start.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import paddleocr
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrError("PaddleOCR is not installed. Run: pip install -r requirements.txt") from exc

        version = getattr(paddleocr, "__version__", "3.0.0")
        self._major_version = int(version.split(".")[0])
        logger.info("Loading PaddleOCR %s (lang=%s, device=%s).", version, self.config.ocr_lang, self.config.ocr_device)

        try:
            if self._major_version >= 3:
                self._ocr = PaddleOCR(
                    lang=self.config.ocr_lang,
                    device=self.config.ocr_device,
                    # Keep the page as it is, so the boxes match our image.
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            else:
                self._ocr = PaddleOCR(
                    lang=self.config.ocr_lang,
                    use_angle_cls=False,
                    use_gpu=self.config.ocr_device != "cpu",
                    show_log=False,
                )
        except Exception as exc:
            raise OcrError(f"Could not load PaddleOCR: {exc}") from exc
        logger.info("PaddleOCR loaded.")

    def read(self, image: Image.Image, page: int = 0) -> list[OcrLine]:
        """Find all text lines on one page image.

        Args:
            image: The page as a PIL image.
            page: Page index, stored in each box.

        Returns:
            Text lines in reading order (top to bottom), low-confidence lines removed.

        Raises:
            OcrError: If PaddleOCR fails.
        """
        self.load()
        # PaddleOCR expects a BGR numpy array (OpenCV style).
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])

        try:
            if self._major_version >= 3:
                lines = self._read_v3(bgr, page)
            else:
                lines = self._read_v2(bgr, page)
        except OcrError:
            raise
        except Exception as exc:
            raise OcrError(f"OCR failed on page {page}: {exc}") from exc

        kept = [line for line in lines if line.confidence >= self.config.ocr_min_confidence]
        logger.info("OCR page %d: %d lines (%d kept).", page, len(lines), len(kept))
        return kept

    def _read_v3(self, bgr: np.ndarray, page: int) -> list[OcrLine]:
        """Read text with the PaddleOCR 3.x API."""
        results = self._ocr.predict(bgr)
        lines: list[OcrLine] = []
        for result in results:
            data = self._result_dict(result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("rec_polys", [])
            for text, score, poly in zip(texts, scores, polys):
                text = str(text).strip()
                if text:
                    lines.append(OcrLine(text, _box_from_points(poly, page), float(score)))
        return lines

    @staticmethod
    def _result_dict(result: Any) -> dict[str, Any]:
        """Get the data dict from a PaddleOCR 3.x result object."""
        if hasattr(result, "get") and result.get("rec_texts") is not None:
            return result
        json_data: Optional[dict[str, Any]] = getattr(result, "json", None)
        if isinstance(json_data, dict):
            return json_data.get("res", json_data)
        return {}

    def _read_v2(self, bgr: np.ndarray, page: int) -> list[OcrLine]:
        """Read text with the old PaddleOCR 2.x API."""
        results = self._ocr.ocr(bgr, cls=False)
        lines: list[OcrLine] = []
        for item in (results[0] if results else None) or []:
            points, (text, score) = item
            text = str(text).strip()
            if text:
                lines.append(OcrLine(text, _box_from_points(points, page), float(score)))
        return lines


_default_engine: Optional[OcrEngine] = None


def get_ocr_engine() -> OcrEngine:
    """Return the shared OCR engine (created once per process)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = OcrEngine()
    return _default_engine
