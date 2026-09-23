"""Application settings for InvoiceAI.

Every setting can be changed with an environment variable, for example:
    INVOICEAI_DTYPE=bfloat16
Set the variable before importing the app.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return "cuda" if an NVIDIA GPU is available, otherwise "cpu"."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is not installed. Falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Settings:
    """All tunable settings in one place.

    Attributes:
        model_name: Hugging Face model id of the vision-language model.
        device: "cuda" or "cpu" for the vision-language model.
        dtype: Model precision on GPU: "float16", "bfloat16" or "float32".
            On CPU the model always runs in float32.
        max_image_side: Longest image side in pixels. Images are resized once,
            and the same image is used for OCR, the model and the drawn boxes.
        min_pixels: Minimum image size (in pixels) for the Qwen2.5-VL processor.
        max_pixels: Maximum image size (in pixels) for the Qwen2.5-VL processor.
            Lower = faster and less GPU memory, but small text may be missed.
        max_new_tokens: Maximum length of the model answer.
        pdf_dpi: Resolution used to turn PDF pages into images.
        max_pdf_pages: Only the first N pages of a PDF are processed.
        ocr_lang: PaddleOCR language code.
        ocr_device: Device for PaddleOCR. CPU by default (see requirements.txt).
        ocr_min_confidence: OCR lines below this confidence are ignored.
        match_high_threshold: Match score (0-100) needed for "high" confidence.
        match_medium_threshold: Match score (0-100) needed for "medium" confidence.
    """

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cpu"
    dtype: str = "float16"
    max_image_side: int = 1600
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 1536
    pdf_dpi: int = 200
    max_pdf_pages: int = 3
    ocr_lang: str = "en"
    ocr_device: str = "cpu"
    ocr_min_confidence: float = 0.5
    match_high_threshold: float = 90.0
    match_medium_threshold: float = 70.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, with safe defaults."""
        return cls(
            model_name=os.getenv("INVOICEAI_MODEL", cls.model_name),
            device=os.getenv("INVOICEAI_DEVICE") or detect_device(),
            dtype=os.getenv("INVOICEAI_DTYPE", cls.dtype),
            max_image_side=int(os.getenv("INVOICEAI_MAX_IMAGE_SIDE", cls.max_image_side)),
            min_pixels=int(os.getenv("INVOICEAI_MIN_PIXELS", cls.min_pixels)),
            max_pixels=int(os.getenv("INVOICEAI_MAX_PIXELS", cls.max_pixels)),
            max_new_tokens=int(os.getenv("INVOICEAI_MAX_NEW_TOKENS", cls.max_new_tokens)),
            pdf_dpi=int(os.getenv("INVOICEAI_PDF_DPI", cls.pdf_dpi)),
            max_pdf_pages=int(os.getenv("INVOICEAI_MAX_PDF_PAGES", cls.max_pdf_pages)),
            ocr_lang=os.getenv("INVOICEAI_OCR_LANG", cls.ocr_lang),
            ocr_device=os.getenv("INVOICEAI_OCR_DEVICE", cls.ocr_device),
            ocr_min_confidence=float(
                os.getenv("INVOICEAI_OCR_MIN_CONFIDENCE", cls.ocr_min_confidence)
            ),
            match_high_threshold=float(
                os.getenv("INVOICEAI_MATCH_HIGH", cls.match_high_threshold)
            ),
            match_medium_threshold=float(
                os.getenv("INVOICEAI_MATCH_MEDIUM", cls.match_medium_threshold)
            ),
        )


settings = Settings.from_env()
