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
        device: "cuda" or "cpu".
        dtype: Model precision on GPU: "float16", "bfloat16" or "float32".
            On CPU the model always runs in float32.
        max_image_side: Longest image side in pixels before the image goes to the model.
        min_pixels: Minimum image size (in pixels) for the Qwen2.5-VL processor.
        max_pixels: Maximum image size (in pixels) for the Qwen2.5-VL processor.
            Lower = faster and less GPU memory, but small text may be missed.
        max_new_tokens: Maximum length of the model answer.
    """

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cpu"
    dtype: str = "float16"
    max_image_side: int = 1600
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    max_new_tokens: int = 1536

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
        )


settings = Settings.from_env()
