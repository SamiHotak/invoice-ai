"""GPU helpers.

1. require_gpu(): stop with a clear message when there is no GPU.
   Without a GPU the 3B model loads on the CPU in float32 (about 14 GB of RAM).
   On Colab this kills the process without any error message.

2. preload_nvrtc_builtins(): work around a CUDA 13 library-path problem.
   PyTorch compiles some small GPU kernels at runtime with NVRTC. NVRTC then
   opens "libnvrtc-builtins.so.13.x" by name, but Linux does not search the
   folder where pip installs it (site-packages/nvidia/cu13/lib). The result is
   "nvrtc: error: failed to open libnvrtc-builtins.so.13.0" in the middle of
   an extraction (seen on Google Colab with torch 2.11 + CUDA 13).
   Loading the library once by its full path fixes it for the whole process.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import sys
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class NoGpuError(RuntimeError):
    """Raised when the model would have to run on the CPU."""


def require_gpu(device: str, allow_cpu: bool = False) -> None:
    """Raise NoGpuError if the model is about to be loaded on the CPU.

    Args:
        device: The device from the settings ("cuda" or "cpu").
        allow_cpu: Set INVOICEAI_ALLOW_CPU=1 to run on the CPU anyway
            (very slow, needs about 16 GB of RAM).
    """
    if device == "cpu" and not allow_cpu:
        raise NoGpuError(
            "No GPU found. InvoiceAI needs an NVIDIA GPU (for example a T4). "
            "In Google Colab: Runtime -> Change runtime type -> T4 GPU. "
            "In Docker: start the container with GPU access (--gpus all). "
            "To run on the CPU anyway (very slow, about 16 GB RAM), set INVOICEAI_ALLOW_CPU=1."
        )


def _library_folders(extra: Optional[Iterable[str]] = None) -> list[str]:
    """Folders where pip installs packages (site-packages), without duplicates."""
    folders = list(extra or []) + list(sys.path)
    seen: set[str] = set()
    result = []
    for folder in folders:
        if folder and folder not in seen and os.path.isdir(folder):
            seen.add(folder)
            result.append(folder)
    return result


def preload_nvrtc_builtins(cuda_version: Optional[str], folders: Optional[Iterable[str]] = None) -> list[str]:
    """Load libnvrtc-builtins by full path, so NVRTC can find it later.

    Args:
        cuda_version: torch.version.cuda, e.g. "13.0". None (CPU build) does nothing.
        folders: Where to search. Default: all folders on sys.path.

    Returns:
        The library files that were loaded (empty if none were needed or found).
    """
    if not cuda_version or not sys.platform.startswith("linux"):
        return []
    major = cuda_version.split(".")[0]

    loaded: list[str] = []
    for folder in _library_folders(folders):
        # e.g. nvidia/cu13/lib/libnvrtc-builtins.so.13.0 or nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.9
        for path in sorted(glob.glob(os.path.join(folder, "nvidia", "*", "lib", f"libnvrtc-builtins.so.{major}*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.debug("Could not preload %s: %s", path, exc)
                continue
            loaded.append(path)
    if loaded:
        logger.info("Preloaded NVRTC builtins: %s", ", ".join(os.path.basename(p) for p in loaded))
    return loaded
