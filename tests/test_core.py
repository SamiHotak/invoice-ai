"""Tests for small core helpers: GPU checks, file signatures, fallback messages."""

from datetime import date

import pytest

from app.fallback import fallback_warning
from app.gpu import NoGpuError, preload_nvrtc_builtins, require_gpu
from app.pipeline import has_valid_signature


def test_require_gpu_stops_on_cpu():
    with pytest.raises(NoGpuError, match="No GPU found"):
        require_gpu("cpu")


def test_require_gpu_allows_gpu_or_explicit_cpu():
    require_gpu("cuda")
    require_gpu("cpu", allow_cpu=True)


def test_preload_does_nothing_without_cuda(tmp_path):
    assert preload_nvrtc_builtins(None) == []
    assert preload_nvrtc_builtins("13.0", folders=[str(tmp_path)]) == []  # nothing to load there


@pytest.mark.parametrize(
    ("suffix", "head", "expected"),
    [
        (".pdf", b"%PDF-1.7 ...", True),
        (".PNG", b"\x89PNG\r\n\x1a\n....", True),
        (".jpg", b"\xff\xd8\xff\xe0....", True),
        (".jpeg", b"\xff\xd8\xff\xe1....", True),
        (".pdf", b"hello world", False),  # renamed text file
        (".png", b"\xff\xd8\xff\xe0....", False),  # JPG renamed to PNG
        (".txt", b"hello", False),  # unsupported type
    ],
)
def test_file_signatures(suffix, head, expected):
    assert has_valid_signature(suffix, head) is expected


def test_fallback_warning_is_readable():
    assert fallback_warning("total_amount", 28.11, 12.0) == (
        "Total: the model's value 28.11 was not found on the document, so 12.00 was taken from the OCR text."
    )
    assert fallback_warning("invoice_date", None, date(2018, 3, 5)) == (
        "Invoice date: the model found no value, so 2018-03-05 was taken from the OCR text."
    )


def test_extractor_refuses_cpu_before_loading_anything():
    from app.config import Settings
    from app.extractor import InvoiceExtractor

    with pytest.raises(NoGpuError):
        InvoiceExtractor(Settings(device="cpu")).load()


def test_unreadable_image_has_its_own_error(tmp_path):
    from app.extractor import ExtractionError, UnreadableImageError, prepare_image

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n not really an image")
    with pytest.raises(UnreadableImageError):
        prepare_image(broken, max_side=1600)
    assert issubclass(UnreadableImageError, ExtractionError)  # old code still catches it
