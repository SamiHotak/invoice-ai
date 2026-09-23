"""Shared test helpers. No GPU, no model: the pipeline is faked."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.extractor import ExtractionError  # noqa: E402
from app.pipeline import ProcessedDocument  # noqa: E402
from app.schema import BoundingBox, DocumentResult, FieldResult, Invoice  # noqa: E402


def make_result(file_name: str = "receipt.jpg") -> DocumentResult:
    """A realistic result: one medium field (fallback) and one low field."""
    invoice = Invoice.model_validate({
        "vendor_name": "BOOK TA .K (TAMAN DAYA) SDN BHD",
        "vendor_address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA",
        "invoice_number": "TD01167104",
        "invoice_date": "2018-12-25",
        "currency": "MYR",
        "line_items": [
            {"description": "KF MODELLING CLAY KIDDY FISH", "quantity": 1, "unit_price": 9.0, "total": 9.0},
            {"description": "GIFT BAG", "quantity": 2, "unit_price": 1.5, "total": 3.0},
        ],
        "subtotal": None,
        "tax": 0.0,
        "total_amount": 12.0,
    })
    box = BoundingBox(x0=10, y0=10, x1=100, y1=30)
    fields = {
        "vendor_name": FieldResult(value=invoice.vendor_name, confidence="high", box=box, match_score=100),
        "vendor_address": FieldResult(value=invoice.vendor_address, confidence="high", box=box, match_score=95),
        "invoice_number": FieldResult(value=invoice.invoice_number, confidence="high", box=box, match_score=100),
        "invoice_date": FieldResult(value=invoice.invoice_date, confidence="high", box=box, match_score=100),
        "currency": FieldResult(value="MYR", confidence="high", box=box, match_score=100),
        "subtotal": FieldResult(value=None, confidence=None),
        "tax": FieldResult(value=0.0, confidence="low", match_score=0),
        "total_amount": FieldResult(value=12.0, confidence="medium", box=box, source="ocr_fallback"),
    }
    line_items = [
        {"description": FieldResult(value="KF MODELLING CLAY KIDDY FISH", confidence="high", box=box),
         "quantity": FieldResult(value=1.0, confidence="high", box=box),
         "unit_price": FieldResult(value=9.0, confidence="high", box=box),
         "total": FieldResult(value=9.0, confidence="high", box=box)},
        {"description": FieldResult(value="GIFT BAG", confidence="medium", box=box),
         "quantity": FieldResult(value=2.0, confidence="low"),
         "unit_price": FieldResult(value=1.5, confidence="high", box=box),
         "total": FieldResult(value=3.0, confidence="high", box=box)},
    ]
    return DocumentResult(
        file_name=file_name, num_pages=1, invoice=invoice, fields=fields, line_items=line_items,
        ocr_line_count=25, processing_seconds=14.2,
        warnings=["Total: the model's value 28.11 was not found on the document, so 12.00 was taken from the OCR text."],
    )


class _FakeExtractor:
    prompt_version = "v3"


class FakePipeline:
    """Behaves like InvoicePipeline, without models. Files with 'broken' in the name fail."""

    def __init__(self, fail_load: bool = False) -> None:
        self.extractor = _FakeExtractor()
        self.fail_load = fail_load
        self.processed: list[str] = []

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("CUDA not available")

    def process_file(self, path: Path) -> ProcessedDocument:
        path = Path(path)
        assert path.is_file(), "the API must save the upload before processing"
        self.processed.append(path.name)
        if "broken" in path.name:
            raise ExtractionError("Model did not return valid JSON after one retry")
        return ProcessedDocument(result=make_result(path.name), pages=[])


def image_bytes(fmt: str = "PNG") -> bytes:
    """A small real image file."""
    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buffer, format=fmt)
    return buffer.getvalue()


def pdf_bytes() -> bytes:
    """A small real one-page PDF."""
    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buffer, format="PDF")
    return buffer.getvalue()


@pytest.fixture
def result() -> DocumentResult:
    return make_result()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(device="cpu", max_upload_mb=1, max_batch_files=3)


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def client(fake_pipeline: FakePipeline, test_settings: Settings):
    """API test client with the fake pipeline (startup runs, models 'load' instantly)."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app(pipeline=fake_pipeline, config=test_settings)) as test_client:
        yield test_client


def file_part(name: str, content: bytes, field: str = "file") -> tuple[str, tuple[str, bytes, str]]:
    """One multipart file entry for the test client."""
    return (field, (name, content, "application/octet-stream"))


@pytest.fixture
def png() -> bytes:
    return image_bytes("PNG")


@pytest.fixture
def jpg() -> bytes:
    return image_bytes("JPEG")


@pytest.fixture
def pdf() -> bytes:
    return pdf_bytes()


