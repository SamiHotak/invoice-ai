"""Tests for the Gradio UI logic (no GPU, no browser: the pipeline is faked)."""

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.export import FailedDocument
from app.pipeline import ProcessedDocument
from app.schema import FieldResult
from tests.conftest import make_result
from ui.gradio_app import (
    InvoiceApp,
    UploadError,
    build_app,
    check_upload,
    fields_table_html,
    fields_to_check,
    line_items_table_html,
    render_document,
    status_markdown,
    warnings_html,
    write_exports,
)

SETTINGS = Settings(device="cpu", max_upload_mb=1, max_batch_files=3)


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_check_upload_accepts_real_files(tmp_path, png, pdf):
    check_upload(_write(tmp_path, "a.png", png), SETTINGS)
    check_upload(_write(tmp_path, "b.pdf", pdf), SETTINGS)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("notes.txt", b"hello", "not supported"),
        ("empty.png", b"", "empty"),
        ("renamed.pdf", b"just text", "renamed"),
        ("big.png", b"\x89PNG\r\n\x1a\n" + b"0" * (2 * 1024 * 1024), "limit is 1 MB"),
    ],
)
def test_check_upload_rejects_bad_files(tmp_path, name, content, message):
    with pytest.raises(UploadError, match=message):
        check_upload(_write(tmp_path, name, content), SETTINGS)


def test_one_bad_file_does_not_stop_the_others(tmp_path, png, fake_pipeline):
    logic = InvoiceApp(fake_pipeline, SETTINGS)
    good = logic.process_one(_write(tmp_path, "good.png", png))
    broken = logic.process_one(_write(tmp_path, "broken.png", png))
    renamed = logic.process_one(_write(tmp_path, "renamed.png", b"not an image"))

    assert good.result.file_name == "good.png"
    assert isinstance(broken, FailedDocument) and "could not read this document" in broken.error
    assert isinstance(renamed, FailedDocument) and "renamed" in renamed.error
    assert fake_pipeline.processed == ["good.png", "broken.png"]  # renamed file never reaches the model


def test_fields_table_shows_confidence_and_fallback(result):
    table = fields_table_html(result)
    assert "OCR fallback" in table
    assert "iai-badge-low" in table and "iai-badge-medium" in table
    assert "Tax" in fields_to_check(result) and "Total" in fields_to_check(result)


def test_html_is_escaped():
    result = make_result("<script>x</script>.png")
    result.fields["vendor_name"] = FieldResult(value="<b>Shop</b>", confidence="high")
    _, summary, warnings, fields, items, _ = render_document(ProcessedDocument(result=result, pages=[]), True)
    html_parts = summary + warnings + fields + items
    assert "<script>" not in html_parts and "<b>Shop</b>" not in html_parts
    assert "&lt;b&gt;Shop&lt;/b&gt;" in html_parts


def test_line_items_and_warnings(result):
    assert "iai-cell-low" in line_items_table_html(result)
    assert "Please check" in warnings_html(result)


def test_failed_document_view():
    images, summary, *_, raw = render_document(FailedDocument("bad.pdf", "bad.pdf: broken"), False)
    assert images == [] and "bad.pdf was not processed" in summary
    assert raw["status"] == "error"


def test_exports_are_written(result):
    excel_path, json_path = write_exports([result, FailedDocument("bad.pdf", "broken")])
    assert Path(excel_path).stat().st_size > 0
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert [d["status"] for d in data] == ["ok", "error"]


def test_status_text_counts_files():
    outcomes = [ProcessedDocument(result=make_result("a.png"), pages=[]), FailedDocument("b.pdf", "x")]
    text = status_markdown(outcomes, 30.0)
    assert "1 of 2" in text and "30.0 s" in text and "could not be processed" in text


def test_app_builds_without_model(fake_pipeline, tmp_path):
    demo = build_app(fake_pipeline, SETTINGS, samples_dir=tmp_path)
    assert demo.title == "InvoiceAI"
