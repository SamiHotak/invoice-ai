"""InvoiceAI web interface (Gradio).

Upload invoices or receipts (JPG, PNG or PDF) and see:
    - the page image with colored boxes around the found values,
    - a table of fields with their confidence and source,
    - the warnings and the line items,
    - the raw JSON,
and download all results as Excel or JSON.

The UI calls the pipeline directly (no API in between), so it needs a GPU.

Run:
    python -m ui.gradio_app            # local:  http://localhost:7860
    python -m ui.gradio_app --share    # public link (e.g. in Google Colab)

In a notebook:
    from app.pipeline import get_pipeline
    from ui.gradio_app import launch_app
    pipeline = get_pipeline()
    pipeline.load()
    demo = launch_app(pipeline, share=True)
"""

from __future__ import annotations

import argparse
import html
import logging
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import gradio as gr
from PIL import Image, UnidentifiedImageError

from app import __version__
from app.config import Settings, settings as default_settings
from app.export import FailedDocument, to_dict, to_excel, to_json
from app.extractor import ExtractionError
from app.pdf_utils import PdfError
from app.pipeline import (
    SUPPORTED_EXTENSIONS,
    InvoicePipeline,
    ProcessedDocument,
    UnsupportedFileError,
    get_pipeline,
)
from app.schema import DocumentResult, FieldResult

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
MAX_EXAMPLES = 4

Outcome = Union[ProcessedDocument, FailedDocument]

# Display names, in the order they are shown.
FIELD_LABELS: dict[str, str] = {
    "vendor_name": "Vendor",
    "vendor_address": "Vendor address",
    "invoice_number": "Invoice number",
    "invoice_date": "Date",
    "currency": "Currency",
    "subtotal": "Subtotal",
    "tax": "Tax",
    "total_amount": "Total",
}
LINE_ITEM_LABELS: dict[str, str] = {
    "description": "Description",
    "quantity": "Qty",
    "unit_price": "Unit price",
    "total": "Total",
}
MONEY_FIELDS = frozenset({"subtotal", "tax", "total_amount", "unit_price", "total"})
NUMBER_FIELDS = MONEY_FIELDS | {"quantity"}

CONFIDENCE_TEXT: dict[str, str] = {
    "high": "High",
    "medium": "Check",
    "low": "Not found",
}

# First bytes of each supported file type, so a renamed file is caught early.
_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}

# Numbers from eval/results.md (SROIE test split, 100 receipts, prompt v3 + OCR fallback).
EVAL_FACTS: list[tuple[str, str]] = [
    ("Total amount correct", "95%"),
    ("Date correct", "97%"),
    ("Correct when marked high", "98.9%"),
    ("Time per receipt (T4 GPU)", "~22 s"),
]


# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#0f5c5c",
    button_primary_background_fill_hover="#0c4a4a",
    button_primary_text_color="#ffffff",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.iai-num { font-variant-numeric: tabular-nums; }

/* Header: title on the left, a small "receipt" with the test results on the right. */
.iai-hero { display: flex; gap: 32px; align-items: flex-start; justify-content: space-between;
            flex-wrap: wrap; padding: 8px 4px 4px; }
.iai-hero h1 { font-size: 2.4rem; line-height: 1.1; margin: 0 0 8px; font-weight: 700;
               letter-spacing: -0.02em; color: var(--body-text-color); }
.iai-hero p { margin: 0; max-width: 60ch; line-height: 1.5; color: var(--body-text-color-subdued); }
.iai-receipt { min-width: 280px; padding: 14px 18px 18px; font-size: 0.9rem;
               background: var(--background-fill-secondary); color: var(--body-text-color);
               border: 1px solid var(--border-color-primary); border-bottom: none;
               -webkit-mask: radial-gradient(circle 6px at 50% 100%, transparent 5.5px, #000 6px)
                             0 0 / 16px 100% repeat-x;
                       mask: radial-gradient(circle 6px at 50% 100%, transparent 5.5px, #000 6px)
                             0 0 / 16px 100% repeat-x; }
.iai-receipt .iai-receipt-title { font-weight: 600; margin-bottom: 6px; }
.iai-receipt .iai-row { display: flex; align-items: baseline; gap: 6px; padding: 2px 0; }
.iai-receipt .iai-dots { flex: 1; border-bottom: 1px dotted var(--body-text-color-subdued);
                         transform: translateY(-4px); }
.iai-receipt .iai-note { margin-top: 8px; font-size: 0.78rem; color: var(--body-text-color-subdued); }

/* Tables */
.iai-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; border: none !important; }
.iai-table th, .iai-table td { border: none !important; background: transparent; }
.iai-table th { text-align: left; font-weight: 600; padding: 8px 10px;
                border-bottom: 2px solid var(--border-color-primary) !important; }
.iai-table td { padding: 8px 10px; vertical-align: top;
                border-bottom: 1px solid var(--border-color-primary) !important; }
.iai-table td.iai-right, .iai-table th.iai-right { text-align: right; }
.iai-table td.iai-label { font-weight: 600; white-space: nowrap; }
.iai-empty { color: var(--body-text-color-subdued); }
.iai-table td.iai-cell-medium { background: rgba(234, 179, 8, 0.20); }
.iai-table td.iai-cell-low { background: rgba(220, 38, 38, 0.16); }

/* Badges: same colors as the boxes on the image */
.iai-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.8rem;
             font-weight: 600; white-space: nowrap; }
.iai-badge-high { background: #16a34a; color: #ffffff; }
.iai-badge-medium { background: #eab308; color: #1c1917; }
.iai-badge-low { background: #dc2626; color: #ffffff; }
.iai-source { display: inline-block; margin-left: 6px; font-size: 0.78rem; padding: 1px 6px;
              border: 1px solid var(--border-color-primary); border-radius: 4px;
              color: var(--body-text-color-subdued); }

/* Document summary, warnings, errors */
.iai-summary { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.iai-summary strong { font-size: 1.05rem; word-break: break-all; }
.iai-meta { color: var(--body-text-color-subdued); }
.iai-box { padding: 10px 14px; border-radius: 6px; border-left: 4px solid; margin: 0; }
.iai-box ul { margin: 4px 0 0 18px; padding: 0; }
.iai-box-warn { border-color: #eab308; background: rgba(234, 179, 8, 0.12); }
.iai-box-ok { border-color: #16a34a; background: rgba(22, 163, 74, 0.10); }
.iai-box-error { border-color: #dc2626; background: rgba(220, 38, 38, 0.10); }
.iai-legend { font-size: 0.85rem; line-height: 1.9; color: var(--body-text-color-subdued); }
.iai-legend .iai-badge { margin-right: 6px; }
.iai-placeholder { padding: 32px 16px; text-align: center; color: var(--body-text-color-subdued);
                   border: 1px dashed var(--border-color-primary); border-radius: 6px; }
"""


def _hero_html() -> str:
    """Header: title, one-line description and the test results as a small receipt."""
    rows = "".join(
        f'<div class="iai-row"><span>{html.escape(label)}</span><span class="iai-dots"></span>'
        f'<span class="iai-num">{html.escape(value)}</span></div>'
        for label, value in EVAL_FACTS
    )
    return f"""
<div class="iai-hero">
  <div>
    <h1>InvoiceAI</h1>
    <p>Turn invoice and receipt images or PDFs into clean data. Every value shows where it was
    found on the page and how sure the system is, so you only check what needs checking.</p>
  </div>
  <div class="iai-receipt" aria-label="Test results">
    <div class="iai-receipt-title">Tested on 100 real receipts</div>
    {rows}
    <div class="iai-note">SROIE test set, English receipts, no fine-tuning. v{html.escape(__version__)}</div>
  </div>
</div>"""


LEGEND_HTML = """
<div class="iai-legend">
  <div><span class="iai-badge iai-badge-high">High</span>Found on the document.</div>
  <div><span class="iai-badge iai-badge-medium">Check</span>Weak match, or taken from the OCR text. Please check.</div>
  <div><span class="iai-badge iai-badge-low">Not found</span>Not found on the document. Please check.</div>
</div>"""

PLACEHOLDER_HTML = (
    '<div class="iai-placeholder">Upload a file and click <b>Extract data</b>, '
    "or try one of the examples.</div>"
)


# --------------------------------------------------------------------------- #
# Upload checks and error messages
# --------------------------------------------------------------------------- #

class UploadError(ValueError):
    """An uploaded file cannot be processed. The message is shown to the user."""


def check_upload(path: Path, config: Settings) -> None:
    """Check one uploaded file before it goes to the model.

    Raises:
        UploadError: With a message the user can understand.
    """
    name = path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS))
        raise UploadError(f"{name}: this file type is not supported. Please upload {allowed}.")
    if not path.is_file():
        raise UploadError(f"{name}: the file could not be found. Please upload it again.")

    size = path.stat().st_size
    if size == 0:
        raise UploadError(f"{name}: the file is empty.")
    if size > config.max_upload_mb * 1024 * 1024:
        raise UploadError(
            f"{name}: the file is {size / 1024 / 1024:.1f} MB. The limit is {config.max_upload_mb} MB."
        )

    with path.open("rb") as handle:
        head = handle.read(16)
    if not head.startswith(_MAGIC_BYTES[suffix]):
        raise UploadError(
            f"{name}: the content is not a real {suffix.lstrip('.').upper()} file "
            "(maybe it was renamed). Please upload the original file."
        )


def friendly_error(file_name: str, exc: Exception) -> str:
    """Turn an exception into a short message for the user (details go to the log)."""
    if isinstance(exc, UploadError):
        return str(exc)
    if isinstance(exc, UnsupportedFileError):
        return f"{file_name}: this file type is not supported. Please upload JPG, PNG or PDF."
    if isinstance(exc, PdfError):
        return f"{file_name}: the PDF could not be read. It may be broken or password-protected."
    if isinstance(exc, (UnidentifiedImageError, Image.DecompressionBombError)):
        return f"{file_name}: the image could not be read. Please upload a normal JPG or PNG."
    if isinstance(exc, ExtractionError):
        return f"{file_name}: the AI model could not read this document. Try a sharper, straighter image."
    if type(exc).__name__ == "OutOfMemoryError":  # torch.cuda.OutOfMemoryError
        return f"{file_name}: the GPU ran out of memory. Try a smaller image or a PDF with fewer pages."
    return f"{file_name}: something went wrong while processing this file."


# --------------------------------------------------------------------------- #
# HTML building
# --------------------------------------------------------------------------- #

def format_value(name: str, value: Any) -> str:
    """Show a value the way a person expects (money with 2 decimals, ISO dates)."""
    if value is None or value == "":
        return '<span class="iai-empty">—</span>'
    if name in NUMBER_FIELDS and isinstance(value, (int, float)):
        if name == "quantity" and float(value).is_integer():
            return f'<span class="iai-num">{int(value)}</span>'
        return f'<span class="iai-num">{value:,.2f}</span>'
    if isinstance(value, date):
        return html.escape(value.isoformat())
    return html.escape(str(value))


def confidence_badge(confidence: Optional[str]) -> str:
    """Colored badge for high / medium / low."""
    if confidence not in CONFIDENCE_TEXT:
        return '<span class="iai-empty">—</span>'
    return f'<span class="iai-badge iai-badge-{confidence}">{CONFIDENCE_TEXT[confidence]}</span>'


def fields_to_check(result: DocumentResult) -> list[str]:
    """Names of the top-level fields with medium or low confidence."""
    return [
        FIELD_LABELS.get(name, name)
        for name, field in result.fields.items()
        if field.confidence in ("medium", "low")
    ]


def fields_table_html(result: DocumentResult) -> str:
    """Field / value / confidence table. Marks values that came from the OCR fallback."""
    rows = []
    names = list(FIELD_LABELS) + [n for n in result.fields if n not in FIELD_LABELS]
    for name in names:
        field = result.fields.get(name, FieldResult(value=None))
        source = ""
        if field.source == "ocr_fallback":
            source = (
                '<span class="iai-source" title="The model value was not found on the page, '
                'so this value was read from the OCR text.">OCR fallback</span>'
            )
        rows.append(
            f'<tr><td class="iai-label">{html.escape(FIELD_LABELS.get(name, name))}</td>'
            f"<td>{format_value(name, field.value)}</td>"
            f"<td>{confidence_badge(field.confidence)}{source}</td></tr>"
        )
    return (
        '<table class="iai-table"><thead><tr><th>Field</th><th>Value</th><th>Confidence</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def line_items_table_html(result: DocumentResult) -> str:
    """Line items table. Uncertain cells are colored like in the Excel file."""
    if not result.line_items:
        return '<div class="iai-placeholder">No line items found on this document.</div>'
    header = "".join(
        f'<th class="{"iai-right" if name in NUMBER_FIELDS else ""}">{label}</th>'
        for name, label in LINE_ITEM_LABELS.items()
    )
    rows = []
    for index, item in enumerate(result.line_items, start=1):
        cells = [f'<td class="iai-num">{index}</td>']
        for name in LINE_ITEM_LABELS:
            field = item.get(name, FieldResult(value=None))
            classes = []
            if name in NUMBER_FIELDS:
                classes.append("iai-right")
            if field.confidence in ("medium", "low") and field.value is not None:
                classes.append(f"iai-cell-{field.confidence}")
            title = f' title="Confidence: {CONFIDENCE_TEXT[field.confidence]}"' if field.confidence else ""
            cells.append(f'<td class="{" ".join(classes)}"{title}>{format_value(name, field.value)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="iai-table"><thead><tr><th>#</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def warnings_html(result: DocumentResult) -> str:
    """Warnings from the pipeline, or a short "all good" note."""
    if not result.warnings:
        return '<div class="iai-box iai-box-ok">No warnings for this document.</div>'
    items = "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
    return f'<div class="iai-box iai-box-warn"><b>Please check</b><ul>{items}</ul></div>'


def summary_html(result: DocumentResult) -> str:
    """One line about the document: name, pages, time and review status."""
    to_check = fields_to_check(result)
    if to_check:
        badge = f'<span class="iai-badge iai-badge-medium">Needs review: {len(to_check)} field(s)</span>'
    else:
        badge = '<span class="iai-badge iai-badge-high">Ready</span>'
    pages = f"{result.num_pages} page" + ("s" if result.num_pages != 1 else "")
    return (
        f'<div class="iai-summary"><strong>{html.escape(result.file_name)}</strong>{badge}'
        f'<span class="iai-meta iai-num">{pages}, processed in {result.processing_seconds:.1f} s</span></div>'
    )


def error_html(failed: FailedDocument) -> str:
    """Error card for a file that could not be processed."""
    return (
        f'<div class="iai-box iai-box-error"><b>{html.escape(failed.file_name)} was not processed.</b>'
        f"<br>{html.escape(failed.error)}</div>"
    )


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #

def _to_paths(files: Optional[Sequence[Any]]) -> list[Path]:
    """Gradio gives file paths (str) or objects with a .name; accept both."""
    paths = []
    for item in files or []:
        name = item if isinstance(item, (str, Path)) else getattr(item, "name", None)
        if name:
            paths.append(Path(name))
    return paths


def _result_of(outcome: Outcome) -> Union[DocumentResult, FailedDocument]:
    """What the export functions expect."""
    return outcome.result if isinstance(outcome, ProcessedDocument) else outcome


def write_exports(outcomes: Sequence[Outcome]) -> tuple[str, str]:
    """Save Excel and JSON files for download. Returns (excel_path, json_path)."""
    results = [_result_of(o) for o in outcomes]
    folder = Path(tempfile.mkdtemp(prefix="invoiceai_"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    excel_path = folder / f"invoiceai_{stamp}.xlsx"
    json_path = folder / f"invoiceai_{stamp}.json"
    excel_path.write_bytes(to_excel(results))
    json_path.write_text(to_json(results), encoding="utf-8")
    return str(excel_path), str(json_path)


class InvoiceApp:
    """The logic behind the UI. Kept apart from the layout so it is easy to test."""

    def __init__(self, pipeline: InvoicePipeline, config: Settings = default_settings) -> None:
        self.pipeline = pipeline
        self.config = config

    def process_one(self, path: Path) -> Outcome:
        """Check and process one file. Never raises: errors become a FailedDocument."""
        try:
            check_upload(path, self.config)
            return self.pipeline.process_file(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the others
            if isinstance(exc, UploadError):
                logger.warning("Rejected upload: %s", exc)
            else:
                logger.exception("Failed to process %s", path.name)
            return FailedDocument(path.name, friendly_error(path.name, exc))

    def process_many(self, paths: Sequence[Path], progress: gr.Progress) -> list[Outcome]:
        """Process files one after another, with a progress bar."""
        outcomes: list[Outcome] = []
        for index, path in enumerate(paths):
            progress(index / len(paths), desc=f"Reading {path.name} ({index + 1} of {len(paths)})")
            outcome = self.process_one(path)
            if isinstance(outcome, FailedDocument):
                gr.Warning(outcome.error)
            outcomes.append(outcome)
        progress(1.0, desc="Done")
        return outcomes


def render_document(outcome: Optional[Outcome], show_line_items: bool) -> tuple:
    """Everything shown for one document: images, summary, warnings, tables, JSON."""
    if outcome is None:
        return [], PLACEHOLDER_HTML, "", "", "", None
    if isinstance(outcome, FailedDocument):
        return [], error_html(outcome), "", "", "", to_dict(outcome)

    result = outcome.result
    images = []
    for page in range(len(outcome.pages)):
        try:
            images.append((outcome.draw(page, show_line_items=show_line_items), f"Page {page + 1}"))
        except Exception:  # noqa: BLE001 - a drawing problem must not hide the data
            logger.exception("Could not draw page %d of %s", page, result.file_name)
            images.append((outcome.pages[page], f"Page {page + 1} (boxes could not be drawn)"))
    return (
        images,
        summary_html(result),
        warnings_html(result),
        fields_table_html(result),
        line_items_table_html(result),
        to_dict(result),
    )


def status_markdown(outcomes: Sequence[Outcome], seconds: float) -> str:
    """Short summary of the whole run, including the processing time."""
    done = [o for o in outcomes if isinstance(o, ProcessedDocument)]
    failed = len(outcomes) - len(done)
    review = sum(1 for o in done if fields_to_check(o.result))
    parts = [f"**{len(done)} of {len(outcomes)}** file(s) processed in **{seconds:.1f} s**"]
    if len(done) > 1:
        parts.append(f"about {seconds / len(outcomes):.1f} s per file")
    text = ", ".join(parts) + "."
    if review:
        text += f"\n\n{review} document(s) need review (see the yellow and red values)."
    if failed:
        text += f"\n\n{failed} file(s) could not be processed. Details are in the document list."
    return text


def _choices(outcomes: Sequence[Outcome]) -> list[tuple[str, int]]:
    """Labels for the document picker."""
    labels = []
    for index, outcome in enumerate(outcomes):
        if isinstance(outcome, FailedDocument):
            label = f"{index + 1}. {outcome.file_name} (failed)"
        elif fields_to_check(outcome.result):
            label = f"{index + 1}. {outcome.result.file_name} (needs review)"
        else:
            label = f"{index + 1}. {outcome.result.file_name}"
        labels.append((label, index))
    return labels


def find_examples(samples_dir: Path = SAMPLES_DIR, limit: int = MAX_EXAMPLES) -> list[Path]:
    """Sample files for the "Try an example" buttons."""
    if not samples_dir.is_dir():
        return []
    files = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return files[:limit]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def build_app(
    pipeline: Optional[InvoicePipeline] = None,
    config: Settings = default_settings,
    samples_dir: Path = SAMPLES_DIR,
) -> gr.Blocks:
    """Create the Gradio app. The pipeline is created on first use if not given."""
    logic = InvoiceApp(pipeline or get_pipeline(), config)
    examples = find_examples(samples_dir)

    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="InvoiceAI", analytics_enabled=False) as demo:
        outcomes_state = gr.State([])

        gr.HTML(_hero_html())

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                files_input = gr.File(
                    label="Invoices or receipts (JPG, PNG, PDF)",
                    file_count="multiple",
                    file_types=sorted(SUPPORTED_EXTENSIONS),
                    height=180,
                )
                with gr.Row():
                    extract_btn = gr.Button("Extract data", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                example_buttons: list[tuple[gr.Button, Path]] = []
                if examples:
                    gr.Markdown("Or try an example:")
                    with gr.Row():
                        for path in examples:
                            example_buttons.append((gr.Button(path.name, size="sm"), path))

                status = gr.Markdown("")
                with gr.Row():
                    excel_btn = gr.DownloadButton("Download Excel", interactive=False)
                    json_btn = gr.DownloadButton("Download JSON", interactive=False)
                gr.HTML(LEGEND_HTML)

            with gr.Column(scale=7, min_width=360):
                doc_picker = gr.Dropdown(label="Document", choices=[], visible=False, interactive=True)
                summary = gr.HTML(PLACEHOLDER_HTML)
                gallery = gr.Gallery(
                    label="Found values on the page",
                    columns=1,
                    height=640,
                    object_fit="contain",
                    preview=False,
                )
                show_items = gr.Checkbox(label="Show boxes for line items", value=False)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.Markdown("### Extracted fields")
                warnings_box = gr.HTML("")
                fields_html = gr.HTML("")
            with gr.Column(scale=6):
                gr.Markdown("### Line items")
                items_html = gr.HTML("")

        with gr.Accordion("Raw JSON", open=False):
            raw_json = gr.JSON(label="Result of the selected document")

        view_outputs = [gallery, summary, warnings_box, fields_html, items_html, raw_json]
        run_outputs = [outcomes_state, doc_picker, *view_outputs, status, excel_btn, json_btn]

        def run(paths: list[Path], show: bool, progress: gr.Progress) -> tuple:
            if not paths:
                raise gr.Error("Please upload at least one JPG, PNG or PDF file.")
            if len(paths) > config.max_batch_files:
                raise gr.Error(f"Please upload at most {config.max_batch_files} files at once.")
            start = time.perf_counter()
            outcomes = logic.process_many(paths, progress)
            seconds = time.perf_counter() - start
            excel_path, json_path = write_exports(outcomes)
            return (
                outcomes,
                gr.Dropdown(choices=_choices(outcomes), value=0, visible=len(outcomes) > 1),
                *render_document(outcomes[0], show),
                status_markdown(outcomes, seconds),
                gr.DownloadButton(value=excel_path, interactive=True),
                gr.DownloadButton(value=json_path, interactive=True),
            )

        def run_uploads(files: Optional[list[Any]], show: bool, progress=gr.Progress()) -> tuple:
            return run(_to_paths(files), show, progress)

        def show_document(outcomes: list[Outcome], index: Optional[int], show: bool) -> tuple:
            if not outcomes:
                return render_document(None, show)
            index = index if isinstance(index, int) and 0 <= index < len(outcomes) else 0
            return render_document(outcomes[index], show)

        def clear() -> tuple:
            return (
                None,
                [],
                gr.Dropdown(choices=[], value=None, visible=False),
                *render_document(None, False),
                "",
                gr.DownloadButton(value=None, interactive=False),
                gr.DownloadButton(value=None, interactive=False),
            )

        extract_btn.click(run_uploads, [files_input, show_items], run_outputs)
        for button, path in example_buttons:

            def run_example(show: bool, progress=gr.Progress(), _path: Path = path) -> tuple:
                return run([_path], show, progress)

            button.click(run_example, [show_items], run_outputs)

        doc_picker.change(show_document, [outcomes_state, doc_picker, show_items], view_outputs)
        show_items.change(show_document, [outcomes_state, doc_picker, show_items], view_outputs)
        clear_btn.click(clear, None, [files_input, *run_outputs])

    return demo


def launch_app(
    pipeline: Optional[InvoicePipeline] = None,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    **launch_kwargs: Any,
) -> gr.Blocks:
    """Build and start the app. One document uses the GPU at a time (queue limit 1)."""
    demo = build_app(pipeline)
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        allowed_paths=[str(SAMPLES_DIR)],
        show_api=False,
        **launch_kwargs,
    )
    return demo


def main() -> None:
    """Command line entry point: load the model, then start the UI."""
    parser = argparse.ArgumentParser(description="InvoiceAI web interface")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on")
    parser.add_argument("--port", type=int, default=7860, help="port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipeline = get_pipeline()
    logger.info("Loading models (the first time this downloads about 7 GB)...")
    pipeline.load()
    launch_app(pipeline, share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
