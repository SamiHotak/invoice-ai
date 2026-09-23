"""InvoiceAI REST API (FastAPI).

Endpoints:
    GET  /health          - is the service ready?
    POST /extract         - one file  -> JSON result
    POST /extract/batch   - many files -> list of results (one bad file does not stop the others)
    POST /export/excel    - many files -> Excel download
    GET  /docs            - interactive API documentation (automatic)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import Settings, settings as default_settings
from app.export import FailedDocument, to_dict, to_excel
from app.extractor import ExtractionError, UnreadableImageError
from app.pdf_utils import PdfError
from app.pipeline import (
    SUPPORTED_EXTENSIONS,
    InvoicePipeline,
    UnsupportedFileError,
    get_pipeline,
    has_valid_signature,
)
from app.schema import DocumentResult

logger = logging.getLogger(__name__)

EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class ApiError(HTTPException):
    """HTTP error with a short machine-readable code and a human message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail={"error": code, "message": message})


def safe_file_name(name: Optional[str]) -> str:
    """Keep only the base name and safe characters (no paths, no surprises)."""
    base = Path(name or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return cleaned or "upload"


class DocumentService:
    """Validates uploads and runs the pipeline, one document at a time.

    The GPU model is not thread-safe, so a lock makes sure only one
    document is processed at a time, even if requests arrive together.
    """

    def __init__(self, pipeline: InvoicePipeline, config: Settings) -> None:
        self.pipeline = pipeline
        self.config = config
        self.ready = False
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the models. On failure the API still starts and /health shows the error."""
        try:
            self.pipeline.load()
            self.ready = True
            logger.info("Models loaded, API is ready.")
        except Exception as exc:  # show the reason in /health instead of crashing
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Could not load the models.")

    @property
    def max_bytes(self) -> int:
        return self.config.max_upload_mb * 1024 * 1024

    async def read_upload(self, upload: UploadFile) -> tuple[str, bytes]:
        """Read and check one upload. Returns (safe file name, content).

        Raises:
            ApiError: wrong type (415), too large (413) or empty (400).
        """
        name = safe_file_name(upload.filename)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ApiError(415, "unsupported_file_type", f"'{name}': unsupported file type. Allowed: {allowed}.")

        content = await upload.read(self.max_bytes + 1)
        if not content:
            raise ApiError(400, "empty_file", f"'{name}' is empty.")
        if len(content) > self.max_bytes:
            raise ApiError(413, "file_too_large", f"'{name}' is larger than {self.config.max_upload_mb} MB.")
        if not has_valid_signature(suffix, content[:16]):
            raise ApiError(415, "file_type_mismatch", f"'{name}' does not look like a real {suffix} file.")
        return name, content

    def _process_bytes(self, name: str, content: bytes) -> DocumentResult:
        """Save to a temp folder and run the pipeline (runs in a worker thread)."""
        with self._lock, tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / name
            path.write_bytes(content)
            return self.pipeline.process_file(path).result

    async def process(self, name: str, content: bytes) -> DocumentResult:
        """Run the pipeline on one checked upload.

        Raises:
            ApiError: models not ready (503), unreadable file (422) or extraction failed (500).
        """
        if not self.ready:
            raise ApiError(503, "not_ready", self.load_error or "Models are still loading. Try again soon.")
        try:
            return await run_in_threadpool(self._process_bytes, name, content)
        except (PdfError, UnsupportedFileError, UnreadableImageError) as exc:
            raise ApiError(422, "unreadable_file", f"'{name}': {exc}") from exc
        except ExtractionError as exc:
            raise ApiError(422, "extraction_failed", f"'{name}': {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error on %s", name)
            raise ApiError(500, "internal_error", f"'{name}': unexpected error ({type(exc).__name__}).") from exc

    async def process_many(self, uploads: list[UploadFile]) -> list[DocumentResult | FailedDocument]:
        """Process several uploads. A bad file becomes a FailedDocument, the rest continue."""
        if not uploads:
            raise ApiError(400, "no_files", "Please upload at least one file.")
        if len(uploads) > self.config.max_batch_files:
            raise ApiError(413, "too_many_files", f"At most {self.config.max_batch_files} files per request.")
        if not self.ready:
            raise ApiError(503, "not_ready", self.load_error or "Models are still loading. Try again soon.")

        results: list[DocumentResult | FailedDocument] = []
        for upload in uploads:
            try:
                name, content = await self.read_upload(upload)
                results.append(await self.process(name, content))
            except ApiError as exc:
                results.append(FailedDocument(safe_file_name(upload.filename), exc.detail["message"]))
        return results


def _mark_uploads_as_binary(node: Any) -> None:
    """Add "format": "binary" to file fields in the OpenAPI schema.

    FastAPI describes uploads with "contentMediaType" only. The Swagger UI on
    /docs then shows text boxes instead of file pickers for lists of files.
    Adding "format": "binary" makes every upload field a file picker.
    """
    if isinstance(node, dict):
        if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
            node.setdefault("format", "binary")
        for value in node.values():
            _mark_uploads_as_binary(value)
    elif isinstance(node, list):
        for value in node:
            _mark_uploads_as_binary(value)


def create_app(pipeline: Optional[InvoicePipeline] = None, config: Settings = default_settings) -> FastAPI:
    """Build the FastAPI app. Tests pass a fake pipeline, so no GPU is needed."""

    service = DocumentService(pipeline or get_pipeline(), config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(service.load)  # load once at startup, not per request
        yield

    api = FastAPI(
        title="InvoiceAI API",
        version=__version__,
        description=(
            "Extract structured data from invoices and receipts (JPG, PNG, PDF): vendor, date, "
            "line items, tax and total. Every field has a confidence level (high / medium / low) "
            "and the position where it was found on the page."
        ),
        lifespan=lifespan,
    )
    api.state.service = service

    def openapi_with_file_pickers() -> dict[str, Any]:
        if api.openapi_schema is None:
            schema = get_openapi(title=api.title, version=api.version, description=api.description, routes=api.routes)
            _mark_uploads_as_binary(schema)
            api.openapi_schema = schema
        return api.openapi_schema

    api.openapi = openapi_with_file_pickers  # type: ignore[method-assign]

    @api.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @api.get("/health", tags=["system"], summary="Service status")
    def health() -> JSONResponse:
        """`ok` when the models are loaded; `loading` or `error` otherwise."""
        state = "ok" if service.ready else ("error" if service.load_error else "loading")
        body = {
            "status": state,
            "version": __version__,
            "models_loaded": service.ready,
            "error": service.load_error,
            "device": config.device,
            "prompt_version": getattr(service.pipeline.extractor, "prompt_version", None),
            "max_upload_mb": config.max_upload_mb,
        }
        return JSONResponse(body, status_code=200 if service.ready else 503)

    @api.post("/extract", tags=["extraction"], summary="Extract data from one file")
    async def extract(file: UploadFile = File(..., description="Invoice or receipt: JPG, PNG or PDF")) -> dict[str, Any]:
        """Returns the extracted invoice, a confidence level and box for every field, and warnings."""
        name, content = await service.read_upload(file)
        return to_dict(await service.process(name, content))

    @api.post("/extract/batch", tags=["extraction"], summary="Extract data from many files")
    async def extract_batch(
        files: list[UploadFile] = File(..., description=f"Up to {config.max_batch_files} files"),
    ) -> list[dict[str, Any]]:
        """One entry per file, in the same order. Failed files have `status: error` and a message."""
        return [to_dict(item) for item in await service.process_many(files)]

    @api.post("/export/excel", tags=["export"], summary="Extract many files and download Excel")
    async def export_excel(
        files: list[UploadFile] = File(..., description=f"Up to {config.max_batch_files} files"),
    ) -> Response:
        """Excel with sheets "Invoices", "Line Items" and "Legend". Uncertain values are colored."""
        results = await service.process_many(files)
        return Response(
            content=to_excel(results),
            media_type=EXCEL_MEDIA_TYPE,
            headers={"Content-Disposition": 'attachment; filename="invoices.xlsx"'},
        )

    return api


app = create_app()
