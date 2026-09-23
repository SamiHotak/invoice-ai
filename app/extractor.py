"""Invoice data extraction with the Qwen2.5-VL vision-language model.

Usage:
    from app.extractor import extract
    invoice = extract("samples/receipt_01.jpg")       # one image
    invoice = extract([page_1, page_2])                # pages of one document
    print(invoice.model_dump_json(indent=2))

The model is loaded only once (on the first call) and then reused.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from PIL import Image, ImageOps
from pydantic import ValidationError

from app.config import Settings, settings as default_settings
from app.schema import Invoice

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, str, Path]
ImagesInput = Union[ImageInput, Sequence[ImageInput]]

EXTRACTION_PROMPT = """You are an expert at reading invoices and receipts in German and English.
Look at the document image and extract the data below.

Return ONLY one JSON object. No explanation, no markdown, no code fences.
Use exactly this structure:

{
  "vendor_name": string or null,
  "vendor_address": string or null,
  "invoice_number": string or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "currency": string or null,
  "line_items": [
    {"description": string or null, "quantity": number or null, "unit_price": number or null, "total": number or null}
  ],
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null
}

Rules:
- Copy text exactly as printed (vendor name, address, invoice number, item names). Do not translate.
- invoice_date: format YYYY-MM-DD. German dates are day.month.year (15.03.2024 -> 2024-03-15).
- Numbers: plain JSON numbers with a dot as decimal separator. No currency symbols, no thousands separators. "1.234,56" -> 1234.56.
- currency: 3-letter ISO code (EUR, USD, GBP, CHF, MYR). "€" -> "EUR".
- tax: total tax amount (VAT, MwSt, USt, GST). If there are several tax rates, add them together.
- subtotal: amount before tax (Netto). total_amount: final amount to pay (Gesamt, Summe, Total, Brutto).
- line_items: one entry per product or service. Do not include tax lines, totals or payment lines.
- If a value is not on the document, use null. Never guess or invent values."""

# v2: English-only, with clearer rules for the fields the model got wrong in
# early tests (a person's name used as vendor, company name inside the address,
# cash paid used as total). Compare v1 vs v2 with eval/evaluate.py.
EXTRACTION_PROMPT_V2 = """You are an expert at reading English invoices and receipts.
Look at the document image and extract the data below.

Return ONLY one JSON object. No explanation, no markdown, no code fences.
Use exactly this structure:

{
  "vendor_name": string or null,
  "vendor_address": string or null,
  "invoice_number": string or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "currency": string or null,
  "line_items": [
    {"description": string or null, "quantity": number or null, "unit_price": number or null, "total": number or null}
  ],
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null
}

Rules:
- vendor_name: the business that issued the document (the seller). It is usually printed in large letters near the top and often ends with words like SDN BHD, BHD, LTD, INC, LLC, ENTERPRISE, TRADING, RESTAURANT or STORE. A person's name on its own (for example a customer or cashier) is NOT the vendor. If the name spans two lines, join them. Copy it exactly as printed.
- vendor_address: only the seller's postal address (building, street, area, postcode, city, state), joined with ", ". Do NOT include the company name, registration numbers, phone, fax, email or GST/tax IDs.
- invoice_number: the receipt, invoice, bill or document number (for example "Invoice No", "Receipt #", "Doc No", "Bill No").
- invoice_date: the transaction date, format YYYY-MM-DD. Most dates are day/month/year (25/12/2018 -> 2018-12-25). If the second number is above 12 (12/28/2017), the date is month/day/year.
- total_amount: the final amount the customer must pay, including tax and after any rounding adjustment. Look for TOTAL, NETT TOTAL, GRAND TOTAL, TOTAL AMOUNT, AMOUNT DUE or TOTAL (INCL GST). Do NOT use the cash paid, the change, or the subtotal.
- subtotal: amount before tax, only if printed. tax: total tax amount (GST, SST, VAT), only if printed.
- Numbers: plain JSON numbers with a dot as decimal separator, no currency symbols, no thousands separators (1,234.50 -> 1234.5).
- currency: 3-letter ISO code (MYR, USD, EUR, GBP, SGD). "RM" -> "MYR".
- line_items: one entry per product or service. Do not include tax lines, totals, rounding or payment lines.
- If a value is not on the document, use null. Never guess or invent values."""

# v3: v2 plus three rules from the round-1 error analysis on SROIE train:
# a person's name printed ABOVE the shop name, invented years for 2-digit
# years, and totals that were calculated instead of copied.
EXTRACTION_PROMPT_V3 = EXTRACTION_PROMPT_V2.replace(
    "A person's name on its own (for example a customer or cashier) is NOT the vendor.",
    "A person's name on its own (for example a customer or cashier) is NOT the vendor. "
    "Many receipts print a person's name on the first line, ABOVE the shop name: skip it "
    "and use the business name below it.",
).replace(
    "If the second number is above 12 (12/28/2017), the date is month/day/year.",
    "If the second number is above 12 (12/28/2017), the date is month/day/year. "
    "If the year has only 2 digits, it means 20xx (12-01-19 -> 2019-01-12). "
    "Never change or guess the year: use the digits that are printed.",
).replace(
    "Do NOT use the cash paid, the change, or the subtotal.",
    "Do NOT use the cash paid, the change, or the subtotal. "
    "Copy the number exactly as printed next to the total label. "
    "Never add up, calculate or round numbers yourself.",
)

PROMPTS: dict[str, str] = {"v1": EXTRACTION_PROMPT, "v2": EXTRACTION_PROMPT_V2, "v3": EXTRACTION_PROMPT_V3}


FIX_JSON_PROMPT = """The text below should be one valid JSON object, but it has an error: {error}

Fix it. Keep the same keys and values. Return ONLY the corrected JSON object.
No explanation, no markdown, no code fences.

Text:
{raw}"""


class ExtractionError(RuntimeError):
    """Raised when the model output cannot be turned into a valid Invoice."""


# ---------------------------------------------------------------------------
# Helpers (no model needed, easy to unit test)
# ---------------------------------------------------------------------------


def prepare_image(image: ImageInput, max_side: int) -> Image.Image:
    """Load an image, fix phone rotation, convert to RGB and limit its size.

    Args:
        image: A PIL image or a path to an image file.
        max_side: The longest side is scaled down to this many pixels.

    Raises:
        ExtractionError: If the file cannot be opened as an image.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise ExtractionError(f"File not found: {path}")
        try:
            image = Image.open(path)
            image.load()
        except (OSError, Image.UnidentifiedImageError) as exc:
            raise ExtractionError(f"Could not read image: {path}") from exc

    image = ImageOps.exif_transpose(image)  # photos from phones are often rotated
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _load_json(raw: str) -> Any:
    """Parse JSON from model output. Strips code fences and fixes small issues."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ExtractionError("No JSON object found in the model output.")
    text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Small repairs, only used when normal parsing failed.
    repaired = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    repaired = re.sub(r"\bNone\b", "null", repaired)  # Python-style null
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Invalid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from exc


def parse_invoice_json(raw: str) -> Invoice:
    """Turn raw model output into a validated Invoice.

    Raises:
        ExtractionError: If the output is not valid JSON or does not match the schema.
    """
    data = _load_json(raw)
    if not isinstance(data, dict):
        raise ExtractionError("Expected a JSON object, got a different JSON type.")
    try:
        return Invoice.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:3]
        )
        raise ExtractionError(f"JSON does not match the schema: {details}") from exc


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class InvoiceExtractor:
    """Wraps Qwen2.5-VL: loads the model once and extracts invoices from images."""

    def __init__(self, config: Settings = default_settings) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self.last_raw_output: Optional[str] = None  # useful for debugging
        self.prompt_version = config.prompt_version
        self.set_prompt_version(config.prompt_version)  # validates the name

    def set_prompt_version(self, version: str) -> None:
        """Switch the extraction prompt without reloading the model."""
        if version not in PROMPTS:
            raise ValueError(f"Unknown prompt version {version!r}. Use one of: {list(PROMPTS)}")
        self.prompt_version = version

    @property
    def is_loaded(self) -> bool:
        """True if the model is already in memory."""
        return self._model is not None

    def load(self) -> None:
        """Load the model and processor. Safe to call many times."""
        if self.is_loaded:
            return

        # Heavy imports happen here, so the rest of the app (and tests) work without a GPU.
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = self._resolve_dtype(torch)
        logger.info(
            "Loading %s on %s (%s). The first time this downloads about 7 GB.",
            self.config.model_name,
            self.config.device,
            dtype,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )
        logger.info("Model loaded.")

    def _resolve_dtype(self, torch: Any) -> Any:
        """Pick the torch dtype. CPU always uses float32."""
        if self.config.device == "cpu":
            return torch.float32
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.config.dtype not in dtypes:
            raise ValueError(f"Unknown dtype {self.config.dtype!r}. Use one of: {list(dtypes)}")
        return dtypes[self.config.dtype]

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        """Run the model on a chat message list and return the answer text."""
        import torch
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # deterministic output
            )

        new_tokens = output_ids[:, inputs.input_ids.shape[1] :]
        answer = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return answer.strip()

    def extract(self, images: ImagesInput) -> Invoice:
        """Extract structured invoice data from one document.

        Args:
            images: One image, or a list of page images of the SAME document
                (e.g. the pages of a PDF). Each can be a PIL image or a file path.

        Returns:
            A validated Invoice.

        Raises:
            ExtractionError: If an image cannot be read, or the model does not
                return valid JSON even after one retry.
        """
        self.load()
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        pil_images = [prepare_image(image, self.config.max_image_side) for image in images]
        if not pil_images:
            raise ExtractionError("No images given.")

        base_prompt = PROMPTS[self.prompt_version]
        prompt = base_prompt
        if len(pil_images) > 1:
            prompt = (
                f"The document has {len(pil_images)} pages, shown in order. "
                "Combine them into one result.\n\n" + base_prompt
            )
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        raw = self._generate(messages)
        self.last_raw_output = raw
        logger.debug("Raw model output: %s", raw)

        try:
            return parse_invoice_json(raw)
        except ExtractionError as exc:
            first_error = str(exc)
            logger.warning("Model output was not valid (%s). Retrying once.", first_error)

        # Retry: ask the model to fix its own JSON (text only, so it is faster).
        fix_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FIX_JSON_PROMPT.format(error=first_error, raw=raw)}
                ],
            }
        ]
        fixed = self._generate(fix_messages)
        self.last_raw_output = fixed
        try:
            return parse_invoice_json(fixed)
        except ExtractionError as exc:
            raise ExtractionError(
                f"Model did not return valid JSON after one retry: {exc}"
            ) from exc


_default_extractor: Optional[InvoiceExtractor] = None


def get_extractor() -> InvoiceExtractor:
    """Return the shared extractor (created once per process)."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = InvoiceExtractor()
    return _default_extractor


def extract(images: ImagesInput) -> Invoice:
    """Extract structured invoice data from one document (uses the shared model)."""
    return get_extractor().extract(images)
