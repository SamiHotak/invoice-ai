"""Pydantic v2 data models for extracted invoices.

All fields are optional, because not every invoice or receipt has every value.
The validators clean up typical model output (German number format, currency
symbols, different date formats) so the final data is always consistent.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# Day-first formats come before month-first ones, because German and most
# non-US receipts write the day first (15.03.2024, 15/03/2024).
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %b %y",  # 28 MAR 18
    "%d %B %y",
    "%Y%m%d",  # 20180304
    # Month-first (US) formats are tried last, only when day-first fails (e.g. 12/28/2017).
    "%m/%d/%Y",
    "%m/%d/%y",
)

_CURRENCY_SYMBOLS: dict[str, str] = {
    "€": "EUR",
    "EURO": "EUR",
    "$": "USD",
    "US$": "USD",
    "£": "GBP",
    "RM": "MYR",
    "FR.": "CHF",
}


def parse_number(value: Any) -> Optional[float]:
    """Turn a number or a number-like string into a float.

    Handles German and English formats, for example:
        "1.234,56 €" -> 1234.56
        "1,234.56"   -> 1234.56
        "12,50"      -> 12.5
        "-3,00"      -> -3.0

    Returns None if no number can be found.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    negative = text.startswith("-") or text.endswith("-") or (
        text.startswith("(") and text.endswith(")")
    )
    cleaned = re.sub(r"[^\d.,]", "", text)
    if not re.search(r"\d", cleaned):
        return None

    if "," in cleaned and "." in cleaned:
        # The separator that comes last is the decimal separator.
        decimal_sep = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        cleaned = cleaned.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif "," in cleaned or "." in cleaned:
        sep = "," if "," in cleaned else "."
        parts = cleaned.split(sep)
        if len(parts) > 2:
            # "1.234.567" -> only thousands separators
            cleaned = "".join(parts)
        else:
            int_part, frac_part = parts
            # "1.234" is one thousand (money rarely has 3 decimals),
            # but "0.250" (e.g. kg) stays a decimal.
            if len(frac_part) == 3 and int_part not in ("", "0"):
                cleaned = int_part + frac_part
            else:
                cleaned = f"{int_part or '0'}.{frac_part or '0'}"

    try:
        number = float(cleaned)
    except ValueError:
        logger.warning("Could not parse number from %r", value)
        return None
    return -number if negative else number


def parse_date(value: Any) -> Optional[date]:
    """Turn a date string in a common format into a date object.

    Returns None (and logs a warning) if the format is unknown.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    # Try the full string first, then only the first part (drops a time like "14:32").
    candidates = [text]
    if " " in text:
        candidates.append(text.split()[0])

    for candidate in candidates:
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
            if parsed.year >= 1900:  # guard against "19" being read as year 19
                return parsed

    logger.warning("Unknown date format: %r", value)
    return None


def _clean_text(value: Any) -> Optional[str]:
    """Convert any value to a stripped string, or None if empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


class LineItem(BaseModel):
    """One product or service line on an invoice."""

    model_config = ConfigDict(extra="ignore")

    description: Optional[str] = Field(default=None, description="Item name as printed.")
    quantity: Optional[float] = Field(default=None, description="Number of units.")
    unit_price: Optional[float] = Field(default=None, description="Price for one unit.")
    total: Optional[float] = Field(default=None, description="Line total.")

    @field_validator("description", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Optional[str]:
        return _clean_text(value)

    @field_validator("quantity", "unit_price", "total", mode="before")
    @classmethod
    def _validate_numbers(cls, value: Any) -> Optional[float]:
        return parse_number(value)


class Invoice(BaseModel):
    """Structured data extracted from one invoice or receipt."""

    model_config = ConfigDict(extra="ignore")

    vendor_name: Optional[str] = Field(default=None, description="Seller / company name.")
    vendor_address: Optional[str] = Field(default=None, description="Seller address.")
    invoice_number: Optional[str] = Field(default=None, description="Invoice or receipt number.")
    invoice_date: Optional[date] = Field(default=None, description="Invoice date (ISO format).")
    currency: Optional[str] = Field(default=None, description="ISO 4217 code, e.g. EUR.")
    line_items: list[LineItem] = Field(default_factory=list, description="Product lines.")
    subtotal: Optional[float] = Field(default=None, description="Amount before tax.")
    tax: Optional[float] = Field(default=None, description="Total tax amount.")
    total_amount: Optional[float] = Field(default=None, description="Final amount to pay.")

    @field_validator("vendor_name", "vendor_address", "invoice_number", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Optional[str]:
        return _clean_text(value)

    @field_validator("invoice_date", mode="before")
    @classmethod
    def _validate_date(cls, value: Any) -> Optional[date]:
        return parse_date(value)

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, value: Any) -> Optional[str]:
        text = _clean_text(value)
        if text is None:
            return None
        text = text.upper()
        return _CURRENCY_SYMBOLS.get(text, text)

    @field_validator("line_items", mode="before")
    @classmethod
    def _validate_line_items(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator("subtotal", "tax", "total_amount", mode="before")
    @classmethod
    def _validate_numbers(cls, value: Any) -> Optional[float]:
        return parse_number(value)


# ---------------------------------------------------------------------------
# Day 2: confidence, boxes and the full document result
# ---------------------------------------------------------------------------

Confidence = Literal["high", "medium", "low"]


class BoundingBox(BaseModel):
    """A rectangle on one page, in pixels of the processed page image."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int = Field(default=0, description="Page index, starting at 0.")

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y1 - self.y0

    @property
    def center_y(self) -> float:
        """Vertical center of the box."""
        return (self.y0 + self.y1) / 2

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Smallest box that contains both boxes (same page)."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            page=self.page,
        )


class FieldResult(BaseModel):
    """One extracted value, with how sure we are and where it was found.

    confidence:
        high   - value found in the OCR text with a strong match
        medium - weak match
        low    - not found in the OCR text (possible model mistake)
        None   - the model found no value for this field
    """

    value: Any = None
    confidence: Optional[Confidence] = None
    box: Optional[BoundingBox] = None
    match_score: Optional[float] = Field(default=None, description="Best match score 0-100.")
    matched_text: Optional[str] = Field(default=None, description="OCR text that matched.")


class DocumentResult(BaseModel):
    """Everything the pipeline returns for one file."""

    file_name: str
    num_pages: int
    invoice: Invoice
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    line_items: list[dict[str, FieldResult]] = Field(default_factory=list)
    ocr_line_count: int = 0
    processing_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
