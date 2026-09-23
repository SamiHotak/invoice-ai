"""Link extracted values to OCR text boxes.

For every value the model extracted, we look for the same value in the OCR
text. This gives us two things:
1. A box, so the UI can show where the value is on the page.
2. A confidence level. A value the OCR cannot find may be a model mistake.

Note: "high" means "this value is printed on the document". It does not
prove it is the right field (e.g. a person's name used as vendor name).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterator, Optional, Sequence

from rapidfuzz import fuzz

from app.config import Settings, settings as default_settings
from app.ocr import OcrLine
from app.schema import BoundingBox, FieldResult, Invoice, LineItem, parse_date, parse_number

logger = logging.getLogger(__name__)

# Invoice fields shown with a box (line items are handled separately).
TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "vendor_name",
    "vendor_address",
    "invoice_number",
    "invoice_date",
    "currency",
    "subtotal",
    "tax",
    "total_amount",
)
# Totals are usually at the bottom, so we prefer the lowest match.
BOTTOM_FIELDS: frozenset[str] = frozenset({"subtotal", "tax", "total_amount"})
# Addresses and long names can span several OCR lines.
MAX_WINDOW_LINES = 6
# Short text (e.g. "01") is only matched as a whole word, never fuzzy.
MIN_FUZZY_LENGTH = 4

_CURRENCY_ALIASES: dict[str, tuple[str, ...]] = {
    "EUR": ("EUR", "€"),
    "USD": ("USD", "US$", "$"),
    "GBP": ("GBP", "£"),
    "MYR": ("MYR", "RM"),
    "CHF": ("CHF",),
    "SGD": ("SGD", "S$"),
}
_NUMBER_TOKEN = re.compile(r"-?\d[\d.,]*")
DATE_TOKEN = re.compile(
    r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}"  # 25/12/2018, 2018-12-25
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}"  # 25 Dec 2018
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"  # Dec 25, 2018
)


@dataclass(frozen=True)
class _Window:
    """One or more neighbouring OCR lines joined together."""

    text: str
    box: BoundingBox
    size: int


@dataclass(frozen=True)
class _Match:
    """The best OCR window found for one value."""

    window: _Window
    score: float


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase and collapse spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _whole_word(needle: str, haystack: str) -> bool:
    """True if needle appears in haystack as a separate word/number."""
    pattern = rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])"
    return re.search(pattern, haystack) is not None


def score_text(value: str, text: str) -> float:
    """Fuzzy score (0-100) for a text value inside an OCR text."""
    needle, haystack = _normalize(value), _normalize(text)
    if not needle or not haystack:
        return 0.0
    if len(needle) < MIN_FUZZY_LENGTH:
        return 100.0 if _whole_word(needle, haystack) else 0.0
    if len(haystack) >= len(needle):
        # Value can be part of a longer line, e.g. "Invoice No: TD01167104".
        return float(fuzz.partial_ratio(needle, haystack))
    # OCR text is shorter than the value: compare the full strings,
    # so one address line does not count as the whole address.
    return float(fuzz.ratio(needle, haystack))


def score_number(value: float, text: str) -> float:
    """100 if the number appears in the text (any common format), else 0."""
    for token in _NUMBER_TOKEN.findall(text):
        number = parse_number(token)
        if number is not None and abs(abs(number) - abs(value)) < 0.005:
            return 100.0
    return 0.0


def score_date(value: date, text: str) -> float:
    """100 if the date appears in the text (any common format), else 0."""
    for token in DATE_TOKEN.findall(text):
        if parse_date(token, warn=False) == value:
            return 100.0
    return 0.0


def score_currency(value: str, text: str) -> float:
    """100 if the currency code or its symbol appears in the text, else 0."""
    haystack = _normalize(text)
    for alias in _CURRENCY_ALIASES.get(value.upper(), (value,)):
        alias = alias.lower()
        found = _whole_word(alias, haystack) if alias.isalpha() else alias in haystack
        if found:
            return 100.0
    return 0.0


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def _windows(lines: Sequence[OcrLine], max_size: int) -> Iterator[_Window]:
    """All groups of 1..max_size neighbouring lines on the same page."""
    for start in range(len(lines)):
        box = lines[start].box
        texts: list[str] = []
        for size in range(1, max_size + 1):
            end = start + size
            if end > len(lines) or lines[end - 1].page != lines[start].page:
                break
            if size > 1:
                box = box.union(lines[end - 1].box)
            texts.append(lines[end - 1].text)
            yield _Window(" ".join(texts), box, size)


def _near(box: BoundingBox, anchor: BoundingBox) -> bool:
    """True if box is on the anchor's row or the row just below/above it."""
    if box.page != anchor.page:
        return False
    tolerance = 3 * max(anchor.height, box.height, 1.0)
    return abs(box.center_y - anchor.center_y) <= tolerance


def _best_match(
    value: Any,
    lines: Sequence[OcrLine],
    scorer: Callable[[Any, str], float],
    *,
    multi_line: bool,
    prefer_bottom: bool = False,
    anchor: Optional[BoundingBox] = None,
    only_near_anchor: bool = False,
) -> Optional[_Match]:
    """Find the OCR window that matches the value best.

    Ties are broken by: smaller window first, then position
    (lowest on the page for totals, closest to the anchor for line items,
    otherwise the first one from the top).
    """
    best: Optional[_Match] = None
    best_key: Optional[tuple[float, ...]] = None
    max_size = MAX_WINDOW_LINES if multi_line else 1

    for window in _windows(lines, max_size):
        if only_near_anchor and (anchor is None or not _near(window.box, anchor)):
            continue
        score = scorer(value, window.text)
        if score <= 0:
            continue

        if anchor is not None:
            position = -abs(window.box.center_y - anchor.center_y) - 1e6 * abs(
                window.box.page - anchor.page
            )
        elif prefer_bottom:
            position = window.box.page * 1e6 + window.box.center_y
        else:
            position = -(window.box.page * 1e6 + window.box.center_y)

        key = (round(score, 1), -window.size, position)
        if best_key is None or key > best_key:
            best, best_key = _Match(window, score), key
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FieldMatcher:
    """Turns an Invoice plus OCR lines into FieldResults with confidence and boxes."""

    def __init__(self, config: Settings = default_settings) -> None:
        self.config = config

    def confidence_for(self, score: float) -> str:
        """Map a match score (0-100) to "high", "medium" or "low"."""
        if score >= self.config.match_high_threshold:
            return "high"
        if score >= self.config.match_medium_threshold:
            return "medium"
        return "low"

    def _to_result(self, value: Any, match: Optional[_Match]) -> FieldResult:
        """Build a FieldResult. Low confidence means no box is shown."""
        if value is None:
            return FieldResult(value=None, confidence=None)
        if match is None:
            return FieldResult(value=value, confidence="low", match_score=0.0)
        confidence = self.confidence_for(match.score)
        return FieldResult(
            value=value,
            confidence=confidence,
            box=match.window.box if confidence != "low" else None,
            match_score=round(match.score, 1),
            matched_text=match.window.text,
        )

    def match_value(
        self,
        field_name: str,
        value: Any,
        lines: Sequence[OcrLine],
        anchor: Optional[BoundingBox] = None,
    ) -> FieldResult:
        """Match one value against the OCR lines."""
        if value is None or not lines:
            return self._to_result(value, None)

        if field_name == "currency":
            match = _best_match(str(value), lines, score_currency, multi_line=False)
        elif isinstance(value, date):
            match = _best_match(value, lines, score_date, multi_line=False)
        elif isinstance(value, (int, float)):
            match = _best_match(
                float(value),
                lines,
                score_number,
                multi_line=False,
                prefer_bottom=field_name in BOTTOM_FIELDS,
                anchor=anchor,
                # A quantity like "1" appears everywhere, so only look on the item's row.
                only_near_anchor=field_name == "quantity",
            )
        else:
            match = _best_match(
                str(value),
                lines,
                score_text,
                multi_line=True,
                prefer_bottom=field_name in BOTTOM_FIELDS,
            )
        return self._to_result(value, match)

    def match_line_item(self, item: LineItem, lines: Sequence[OcrLine]) -> dict[str, FieldResult]:
        """Match all fields of one line item. Numbers near the description are preferred."""
        description = self.match_value("description", item.description, lines)
        anchor = description.box
        return {
            "description": description,
            "quantity": self.match_value("quantity", item.quantity, lines, anchor=anchor),
            "unit_price": self.match_value("unit_price", item.unit_price, lines, anchor=anchor),
            "total": self.match_value("total", item.total, lines, anchor=anchor),
        }

    def match_invoice(
        self, invoice: Invoice, lines: Sequence[OcrLine]
    ) -> tuple[dict[str, FieldResult], list[dict[str, FieldResult]]]:
        """Match every field of an invoice.

        Returns:
            (top-level field results, one dict of field results per line item)
        """
        fields = {
            name: self.match_value(name, getattr(invoice, name), lines) for name in TOP_LEVEL_FIELDS
        }
        items = [self.match_line_item(item, lines) for item in invoice.line_items]
        return fields, items
