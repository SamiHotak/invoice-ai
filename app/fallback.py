"""OCR fallback for the total amount and the date.

Evaluation on SROIE showed: when the model's total or date is NOT found in
the OCR text (confidence "low"), it is usually wrong. The model sometimes
calculates a total instead of copying it, or invents the year of a date.
In those cases we look for the value in the OCR text directly:
- total: the amount next to a label like "TOTAL", "NETT TOTAL", "AMOUNT DUE"
- date:  a date token, preferring lines that contain the word "date"

Values found this way get confidence "medium" and source "ocr_fallback",
because they come from a simple rule, not from the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from app.matcher import _DATE_TOKEN
from app.ocr import OcrLine
from app.schema import BoundingBox, FieldResult, Invoice, parse_date, parse_number

logger = logging.getLogger(__name__)

# Lower number = stronger label for the final amount.
TOTAL_LABELS: tuple[tuple[str, int], ...] = (
    ("rounded total", 0),
    ("total rounded", 0),
    ("nett total", 1),
    ("net total", 1),
    ("grand total", 1),
    ("total amount", 2),
    ("amount due", 2),
    ("total payable", 2),
    ("total incl", 3),
    ("total (incl", 3),
    ("total sales", 3),
    ("total", 4),
)
# Lines with these words are not the final amount.
NOT_TOTAL: tuple[str, ...] = (
    "sub total", "subtotal", "sub-total", "total qty", "total quantity", "total item",
    "total discount", "total saving", "total gst", "total tax", "excl", "total no",
    "no. of", "total number", "total excluding",
)
# Money: exactly 2 decimals, e.g. 9.00, 1,234.50, RM9.00, $8.20 (not 9.000 or 25/12/2018).
_MONEY = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}(?![\d])")
MIN_YEAR = 1990


@dataclass(frozen=True)
class FoundValue:
    """A value found in the OCR text."""

    value: object
    box: BoundingBox
    text: str


def _money_values(text: str) -> list[float]:
    """All money amounts in a text, left to right."""
    values = [parse_number(token) for token in _MONEY.findall(text)]
    return [v for v in values if v is not None]


def _same_row(line: OcrLine, other: OcrLine) -> bool:
    """True if other is on the same text row as line."""
    if other.page != line.page:
        return False
    tolerance = 0.6 * max(line.box.height, other.box.height, 1.0)
    return abs(other.box.center_y - line.box.center_y) <= tolerance


def _amount_for_label(lines: Sequence[OcrLine], index: int) -> Optional[FoundValue]:
    """Find the amount that belongs to the label line at lines[index].

    Order: on the label line itself -> right of it on the same row ->
    the next line, but only if that line is just an amount (e.g. "9.00").
    """
    label = lines[index]
    amounts = _money_values(label.text)
    if amounts:
        return FoundValue(amounts[-1], label.box, label.text)

    right = [
        other for other in lines
        if other is not label and _same_row(label, other) and other.box.x0 >= label.box.x0
    ]
    for other in sorted(right, key=lambda l: l.box.x0, reverse=True):
        amounts = _money_values(other.text)
        if amounts:
            return FoundValue(amounts[-1], label.box.union(other.box), f"{label.text} {other.text}")

    if index + 1 < len(lines):
        nxt = lines[index + 1]
        only_amount = re.fullmatch(r"\s*(?:rm|\$|€|£|usd|myr)?\s*[\d.,]+\s*\w?\s*", nxt.text.lower())
        amounts = _money_values(nxt.text)
        if only_amount and amounts and nxt.page == label.page:
            return FoundValue(amounts[-1], label.box.union(nxt.box), f"{label.text} {nxt.text}")
    return None


def find_total(lines: Sequence[OcrLine]) -> Optional[FoundValue]:
    """Find the final total in OCR lines, or None.

    Picks the strongest label (see TOTAL_LABELS); if several have the same
    strength, the one lowest on the page (totals are near the bottom).
    """
    best: Optional[tuple[tuple[int, float], FoundValue]] = None
    for index, line in enumerate(lines):
        text = line.text.lower()
        if "total" not in text and "amount due" not in text:
            continue
        if any(word in text for word in NOT_TOTAL):
            continue
        priority = next(rank for label, rank in TOTAL_LABELS if label in text or label == "total")
        found = _amount_for_label(lines, index)
        if found is None or found.value <= 0:
            continue
        key = (priority, -(line.page * 1e6 + line.box.center_y))
        if best is None or key < best[0]:
            best = (key, found)
    return best[1] if best else None


def find_date(lines: Sequence[OcrLine], latest_year: Optional[int] = None) -> Optional[FoundValue]:
    """Find the document date in OCR lines, or None.

    Prefers lines with the word "date"; otherwise the first date from the top.
    """
    latest_year = latest_year or date.today().year + 1
    best: Optional[tuple[tuple[int, int], FoundValue]] = None
    for index, line in enumerate(lines):
        for token in _DATE_TOKEN.findall(line.text):
            parsed = parse_date(token, warn=False)
            if parsed is None or not MIN_YEAR <= parsed.year <= latest_year:
                continue
            key = (0 if "date" in line.text.lower() else 1, index)
            if best is None or key < best[0]:
                best = (key, FoundValue(parsed, line.box, line.text))
    return best[1] if best else None


class OcrFallback:
    """Replaces low-confidence totals and dates with values from the OCR text."""

    def apply(
        self,
        invoice: Invoice,
        fields: dict[str, FieldResult],
        lines: Sequence[OcrLine],
    ) -> tuple[Invoice, dict[str, FieldResult], list[str]]:
        """Return the (maybe) updated invoice, field results and warnings."""
        warnings: list[str] = []
        if not lines:
            return invoice, fields, warnings

        fields = dict(fields)
        updates: dict[str, object] = {}
        for name, finder in (("total_amount", find_total), ("invoice_date", find_date)):
            current = fields.get(name)
            if current is not None and current.confidence not in (None, "low"):
                continue  # the model value was found on the page: keep it
            found = finder(lines)
            if found is None or found.value == getattr(invoice, name):
                continue
            model_value = getattr(invoice, name)
            updates[name] = found.value
            fields[name] = FieldResult(
                value=found.value,
                confidence="medium",
                box=found.box,
                matched_text=found.text,
                source="ocr_fallback",
            )
            warnings.append(
                f"{name}: model value {model_value} was not found on the document; "
                f"used {found.value} from the OCR text instead."
            )
            logger.info("Fallback %s: %s -> %s (%r)", name, model_value, found.value, found.text)

        if updates:
            invoice = invoice.model_copy(update=updates)
        return invoice, fields, warnings
