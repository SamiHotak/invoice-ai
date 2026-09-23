"""Tests for value cleaning and the Pydantic models."""

from datetime import date

import pytest

from app.schema import FieldResult, Invoice, parse_date, parse_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234,56 €", 1234.56),
        ("1,234.56", 1234.56),
        ("12,50", 12.5),
        ("-3,00", -3.0),
        ("RM 9.00", 9.0),
        ("$8.20", 8.2),
        ("0.250", 0.25),
        (7, 7.0),
        ("", None),
        ("EUR", None),
        (None, None),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2018-12-25", date(2018, 12, 25)),
        ("25/12/2018", date(2018, 12, 25)),
        ("12-01-19", date(2019, 1, 12)),  # 2-digit year must not become year 19
        ("05 MAR 2018", date(2018, 3, 5)),
        ("28 MAR 18", date(2018, 3, 28)),
        ("20180304", date(2018, 3, 4)),
        ("12/28/2017", date(2017, 12, 28)),  # month-first only when day-first is impossible
        ("25/12/2018 8:13:39 PM", date(2018, 12, 25)),
        ("05 MAR 2018 18:24", date(2018, 3, 5)),  # date with a time attached
        ("05 Mar 2018 6:24 PM", date(2018, 3, 5)),
        ("2018-12-25T18:24:00", date(2018, 12, 25)),
        ("Mar 5, 2018 18:24", date(2018, 3, 5)),
        ("not a date", None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw, warn=False) == expected


def test_invoice_cleans_messy_model_output():
    invoice = Invoice.model_validate({
        "vendor_name": "  REWE  ",
        "invoice_number": 4711,  # model sometimes returns a number
        "invoice_date": "15.03.2024",
        "currency": "€",
        "line_items": [{"description": "Milk", "quantity": "2", "unit_price": "1,19", "total": "2,38"}],
        "total_amount": "2,38",
        "unknown_field": "ignored",
    })
    assert invoice.vendor_name == "REWE"
    assert invoice.invoice_number == "4711"
    assert invoice.invoice_date == date(2024, 3, 15)
    assert invoice.currency == "EUR"
    assert invoice.line_items[0].unit_price == 1.19
    assert invoice.total_amount == 2.38


def test_invoice_all_fields_optional():
    invoice = Invoice.model_validate({"line_items": None, "vendor_name": "null"})
    assert invoice.line_items == []
    assert invoice.vendor_name is None


def test_field_result_defaults():
    field = FieldResult(value=1.0)
    assert field.source == "model"
    assert field.confidence is None and field.box is None
