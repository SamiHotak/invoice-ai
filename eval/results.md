# Evaluation on SROIE (test split)

- Documents: **100** (failed: 0)
- Model: `Qwen/Qwen2.5-VL-3B-Instruct` (no fine-tuning), prompt `v2`
- OCR: PP-OCR models via RapidOCR (CPU)
- Average time: **22.28 s** per document on Tesla T4
- Dataset: [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE), rows 0-99
- Date: 2026-09-23

| Field | SROIE label | Exact match | Fuzzy match (≥90) | Avg fuzzy score | Docs |
|---|---|---|---|---|---|
| vendor_name | company | 89.0% | 92.0% | 95.8 | 100 |
| invoice_date | date | 97.0% | 98.0% | 98.7 | 100 |
| vendor_address | address | 79.0% | 97.0% | 98.2 | 100 |
| total_amount | total | 82.0% | 82.0% | 91.0 | 100 |

Exact match ignores case, spaces and punctuation for text; dates and amounts are compared as values.

## Does the confidence level help?

Exact-match accuracy for each confidence level (count in brackets).

| Field | High | Medium | Low |
|---|---|---|---|
| vendor_name | 89.7% (97) | 100.0% (1) | 100.0% (1) |
| invoice_date | 100.0% (94) | – (0) | 60.0% (5) |
| vendor_address | 83.2% (95) | 0.0% (3) | 0.0% (1) |
| total_amount | 98.8% (80) | – (0) | 15.8% (19) |
