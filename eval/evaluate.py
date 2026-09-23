"""Evaluate the InvoiceAI pipeline on the SROIE receipt dataset.

SROIE has 4 labeled fields per receipt. We map them to our schema:
    company -> vendor_name
    date    -> invoice_date
    address -> vendor_address
    total   -> total_amount

Metrics per field:
    exact match  - same value after normalizing (see _compact_text / dates / amounts)
    fuzzy match  - fuzzy score >= 90 (small OCR-style differences are OK)
    avg fuzzy    - average fuzzy score, 0-100
We also report how accurate each confidence level (high/medium/low) is.

Predictions are saved after every document, so a Colab disconnect does not
lose work: run again with the same output folder and it continues.

Usage (from the repo root):
    python -m eval.evaluate --split test --limit 100 --prompt v2 --output-dir eval
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from PIL import Image
from rapidfuzz import fuzz

# Allow "python eval/evaluate.py" as well as "python -m eval.evaluate".
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schema import parse_date, parse_number  # noqa: E402

logger = logging.getLogger(__name__)

DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"
FUZZY_THRESHOLD = 90.0

# SROIE field -> our schema field
FIELD_MAP: dict[str, str] = {
    "company": "vendor_name",
    "date": "invoice_date",
    "address": "vendor_address",
    "total": "total_amount",
}
TEXT_FIELDS = frozenset({"company", "address"})


@dataclass
class Sample:
    """One labeled receipt."""

    key: str
    image: Image.Image
    labels: dict[str, str]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_sroie(split: str = "test", limit: Optional[int] = 100, offset: int = 0) -> Iterator[Sample]:
    """Yield SROIE samples from Hugging Face, in a fixed order (same docs every run)."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=split)
    end = len(dataset) if limit is None else min(len(dataset), offset + limit)
    logger.info("Loaded %s split '%s': %d rows, using rows %d-%d.", DATASET_NAME, split, len(dataset), offset, end - 1)
    for index in range(offset, end):
        row = dataset[index]
        yield Sample(key=row["key"], image=row["image"], labels=dict(row["entities"]))


# ---------------------------------------------------------------------------
# Normalizing and scoring
# ---------------------------------------------------------------------------


def _loose_text(value: str) -> str:
    """Lowercase and collapse spaces (used for fuzzy scores)."""
    return " ".join(str(value).lower().split())


def _compact_text(value: str) -> str:
    """Lowercase, keep only letters, digits and '&' (used for exact match).

    So "NO.53 55,57 & 59" and "no 53 55, 57 &59" count as the same text.
    """
    return "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "&")


def score_field(sroie_field: str, predicted: Any, label: str) -> Optional[tuple[bool, float]]:
    """Compare one predicted value with the label.

    Returns:
        (exact_match, fuzzy_score 0-100), or None if the label is empty/unusable
        (then the field is skipped for this document).
    """
    if label is None or not str(label).strip():
        return None

    if sroie_field == "date":
        label_date = parse_date(label)
        if label_date is None:
            return None
        pred_date = predicted if isinstance(predicted, date) else parse_date(predicted)
        if pred_date is None:
            return False, 0.0
        if pred_date == label_date:
            return True, 100.0
        return False, float(fuzz.ratio(pred_date.isoformat(), label_date.isoformat()))

    if sroie_field == "total":
        label_amount = parse_number(label)
        if label_amount is None:
            return None
        pred_amount = parse_number(predicted)
        if pred_amount is None:
            return False, 0.0
        if abs(pred_amount - label_amount) < 0.005:
            return True, 100.0
        return False, float(fuzz.ratio(f"{pred_amount:.2f}", f"{label_amount:.2f}"))

    # Text fields: company, address
    if predicted is None or not str(predicted).strip():
        return False, 0.0
    exact = _compact_text(predicted) == _compact_text(label)
    score = 100.0 if exact else float(fuzz.ratio(_loose_text(predicted), _loose_text(label)))
    return exact, score


def _json_value(value: Any) -> Any:
    """Make a value JSON friendly."""
    if isinstance(value, date):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _load_done(predictions_path: Path) -> dict[str, dict[str, Any]]:
    """Read predictions saved by an earlier (maybe interrupted) run."""
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.is_file():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["key"]] = record
    return done


def evaluate_sample(pipeline: Any, sample: Sample) -> dict[str, Any]:
    """Run the pipeline on one sample and score it. Never raises."""
    start = time.perf_counter()
    record: dict[str, Any] = {
        "key": sample.key, "labels": sample.labels, "predicted": {}, "confidence": {}, "source": {}
    }
    try:
        doc = pipeline.process_pages([sample.image], f"{sample.key}.jpg")
        for sroie_field, our_field in FIELD_MAP.items():
            field_result = doc.result.fields.get(our_field)
            record["predicted"][sroie_field] = _json_value(getattr(doc.result.invoice, our_field))
            record["confidence"][sroie_field] = field_result.confidence if field_result else None
            record["source"][sroie_field] = field_result.source if field_result else None
        record["error"] = None
    except Exception as exc:  # keep going: one bad receipt must not stop the run
        logger.exception("Failed on %s", sample.key)
        record["error"] = f"{type(exc).__name__}: {exc}"
        _free_gpu_memory()
    record["seconds"] = round(time.perf_counter() - start, 2)

    record["scores"] = {}
    for sroie_field in FIELD_MAP:
        result = score_field(sroie_field, record["predicted"].get(sroie_field), sample.labels.get(sroie_field))
        record["scores"][sroie_field] = None if result is None else {"exact": result[0], "fuzzy": round(result[1], 1)}
    return record


def _free_gpu_memory() -> None:
    """Release cached GPU memory after an error (e.g. out of memory)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn per-document records into per-field metrics."""
    fields: dict[str, Any] = {}
    confidence: dict[str, Any] = {}
    fallback: dict[str, Any] = {}
    for sroie_field, our_field in FIELD_MAP.items():
        scored = [r for r in records if r["scores"].get(sroie_field) is not None]
        n = len(scored)
        exact = sum(r["scores"][sroie_field]["exact"] for r in scored)
        fuzzy_ok = sum(r["scores"][sroie_field]["fuzzy"] >= FUZZY_THRESHOLD for r in scored)
        fields[our_field] = {
            "sroie_field": sroie_field,
            "evaluated": n,
            "skipped_no_label": len(records) - n,
            "exact_match": round(exact / n, 4) if n else None,
            "fuzzy_match": round(fuzzy_ok / n, 4) if n else None,
            "avg_fuzzy_score": round(statistics.mean(r["scores"][sroie_field]["fuzzy"] for r in scored), 1) if n else None,
        }
        levels: dict[str, Any] = {}
        for level in ("high", "medium", "low"):
            group = [r for r in scored if r["confidence"].get(sroie_field) == level]
            correct = sum(r["scores"][sroie_field]["exact"] for r in group)
            levels[level] = {"count": len(group), "exact_match": round(correct / len(group), 4) if group else None}
        confidence[our_field] = levels

        used = [r for r in scored if (r.get("source") or {}).get(sroie_field) == "ocr_fallback"]
        if used:
            fallback[our_field] = {
                "used": len(used),
                "exact_match": round(sum(r["scores"][sroie_field]["exact"] for r in used) / len(used), 4),
            }

    ok = [r for r in records if r["error"] is None]
    times = [r["seconds"] for r in ok]
    return {
        "documents": len(records),
        "failed": len(records) - len(ok),
        "avg_seconds_per_doc": round(statistics.mean(times), 2) if times else None,
        "median_seconds_per_doc": round(statistics.median(times), 2) if times else None,
        "fields": fields,
        "confidence": confidence,
        "fallback": fallback,
    }


def _pct(value: Optional[float]) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def to_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    """Readable report for the README."""
    lines = [
        f"# Evaluation on SROIE ({meta['split']} split)",
        "",
        f"- Documents: **{summary['documents']}** (failed: {summary['failed']})",
        f"- Model: `{meta['model']}` (no fine-tuning), prompt `{meta['prompt_version']}`, "
        f"OCR fallback {'on' if meta.get('ocr_fallback') else 'off'}",
        "- OCR: PP-OCR models via RapidOCR (CPU)",
        f"- Average time: **{summary['avg_seconds_per_doc']} s** per document on {meta['device']}",
        f"- Dataset: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME}), rows {meta['offset']}-{meta['offset'] + summary['documents'] - 1}",
        f"- Date: {meta['date']}",
        "",
        "| Field | SROIE label | Exact match | Fuzzy match (≥90) | Avg fuzzy score | Docs |",
        "|---|---|---|---|---|---|",
    ]
    for our_field, m in summary["fields"].items():
        lines.append(
            f"| {our_field} | {m['sroie_field']} | {_pct(m['exact_match'])} | {_pct(m['fuzzy_match'])} "
            f"| {m['avg_fuzzy_score'] if m['avg_fuzzy_score'] is not None else '–'} | {m['evaluated']} |"
        )
    lines += [
        "",
        "Exact match ignores case, spaces and punctuation for text; dates and amounts are compared as values.",
        "",
        "## Does the confidence level help?",
        "",
        "Exact-match accuracy for each confidence level (count in brackets).",
        "",
        "| Field | High | Medium | Low |",
        "|---|---|---|---|",
    ]
    for our_field, levels in summary["confidence"].items():
        cells = [f"{_pct(levels[l]['exact_match'])} ({levels[l]['count']})" for l in ("high", "medium", "low")]
        lines.append(f"| {our_field} | " + " | ".join(cells) + " |")

    if summary.get("fallback"):
        lines += [
            "",
            "## OCR fallback",
            "",
            "When the model's total or date was not found on the document, the value was taken",
            "from the OCR text instead (these fields show as \"medium\" confidence above).",
            "",
            "| Field | Used on | Exact match when used |",
            "|---|---|---|",
        ]
        for our_field, stats in summary["fallback"].items():
            lines.append(f"| {our_field} | {stats['used']} docs | {_pct(stats['exact_match'])} |")
    lines.append("")
    return "\n".join(lines)


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "CPU"


def run_evaluation(
    pipeline: Any,
    samples: Iterable[Sample],
    output_dir: str | Path = "eval",
    prompt_version: Optional[str] = None,
    split: str = "test",
    offset: int = 0,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate the pipeline and write predictions.jsonl, results.json and results.md.

    Args:
        pipeline: An InvoicePipeline (or anything with process_pages()).
        samples: Labeled receipts, e.g. from load_sroie().
        output_dir: Folder for the output files.
        prompt_version: Switch the extraction prompt for this run (None = keep current).
        split, offset: Only used for the report text.
        resume: Skip documents already in predictions.jsonl.

    Returns:
        The summary dict (also saved in results.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    if prompt_version is not None:
        pipeline.extractor.set_prompt_version(prompt_version)
    used_prompt = getattr(pipeline.extractor, "prompt_version", "unknown")

    done = _load_done(predictions_path) if resume else {}
    if not resume and predictions_path.exists():
        predictions_path.unlink()
    if done:
        logger.info("Resuming: %d documents already done.", len(done))

    records: list[dict[str, Any]] = []
    with predictions_path.open("a", encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            if sample.key in done:
                records.append(done[sample.key])
                continue
            record = evaluate_sample(pipeline, sample)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            records.append(record)
            status = "ERROR" if record["error"] else " ".join(
                f"{f}={'ok' if (s or {}).get('exact') else ('-' if s is None else 'x')}"
                for f, s in record["scores"].items()
            )
            logger.info("[%d] %s  %.1fs  %s", i, sample.key, record["seconds"], status)

    summary = summarize(records)
    meta = {
        "dataset": DATASET_NAME,
        "split": split,
        "offset": offset,
        "prompt_version": used_prompt,
        "ocr_fallback": bool(getattr(pipeline, "use_ocr_fallback", False)),
        "model": getattr(getattr(pipeline, "config", None), "model_name", "unknown"),
        "device": _device_name(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    report = {"meta": meta, "summary": summary}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "results.md").write_text(to_markdown(summary, meta), encoding="utf-8")
    logger.info("Saved results to %s", output_dir)
    return report


def worst_examples(output_dir: str | Path, field: str, n: int = 10) -> list[dict[str, Any]]:
    """Wrong predictions for one SROIE field, lowest fuzzy score first (to study errors)."""
    records = list(_load_done(Path(output_dir) / "predictions.jsonl").values())
    wrong = [r for r in records if r["scores"].get(field) and not r["scores"][field]["exact"]]
    wrong.sort(key=lambda r: r["scores"][field]["fuzzy"])
    return [
        {"key": r["key"], "label": r["labels"].get(field), "predicted": r["predicted"].get(field),
         "fuzzy": r["scores"][field]["fuzzy"], "confidence": r["confidence"].get(field)}
        for r in wrong[:n]
    ]


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Evaluate InvoiceAI on SROIE.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=100, help="Number of documents (default 100).")
    parser.add_argument("--offset", type=int, default=0, help="Start at this row.")
    parser.add_argument("--prompt", default=None, help="Prompt version: v1, v2 or v3.")
    parser.add_argument("--no-fallback", action="store_true", help="Switch off the OCR fallback.")
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--no-resume", action="store_true", help="Start from zero.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from app.pipeline import get_pipeline

    pipeline = get_pipeline()
    pipeline.use_ocr_fallback = not args.no_fallback
    pipeline.load()
    report = run_evaluation(
        pipeline,
        load_sroie(args.split, args.limit, args.offset),
        output_dir=args.output_dir,
        prompt_version=args.prompt,
        split=args.split,
        offset=args.offset,
        resume=not args.no_resume,
    )
    print(to_markdown(report["summary"], report["meta"]))


if __name__ == "__main__":
    main()
