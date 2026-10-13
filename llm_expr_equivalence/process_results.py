"""
Process LLM equivalence prediction JSONL files: re-extract missing predictions,
report accuracy + confusion matrix as markdown, optionally split correct/incorrect.

Created by: Alex Myasnikov
Date: May 11, 2026
"""
import argparse
import json
import re
import sys
from pathlib import Path

from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsBase
from llm_expr_equivalence.schemas.common import Trichotomy

_BOXED_TEXT_RE = re.compile(
    r"\\boxed\s*\{\s*\\text\s*\{\s*(YES|NO|DONT\s*KNOW|DONTKNOW)\s*\}\s*\}",
    re.IGNORECASE,
)
_BOXED_BARE_RE = re.compile(
    r"\\boxed\s*\{\s*(YES|NO|DONT\s*KNOW|DONTKNOW)\s*\}",
    re.IGNORECASE,
)


def _normalize(token: str) -> Trichotomy:
    t = re.sub(r"\s+", "", token).upper()
    if t == "YES":
        return Trichotomy.YES
    if t == "NO":
        return Trichotomy.NO
    return Trichotomy.DONOTKNOW


def extract_boxed(content: str) -> Trichotomy | None:
    """Try \\boxed{\\text{...}} then \\boxed{...}; last occurrence wins."""
    if not content:
        return None
    for pattern in (_BOXED_TEXT_RE, _BOXED_BARE_RE):
        matches = pattern.findall(content)
        if matches:
            return _normalize(matches[-1])
    return None


def recover_prediction(completion: str) -> Trichotomy | None:
    tri = LLMEvaluateResultsBase.extract_result(completion)
    if tri is not None:
        return tri
    return extract_boxed(completion)


def process(records: list[dict]) -> tuple[list[dict], dict]:
    recovered = 0
    still_null = 0
    confusion: dict[str, dict[str, int]] = {
        "YES": {"YES": 0, "NO": 0, "DONTKNOW": 0},
        "NO": {"YES": 0, "NO": 0, "DONTKNOW": 0},
    }
    correct_count = 0

    for rec in records:
        if rec.get("prediction") is None:
            tri = recover_prediction(rec.get("completion") or "")
            if tri is not None:
                rec["prediction"] = str(tri)
                recovered += 1

        gt = rec.get("ground_truth")
        pred = rec.get("prediction")
        pred_bucket = pred if pred in ("YES", "NO", "DONTKNOW") else "DONTKNOW"
        if pred is None:
            still_null += 1

        rec["correct"] = (pred is not None and pred == gt)
        if rec["correct"]:
            correct_count += 1

        if gt in confusion:
            confusion[gt][pred_bucket] += 1

    cm = confusion
    correct_yes = cm["YES"]["YES"]
    correct_no = cm["NO"]["NO"]
    incorrect_yes = cm["YES"]["NO"] + cm["YES"]["DONTKNOW"]
    incorrect_no = cm["NO"]["YES"] + cm["NO"]["DONTKNOW"]

    total = len(records)
    stats = {
        "total": total,
        "recovered": recovered,
        "still_null": still_null,
        "accuracy": correct_count / total if total else 0.0,
        "correct_yes": correct_yes,
        "correct_no": correct_no,
        "incorrect_yes": incorrect_yes,
        "incorrect_no": incorrect_no,
        "confusion": confusion,
    }
    return records, stats


def render_markdown(name: str, stats: dict) -> str:
    cm = stats["confusion"]
    return (
        f"# LLM Prediction Analysis: {name}\n\n"
        f"## Summary\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Total records | {stats['total']} |\n"
        f"| Predictions recovered from completion | {stats['recovered']} |\n"
        f"| Still unparseable (lumped into DONTKNOW) | {stats['still_null']} |\n"
        f"| Accuracy | {stats['accuracy']:.4f} |\n"
        f"| Correct, ground_truth=YES | {stats['correct_yes']} |\n"
        f"| Correct, ground_truth=NO | {stats['correct_no']} |\n"
        f"| Incorrect (incl. DONTKNOW), ground_truth=YES | {stats['incorrect_yes']} |\n"
        f"| Incorrect (incl. DONTKNOW), ground_truth=NO | {stats['incorrect_no']} |\n\n"
        f"## Confusion Matrix\n\n"
        f"| ground_truth \\ prediction | YES | NO | DONTKNOW |\n"
        f"|---|---|---|---|\n"
        f"| YES | {cm['YES']['YES']} | {cm['YES']['NO']} | {cm['YES']['DONTKNOW']} |\n"
        f"| NO  | {cm['NO']['YES']} | {cm['NO']['NO']} | {cm['NO']['DONTKNOW']} |\n"
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_split(records: list[dict], prefix: str) -> None:
    correct_path = Path(f"{prefix}.correct.jsonl")
    incorrect_path = Path(f"{prefix}.incorrect.jsonl")
    with correct_path.open("w") as fc, incorrect_path.open("w") as fi:
        for rec in records:
            (fc if rec["correct"] else fi).write(json.dumps(rec) + "\n")


def split_original(
    original_path: Path, records: list[dict], prefix: str
) -> tuple[int, int, int]:
    """Split original dataset by matching exprs_hash -> uuid in predictions.

    Returns (correct_count, incorrect_count, unmatched_count).
    """
    verdict_by_id: dict[str, bool] = {
        rec["uuid"]: bool(rec["correct"]) for rec in records if "uuid" in rec
    }
    correct_path = Path(f"{prefix}.original.correct.jsonl")
    incorrect_path = Path(f"{prefix}.original.incorrect.jsonl")
    c = i = u = 0
    with original_path.open() as f_in, \
            correct_path.open("w") as fc, \
            incorrect_path.open("w") as fi:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            verdict = verdict_by_id.get(rec.get("exprs_hash"))
            if verdict is None:
                u += 1
                continue
            (fc if verdict else fi).write(json.dumps(rec) + "\n")
            if verdict:
                c += 1
            else:
                i += 1
    return c, i, u


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM equivalence prediction JSONL files."
    )
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument(
        "--split-prefix",
        type=str,
        default=None,
        help="If set, write PREFIX.correct.jsonl and PREFIX.incorrect.jsonl",
    )
    parser.add_argument(
        "--original-data",
        type=Path,
        default=None,
        help=(
            "Path to original dataset JSONL (records keyed by exprs_hash). "
            "Requires --split-prefix. Writes "
            "PREFIX.original.correct.jsonl and PREFIX.original.incorrect.jsonl, "
            "matching exprs_hash to prediction uuid."
        ),
    )
    args = parser.parse_args()

    if args.original_data and not args.split_prefix:
        parser.error("--original-data requires --split-prefix")

    records = load_jsonl(args.input)
    records, stats = process(records)
    sys.stdout.write(render_markdown(args.input.name, stats))

    if args.split_prefix:
        write_split(records, args.split_prefix)

    if args.original_data:
        c, i, u = split_original(args.original_data, records, args.split_prefix)
        sys.stdout.write(
            f"\n## Original Dataset Split\n\n"
            f"| Bucket | Count |\n"
            f"|---|---|\n"
            f"| Correct | {c} |\n"
            f"| Incorrect | {i} |\n"
            f"| Unmatched (no prediction for exprs_hash) | {u} |\n"
        )


if __name__ == "__main__":
    main()
