"""
Split a GRPO rewards log JSONL into N equal time-ordered buckets and print
markdown table of progress stats (reward mean/median, </think> closure rate,
clipped-mid-think rate, answer-correct rate, recognized/error counts).

Usage:
    python scripts/training_progress_stats.py PATH [--buckets N]

Created by: Alex Myasnikov
Date: May 11, 2026
"""
import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def bucketize(rows: list[dict], n: int) -> list[list[dict]]:
    size = len(rows) / n
    return [rows[int(i * size) : int((i + 1) * size)] for i in range(n)]


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    rewards = [r["reward"] for r in rows]
    nz = [r for r in rewards if r != 0]
    sols = [r.get("llm_solution") or "" for r in rows]
    preds = [r.get("prediction") for r in rows]
    gts = [r.get("ground_truth") for r in rows]
    closed = sum("</think>" in s for s in sols)
    clipped = sum("<think>" in s and "</think>" not in s for s in sols)
    correct = sum(p is not None and p == g for p, g in zip(preds, gts, strict=False))
    recognized = sum(p is not None for p in preds)
    return {
        "n": n,
        "reward mean": sum(rewards) / n,
        "reward median": statistics.median(rewards),
        "reward (nonzero only)": (sum(nz) / len(nz)) if nz else float("nan"),
        "zero-reward rate": sum(1 for r in rewards if r == 0) / n,
        "</think> closed": closed / n,
        "clipped mid-think": clipped / n,
        "answer-correct": correct / n,
        "prediction recognized": recognized,
        "prediction error": n - recognized,
    }


PCT_ROWS = {"zero-reward rate", "</think> closed", "clipped mid-think", "answer-correct"}
INT_ROWS = {"prediction recognized", "prediction error"}


def fmt(metric: str, v: float) -> str:
    if metric in INT_ROWS:
        return str(int(v))
    if metric in PCT_ROWS:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def delta(metric: str, a: float, b: float) -> str:
    if metric in INT_ROWS:
        return f"{int(b - a):+d}"
    if metric in PCT_ROWS:
        return f"{(b - a) * 100:+.1f} pp"
    if a == 0:
        return "n/a"
    return f"{(b - a) / abs(a) * 100:+.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--buckets", type=int, default=4)
    args = ap.parse_args()

    rows = load(args.path)
    buckets = bucketize(rows, args.buckets)
    s = [stats(b) for b in buckets]
    metrics = [k for k in s[0] if k != "n"]

    headers = (
        ["metric"]
        + [f"Q{i + 1}" + (" (early)" if i == 0 else " (latest)" if i == len(s) - 1 else "") for i in range(len(s))]
        + [f"Δ Q1→Q{len(s)}"]
    )

    print(f"# Training progress — `{args.path.name}`\n")
    print(f"_{len(rows)} rows, {args.buckets} buckets of ~{len(rows) // args.buckets}_\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for m in metrics:
        cells = [m] + [fmt(m, b[m]) for b in s] + [delta(m, s[0][m], s[-1][m])]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
