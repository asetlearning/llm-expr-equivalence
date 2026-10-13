"""
Join multiple JSONL files, dropping duplicates by a key field
(default `exprs_hash`, which matches the original-data split format).

Usage:
    python scripts/join_jsonl.py OUTPUT INPUT1 INPUT2 [INPUT...] [--key FIELD]

Created by: Alex Myasnikov
Date: May 11, 2026
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate JSONL files, dedup by key.")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--key", default="exprs_hash")
    args = parser.parse_args()

    seen: set[str] = set()
    kept = dups = 0
    with args.output.open("w") as out:
        for path in args.inputs:
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    k = rec.get(args.key)
                    if k in seen:
                        dups += 1
                        continue
                    seen.add(k)
                    out.write(json.dumps(rec) + "\n")
                    kept += 1
    print(f"Wrote {kept} records to {args.output} ({dups} duplicates dropped)")


if __name__ == "__main__":
    main()
