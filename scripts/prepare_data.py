#!/usr/bin/env python3
"""Prepare legally acquired source rows for the frozen E028 contract.

Raw third-party datasets are intentionally not bundled.  This utility checks
that a user-provided JSONL has the required fields and emits a derived hash
manifest; it never downloads or silently substitutes a dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    h = hashlib.sha256(); rows = 0; labels = {}
    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line); rows += 1
            for field in ("claim", "evidence", "label_id"):
                if field not in row: raise ValueError(f"missing {field} at row {rows}")
            h.update(line.encode("utf-8"))
            labels[str(row["label_id"])] = labels.get(str(row["label_id"]), 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "label_counts": labels, "sha256_jsonl_bytes": h.hexdigest()}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
