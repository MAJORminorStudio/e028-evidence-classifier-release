#!/usr/bin/env python3
"""CPU-only structural validation for the E028 release artifact."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.release.resolve()
    required = [root / "adapter/adapter_model.safetensors", root / "classifier/classifier_head.safetensors", root / "config/tokenizer/tokenizer.json", root / "config/base-model.json"]
    missing = [str(item) for item in required if not item.is_file()]
    payload = {"status": "PASS" if not missing else "FAIL", "missing": missing, "sha256": {str(item.relative_to(root)): digest(item) for item in required if item.is_file()}}
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not missing else 1)


if __name__ == "__main__":
    main()
