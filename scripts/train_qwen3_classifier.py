#!/usr/bin/env python3
"""Reproduce the frozen E028 training contract on user-supplied local data.

The command is intentionally never invoked by E029. It requires the user to
provide legally acquired JSONL rows and a local snapshot of the pinned base
model; it does not download data or silently substitute paths.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/training-config.json")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for path in (args.train, args.validation, args.model_dir):
        if not path.exists(): raise FileNotFoundError(path)
    if config["model_revision"] != "49e3418fbbbca6ecbdf9608b4d22e5a407081db4": raise ValueError("base revision is not the frozen E028 revision")
    if config["num_train_epochs"] != 2 or config["learning_rate"] != 0.0002 or config["seed"] != 17: raise ValueError("training config is not the frozen E028 contract")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import e026_train
    config.update({"train_path": str(args.train.resolve()), "validation_path": str(args.validation.resolve()), "model_dir": str(args.model_dir.resolve()), "output_dir": str(args.output.resolve())})
    e026_train.ROOT = root
    e026_train.train(config)


if __name__ == "__main__": main()
