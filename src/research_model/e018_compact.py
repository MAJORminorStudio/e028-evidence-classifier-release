"""E018 compact evidence-support training/evaluation contracts.

This module deliberately keeps the frozen data contract independent of the
optional CUDA/Transformers stack so that all validation and scoring can run on
CPU-only hosts.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
SMOKE_LIFECYCLE = ("PREPARED", "MODEL_READY", "QLORA_READY", "BATCH_READY", "FORWARD_COMPLETE", "LOSS_VALIDATED", "BACKWARD_COMPLETE", "GRADIENTS_VALIDATED", "OPTIMIZER_COMPLETE", "CHECKPOINT_SAVED", "CHECKPOINT_FINALIZED", "CHECKPOINT_RELOADED", "RESUME_VALIDATED", "EVALUATOR_VALIDATED", "METRICS_WRITTEN", "COMPLETE")

def validate_smoke_lifecycle(states):
    if list(states) != list(SMOKE_LIFECYCLE):
        raise E018DataError(f"invalid smoke lifecycle: {states}")
    return True
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
CORE_COUNTS = {
    "train": {"SUPPORTED": 800, "INSUFFICIENT_EVIDENCE": 119, "CONTRADICTED": 16},
    "validation": {"SUPPORTED": 96, "INSUFFICIENT_EVIDENCE": 0, "CONTRADICTED": 1},
    "test": {"SUPPORTED": 219, "INSUFFICIENT_EVIDENCE": 27, "CONTRADICTED": 3},
}


class E018DataError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path, *, expected_split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        raise E018DataError(f"missing dataset: {p}")
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise E018DataError(f"{p}:{line_no}: invalid JSON: {exc}") from exc
        validate_row(row, expected_split=expected_split, path=p, line_no=line_no)
        rows.append(row)
    if not rows:
        raise E018DataError(f"empty dataset: {p}")
    return rows


def validate_row(row: dict[str, Any], *, expected_split: str | None = None,
                 path: Path | None = None, line_no: int | None = None) -> None:
    where = f"{path}:{line_no}: " if path and line_no else ""
    required = {"question", "context", "evidence", "target"}
    missing = required - set(row)
    if missing:
        raise E018DataError(f"{where}missing fields: {sorted(missing)}")
    if not (row.get("example_id") or row.get("challenge_id")):
        raise E018DataError(f"{where}missing example_id/challenge_id")
    if expected_split and row.get("split") != expected_split:
        raise E018DataError(f"{where}expected split {expected_split!r}, got {row.get('split')!r}")
    target = row["target"]
    if not isinstance(target, dict) or set(target) != {"status", "evidence_ids", "claim"}:
        raise E018DataError(f"{where}target must contain exactly status/evidence_ids/claim")
    status = target["status"]
    if status not in LABELS:
        raise E018DataError(f"{where}unknown status {status!r}")
    if not isinstance(target["evidence_ids"], list) or any(not isinstance(x, str) for x in target["evidence_ids"]):
        raise E018DataError(f"{where}evidence_ids must be a list of strings")
    if status == "INSUFFICIENT_EVIDENCE" and (target["evidence_ids"] or target["claim"] is not None):
        raise E018DataError(f"{where}insufficient-evidence target must have empty evidence_ids and null claim")
    if status != "INSUFFICIENT_EVIDENCE" and (not target["evidence_ids"] or not isinstance(target["claim"], str) or not target["claim"].strip()):
        raise E018DataError(f"{where}supported/contradicted target requires evidence_ids and a nonempty claim")
    if not isinstance(row["evidence"], list):
        raise E018DataError(f"{where}evidence must be a list")


def target_json(target: dict[str, Any]) -> str:
    validate_row({"example_id": "_", "question": "_", "context": "_", "evidence": [], "target": target})
    return json.dumps({"status": target["status"], "evidence_ids": target["evidence_ids"], "claim": target["claim"]}, ensure_ascii=False, separators=(",", ":"))


def prompt_text(row: dict[str, Any]) -> str:
    return ("Classify the claim support using only the supplied evidence.\n"
            "Return one compact JSON object with keys status, evidence_ids, and claim.\n\n"
            f"Question:\n{row['question']}\n\nEvidence context:\n{row['context']}\n\nAnswer:\n")


def build_training_text(row: dict[str, Any]) -> tuple[str, str]:
    return prompt_text(row), target_json(row["target"])


def make_supervision_mask(prompt_ids: list[int], target_ids: list[int], eos_id: int | None) -> tuple[list[int], list[int]]:
    if not target_ids:
        raise E018DataError("zero-target example")
    target = list(target_ids) + ([] if eos_id is None else [eos_id])
    return prompt_ids + target, ([-100] * len(prompt_ids)) + target


def collate_compact_features(features: list[dict[str, Any]], *, pad_token_id: int,
                             return_tensors: str | None = None) -> dict[str, Any]:
    """Pad already-tokenized E018 features without changing supervision boundaries."""
    if not features:
        raise E018DataError("cannot collate an empty batch")
    if pad_token_id is None:
        raise E018DataError("tokenizer has no pad_token_id; configure one explicitly before batching")
    lengths = [len(f.get("input_ids", [])) for f in features]
    if any(length <= 0 for length in lengths):
        raise E018DataError("empty input sequence")
    width = max(lengths)
    ids, masks, labels = [], [], []
    for feature, length in zip(features, lengths):
        input_ids = list(feature["input_ids"])
        label_ids = list(feature["labels"])
        attention = list(feature.get("attention_mask", [1] * length))
        if len(label_ids) != length or len(attention) != length:
            raise E018DataError("feature tensors do not share a length")
        supervised = [i for i, value in enumerate(label_ids) if value != -100]
        if not supervised:
            raise E018DataError(f"zero-supervision example: {feature.get('example_id', '<unknown>')}")
        if any(value != 1 for value in attention):
            raise E018DataError("un-padded feature attention mask must contain only ones")
        ids.append(input_ids + [pad_token_id] * (width - length))
        masks.append(attention + [0] * (width - length))
        labels.append(label_ids + [-100] * (width - length))
    if any(not (len(i) == len(m) == len(l) == width) for i, m, l in zip(ids, masks, labels)):
        raise E018DataError("collated tensor shape mismatch")
    batch: dict[str, Any] = {"input_ids": ids, "attention_mask": masks, "labels": labels}
    if return_tensors == "pt":
        try:
            import torch
        except ImportError as exc:
            raise E018DataError("return_tensors='pt' requires torch") from exc
        batch = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return batch


def validate_padded_batch(batch: dict[str, Any], *, pad_token_id: int) -> None:
    """Validate padding by attention-mask position, never by token identity."""
    try:
        input_ids, attention, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        if not hasattr(input_ids, "shape"):
            shapes = {(len(input_ids), len(input_ids[0])), (len(attention), len(attention[0])), (len(labels), len(labels[0]))}
            if len(shapes) != 1: raise E018DataError("input_ids, attention_mask, and labels shapes differ")
            for ids, mask, labs in zip(input_ids, attention, labels):
                if any(value not in (0, 1) for value in mask): raise E018DataError("attention_mask must contain only 0/1")
                if not any(value == 1 for value in mask): raise E018DataError("example has no real tokens")
                for token, flag, label in zip(ids, mask, labs):
                    if flag == 0 and (token != pad_token_id or label != -100): raise E018DataError("invalid list padding invariant")
            return
        shape = input_ids.shape
        if attention.shape != shape or labels.shape != shape:
            raise E018DataError("input_ids, attention_mask, and labels shapes differ")
        if not ((attention == 0) | (attention == 1)).all():
            raise E018DataError("attention_mask must contain only 0/1")
        padding = attention == 0
        real = attention == 1
        if not (labels[padding] == -100).all():
            raise E018DataError("padding labels must be -100")
        if not (input_ids[padding] == pad_token_id).all():
            raise E018DataError("padding input_ids must use pad_token_id")
        if not real.any(dim=-1).all():
            raise E018DataError("example has no real tokens")
    except AttributeError as exc:
        raise E018DataError("padded-batch invariant requires tensor inputs") from exc


class CompactDataCollator:
    """Trainer-compatible adapter around the tested E018 collator."""
    def __init__(self, tokenizer: Any, *, return_tensors: str = "pt") -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise E018DataError("E018 requires a tokenizer pad_token_id before training")
        self.pad_token_id = int(pad_token_id)
        self.return_tensors = return_tensors

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_compact_features(features, pad_token_id=self.pad_token_id, return_tensors=self.return_tensors)


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def resolve_frozen_root(manifest_path: str | Path) -> Path:
    return Path(manifest_path).resolve().parent


def load_core(manifest_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = resolve_frozen_root(manifest_path)
    manifest = load_manifest(manifest_path)
    # E019 deliberately reuses the E018 trainer/evaluator contracts but has a
    # different frozen training composition. Keep the E018 default immutable
    # while allowing an explicit manifest to name the E019 train/validation
    # files; the frozen E018 test/challenge remain separate evaluation inputs.
    if manifest.get("identity") == "E019-contradiction-supervision-v1":
        files = manifest.get("training_files", {})
        if set(files) != {"train", "validation"}:
            raise E018DataError("E019 manifest must declare train and validation files")
        return {
            "train": load_jsonl(root / files["train"], expected_split="train"),
            "validation": load_jsonl(root / files["validation"], expected_split="validation"),
            "test": [],
        }
    result = {s: load_jsonl(root / "core" / f"E018-candidate-{s}.jsonl", expected_split=s) for s in ("train", "validation", "test")}
    for split, rows in result.items():
        counts = Counter(r["target"]["status"] for r in rows)
        if counts != Counter(CORE_COUNTS[split]):
            raise E018DataError(f"{split} counts mismatch: {dict(counts)}")
    return result


def load_challenge(manifest_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = resolve_frozen_root(manifest_path)
    return (load_jsonl(root / "challenge" / "contradiction-challenge.jsonl"),
            load_jsonl(root / "challenge" / "matched-supported-controls.jsonl"))


def parse_prediction(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw.strip())
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    if not isinstance(value, dict) or set(value) != {"status", "evidence_ids", "claim"}:
        return None, "invalid_schema"
    try:
        validate_row({"example_id": "prediction", "question": "_", "context": "_", "evidence": [], "target": value})
    except E018DataError as exc:
        return None, f"invalid_target:{exc}"
    return value, None


def classification_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    invalid = 0
    for row in rows:
        actual = row["target"]["status"]
        pred = row.get("prediction", {}).get("status") if isinstance(row.get("prediction"), dict) else None
        if pred not in LABELS:
            invalid += 1
        else:
            confusion[actual][pred] += 1
    valid = len(rows) - invalid
    per_class = {}
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[a][label] for a in LABELS if a != label)
        fn = sum(confusion[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
        f1s.append(f1)
    correct = sum(confusion[l][l] for l in LABELS)
    return {"n": len(rows), "valid": valid, "invalid": invalid, "accuracy": correct / len(rows) if rows else 0.0,
            "macro_f1": sum(f1s) / len(f1s), "per_class": per_class, "confusion_matrix": confusion}


def challenge_metrics(challenge_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = classification_metrics(challenge_rows); k = classification_metrics(control_rows)
    def status(row):
        prediction = row.get("prediction")
        return prediction.get("status") if isinstance(prediction, dict) else None
    c_correct = sum(status(r) == r["target"]["status"] for r in challenge_rows)
    contradiction_recall = sum(status(r) == "CONTRADICTED" for r in challenge_rows) / len(challenge_rows) if challenge_rows else 0.0
    pairs = []
    controls = {r.get("challenge_id"): r for r in control_rows}
    for r in challenge_rows:
        q = controls.get(r.get("challenge_id"))
        if q:
            pairs.append({"challenge_id": r.get("challenge_id"), "challenge_predicted_contradicted": status(r) == "CONTRADICTED", "control_predicted_supported": status(q) == "SUPPORTED", "paired_correct": status(r) == "CONTRADICTED" and status(q) == "SUPPORTED"})
    false_labels = Counter(status(r) for r in challenge_rows if status(r) != "CONTRADICTED")
    return {"challenge": c, "controls": k, "contradiction_accuracy": c_correct / len(challenge_rows) if challenge_rows else 0.0, "contradiction_recall": contradiction_recall, "false_label_distribution": dict(false_labels), "paired_discrimination": sum(p["paired_correct"] for p in pairs) / len(pairs) if pairs else 0.0, "pairs": pairs}


def write_run_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {p}")
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
