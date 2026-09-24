#!/usr/bin/env python3
"""E022 canonical identifiable-input contract.

This module reads frozen E018/E019 records and creates a new, explicit
claim/evidence representation.  It never edits the source records and never
renders labels or transformation metadata into the model prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MAX_SEQ_LENGTH = 4608
SEED = 20260923

E019_MANIFEST = ROOT / "experiments/E019_contradiction_supervision_2026-09-22/E019-DATASET-MANIFEST.json"
E018_ROOT = ROOT / "experiments/E018_evidence_support_classification_2026-09-21/E018-final-frozen-v1"
E021_PROBES = ROOT / "experiments/E021_explicit_classifier_2026-09-23/E021-FROZEN-TRAINING-PROBES.json"
TOKENIZER_DIR = ROOT / "experiments/E021_explicit_classifier_2026-09-23/E021-TRAINING-OUTPUT/e021-explicit-classifier-base-qlora/checkpoint-68/tokenizer"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalized(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def row_id(row: dict[str, Any], role: str | None = None) -> str:
    base = str(row.get("example_id") or row.get("challenge_id") or row.get("control_id") or "<missing-id>")
    if role in {"challenge", "matched_controls"}:
        return f"{base}:{role}"
    return base


def status(row: dict[str, Any]) -> str:
    return str(row.get("target", {}).get("status", "<missing>"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_frozen_sources() -> dict[str, Any]:
    from research_model.e018_compact import load_challenge, load_core, load_jsonl as load_validated_jsonl

    e019 = load_core(E019_MANIFEST)
    core = load_validated_jsonl(E018_ROOT / "core/E018-candidate-test.jsonl", expected_split="test")
    challenge, controls = load_challenge(E018_ROOT / "E018-FINAL-MANIFEST.json")
    return {"e019_train": e019["train"], "e019_legacy_validation": e019["validation"], "core_test": core, "challenge": challenge, "matched_controls": controls}


def source_group(row: dict[str, Any], role: str) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    paper = provenance.get("paper_id") or provenance.get("source_group")
    if paper:
        return f"paper:{paper}"
    parent = provenance.get("parent_example_id")
    if parent:
        return f"parent:{parent}"
    question = provenance.get("question_id")
    if question:
        return f"question:{question}"
    return f"record:{row_id(row, role)}"


def compact_provenance(row: dict[str, Any]) -> dict[str, Any]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    keys = ("paper_id", "source_group", "source_family", "question_id", "parent_example_id", "parent_split", "source_split", "source")
    return {key: provenance[key] for key in keys if key in provenance}


def evidence_items(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("evidence") if isinstance(row.get("evidence"), list) else []
    items = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        identifier = str(item.get("id") or f"evidence_{index}")
        items.append({"id": identifier, "text": text})
    return items


def canonical_claim_and_reason(row: dict[str, Any]) -> tuple[str, str, str | None]:
    label = status(row)
    target_claim = row.get("target", {}).get("claim")
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    transformed = provenance.get("transformed_proposition")
    if label in {"SUPPORTED", "CONTRADICTED"}:
        claim = str(target_claim or "").strip()
        if not claim:
            return "", "target.claim", "DROP_MISSING_CLAIM"
        if label == "CONTRADICTED" and transformed:
            if normalized(claim) != normalized(str(transformed)):
                return claim, "target.claim", "DROP_OTHER_IDENTIFIABILITY_FAILURE"
            return claim, "provenance.transformed_proposition=target.claim", None
        return claim, "target.claim", None
    if label == "INSUFFICIENT_EVIDENCE":
        claim = str(row.get("question", "")).strip()
        if not claim:
            return "", "question", "DROP_MISSING_CLAIM"
        return claim, "question_as_judged_proposition", None
    return "", "target.status", "DROP_OTHER_IDENTIFIABILITY_FAILURE"


def canonical_evidence_and_reason(row: dict[str, Any]) -> tuple[str, list[dict[str, str]], str, str | None]:
    label = status(row)
    items = evidence_items(row)
    if label in {"SUPPORTED", "CONTRADICTED"}:
        if not items:
            return "", [], "row.evidence", "DROP_MISSING_EVIDENCE"
        required_ids = {str(item) for item in (row.get("target", {}).get("evidence_ids", []) or [])}
        available_ids = {item["id"] for item in items}
        if required_ids - available_ids:
            return "", [], "row.evidence", "DROP_MISSING_EVIDENCE"
        rendered = "\n\n".join(f"[{item['id']}] {item['text']}" for item in items)
        return rendered, items, "row.evidence.text", None
    context = str(row.get("context", "")).strip()
    if not context:
        return "", [], "row.context", "DROP_MISSING_EVIDENCE"
    item = {"id": "context", "text": context}
    return f"[context] {context}", [item], "row.context", None


def canonical_prompt(claim: str, evidence: str) -> str:
    return (
        "Determine the relationship between the claim and the evidence.\n"
        "The evidence may support the claim, contradict the claim, or be insufficient to decide.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Choose exactly one class: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE.\n"
    )


def canonicalize(row: dict[str, Any], *, role: str, source_split: str) -> tuple[dict[str, Any] | None, str | None]:
    claim, claim_source, claim_reason = canonical_claim_and_reason(row)
    if claim_reason:
        return None, claim_reason
    evidence, items, evidence_source, evidence_reason = canonical_evidence_and_reason(row)
    if evidence_reason:
        return None, evidence_reason
    label = status(row)
    identifier = row_id(row, role)
    raw_id = row_id(row)
    record = {
        "example_id": identifier,
        "source_example_id": raw_id,
        "source_role": role,
        "source_split": source_split,
        "source_group": source_group(row, role),
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "canonical_claim": claim,
        "canonical_claim_source": claim_source,
        "canonical_evidence": evidence,
        "canonical_evidence_source": evidence_source,
        "canonical_evidence_ids": [item["id"] for item in items],
        "canonical_evidence_sha256": sha256_text(evidence),
        "canonical_prompt": canonical_prompt(claim, evidence),
        "source_target_claim": row.get("target", {}).get("claim"),
        "source_target_evidence_ids": row.get("target", {}).get("evidence_ids", []),
        "source_record_sha256": sha256_json(row),
        "source_provenance": compact_provenance(row),
    }
    record["canonical_prompt_sha256"] = sha256_text(record["canonical_prompt"])
    record["raw_transformed_proposition_present"] = bool((row.get("provenance") or {}).get("transformed_proposition"))
    return record, None


def canonicalize_split(rows: Iterable[dict[str, Any]], *, role: str, source_split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained, dropped = [], []
    for row in rows:
        record, reason = canonicalize(row, role=role, source_split=source_split)
        if record is not None:
            retained.append(record)
        else:
            dropped.append({"source_example_id": row_id(row), "source_role": role, "source_split": source_split, "label": status(row), "reason": reason, "source_record_sha256": sha256_json(row)})
    return retained, dropped


def class_counts(rows: Iterable[dict[str, Any]], key: str = "label") -> dict[str, int]:
    counts = Counter(str(row.get(key)) for row in rows)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
