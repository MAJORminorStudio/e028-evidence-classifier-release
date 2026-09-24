#!/usr/bin/env python3
"""E022 checkpoint serialization around the established E021 classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from research_model.e021_classifier import E021Classifier, classifier_parameter_counts, load_classifier_head


LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
CLASS_MAPPING = {label: index for index, label in enumerate(LABELS)}


def save_e022_checkpoint(model: E021Classifier, tokenizer: Any, checkpoint_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    from safetensors.torch import save_file

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(root / "adapter", safe_serialization=True, selected_adapters=["default"])
    tokenizer.save_pretrained(root / "tokenizer")
    state = {key: value.detach().cpu().contiguous() for key, value in model.classifier_head.state_dict().items()}
    save_file(state, str(root / "classifier_head.safetensors"))
    (root / "E022-CLASSIFIER-CONFIG.json").write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "E022-CLASS-MAPPING.json").write_text(json.dumps({"label_to_id": CLASS_MAPPING, "id_to_label": {str(index): label for label, index in CLASS_MAPPING.items()}}, indent=2) + "\n", encoding="utf-8")
    return root


def load_e022_model(model_dir: str | Path, checkpoint_dir: str | Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    backbone = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    backbone = PeftModel.from_pretrained(backbone, str(Path(checkpoint_dir) / "adapter"), local_files_only=True)
    hidden_size = int(backbone.get_base_model().config.hidden_size)
    model = E021Classifier(backbone, hidden_size=hidden_size)
    device = next(parameter.device for parameter in model.backbone.parameters() if parameter.device.type != "meta")
    model.classifier_head.to(device=device, dtype=torch.float32)
    load_classifier_head(model, checkpoint_dir, map_location="cpu")
    model.eval()
    return model, tokenizer, device

