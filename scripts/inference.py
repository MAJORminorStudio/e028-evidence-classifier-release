#!/usr/bin/env python3
"""Deterministic GPU inference for the E028 three-way evidence classifier.

The base model must be acquired separately at the exact pinned revision.  The
release contains the LoRA adapter, classifier head, tokenizer, and formatter,
but never redistributes Qwen base weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from peft import PeftModel
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from canonical_formatter import format_claim_evidence

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--claim", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    release = args.release.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(release / "config/tokenizer"), local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    causal = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    backbone = PeftModel.from_pretrained(causal, str(release / "adapter"), local_files_only=True, is_trainable=False)
    body = getattr(backbone.get_base_model(), "model", None) or getattr(backbone.get_base_model(), "transformer", None)
    hidden = int(backbone.get_base_model().config.hidden_size)
    head = nn.Linear(hidden, 3, bias=True)
    head.load_state_dict(load_file(str(release / "classifier/classifier_head.safetensors"), device="cpu"))
    device = next(parameter.device for parameter in body.parameters() if parameter.device.type != "meta")
    text = format_claim_evidence(args.claim, args.evidence)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=4608)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    body.eval(); head.to(device=device, dtype=torch.float32).eval()
    with torch.inference_mode():
        output = body(**encoded, output_hidden_states=True, return_dict=True)
        last = encoded["attention_mask"].sum(dim=1).long() - 1
        hidden_state = output.hidden_states[-1][torch.arange(output.hidden_states[-1].shape[0], device=device), last]
        logits = head(hidden_state).float()
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    index = int(max(range(3), key=lambda i: probabilities[i]))
    print(json.dumps({"label": LABELS[index], "probabilities": dict(zip(LABELS, probabilities)), "prompt": text}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
