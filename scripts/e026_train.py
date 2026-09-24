#!/usr/bin/env python3
"""Train exactly one preregistered E026 QLoRA classifier candidate."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "e023"))
sys.path.insert(0, str(ROOT / "src"))

from e022_common import LABELS, class_counts, load_jsonl  # noqa: E402
from e022_model import E021Classifier, classifier_parameter_counts, save_e022_checkpoint  # noqa: E402
from research_model.e018_training import build_training_arguments  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def encode_rows(tokenizer, rows: list[dict[str, Any]], max_seq_length: int) -> list[dict[str, Any]]:
    features = []
    for row in rows:
        ids = list(tokenizer(row["canonical_prompt"], add_special_tokens=False, truncation=False)["input_ids"])
        if not ids:
            raise ValueError(f"empty E026 prompt: {row['example_id']}")
        if len(ids) > max_seq_length:
            raise ValueError(f"E026 prompt exceeds max_seq_length: {row['example_id']} length={len(ids)}")
        features.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": int(row["label_id"]), "example_id": row["example_id"]})
    return features


class PromptDataset:
    def __init__(self, features: list[dict[str, Any]]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index]


class E026DataCollator:
    def __init__(self, tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("E026 requires tokenizer.pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        return {
            "input_ids": torch.tensor([feature["input_ids"] + [self.pad_token_id] * (width - len(feature["input_ids"])) for feature in features], dtype=torch.long),
            "attention_mask": torch.tensor([feature["attention_mask"] + [0] * (width - len(feature["attention_mask"])) for feature in features], dtype=torch.long),
            "labels": torch.tensor([int(feature["labels"]) for feature in features], dtype=torch.long),
        }


def class_metrics(predictions, labels) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(labels)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, guess in zip(labels.tolist(), predicted.tolist()):
        confusion[int(actual), int(guess)] += 1
    f1s = []
    metrics: dict[str, Any] = {"accuracy": float((predicted == labels).mean())}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"{label.lower()}_precision"] = precision
        metrics[f"{label.lower()}_recall"] = recall
        metrics[f"{label.lower()}_f1"] = f1
        f1s.append(f1)
    metrics["macro_f1"] = float(sum(f1s) / len(f1s))
    metrics["confusion_matrix"] = confusion.tolist()
    metrics["predicted_class_counts"] = {label: int((predicted == index).sum()) for index, label in enumerate(LABELS)}
    return metrics


def gpu_snapshot() -> dict[str, str]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        return {"raw": result.stdout.strip()}
    except OSError:
        return {"raw": "unavailable"}


def train(config: dict[str, Any]) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed

    output = Path(config["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty E026 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(Path(config["train_path"]))
    validation_rows = load_jsonl(Path(config["validation_path"]))
    set_seed(int(config["seed"]))
    random.seed(int(config["seed"]))

    model_source = str(config["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "right"
    train_features = encode_rows(tokenizer, train_rows, int(config["max_seq_length"]))
    validation_features = encode_rows(tokenizer, validation_rows, int(config["max_seq_length"]))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    backbone = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    hidden_size = int(backbone.config.hidden_size)
    if hidden_size != 4096:
        raise RuntimeError(f"E026 expected hidden size 4096, got {hidden_size}")
    backbone.config.use_cache = False
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type=TaskType.CAUSAL_LM)
    backbone = get_peft_model(backbone, lora_config)
    model = E021Classifier(backbone, hidden_size=hidden_size)
    backbone_device = next(parameter.device for parameter in model.backbone.parameters() if parameter.device.type != "meta")
    model.classifier_head.to(device=backbone_device, dtype=torch.float32)
    model.train()
    counts = classifier_parameter_counts(model)
    if counts["other_trainable_parameters"] != 0 or counts["trainable_classifier_parameters"] != hidden_size * 3 + 3 or counts["trainable_lora_parameters"] <= 0:
        raise RuntimeError(f"unexpected E026 trainable parameter contract: {counts}")

    train_config = dict(config, train_examples=len(train_rows), world_size=1)
    args, schedule = build_training_arguments(train_config, str(output), TrainingArguments)
    args.remove_unused_columns = False
    boundary_steps = {int(schedule["total_optimizer_steps"] // int(config["num_train_epochs"])), int(schedule["total_optimizer_steps"])}
    boundary_records: list[dict[str, Any]] = []

    def latest_eval_metrics(state) -> dict[str, Any]:
        for item in reversed(state.log_history):
            if "eval_loss" in item:
                return {key: value for key, value in item.items() if key.startswith("eval_")}
        return {}

    class E026Trainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            self.gradient_audit = {"classifier": None, "lora": None, "step": None}
            super().__init__(*trainer_args, **trainer_kwargs)

        def training_step(self, model, inputs, *step_args, **step_kwargs):
            loss = super().training_step(model, inputs, *step_args, **step_kwargs)
            if self.gradient_audit["step"] is None:
                classifier_norm = 0.0
                lora_norm = 0.0
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    value = float(parameter.grad.detach().float().norm().cpu())
                    if name.startswith("classifier_head."):
                        classifier_norm += value
                    elif ".lora_" in name:
                        lora_norm += value
                self.gradient_audit = {"classifier": classifier_norm, "lora": lora_norm, "step": int(self.state.global_step) + 1}
            return loss

        def _save_checkpoint(self, model, trial):
            step = int(self.state.global_step)
            if step not in boundary_steps:
                return
            epoch = 1 if step == min(boundary_steps) else 2
            checkpoint = output / f"checkpoint-{step}"
            validation_metrics = latest_eval_metrics(self.state)
            payload = {
                "identity": "E026-checkpoint-v1",
                "base_model": config["base_model"],
                "model_revision": config["model_revision"],
                "tokenizer_revision": config["tokenizer_revision"],
                "hidden_size": hidden_size,
                "num_labels": 3,
                "class_mapping": {label: index for index, label in enumerate(LABELS)},
                "pooling": "last_non_padding_token",
                "max_seq_length": int(config["max_seq_length"]),
                "loss": "ordinary_cross_entropy",
                "class_weighting": "none",
                "epoch": epoch,
                "global_step": step,
                "validation_metrics": validation_metrics,
                "selection_set": "E026-VALIDATION.jsonl only",
            }
            save_e022_checkpoint(model, tokenizer, checkpoint, payload)
            write_json(checkpoint / "E026-CHECKPOINT-MANIFEST.json", {"identity": "E026-checkpoint-manifest-v1", "epoch": epoch, "global_step": step, "validation_metrics": validation_metrics, "adapter": str((checkpoint / "adapter").resolve()), "classifier_head": str((checkpoint / "classifier_head.safetensors").resolve()), "tokenizer": str((checkpoint / "tokenizer").resolve())})
            boundary_records.append({"epoch": epoch, "global_step": step, "checkpoint": str(checkpoint.resolve()), "validation_metrics": validation_metrics})

    class TelemetryCallback(TrainerCallback):
        def __init__(self):
            self.gpu = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.gpu.append({"step": int(state.global_step), "snapshot": gpu_snapshot()})
            return control

    telemetry = TelemetryCallback()
    trainer = E026Trainer(model=model, args=args, train_dataset=PromptDataset(train_features), eval_dataset=PromptDataset(validation_features), data_collator=E026DataCollator(tokenizer), compute_metrics=lambda prediction: class_metrics(prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions, prediction.label_ids), callbacks=[telemetry])
    run_manifest = {
        "identity": "E026-primary-candidate-run-v1",
        "config": str((ROOT / "E026-TRAINING-CONFIG.json").resolve()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "class_counts": {"train": class_counts(train_rows), "validation": class_counts(validation_rows)},
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "hidden_size": hidden_size,
        "class_mapping": {label: index for index, label in enumerate(LABELS)},
        "pooling": "last_non_padding_token",
        "loss": "ordinary_cross_entropy",
        "class_weighting": "none",
        "canonical_prompt_renderer": "E023 canonical claim + canonical evidence; no answer appended",
        "schedule": schedule,
        "parameter_counts": counts,
        "num_train_epochs": int(config["num_train_epochs"]),
        "epoch_boundary_steps": sorted(boundary_steps),
        "validation_only_checkpoint_selection": "highest development macro-F1; tie lowest development CE; exact tie later epoch",
        "e025_holdout_used_for_selection": False,
    }
    write_json(output / "E026-RUN-MANIFEST.json", run_manifest)
    started = time.time()
    initial_gpu = gpu_snapshot()
    result = trainer.train()
    final_gpu = gpu_snapshot()
    if sorted(record["global_step"] for record in boundary_records) != sorted(boundary_steps):
        raise RuntimeError(f"E026 expected boundary artifacts at {sorted(boundary_steps)}, got {boundary_records}")
    validation_history = sorted(boundary_records, key=lambda item: item["epoch"])

    def selection_key(record: dict[str, Any]):
        metrics = record["validation_metrics"]
        return (float(metrics.get("eval_macro_f1", float("-inf"))), -float(metrics.get("eval_loss", float("inf"))), int(record["epoch"]))

    selected = max(validation_history, key=selection_key)
    selection = {"identity": "E026-development-only-checkpoint-selection-v1", "policy": "highest development macro-F1; tie lowest development CE; exact tie later epoch", "candidates": [{"epoch": item["epoch"], "global_step": item["global_step"], "checkpoint": item["checkpoint"], "development_macro_f1": item["validation_metrics"].get("eval_macro_f1"), "development_ce": item["validation_metrics"].get("eval_loss")} for item in validation_history], "selected_epoch": selected["epoch"], "selected_global_step": selected["global_step"], "selected_checkpoint": selected["checkpoint"]}
    write_json(output / "E026-CHECKPOINT-SELECTION.json", selection)
    training_result = {"identity": "E026-training-result-v1", "train_metrics": dict(result.metrics), "development_history": validation_history, "checkpoint_selection": selection, "final_development_metrics": validation_history[-1]["validation_metrics"], "global_step": int(trainer.state.global_step), "elapsed_seconds": time.time() - started, "gradient_audit": trainer.gradient_audit, "gpu": {"initial": initial_gpu, "samples": telemetry.gpu, "final": final_gpu, "torch_device": str(backbone_device), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None, "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None}}
    write_json(output / "E026-TRAINING-RESULT.json", training_result)
    write_json(output / "E026-VALIDATION-HISTORY.json", validation_history)
    with (output / "E026-TRAINING-LOG.jsonl").open("w", encoding="utf-8") as handle:
        for item in trainer.state.log_history:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "checkpoints": [item["checkpoint"] for item in validation_history], "global_step": trainer.state.global_step, "development_macro_f1": [item["validation_metrics"].get("eval_macro_f1") for item in validation_history], "parameter_counts": counts}, indent=2), flush=True)


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
