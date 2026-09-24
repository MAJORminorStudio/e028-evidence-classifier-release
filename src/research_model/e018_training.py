"""Shared E018 production Trainer compatibility helpers."""
from __future__ import annotations

import inspect
import math
from typing import Any


def planned_optimizer_steps(
    num_examples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    world_size: int = 1,
) -> int:
    """Mirror Transformers Trainer's finite-dataloader step calculation."""
    if min(num_examples, per_device_batch_size, gradient_accumulation_steps, world_size) <= 0:
        raise ValueError("training dimensions must be positive")
    batches = math.ceil(num_examples / (per_device_batch_size * world_size))
    updates_per_epoch = max(
        batches // gradient_accumulation_steps
        + int(batches % gradient_accumulation_steps > 0),
        1,
    )
    return math.ceil(num_train_epochs * updates_per_epoch)


def equivalent_warmup_steps(total_optimizer_steps: int, warmup_ratio: float) -> int:
    """Convert the frozen ratio using the historical HF ceil convention."""
    if total_optimizer_steps < 0 or not 0 <= warmup_ratio <= 1:
        raise ValueError("invalid warmup inputs")
    return math.ceil(total_optimizer_steps * warmup_ratio)


def build_training_arguments(config: dict[str, Any], output_dir: str, training_arguments_cls: Any):
    """Build the exact production TrainingArguments with 5.17 compatibility.

    Transformers 5.17 removed ``warmup_ratio`` from the constructor.  The
    frozen E018 ratio is therefore converted to the equivalent integer step
    count using the same finite-dataloader calculation used by Trainer.
    """
    total_steps = planned_optimizer_steps(
        config["train_examples"],
        config["per_device_train_batch_size"],
        config["gradient_accumulation_steps"],
        config["num_train_epochs"],
        config.get("world_size", 1),
    )
    warmup_steps = equivalent_warmup_steps(total_steps, config["warmup_ratio"])
    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "num_train_epochs": config["num_train_epochs"],
        "warmup_steps": warmup_steps,
        "weight_decay": config["weight_decay"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "optim": config["optim"],
        "bf16": config["bf16"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "logging_steps": config["logging_steps"],
        "eval_strategy": "steps",
        "eval_steps": config["eval_steps"],
        "save_strategy": "steps",
        "save_steps": config["save_steps"],
        "save_total_limit": config["save_total_limit"],
        "report_to": [],
        "seed": config["seed"],
    }
    supported = set(inspect.signature(training_arguments_cls).parameters)
    unknown = set(kwargs) - supported
    if unknown:
        raise TypeError(f"unsupported TrainingArguments fields: {sorted(unknown)}")
    args = training_arguments_cls(**kwargs)
    return args, {"total_optimizer_steps": total_steps, "warmup_steps": warmup_steps}
