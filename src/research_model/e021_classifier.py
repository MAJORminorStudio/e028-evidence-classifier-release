"""E021 direct three-way evidence-consistency classifier.

The wrapper deliberately keeps the frozen E019 prompt/data contract separate
from textual answer generation.  It pools the final real prompt-token hidden
state and predicts the fixed integer status mapping directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

from .e018_compact import ID_TO_LABEL, LABELS, LABEL_TO_ID, E018DataError, prompt_text


CLASS_MAPPING = dict(LABEL_TO_ID)
POOLING = "last_non_padding_token"


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final real-token index for either left or right padding."""
    if attention_mask.ndim != 2:
        raise E018DataError(f"attention_mask must be [batch,time], got {tuple(attention_mask.shape)}")
    mask = attention_mask.to(dtype=torch.long)
    lengths = mask.sum(dim=1)
    if bool((lengths <= 0).any().item()):
        raise E018DataError("every classifier example must contain at least one real prompt token")
    return lengths - 1 if bool((mask[:, -1] == 1).all().item()) else _mixed_padding_indices(mask)


def _mixed_padding_indices(mask: torch.Tensor) -> torch.Tensor:
    """Find the last one without assuming a padding side."""
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    return positions.masked_fill(mask == 0, -1).max(dim=1).values


def pool_last_non_padding(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise E018DataError(f"hidden states must be [batch,time,hidden], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise E018DataError("hidden states and attention mask do not share batch/time dimensions")
    indices = last_non_padding_indices(attention_mask)
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, indices, :]


def _base_causal_model(backbone: nn.Module) -> nn.Module:
    """Unwrap PEFT while retaining the same LoRA-instrumented modules."""
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    if hasattr(base, "model") and isinstance(base.model, nn.Module):
        return base.model
    if hasattr(base, "transformer") and isinstance(base.transformer, nn.Module):
        return base.transformer
    return base


def _last_hidden_state(backbone: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Use the transformer body directly so all intermediate hidden states are not retained."""
    body = _base_causal_model(backbone)
    body_kwargs = dict(kwargs)
    body_kwargs.update({"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False, "return_dict": True})
    try:
        outputs = body(**body_kwargs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden
    except (AttributeError, TypeError, ValueError):
        # Some compatible wrappers expose only the causal-LM forward.  The
        # fallback remains correct, although it may retain more output state.
        pass
    outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True, **kwargs)
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise E018DataError("backbone returned neither last_hidden_state nor hidden_states")
        return hidden
    return hidden_states[-1]


class E021Classifier(nn.Module):
    """Qwen causal backbone plus a direct three-way classifier head."""

    def __init__(self, backbone: nn.Module, *, hidden_size: int | None = None, num_labels: int = 3) -> None:
        super().__init__()
        self.backbone = backbone
        config_hidden = getattr(getattr(backbone, "config", None), "hidden_size", None)
        resolved_hidden = int(hidden_size if hidden_size is not None else config_hidden or 0)
        if resolved_hidden <= 0:
            raise E018DataError("backbone config does not expose a positive hidden_size")
        if num_labels != len(LABELS):
            raise E018DataError(f"E021 requires exactly {len(LABELS)} labels")
        self.hidden_size = resolved_hidden
        self.num_labels = int(num_labels)
        self.pooling = POOLING
        self.class_mapping = dict(CLASS_MAPPING)
        self.classifier_head = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: Any):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden = _last_hidden_state(self.backbone, input_ids, attention_mask, **kwargs)
        pooled = pool_last_non_padding(hidden, attention_mask)
        # Keep the small head in its declared dtype while avoiding a dtype
        # mismatch when a quantized/bfloat16 backbone feeds the head.
        pooled = pooled.to(dtype=self.classifier_head.weight.dtype)
        logits = self.classifier_head(pooled).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.to(device=logits.device, dtype=torch.long))
        return SequenceClassifierOutput(loss=loss, logits=logits)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None, every_n_layers: int = 1, offload: bool = False) -> None:
        method = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if method is None:
            raise AttributeError("backbone does not support gradient checkpointing")
        if gradient_checkpointing_kwargs is None:
            method(every_n_layers=every_n_layers, offload=offload)
        else:
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs, every_n_layers=every_n_layers, offload=offload)

    def gradient_checkpointing_disable(self) -> None:
        method = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if method is None:
            raise AttributeError("backbone does not support disabling gradient checkpointing")
        method()


def classifier_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    classifier = sum(int(parameter.numel()) for name, parameter in trainable.items() if name.startswith("classifier_head."))
    lora = sum(int(parameter.numel()) for name, parameter in trainable.items() if ".lora_" in name or name.startswith("lora_"))
    other = sum(int(parameter.numel()) for name, parameter in trainable.items() if not (name.startswith("classifier_head.") or ".lora_" in name or name.startswith("lora_")))
    return {"total_parameters": total, "trainable_parameters": sum(int(p.numel()) for p in trainable.values()), "trainable_lora_parameters": lora, "trainable_classifier_parameters": classifier, "other_trainable_parameters": other}


def classifier_config_payload(*, base_revision: str, tokenizer_revision: str, hidden_size: int, max_seq_length: int) -> dict[str, Any]:
    return {
        "identity": "E021-explicit-three-way-classifier-v1",
        "base_model": "Qwen/Qwen3-8B-Base",
        "model_revision": base_revision,
        "tokenizer_revision": tokenizer_revision,
        "hidden_size": int(hidden_size),
        "num_labels": len(LABELS),
        "class_mapping": CLASS_MAPPING,
        "id_to_label": {str(index): label for index, label in ID_TO_LABEL.items()},
        "pooling": POOLING,
        "prompt_renderer": "research_model.e018_compact.prompt_text",
        "add_special_tokens": False,
        "textual_answer_appended": False,
        "max_seq_length": int(max_seq_length),
        "loss": "torch.nn.CrossEntropyLoss",
        "class_weighting": "none",
    }


def save_classifier_checkpoint(model: E021Classifier, tokenizer: Any, checkpoint_dir: str | Path, config_payload: Mapping[str, Any], *, adapter_name: str = "default") -> Path:
    """Save PEFT adapter, classifier head, tokenizer, and reload metadata."""
    from safetensors.torch import save_file

    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    adapter_dir = root / "adapter"
    model.backbone.save_pretrained(adapter_dir, safe_serialization=True, selected_adapters=[adapter_name] if adapter_name else None)
    tokenizer.save_pretrained(root / "tokenizer")
    head_state = {key: value.detach().cpu().contiguous() for key, value in model.classifier_head.state_dict().items()}
    save_file(head_state, str(root / "classifier_head.safetensors"))
    (root / "E021-CLASSIFIER-CONFIG.json").write_text(json.dumps(dict(config_payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "E021-CLASS-MAPPING.json").write_text(json.dumps({"label_to_id": CLASS_MAPPING, "id_to_label": {str(i): label for i, label in ID_TO_LABEL.items()}}, indent=2) + "\n", encoding="utf-8")
    return root


def load_classifier_head(model: E021Classifier, checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> E021Classifier:
    from safetensors.torch import load_file

    root = Path(checkpoint_dir)
    state = load_file(str(root / "classifier_head.safetensors"), device=str(map_location))
    model.classifier_head.load_state_dict(state)
    return model


def prompt_for_classifier(row: Mapping[str, Any]) -> str:
    """Named helper used by training/evaluation to make the no-label contract explicit."""
    return prompt_text(dict(row))
