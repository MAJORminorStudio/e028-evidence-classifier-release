# Reproducibility

1. Acquire `Qwen/Qwen3-8B-Base` at revision
   `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` under the upstream license.
2. Acquire the source datasets separately and verify the hashes and roles in
   `manifests/data-sources.json`; do not use the E025 holdout for selection.
3. Prepare JSONL with `scripts/prepare_data.py` and verify the generated
   manifest against the supplied training/development manifests.
4. Use `config/training-config.json`, the exact prompt formatter, seed 17,
   ordinary cross-entropy, two epochs, max length 4,608, effective batch size
   16, learning rate 0.0002, and the recorded LoRA/quantization settings.
   `scripts/train_qwen3_classifier.py` validates this contract and documents
   the audited project trainer invocation; E029 does not execute training.
5. Run structural checks with `scripts/validate_artifact.py`. Actual model
   forward inference requires CUDA-capable memory for the 8B base and a
   separately authorized Track B adapter/head release. The public Track A
   repository does not contain those learned binaries.

The release does not duplicate base weights, raw source datasets, or the full
83,065-row holdout predictions. E028 remains the immutable provenance record.
