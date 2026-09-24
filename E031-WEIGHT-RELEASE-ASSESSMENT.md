# E031 learned-artifact release assessment

Date: 2026-09-24  
Frozen release: `releases/e028_evidence_classifier_v1/final/`

## Decision

Both learned artifacts are classified as
`LEGAL_REVIEW_RECOMMENDED_BUT_NOT_EXPLICITLY_PROHIBITED`.

This is not a legal clearance and not a finding of infringement. It records
that the upstream terms reviewed are silent on the specific learned-weight
question, so the artifacts are held in Track B while Track A research
materials are published.

| Artifact | Frozen path | SHA-256 | Size | Status |
| --- | --- | --- | ---: | --- |
| LoRA adapter | `final/adapter/adapter_model.safetensors` | `b8a6dfd754f1cb766d710697d03c5b26a8316abc085c9c1dc999b942e81e5f8c` | 174,655,536 bytes | Legal review recommended; not expressly prohibited |
| Classifier head | `final/classifier/classifier_head.safetensors` | `cea444d633653afbb0697c0357717fc0f6014b25bc0ca4965660807aee77c5a4` | 49,300 bytes | Legal review recommended; not expressly prohibited |

## Basis for the adapter decision

The adapter was learned from the 1,576-row E028 union: QASPER/S2ORC-backed
E023 seed rows plus selected real and synthetic VitaminC augmentation rows.
Qwen3-8B-Base is Apache-2.0, but the Qwen materials do not expressly settle
whether this separate LoRA artifact is an adapted model, nor do the QASPER,
S2ORC, or VitaminC materials expressly settle whether data terms attach to its
parameters. The adapter contains no raw training text.

## Basis for the classifier-head decision

The head is a separate 4,096-to-3 linear classifier trained as part of the
same E028 supervised objective. It is numerically smaller and does not embed
raw rows, but it shares the same training-source provenance and unresolved
learned-parameter interpretation. It must therefore receive the same status,
not an unsupported independent license.

## Release handling

- Track A excludes both binary files and publishes aggregate evidence only.
- Track B retains both files locally, checksummed and unchanged, for counsel
  review or a later authorized release.
- The public study must say “model weights pending license review,” not “model
  weights are prohibited” and not “model weights are cleared.”
- Qwen base weights remain excluded regardless of the Track B decision.

