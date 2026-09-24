# E031 — Release license resolution and publication unblock

Date: 2026-09-24  
Status: preregistered and execution-only  
Frozen subject: E028 Evidence Classifier v1

## Scope

E031 resolves the publication status of the frozen E028 release. It does not
train, fine-tune, evaluate, regenerate, or alter E028. It does not start E032.
The exact frozen artifacts are identified by the E028 release manifest and
checksums. The Qwen base is referenced by revision only; base weights are not
redistributed.

The audit covers four separate objects:

1. source data and source-text provenance;
2. project code and documentation;
3. base-model terms;
4. derived learned artifacts (the LoRA adapter and classifier head).

## Pre-registered questions

- What exact rows and upstream sources contributed gradients to E028?
- Which E025 benchmark sources were evaluation-only rather than training data?
- Do authoritative upstream materials expressly permit, prohibit, or remain
  silent about redistribution of numerical learned adapter/head parameters?
- Can the research report, code, aggregate metrics, figures, manifests, and
  study page be published without publishing restricted raw rows or learned
  weights?

## Decision rule

If an upstream source expressly prohibits redistribution of the adapter or
head, Track B is prohibited. If the authoritative materials are silent on
learned weights but the interpretation is legally material, Track B is held
for legal review while Track A is published. Track A consists only of
research materials that contain no raw training/evaluation rows, paper text,
Qwen base weights, adapter binary, or classifier-head binary.

## Frozen E028 identity

- Base: `Qwen/Qwen3-8B-Base`, revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Selected checkpoint: epoch 2, global step 198.
- Adapter SHA-256:
  `b8a6dfd754f1cb766d710697d03c5b26a8316abc085c9c1dc999b942e81e5f8c`.
- Classifier-head SHA-256:
  `cea444d633653afbb0697c0357717fc0f6014b25bc0ca4965660807aee77c5a4`.

## Execution constraints

- No GPU, pod, inference, benchmark rerun, or training is authorized.
- Do not modify the frozen E028 weights or release package.
- Preserve exact source hashes and provenance records.
- Publish only after local exclusion and checksum verification.
- Record the final pod inventory even though E031 uses no compute.

