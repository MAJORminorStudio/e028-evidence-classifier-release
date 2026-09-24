# Qwen3-8B Evidence Classifier — E028 v1.0.0 research release

This is the public Track A research release for the frozen E028 evidence
classifier. It contains code, aggregate metrics, figures, sanitized manifests,
source provenance, and publication audits. It intentionally does not contain
Qwen base weights, raw training/evaluation rows, the E028 LoRA adapter, or the
separate `Linear(4096, 3)` classifier head.

Model weights are pending license review. The adapter and classifier head are
staged locally in the frozen E028 package and their hashes are recorded in the
E031 audit, but they are not part of this public repository.

The input is a `CLAIM` and supplied `EVIDENCE`.  The output is exactly one of
`SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`.  The model classifies
the relationship relative to the supplied evidence.  It does not retrieve
evidence and does not independently establish real-world truth beyond that
evidence.

The selected frozen checkpoint is E028 epoch 2 / global step 198. The pinned
base revision, adapter/head hashes, formatter, scripts, manifests, metrics,
and figures are recorded in `RELEASE-MANIFEST.json`, `SHA256SUMS`, and the
E031 provenance audit.

The E031 decision is `RELEASE_PARTIAL_TRACK_A_PUBLIC_TRACK_B_HELD`. The
historical E030 manifest is retained for reproducibility; the E031 documents
are the current publication status.

Start with `E031-REPORT.md`, `E031-TRAINING-PROVENANCE.json`,
`E031-LICENSE-ANALYSIS.md`, `E031-DATA-EXCLUSION-AUDIT.md`,
`TECHNICAL-REPORT.md`, and `REPRODUCIBILITY.md`.
