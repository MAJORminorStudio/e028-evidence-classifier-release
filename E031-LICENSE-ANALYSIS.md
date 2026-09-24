# E031 license analysis

Date: 2026-09-24  
Decision vocabulary: `EXPRESSLY_PERMITTED`, `EXPRESSLY_PROHIBITED`,
`LICENSE_SILENT_ON_TRAINED_WEIGHTS`, and
`LEGAL_REVIEW_RECOMMENDED_BUT_NOT_EXPLICITLY_PROHIBITED`.

This is a source and publication audit, not legal advice. It separates the
license of a dataset or code repository from the unresolved question of what
terms, if any, attach to numerical parameters learned from that data.

## Source-by-source result

| Source | E028 role | Authoritative terms found | Trained adapter/head result |
| --- | --- | --- | --- |
| Qwen3-8B-Base | Base model | Apache-2.0; pinned revision only | The upstream terms do not expressly resolve this separately trained adapter/head. |
| QASPER | 976-row E023 seed | CC BY 4.0 dataset terms; underlying papers extracted from S2ORC | `LICENSE_SILENT_ON_TRAINED_WEIGHTS` |
| S2ORC | QASPER dependency | ODC-By 1.0 database terms and source-paper rights | `LICENSE_SILENT_ON_TRAINED_WEIGHTS` |
| VitaminC real annotations | 300 selected E028 augmentation rows | Wikipedia/article terms may apply | `LICENSE_SILENT_ON_TRAINED_WEIGHTS` |
| VitaminC synthetic annotations | 300 selected E028 augmentation rows | Uses or modifies FEVER examples; DATA_LICENSE describes Wikipedia/CC BY-SA fallback for annotations | `LICENSE_SILENT_ON_TRAINED_WEIGHTS` |
| FEVER | E025 evaluation-only and VitaminC dependency | Wikipedia-derived official benchmark material | No E028 gradient contribution; does not govern a new E028 weight release on this record. |
| FEVER2 | E025 evaluation-only | Official task material | No E028 gradient contribution. |
| SciFact | E025 evaluation-only | Claims/evidence CC BY 4.0; abstracts S2ORC ODC-By 1.0; code Apache-2.0 | No E028 gradient contribution. |
| AVeriTeC | E025 evaluation-only | CC BY-NC 4.0 | No E028 gradient contribution. |
| FactKG | E025 evaluation-only | CC BY-NC-SA 4.0 | No E028 gradient contribution. |
| Project code | Code and documentation | Project Apache-2.0 (`/LICENSE`) | Does not override upstream data terms. |

## Exact training provenance

E028 trained on 1,576 rows:

- 976 E023 seed rows: 720 `SUPPORTED`, 149 `CONTRADICTED`, and 107
  `INSUFFICIENT_EVIDENCE`; all are QASPER-backed rows from the QASPER train
  split. E023 has 418 unique `paper:<arxiv_id>` source groups.
- 400 E026 VitaminC augmentation rows: 200 `CONTRADICTED` and 200
  `INSUFFICIENT_EVIDENCE`, with 100 real and 100 synthetic rows per class.
- 200 E028 VitaminC augmentation rows: 200 `SUPPORTED`, with 100 real and
  100 synthetic rows.

The exact row-level source IDs, source-group inventory, canonicalization
fields, and source-file hashes are in
`E031-TRAINING-PROVENANCE.json`. The raw JSONL is retained locally for
provenance checking and is not a Track A publication artifact.

## Questions A–D

### A. Are the learned weights derivative works?

The authoritative sources reviewed do not resolve this question for these
specific numerical parameters. The adapter is a LoRA parameterization trained
against Qwen representations; the classifier head is a separate 4,096-by-3
linear head. Whether either is an Adapted Material/derivative work of the
base model, of source data, or both is jurisdiction- and fact-dependent and
requires counsel if released.

### B. Is redistribution expressly permitted or prohibited?

No reviewed training-source document expressly permits redistribution of this
adapter/head pair, and no reviewed training-source document expressly
prohibits it. The correct recorded state is therefore
`LICENSE_SILENT_ON_TRAINED_WEIGHTS`, escalated to
`LEGAL_REVIEW_RECOMMENDED_BUT_NOT_EXPLICITLY_PROHIBITED` for publication.
This is more precise than treating silence as either permission or a ban.

### C. Do evaluation datasets affect the learned-weight decision?

No. FEVER, FEVER2, SciFact, AVeriTeC, and FactKG were E025 evaluation-only
inputs for this release. They produced benchmark metrics but no E028 gradient
updates. Their raw rows are not included in Track A.

### D. Is share-alike triggered for the learned weights?

No authoritative source reviewed states that its data-annotation share-alike
terms attach to numerical learned adapter/head parameters. VitaminC's data
notice includes a CC BY-SA 3.0 fallback for annotations where Wikipedia terms
are unavailable, but it does not expressly characterize trained weights. Do
not represent the weights as Apache-2.0 or CC BY-SA without legal review.

## Exact unresolved proposition

> Does distributing the 174,655,536-byte E028 LoRA adapter and 49,300-byte
> classifier head, without raw source rows, paper text, or Qwen base weights,
> constitute distribution of Adapted Material or otherwise trigger a source
> data license condition under the QASPER/S2ORC and VitaminC terms?

The reviewed authoritative materials do not answer this proposition. This is
the sole material publication blocker for Track B.

