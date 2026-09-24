# E028 experiment summary and lineage

| Stage | Finding | Status |
|---|---|---|
| E018–E021 | Early semantic conclusions were made on malformed/non-identifiable input. | Invalidated by the E021 audit. |
| E022 | Corrected identifiable input plus explicit classifier. | Corrected chain begins. |
| E023 | Two-epoch corrected classifier confirmation. | Frozen predecessor. |
| E024 | Vanilla baseline and linear-probe comparison. | Baseline context. |
| E025 | Frozen 83,065-example paired external benchmark. | Evaluation lock. |
| E026 | Three-way generalization repair caused supported-class collapse. | Diagnostic failure. |
| E027 | Forensic diagnosis identified source/class confounding. | Mechanism evidence. |
| E028 | Source-complete VitaminC supported augmentation restored the three-way boundary. | Frozen release model. |
| E029 | Release qualification, metrics, figures, and local package freeze. | Completed. |
| E030 | License/security/package gates, CUDA smoke, publication staging, and cleanup. | Partial: public publication blocked by license/auth gates. |

The history is preserved because each failure changes what the next result can
mean. E028 is not a standalone truth verifier; it is the frozen supplied-
evidence classifier described in the model and dataset cards.
