# Dataset card: E028 source-complete evidence classifier data

The training contract combines the corrected E023 evidence-support examples
with source-complete VitaminC augmentation. The intended training composition
is 920 SUPPORTED, 349 CONTRADICTED, and 307 INSUFFICIENT_EVIDENCE examples
(1,576 total), including exactly 100 real and 100 synthetic VitaminC
SUPPORTED additions. The development set has 4,500 rows with 1,500 per
class. The external evaluation is the frozen E025 suite of 83,065 paired rows:
FEVER 13,198; FEVER2 780; VitaminC 55,171; SciFact 188; AVeriTeC 462; and
FactKG 13,266.

Raw source rows and holdout predictions are not bundled. The manifests record
source roles, hashes, class counts, the canonical prompt contract, and the
E025 overlap firewall. Known-prior-exposure and clean groups must remain
separate in any reuse. Evaluation is evidence classification, not retrieval,
fact verification against a live source, or a claim of contamination-free
pretraining.
