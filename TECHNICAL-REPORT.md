# E029 — E028 release qualification and publication freeze

## Outcome

**RELEASE_READY_WITH_LIMITATIONS.** E028 is verified unchanged, the selected
epoch-2/step-198 artifact is packaged without base weights or raw third-party
rows, all required metrics and analyses are preserved, and five publication
figures plus source CSVs are generated. No training, cloud GPU, external
rerun, publication push, or E030 was started.

## Model and training identity

Base: `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; hidden size 4096; last non-padding
pooling; Linear(4096,3) head; QLoRA rank 16 / alpha 32 / dropout 0.05 on the
seven recorded projection/MLP target modules; ordinary cross-entropy; seed 17;
two epochs; max length 4608; effective batch size 16; learning rate 0.0002;
4-bit NF4. Total parameters 4,761,510,915 and trainable parameters 43,659,267.

## Metrics and uncertainty

Development (n=4,500): accuracy 0.743556, macro-F1 0.737989; S recall
0.842000, C recall 0.825333, IE recall 0.563333. Internal aggregate (n=257)
is 1.0 accuracy / 1.0 macro-F1. External (n=83,065): accuracy 0.786890,
macro-F1 0.714953. Clean aggregate (n=69,217): accuracy 0.769291,
macro-F1 0.710075. The paired E023→E028 aggregate has 16,294 old-wrong/new-
correct and 7,445 old-correct/new-wrong transitions, net +8,849 corrections;
macro-F1 gain 0.246034 with 95% CI [0.241870, 0.250129]. Clean macro-F1 gain
is 0.258257 with 95% CI [0.253703, 0.262624].

Contradiction recall on the clean aggregate is 0.855485 for E028 versus
0.572530 for E023 and 0.447662 for vanilla. Insufficient-evidence recall is
0.566593 for E028 versus 0.0 for E023. VitaminC is recorded separately in
`E029-VITAMINC-ANALYSIS.json`.

## Causal/diagnostic interpretation

E026 used C/IE augmentation without source-complete S coverage and collapsed
S recall to 0. E028 added exactly 200 VitaminC S examples (100 real, 100
synthetic) while preserving the corrected identifiable input and classifier
contract. The E026→E028 comparison supports source-complete augmentation as
the explanation for recovery under this intervention; it is not a universal
causal proof.

The E021 input-identifiability audit invalidates the semantic conclusions of
E018–E021. E022 onward is the corrected input chain. The full lineage and
claim register are separate canonical artifacts.

## Contamination and exposure

E028 preserved the E025 firewall. FEVER, SciFact, and AVeriTeC are
known-prior-exposure; FEVER2, VitaminC, and FactKG are the clean aggregate.
The contamination audit records selected-overlap checks and source hashes.
Public-data “clean” does not imply no base-model pretraining exposure.

## Package and validation

The release contains the adapter (166.57 MiB),
classifier head (48.14 KiB),
tokenizer/config, manifests, metrics, figures, model/dataset cards, and
scripts. The Qwen base weights and raw third-party rows are referenced, not
duplicated. Structural validation and three formatter smoke cases pass; a
full 8B CUDA forward smoke remains user-side because E029 does not provision
GPU compute.

## Costs and decision

Historical single-L40S costs for E023–E028 are recorded in
`E029-COST-ACCOUNTING.json`; E027 and E029 are local/no-cloud. E028 wall time
was 02:42:00 and estimated GPU cost was USD 2.943. No further training is
required to package or qualify this release. Future work may test new source
mixtures, calibration, retrieval coupling, domain shifts, and licensed
independent holdouts, but would be a new preregistered experiment after E029.


## E030 release execution

The E030 smoke loaded the pinned public base, staged adapter, classifier head, tokenizer/config, and release formatter. All three deterministic cases passed and formatter parity was byte-equivalent. The pod was deleted after log recovery. The manual license gate remains unresolved for the learned adapter/head, so no public model, repository, or website push was made.
