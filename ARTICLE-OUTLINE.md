# Article outline — E028 source-complete evidence classifier

1. **Title and abstract.** Source-complete augmentation restores three-way
   evidence classification after an asymmetric augmentation collapse.
2. **Problem and scope.** Define supplied-evidence classification, the three
   labels, and the distinction between internal learning and external
   generalization.
3. **Identifiability correction.** Describe the E021 audit, the dropped claim
   and missing contradiction evidence, and why E018–E021 semantic conclusions
   are invalidated.
4. **Methods.** Give the canonical prompt, Qwen3-8B base revision, QLoRA
   configuration, last-token pooling, classifier head, seed, and selection
   policy.
5. **Experiment lineage.** Summarize E022–E028, including E023 corrected
   success, E024 baselines, E025 frozen suite, E026 S collapse, E027 diagnosis,
   and E028 source-complete repair.
6. **Evaluation.** Report development/internal metrics, the full 83,065 paired
   System A/System B external evaluation, clean versus known-prior-exposure
   groups, confidence intervals, contradiction retention, IE recovery, and
   paired transitions.
7. **Mechanistic interpretation.** Present the E026→E028 composition contrast
   as qualified causal support for source-complete augmentation, not universal
   proof.
8. **Limitations and contamination.** Cover public-data exposure, small
   internal sets, supplied-evidence scope, omitted raw sources/base weights,
   and license review.
9. **Reproducibility and release.** Point to the model card, dataset card,
   manifests, checksums, scripts, figures, and exact pinned revision.
10. **Conclusion and future work.** State `RELEASE_READY_WITH_LIMITATIONS`,
    that further training is not required for this freeze, and propose new
    preregistered work on calibration, retrieval, shifts, and licensed
    independent holdouts.
