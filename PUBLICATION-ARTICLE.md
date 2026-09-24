# The input was the missing variable

## Source-complete augmentation restores a three-way evidence classifier after two different class collapses

At first, contradiction learning looked like the hard part of this project.
The model could learn some supported relationships, but the experiments around
contradiction were unstable and the three-way task seemed to resist a clean
boundary. That diagnosis was wrong in a useful way: the model was sometimes
not receiving the information required to identify the class.

This study is about a supplied-evidence classifier. Given a `CLAIM` and
caller-provided `EVIDENCE`, the frozen model predicts `SUPPORTED`,
`CONTRADICTED`, or `INSUFFICIENT_EVIDENCE`. It is not a retrieval system and
does not independently establish real-world truth. Every result below is for
that input contract and for the evaluated benchmark populations.

## The first failure was an input problem

The E021 input-identifiability audit found that earlier examples could drop
the claim or omit the contradiction-bearing evidence while still carrying a
label. That made the semantic conclusions from E018–E021 invalid: a model
cannot reliably learn a claim/evidence relationship from a prompt that does
not consistently identify the claim and evidence needed for the decision.

E022 corrected the representation. E023 then confirmed that the corrected
classifier could learn the three-way task for two epochs. E024 compared the
corrected model with a vanilla baseline and a linear probe. The internal set
was small and eventually saturated, so the next question was generalization,
not another round of training on the same controls.

## The larger benchmark changed the diagnosis again

E025 froze a paired external evaluation of 83,065 examples. The first
corrected model improved some behavior but collapsed on
`INSUFFICIENT_EVIDENCE`. E026 attempted a targeted generalization repair and
created the opposite collapse: supported predictions disappeared. That result
was not evidence that support learning was intrinsically incompatible with
contradiction learning. E027’s forensic audit identified a source/class
confound: the repair had added one class from a source distribution without
matching the source coverage of the other classes.

E028 kept the corrected input, the classifier architecture, and the frozen
selection rule. It added source-complete VitaminC supported coverage—100 real
and 100 synthetic matched examples—while retaining the earlier corrected data.
The comparison is qualified causal evidence for the source-complete
intervention under this training setup; it is not a universal causal proof.

## What the frozen model did

The release model is Qwen3-8B-Base at revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, plus the E028 epoch-2/global-step-
198 LoRA adapter and a separate `Linear(4096, 3)` head. The head pools the
last non-padding hidden state. The prompt formatter is frozen and visible in
the release package.

On the 4,500-example development set, accuracy was 0.743556 and macro-F1 was
0.737989. The 257-example internal aggregate remained 257/257. On the full
paired external suite, E028 reached accuracy 0.786890 and macro-F1 0.714953
over 83,065 examples. The clean subset—69,217 examples—reached accuracy
0.769291 and macro-F1 0.710075.

The clean result is the more useful headline for generalization, but it is not
a claim of uncontaminated pretraining. The clean group is FEVER2, VitaminC,
and FactKG; FEVER, SciFact, and AVeriTeC are labeled known-prior-exposure in
the study records. Public-data exposure and base-model pretraining exposure
remain separate limitations.

For the clean comparison, vanilla macro-F1 was 0.534814, E023 was 0.451817,
and E028 was 0.710075. Clean contradiction recall was 0.447662 for vanilla,
0.572530 for E023, and 0.855485 for E028. On VitaminC alone, E028 accuracy
was 0.784851 and macro-F1 was 0.731225, with recall 0.779096 for supported,
0.871031 for contradicted, and 0.566593 for insufficient evidence.

The paired E023-to-E028 transition record contains 16,294 wrong-to-correct
changes and 7,445 correct-to-wrong changes, a net gain of 8,849 corrections.
The clean paired macro-F1 gain was 0.258257, with a 95% bootstrap confidence
interval of [0.253703, 0.262624]. Those intervals describe the paired
benchmark records; they are not a population-wide guarantee.

## Why the sequence matters

The scientific result is not simply that the last checkpoint scored higher.
The sequence exposed three different failure modes:

1. malformed inputs made early semantic conclusions uninterpretable;
2. a corrected model could learn the task internally but lose
   insufficient-evidence behavior externally;
3. a partial repair could erase supported behavior when source and class were
   confounded.

The final repair addressed the third mechanism by restoring source-complete
coverage. That is why the failed experiments stay in the account. Removing
them would make the final score look more inevitable than it was and would
hide the boundary between an input-identifiability problem and a data-
composition problem.

## Release boundaries

The Track A public package contains the tokenizer/config, formatter and
reproduction scripts, provenance manifests, aggregate metrics, figures,
cards, and checksums. It does not contain the Qwen base weights, raw
third-party rows, adapter binary, or classifier-head binary. Users must obtain
the pinned base and restricted source datasets from their authoritative
locations. The learned artifacts remain local Track B files pending legal
review.

The E030 CUDA smoke reconstructed the package in a clean environment on one
secure L40S. Three deterministic non-holdout cases—one per label—loaded the
base, adapter, head, tokenizer, and formatter and completed successfully.
This is a packaging smoke, not a rerun of the 83,065-example evaluation.

E031 publishes the research materials as Track A. The project code is
Apache-2.0, and Qwen’s model card states Apache-2.0 for the base model, but
the learned adapter and classifier head have mixed-source training provenance
and the reviewed authoritative materials are silent on the specific learned-
weight question. They remain staged locally as Track B artifacts pending legal
review; no public model-weight upload is represented as complete.

## Conclusion

Under the E028 intervention and evaluated benchmarks, source-complete
augmentation restored a usable three-way boundary after two distinct class
collapses. The result is a supplied-evidence classifier with materially better
clean external macro-F1 than E023, while retaining the small internal task.
It is not a universal verifier, a retrieval system, or a perfect fact checker.

No further training is required for this freeze. Future work should be a new
preregistered effort on calibration, retrieval coupling, domain shifts, and
independently licensed holdouts, with model-derivative rights resolved before
public weight distribution.
