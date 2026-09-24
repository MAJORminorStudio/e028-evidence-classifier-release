# Qwen3-8B Evidence Classifier — E028 release candidate

## Model identity

This is a three-way evidence classifier built on `Qwen/Qwen3-8B-Base` at
revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. It uses 4-bit NF4
QLoRA with LoRA rank 16, alpha 32, dropout 0.05, target modules
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, and a trainable
Linear(4096, 3) head. The head pools the last non-padding hidden state.
Total parameters are 4,761,510,915; trainable parameters are 43,659,267,
including 43,646,976 LoRA parameters and 12,291 classifier parameters.

Labels are `SUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT_EVIDENCE` with IDs
0, 1, and 2. The exact prompt formatter is in `scripts/canonical_formatter.py`.

## Intended use and limitations

Use for controlled claim/evidence relationship classification when the evidence
is supplied by the caller. Do not treat the output as a retrieval system, a
source of evidence, a universal truth judgment, or a guarantee against
hallucination or contamination. Internal data are small and saturated; public
benchmarks include known-prior-exposure and clean groups; no claim of
state-of-the-art performance is made.

## Evaluation summary

On the selected epoch-2 development set (n=4,500), accuracy is 0.743556 and
macro-F1 is 0.737989. On the complete frozen external suite (n=83,065),
E028 has accuracy 0.786890 and macro-F1 0.714953. On the clean subset
(n=69,217), accuracy is 0.769291 and macro-F1 is 0.710075. The clean
E023-to-E028 macro-F1 gain is 0.258257 with paired 95% bootstrap CI
[0.253703, 0.262624]. These are benchmark-specific results, not deployment
guarantees.

## Ethical and licensing considerations

The adapter is a research artifact. Review the Qwen license and all source
dataset licenses before redistribution. The package intentionally omits base
weights and raw third-party rows.
