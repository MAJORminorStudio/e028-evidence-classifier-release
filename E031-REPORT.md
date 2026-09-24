# E031 report — release license resolution and publication unblock

Date: 2026-09-24  
Final status: `RELEASE_PARTIAL_TRACK_A_PUBLIC_TRACK_B_HELD`

## Outcome

E031 completed the license/provenance audit for frozen E028 and authorized
publication of sanitized research materials. It did not train, rerun a
benchmark, modify E028, upload weights, or start E032.

## Frozen release and immutability

The base is Qwen/Qwen3-8B-Base revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. E028 remains epoch 2 / global
step 198. Adapter SHA-256 is
`b8a6dfd754f1cb766d710697d03c5b26a8316abc085c9c1dc999b942e81e5f8c`; head
SHA-256 is
`cea444d633653afbb0697c0357717fc0f6014b25bc0ca4965660807aee77c5a4`.

## Training provenance

E028 trained on 1,576 rows: QASPER/S2ORC-backed E023 seed rows (720
SUPPORTED, 149 CONTRADICTED, 107 INSUFFICIENT_EVIDENCE), plus 400 selected
VitaminC augmentation rows from E026 and 200 selected VitaminC augmentation
rows from E028. The exact hashes, source roles, source groups, class counts,
and canonicalization rules are in `E031-TRAINING-PROVENANCE.json`.

FEVER, FEVER2, SciFact, AVeriTeC, and FactKG were E025 evaluation-only
sources. They did not contribute E028 gradients.

## License result

Qwen3-8B-Base is Apache-2.0 upstream. QASPER is identified as CC BY 4.0,
with S2ORC and source-paper dependencies. VitaminC code is MIT, while its
data notice is source-dependent and discusses Wikipedia, FEVER, and a CC BY-SA
fallback for annotations. None of the reviewed training-source materials
expressly permits or prohibits redistribution of this specific numerical
adapter/head pair. Both learned artifacts are therefore
`LEGAL_REVIEW_RECOMMENDED_BUT_NOT_EXPLICITLY_PROHIBITED`.

## Evaluation headline

On the frozen clean paired external evaluation (`n=69,217`), E028 accuracy
was 0.769291 and macro-F1 was 0.710075. Vanilla macro-F1 was 0.534814 and
E023 macro-F1 was 0.451817. E028 clean contradiction recall was 0.855485;
VitaminC accuracy was 0.784851 and macro-F1 was 0.731225. Paired transitions
were 16,294 old-wrong/new-correct and 7,445 old-correct/new-wrong, for a net
gain of 8,849. The paired clean macro-F1 delta was +0.258257 with 95% CI
[0.253703, 0.262624].

These are frozen recorded metrics; E031 did not recompute them.

## Publication disposition

Track A is public: report, article, code, figures, aggregate metrics,
sanitized manifests, attribution, and audit files. Track B is held: the
adapter and classifier head remain local for legal review. Hugging Face is not
uploaded because authentication is absent and Track B is held.

## Verified destinations

- GitHub Track A repository:
  https://github.com/MAJORminorStudio/e028-evidence-classifier-release
- Public research study:
  https://e028-evidence-classifier-study.orange-tern-3473.chatgpt.site/
- Public article:
  https://e028-evidence-classifier-study.orange-tern-3473.chatgpt.site/article.html

The GitHub repository is public and its remote tree contains no raw data
extensions or learned-weight binaries. Both site pages returned HTTP 200 after
the public production deployment.

## Compute and cleanup

E031 used no GPU. Final pod inventory must be recorded as `[]`. No E032 or
training action was started.
