# E031 publication decision

Date: 2026-09-24  
Decision case: Case 2 — publish Track A; hold Track B for legal review

## Decision

`RELEASE_PARTIAL_TRACK_A_PUBLIC_TRACK_B_HELD`

The audit found no express upstream prohibition on distributing the frozen
adapter/head, but it also found no authoritative express permission for this
specific learned-parameter release. Track B therefore remains staged and
checksummed. Track A is publishable because it contains research materials,
aggregate metrics, code, figures, sanitized manifests, and attribution records
without raw rows, source text, base weights, adapter binary, or head binary.

## Track A — publish

- technical report and study narrative;
- article and exact headline metrics;
- code, formatter, configs, validation tools, and figures;
- sanitized release manifest and checksums;
- provenance, license-source register, exclusion audit, and attribution audit;
- public research repository and public static study page.

The public wording is: “Model weights pending license review.”

## Track B — hold

The following remain local and are not pushed to the public repository or
model hub:

- E028 LoRA adapter;
- E028 classifier head.

Both remain unchanged at their frozen E028 hashes. The counsel-facing brief is
`E031-LEGAL-REVIEW-BRIEF.md`.

## Separate Hugging Face state

Hugging Face CLI authentication is not present in this environment. E031 does
not upload the model or weights and does not expose or request a token. If a
later authorized Track B release is approved, the user must authenticate
interactively and then perform a separate upload decision.

