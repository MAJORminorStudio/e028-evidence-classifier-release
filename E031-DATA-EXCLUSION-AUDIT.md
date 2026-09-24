# E031 Track A data-exclusion audit

Date: 2026-09-24  
Audit target: public research repository and public study page

## Result

PASS — no raw training/evaluation rows or Qwen base weights are included in
the Track A publication set. The final E028 package scan found no files with
`.jsonl`, `.csv`, `.parquet`, `.pkl`, or `.pickle` extensions. The public
Track A copy is additionally built from a fresh sanitized directory with the
binary learned weights removed.

## Excluded source material

- `data/raw/qasper` and the exact E023/E028 JSONL training rows;
- the local VitaminC train/dev data and its raw annotations;
- FEVER, FEVER2, SciFact, AVeriTeC, and FactKG raw benchmark rows;
- downloaded or cached scientific-paper text;
- Qwen/Qwen3-8B-Base weights;
- `adapter/adapter_model.safetensors`;
- `classifier/classifier_head.safetensors`.

## Allowed public material

- source IDs, counts, hashes, split names, source roles, and provenance
  summaries;
- aggregate metrics, confidence intervals, regression/contamination analyses,
  figures, and the technical report;
- code, formatter, configs, validation scripts, and sanitized manifests;
- the three synthetic Eiffel Tower CUDA-smoke cases in `CUDA-SMOKE.json`.

The smoke cases are generated diagnostic examples, not rows copied from a
benchmark or training source. The public manifests are identifiers and
hashes, not a substitute for raw data redistribution.

## Learned-artifact handling

The Track A staging copy intentionally omits the two binary learned artifacts.
The frozen local E028 package remains intact for verification and legal review;
this audit does not delete or rewrite it.

## Verification commands

The final audit is reproducible with:

```text
find releases/e028_evidence_classifier_v1/final -type f \\
  \( -name '*.jsonl' -o -name '*.csv' -o -name '*.parquet' \\
     -o -name '*.pkl' -o -name '*.pickle' \)
find releases/e028_evidence_classifier_v1/publication_staging/github-track-a-public \\
  -type f \( -name '*.safetensors' -o -name '*.jsonl' -o -name '*.csv' \\
     -o -name '*.parquet' -o -name '*.pkl' -o -name '*.pickle' \)
```

Both commands must produce no output for a passing Track A audit.

