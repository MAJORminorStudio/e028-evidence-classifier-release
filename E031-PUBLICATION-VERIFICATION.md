# E031 publication verification

Date: 2026-09-24  
Verification completed: 2026-09-24

This file is updated after each external publication action. It records only
Track A destinations; Track B learned binaries remain held.

## Local pre-publication gates

- [x] E031 files complete and JSON parses.
- [x] E028 adapter/head hashes unchanged.
- [x] E028 and E029 checksum files pass.
- [x] final package checksum file passes.
- [x] public Track A copy contains no raw rows or learned binaries.
- [x] GitHub repository reachable and content is Track A only.
- [x] study site deployed and reachable.
- [x] `runpodctl pod list -o json` returns `[]`.

## Destinations

- GitHub repository: https://github.com/MAJORminorStudio/e028-evidence-classifier-release
- Research study page: https://e028-evidence-classifier-study.orange-tern-3473.chatgpt.site/
- Article: https://e028-evidence-classifier-study.orange-tern-3473.chatgpt.site/article.html
- Hugging Face: not uploaded; CLI authentication is absent and Track B is held.

## Verification evidence

- GitHub default branch head: `2329a60ee70a146b203ff23264e4c4e4d62044dc`.
- GitHub repository visibility: public.
- Remote tree scan: no `.safetensors`, `.jsonl`, `.csv`, `.parquet`, `.pkl`, or
  `.pickle` paths.
- Site response: HTTP 200 for `/` and `/article.html`.
- Site deployment: succeeded, Sites version 1, commit
  `884b6a4574bfeaa252b4171c6baefddb1f70c6f4`.
- Site access: public.
- Final pod inventory: `[]`.
