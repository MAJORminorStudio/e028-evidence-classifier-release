# E031 counsel review brief

Prepared: 2026-09-24  
Purpose: narrow legal question for review before any Track B weight upload

## Proposed review question

Does distributing the following numerical artifacts, without raw data, paper
text, or Qwen base weights, constitute distribution of Adapted Material or
otherwise trigger an upstream data-license condition?

- LoRA adapter: 174,655,536 bytes, SHA-256
  `b8a6dfd754f1cb766d710697d03c5b26a8316abc085c9c1dc999b942e81e5f8c`.
- Classifier head: 49,300 bytes, SHA-256
  `cea444d633653afbb0697c0357717fc0f6014b25bc0ca4965660807aee77c5a4`.

## Facts counsel should assume

- Base model: Qwen/Qwen3-8B-Base at revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, Apache-2.0 upstream terms;
  base weights are not redistributed.
- Training set: 1,576 rows — 976 QASPER-backed E023 rows, 400 selected
  VitaminC rows from E026, and 200 selected VitaminC rows from E028.
- VitaminC augmentation is 100 real plus 100 synthetic per selected class;
  synthetic annotations use or modify FEVER examples.
- The public release contains no raw training/evaluation rows or paper text.
- FEVER, FEVER2, SciFact, AVeriTeC, and FactKG were evaluation-only for E028.

## Materials reviewed

- [Qwen3 official repository](https://github.com/QwenLM/Qwen3) and
  [Qwen3-8B-Base model card](https://huggingface.co/Qwen/Qwen3-8B-Base);
- [QASPER dataset card](https://huggingface.co/datasets/allenai/qasper/blob/main/README.md)
  and [S2ORC repository](https://github.com/allenai/s2orc);
- [VitaminC code license](https://github.com/TalSchuster/VitaminC/blob/main/LICENSE)
  and [VitaminC data license](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE);
- evaluation-only source notices listed in `E031-LICENSE-SOURCE-REGISTER.json`.

## Current non-legal disposition

The source materials are silent on the specific learned-weight question. No
express prohibition was found, but no express permission was found either.
Track A is published; Track B is held pending review. This brief should not be
read as a legal opinion or as permission to redistribute the weights.

