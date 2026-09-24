# E031 attribution audit

Date: 2026-09-24  
This record supplies attribution targets for Track A and does not relicense
upstream material.

| Dependency or source | Attribution / citation target | Role | Terms / source |
| --- | --- | --- | --- |
| Qwen3 | Qwen Team; Qwen3 technical report/model card | Base model | Apache-2.0; [official repository](https://github.com/QwenLM/Qwen3) |
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base`, pinned revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | Base model reference | [model card](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| QASPER | Dasigi et al., “QASPER: Dataset of Scientific Research Questions and Answers,” 2021 | E023/E028 training source | [dataset card](https://huggingface.co/datasets/allenai/qasper/blob/main/README.md) |
| S2ORC | Lo et al., “S2ORC: The Semantic Scholar Open Research Corpus,” 2020 | QASPER full-text dependency | [official repository](https://github.com/allenai/s2orc) |
| VitaminC | Schuster, Fisch, and Barzilay, “Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence,” 2021 | E026/E028 training source and E025 evaluation source | [official repository](https://github.com/TalSchuster/VitaminC) |
| FEVER | Thorne et al., “FEVER: a large-scale dataset for Fact Extraction and VERification,” 2018 | E025 evaluation-only; VitaminC synthetic dependency | [official dataset page](https://fever.ai/dataset/fever.html) |
| FEVER2 | FEVER task organizers and official task materials | E025 evaluation-only | [official task page](https://fever.ai/2019/task.html) |
| SciFact | Wadden et al., “Fact or Fiction: Content-Level Textual Entailment for Evaluating Scientific Claims,” 2020 | E025 evaluation-only | [official repository/license](https://github.com/allenai/scifact/blob/master/LICENSE.md) |
| AVeriTeC | AIC fact-checking community / AVeriTeC task organizers | E025 evaluation-only | [official repository](https://github.com/aic-factcheck/aic_averitec) |
| FactKG | Jiho Kim et al., FactKG dataset authors | E025 evaluation-only | [official repository](https://github.com/jiho283/FactKG) |
| Project | MAJOR//minor Studio, E028 Evidence Classifier v1 | Release code and analysis | Local project `LICENSE` (Apache-2.0) |

## Public attribution placement

Track A includes this audit, a third-party license notice, source URLs in the
technical report, and links to the upstream model/data pages. No raw source
text is copied into the public release. The site and article identify the
evaluation sources as evaluation-only and identify the model-weight status as
pending license review.

