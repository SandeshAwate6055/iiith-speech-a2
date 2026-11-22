| Quick navigation | Documentation | Final report | Results and implementation |
|---|---|---|---|
| Assignment 2A | [Alignment README](assignment_2a_mfa/README.md) | [2A PDF](assignment_2a_mfa/report/report_2a.pdf) | [Results](assignment_2a_mfa/outputs/evaluation/summary.md) · [TextGrids](assignment_2a_mfa/outputs/textgrids/) |
| Assignment 2B | [Emotion README](assignment_2b_emotion/README.md) | [2B PDF](assignment_2b_emotion/report/report_2b.pdf) | [Results](assignment_2b_emotion/Results/accuracy_tables.md) · [Plots](assignment_2b_emotion/Results/plots/) · [Model pipelines](assignment_2b_emotion/models/) |
| Source code | [2A scripts](assignment_2a_mfa/scripts/) | [2B implementation](assignment_2b_emotion/src/mer/) | [Speech pipeline](assignment_2b_emotion/models/speech_pipeline/) / [Text pipeline](assignment_2b_emotion/models/text_pipeline/) / [Fusion pipeline](assignment_2b_emotion/models/fusion_pipeline/) |
| Complete report | [Combined PDF - 29 pages](Combined_Report.pdf) | [Reproducibility](#reproducibility-and-repository-structure) | [Findings and limitations](#key-findings-and-limitations) |

# IIITH Speech Processing · Assignment 2

Forced alignment and multimodal emotion recognition, with reproducible pipelines, saved evaluation evidence and final technical reports. Assignment 2A aligns words and phones to audio; Assignment 2B tests how speech, text and fusion generalize across lexical and speaker splits.

[2A requirements and evidence](assignment_2a_mfa/README.md#official-requirements-and-evidence) | [2B requirements and evidence](assignment_2b_emotion/README.md#official-requirements-and-evidence)

## Key achievements

- **2A:** six supplied recordings, zero custom-lexicon OOVs, five evaluated configurations and 30 TextGrids, with genuine Praat inspection. The CTC comparator has word tiers only.
- **2B:** all 2,800 TESS recordings audited; six model families evaluated across Protocols A/B/C, with checkpoint-backed neural verification.
- **Best speech macro-F1:** 99.96% (A), 56.23% (B), 99.86% (C). The speaker-transfer gap is the central finding.

## Protocol comparison

Percentages are unweighted fold means ± population standard deviations. [Full precision CSV](assignment_2b_emotion/Results/accuracy_tables.csv) · [Per-emotion analysis](assignment_2b_emotion/Results/comparative_analysis.md).

### Protocol A · Word-disjoint, shared speakers · 5 folds

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| eGeMAPS + SVM | 98.43 ± 0.58 | 98.43 ± 0.58 |
| TF-IDF + logistic regression | 14.29 ± 0.00 | 3.57 ± 0.00 |
| Concatenation fusion | 99.82 ± 0.11 | 99.82 ± 0.11 |
| Gated cross-attention fusion | 99.79 ± 0.13 | 99.79 ± 0.13 |
| WavLM-BiGRU-ASP | 99.96 ± 0.07 | 99.96 ± 0.07 |
| RoBERTa (top 2 layers fine-tuned) | 14.29 ± 0.00 | 3.57 ± 0.00 |

### Protocol B · Speaker- and word-disjoint · 2 folds

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| eGeMAPS + SVM | 18.04 ± 1.61 | 12.45 ± 0.71 |
| TF-IDF + logistic regression | 14.29 ± 0.00 | 3.57 ± 0.00 |
| Concatenation fusion | 43.75 ± 5.89 | 34.44 ± 4.66 |
| Gated cross-attention fusion | 52.14 ± 5.36 | 46.40 ± 1.44 |
| WavLM-BiGRU-ASP | 60.71 ± 8.21 | 56.23 ± 7.53 |
| RoBERTa (top 2 layers fine-tuned) | 14.29 ± 0.00 | 6.31 ± 0.22 |

### Protocol C · Random stratified reference · 5 folds

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| eGeMAPS + SVM | 98.54 ± 0.50 | 98.55 ± 0.49 |
| TF-IDF + logistic regression | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Concatenation fusion | 99.71 ± 0.27 | 99.72 ± 0.27 |
| Gated cross-attention fusion | 99.82 ± 0.28 | 99.82 ± 0.28 |
| WavLM-BiGRU-ASP | 99.86 ± 0.13 | 99.86 ± 0.13 |
| RoBERTa (top 2 layers fine-tuned) | 14.25 ± 0.62 | 4.34 ± 0.97 |

## Results at a glance

![Macro-F1 comparison across Protocols A, B and C](assignment_2b_emotion/Results/plots/model_comparison_protocols.png)

*Model comparison across Protocols A/B/C — Macro-F1 (%) with population standard deviation.*

![Temporal, contextual and fusion representations: within-fold UMAP](assignment_2b_emotion/Results/plots/embeddings_umap_protoA.png)

The embedding panels use Protocol A fold 0, not pooled cross-fold coordinates. [t-SNE](assignment_2b_emotion/Results/plots/embeddings_tsne_protoA.png) · [Five failure cases](assignment_2b_emotion/Results/failure_cases.md).

## Architecture · Assignment 2A

![Updated Assignment 2A architecture: MFA forced alignment](docs/figures/ARCH_2A.png)

These supplied infographics are used for README display; the PDF reports retain their existing architecture figures.

The final architecture shows **lowercase** transcript normalization, custom dictionary/G2P handling and parallel alignment alternatives. CTC produces **word-only** alignments; the primary MFA path supplies words and phones. [Technical details](assignment_2a_mfa/README.md#architecture-and-technical-design).

## Architecture · Assignment 2B

![Updated Assignment 2B architecture: speech, text and fusion](docs/figures/ARCH_2B.png)

The final architecture traces all five functional blocks. Gated cross-attention operates on projected speech **frame** and text **token** sequences before pooling. Concatenation uses pooled embeddings and was evaluated under all three protocols. [Dimensions and implementation](assignment_2b_emotion/README.md#architecture-and-technical-design).

**Implementation detail:** the infographic's `concat(512 + 256) -> 768` label describes the unprojected pooled branch dimensions. The implemented concatenation path first projects each branch to 256 dimensions, concatenates to 512, then projects to 256. Gated cross-attention and concatenation are separate alternatives, not sequential fusion stages. SpecAugment runs on learned acoustic features during training.

## The two assignments

| Component | Objective | Evidence |
|---|---|---|
| [2A: forced alignment](assignment_2a_mfa/README.md) | Normalize audio/transcripts, resolve OOVs and align words/phones with MFA | [Alignment evaluation](assignment_2a_mfa/outputs/evaluation/summary.md), [Praat observations](assignment_2a_mfa/outputs/evaluation/praat_observations.md) |
| [2B: emotion recognition](assignment_2b_emotion/README.md) | Compare WavLM speech, RoBERTa text, concatenation and gated cross-attention | [Verified evaluations](assignment_2b_emotion/Results/verified_p0/), [confusion matrices and plots](assignment_2b_emotion/Results/plots/) |

The final reports provide clickable contents, PDF bookmarks and page numbering: [2A - 10 pages](assignment_2a_mfa/report/report_2a.pdf), [2B - 19 pages](assignment_2b_emotion/report/report_2b.pdf), and [complete 2A followed by 2B - 29 pages](Combined_Report.pdf).

## Reproducibility and repository structure

Key tracked paths are shown below; individual output files and split manifests are grouped by directory.

```text
iiith-speech-a2/
|-- README.md
|-- Combined_Report.pdf
|-- requirements.txt
|-- .gitignore
|-- assignment_2a_mfa/
|   |-- README.md
|   |-- report/
|   |   `-- report_2a.pdf
|   |-- requirements.txt
|   |-- run_pipeline.ps1
|   |-- configs/
|   |-- dictionaries/
|   |-- scripts/
|   |-- tests/
|   `-- outputs/
|       |-- textgrids/
|       |-- figures/
|       |-- praat_screenshots/
|       `-- evaluation/
|-- assignment_2b_emotion/
|   |-- README.md
|   |-- report/
|   |   `-- report_2b.pdf
|   |-- models/
|   |   |-- speech_pipeline/
|   |   |   |-- train.py
|   |   |   `-- test.py
|   |   |-- text_pipeline/
|   |   |   |-- train.py
|   |   |   `-- test.py
|   |   `-- fusion_pipeline/
|   |       |-- train.py
|   |       `-- test.py
|   |-- src/
|   |   `-- mer/
|   |       |-- data/
|   |       |-- features/
|   |       |-- models/
|   |       |-- training/
|   |       |-- evaluation/
|   |       `-- utils/
|   |-- configs/
|   |-- splits/
|   |-- Results/
|   |   |-- accuracy_tables.csv
|   |   |-- accuracy_tables.md
|   |   |-- comparative_analysis.md
|   |   |-- failure_cases.md
|   |   |-- per_fold/
|   |   |-- plots/
|   |   `-- verified_p0/
|   |-- tests/
|   |-- requirements.txt
|   |-- requirements-lock-windows.txt
|   `-- environment_versions.json
|-- docs/
|   `-- figures/
|       |-- ARCH_2A.png
|       `-- ARCH_2B.png
`-- tools/
```

- `assignment_2a_mfa/` contains corpus preparation, dictionary/G2P and MFA scripts; `outputs/` holds TextGrids, alignment figures, genuine Praat captures and evaluation evidence.
- `assignment_2b_emotion/` contains the required speech/text/fusion train-test entry points, shared `src/mer/` implementation, fixed splits, configurations and dependency records. `Results/` holds verified metrics, per-fold evaluations, plots and failure analysis.
- `docs/figures/` holds README architecture images and existing technical schematics. `tools/` contains report, diagram and evaluation utilities.
- The root combined PDF and component `report/` PDFs are the existing final reports. Raw/licensed datasets, feature caches and large checkpoints are not tracked.


<details>
<summary>Setup, commands and evaluation evidence</summary>

- [2A installation and execution](assignment_2a_mfa/README.md#installation-and-execution).
- [2B reproducibility](assignment_2b_emotion/README.md#reproducibility), [speech](assignment_2b_emotion/models/speech_pipeline/), [text](assignment_2b_emotion/models/text_pipeline/), [fusion](assignment_2b_emotion/models/fusion_pipeline/) entry points.
- [Root dependencies](requirements.txt), [2A tests](assignment_2a_mfa/tests/), [2B tests](assignment_2b_emotion/tests/).
- Raw/licensed audio, datasets, secrets, cached features and large checkpoints are excluded. Local data/models must be supplied to reproduce computation.

</details>

## Key findings and limitations

TESS has only two actors and acted emotions. Protocol C shares speakers and words; its high scores do not establish unseen-speaker generalization. Protocol B is the relevant transfer stress test. The carrier transcript supplies no reliable emotion semantics. Neither fusion variant improves overall macro-F1 over speech alone.

Classical estimators fit train+validation data; neural models train on the training partition with validation selection. This difference limits direct fairness claims. The TF-IDF Protocol C zero score reflects finite split/complement effects, not evidence of emotion recognition. Accuracy chance is 1/7 on balanced labels; macro-F1 has no universal chance value.

Saved neural checkpoint-to-prediction mappings were independently re-evaluated and hash-checked. Complete historical RoBERTa training lineage is not recoverable; final evaluation verification does not establish that lineage. Gated Protocol C epoch history is unavailable. Gate values and low-dimensional projections do not prove causal modality importance or speaker invariance.

2A boundary comparisons measure agreement with the primary MFA output, not accuracy against independently annotated gold boundaries. Visual inspection is documented; no listening-based gold standard or manual boundary corrections are claimed.

## Academic references

- [MFA: McAuliffe et al. (2017)](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html).
- [WavLM: Chen et al.](https://arxiv.org/abs/2110.13900); [attentive statistics pooling: Okabe et al.](https://arxiv.org/abs/1803.10963).
- [RoBERTa: Liu et al.](https://arxiv.org/abs/1907.11692).
- [MulT: Tsai et al.](https://arxiv.org/abs/1906.00295); [gated multimodal units: Arevalo et al.](https://arxiv.org/abs/1702.01992). These motivate the custom fusion architecture; this is not a reproduction of either paper.
- [TESS dataset: Pichora-Fuller and Dupuis](https://doi.org/10.5683/SP2/E8H2MF).
