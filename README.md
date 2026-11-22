# IIITH Speech Processing — Assignment 2

This repository contains the complete implementation, evaluation artifacts, and final reports for **Assignment 2A (Forced Alignment using Montreal Forced Aligner)** and **Assignment 2B (Multimodal Emotion Recognition)**.

## Deliverables & Quick Links

| Component | Deliverable | Location |
|---|---|---|
| **Assignment 2A** | Forced Alignment Guide & Setup | [`assignment_2a_mfa/README.md`](assignment_2a_mfa/README.md) |
| **Assignment 2A** | Final PDF Report (3 pages) | [`assignment_2a_mfa/report/report_2a.pdf`](assignment_2a_mfa/report/report_2a.pdf) |
| **Assignment 2A** | Generated TextGrids (Primary + Extra Credit) | [`assignment_2a_mfa/outputs/textgrids/`](assignment_2a_mfa/outputs/textgrids/) |
| **Assignment 2A** | Alignment Evaluation & Praat Observations | [`assignment_2a_mfa/outputs/evaluation/summary.md`](assignment_2a_mfa/outputs/evaluation/summary.md), [`praat_observations.md`](assignment_2a_mfa/outputs/evaluation/praat_observations.md) |
| **Assignment 2B** | Multimodal Emotion Recognition Guide & Setup | [`assignment_2b_emotion/README.md`](assignment_2b_emotion/README.md) |
| **Assignment 2B** | Final PDF Report (6 pages) | [`assignment_2b_emotion/report/report_2b.pdf`](assignment_2b_emotion/report/report_2b.pdf) |
| **Assignment 2B** | Accuracy Tables & Comparative Analysis | [`assignment_2b_emotion/Results/accuracy_tables.md`](assignment_2b_emotion/Results/accuracy_tables.md), [`comparative_analysis.md`](assignment_2b_emotion/Results/comparative_analysis.md) |
| **Assignment 2B** | Error Analysis (5 Failure Cases) & Plots | [`assignment_2b_emotion/Results/failure_cases.md`](assignment_2b_emotion/Results/failure_cases.md), [`assignment_2b_emotion/Results/plots/`](assignment_2b_emotion/Results/plots/) |

---

## Repository Structure

```text
iiith-speech-a2/
├── README.md
├── requirements.txt
├── assignment_2a_mfa/                  # Assignment 2A: MFA Forced Alignment
│   ├── README.md
│   ├── requirements.txt
│   ├── run_pipeline.ps1
│   ├── configs/pipeline.yaml
│   ├── dictionaries/                   # Custom ARPAbet lexicon, G2P & manual overrides
│   ├── scripts/                        # Corpus prep, lexicon audit, MFA runner, CTC & plots
│   ├── outputs/
│   │   ├── evaluation/                 # Alignment metrics, lexicon audit & Praat observations
│   │   ├── figures/                    # 6 waveform + spectrogram + TextGrid visualizations
│   │   ├── praat_screenshots/          # Genuine Praat editor captures (ISLE 01 & 02)
│   │   └── textgrids/                  # english_us_arpa, adapted_arpa, english_mfa, isle_probe, ctc
│   ├── report/report_2a.pdf
│   └── tests/
├── assignment_2b_emotion/              # Assignment 2B: Multimodal Emotion Recognition (TESS)
│   ├── README.md
│   ├── requirements.txt
│   ├── requirements-lock-windows.txt
│   ├── environment_versions.json
│   ├── configs/                        # speech.yaml, text.yaml, fusion_concat.yaml, fusion.yaml
│   ├── splits/                         # Frozen Protocols A, B, C split manifests
│   ├── models/                         # Required train.py / test.py pipeline entry points
│   │   ├── speech_pipeline/
│   │   ├── text_pipeline/
│   │   └── fusion_pipeline/
│   ├── src/mer/                        # Modular data, feature, model, training & evaluation package
│   ├── Results/                        # Accuracy tables, comparative analysis, verified_p0 & plots
│   ├── report/report_2b.pdf
│   └── tests/
└── tools/
    ├── build_reports.py
    └── finalize_evaluation.py
```

---

## Environment Setup

Use Python 3.10 for the core ML/evaluation environment. From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

- **Assignment 2A (MFA)**: Uses an isolated Conda/Micromamba environment (`montreal-forced-aligner=3.4.4`). See [`assignment_2a_mfa/README.md`](assignment_2a_mfa/README.md) for step-by-step installation, corpus normalization, lexicon building, and alignment commands.
- **Assignment 2B (TESS Emotion Recognition)**: Exact package versions are pinned in [`assignment_2b_emotion/requirements.txt`](assignment_2b_emotion/requirements.txt) and [`assignment_2b_emotion/environment_versions.json`](assignment_2b_emotion/environment_versions.json). See [`assignment_2b_emotion/README.md`](assignment_2b_emotion/README.md) for feature extraction, training, testing, and evaluation commands.

---

## Summary of Results

### Assignment 2A: Forced Alignment (BU Radio & ISLE)
- **Dataset**: 6 provided recordings (3 BU Radio broadcast clips transcoded from NIST SPHERE to 16 kHz mono RIFF PCM16; 3 ISLE non-native learner clips; 97.163 s total duration).
- **Lexicon**: 145 unique corpus word types (248 pronunciations); 11 base OOV types resolved via G2P and curated overrides (`0` final OOVs).
- **Alignment**: Primary `english_us_arpa` (v3.3.0) and speaker-adapted `adapted_arpa` (fMLLR) align 100% of transcript words (`68/73/82/5/5/5`) with zero structural gaps, overlaps, or `spn` intervals. Extra-credit comparisons include `english_mfa` (IPA), `isle_probe`, and a neural `ctc` word-boundary baseline.

### Assignment 2B: Multimodal Emotion Recognition (TESS)
All 12 neural headline configurations (plus classical `eGeMAPS+SVM` and `TF-IDF+LR` baselines and zero-text interventions) were evaluated across frozen splits and independently verified against checkpoint SHA256 hashes in [`assignment_2b_emotion/Results/verified_p0/`](assignment_2b_emotion/Results/verified_p0/):
- **Protocol A** (5-fold word-disjoint, shared speakers)
- **Protocol B** (2-fold speaker-independent LOSO + word-disjoint stress test)
- **Protocol C** (5-fold random stratified optimistic reference)

| System | Architecture | Protocol A Macro-F1 (%) | Protocol B Macro-F1 (%) | Protocol C Macro-F1 (%) |
|---|---|---:|---:|---:|
| **Speech** | WavLM-Base+ (13-layer sum) + BiGRU + ASP | **99.96 ± 0.07** | **56.23 ± 7.53** | **99.86 ± 0.13** |
| **Text** | RoBERTa-base (top-2 layer FT + attention pool) | 3.57 ± 0.00 (14.29% Acc) | 6.31 ± 0.22 (14.29% Acc) | 4.34 ± 0.97 (14.25% Acc) |
| **Concat Fusion** | WavLM-BiGRU-ASP + RoBERTa Concatenation | 99.82 ± 0.11 | 34.44 ± 4.66 | 99.72 ± 0.27 |
| **Gated Fusion** | 4-Head Bidirectional Cross-Attention + Gate | 99.79 ± 0.13 | 46.40 ± 1.44 | 99.82 ± 0.28 |
| *Classical Speech* | *eGeMAPS (88-dim) + RBF SVM (train+val fit)* | *98.43 ± 0.58* | *12.45 ± 0.71* | *98.55 ± 0.49* |
| *Classical Text* | *TF-IDF + Logistic Regression (train+val fit)* | *3.57 ± 0.00 (14.29% Acc)* | *3.57 ± 0.00 (14.29% Acc)* | *0.00 ± 0.00 (0.00% Acc)* |

---

## Verification & Testing

Run the automated test suite (19 unit and split-leakage tests) from the repository root:

```powershell
$env:PYTHONPATH = "assignment_2b_emotion/src;assignment_2a_mfa/scripts"
python -m pytest assignment_2a_mfa/tests assignment_2b_emotion/tests -q
```

To regenerate the consolidated 2B evaluation tables/plots and rebuild both final PDF reports from the verified outputs:

```powershell
python tools/finalize_evaluation.py
python tools/build_reports.py --final --verification-root assignment_2b_emotion/Results/verified_p0
```

## Data & Licensing Policy

Raw/licensed audio recordings (BU Radio, ISLE, TESS), cached feature tensors, and large model checkpoints are excluded from Git tracking via [`.gitignore`](.gitignore). TESS is sourced from Borealis Dataverse (`doi:10.5683/SP2/E8H2MF`, CC BY-NC 4.0).
