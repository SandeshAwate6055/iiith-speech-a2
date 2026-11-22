# IIITH Speech Assignment 2

This repository (`SandeshAwate6055/iiith-speech-a2`) must remain **PRIVATE**. Public-release instructions in PLAN.md are superseded by the user decision in PROGRESS.md. Do not change visibility or publish artifacts without separate authorization.

- [2A: MFA alignment](assignment_2a_mfa/README.md)
- [2B: speech, text and fusion](assignment_2b_emotion/README.md)
- [Private GPU Colab notebook](assignment_2b_emotion/notebooks/P0_colab.ipynb)

## Setup

Use Python 3.10. From the repository root in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Component requirements match the installed evaluation environment. `assignment_2b_emotion/environment_versions.json` records actual versions; `requirements-lock-windows.txt` captures this Windows environment. Install CUDA torch first as above. MFA has a separate micromamba/conda environment; PDF generation uses its isolated report environment. Colab is an unvalidated fallback and was not used for final results.

## Data and repository rules

Never commit raw/licensed audio, datasets, processed audio, features, caches, checkpoints, credentials, or tokens. Keep these on authorized local storage/private Drive. Do not force-add ignored files. Clear notebook outputs and review staged diffs before any separately authorized commit. `.gitignore` is a safeguard, not a complete secret scanner. Share private access with collaborators; do not make the repository public for Colab.

2A uses only the six assignment-provided BU/ISLE recordings, whose audio is licensed. TESS was acquired from Borealis DOI `10.5683/SP2/E8H2MF` (PROGRESS records CC BY-NC 4.0); retain attribution and check the downloaded license for your use. No audio redistribution is authorized.

## Final recovery and results

The crash-recovered checkpoints were re-evaluated into `assignment_2b_emotion/Results/verified_p0/`; each headline run includes checkpoint hashes, prediction hashes and explicit embedding IDs. Historical text-training lineage remains affected by the recorded concurrency incident; old P0 markers are not used as evidence. Only the missing concat-C run was trained during final recovery.

- [2A report](assignment_2a_mfa/report/report_2a.pdf), [alignment evaluation](assignment_2a_mfa/outputs/evaluation/summary.md), and [Praat observations](assignment_2a_mfa/outputs/evaluation/praat_observations.md).
- [2B report](assignment_2b_emotion/report/report_2b.pdf), [accuracy tables](assignment_2b_emotion/Results/accuracy_tables.md), [comparative analysis](assignment_2b_emotion/Results/comparative_analysis.md), and [five failure cases](assignment_2b_emotion/Results/failure_cases.md).
- [PLAN.md](PLAN.md) is the approved historical plan; [PROGRESS.md](PROGRESS.md) is the current recovery record. No public release or submission is performed automatically.

From the repository root, with prepared data and local saved checkpoints:

```powershell
python tools/complete_model_verification.py
python tools/finalize_evaluation.py
python tools/build_reports.py --final --verification-root assignment_2b_emotion/Results/verified_p0
```

The first command preserves completed runs and evaluates saved weights; it trains only the missing concat-C baseline when its completed output is absent. Do not use it as a general retraining command. PDF dependencies are `reportlab`, `pymupdf`, `numpy`, `PyYAML`; the existing isolated interpreter is documented in the report build guide. Raw data, cached features and checkpoints are intentionally absent from Git; component READMEs describe acquisition and fresh training. Use new experiment names for fresh training, never overwrite historical runs.
