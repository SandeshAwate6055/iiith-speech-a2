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

Requirements are a pinned candidate environment, not a validated lockfile. Install MFA separately through conda-forge (see 2A). The current `.venv` launcher points to a missing Python installation in this documentation session; recreate with an installed Python 3.10 when necessary. Colab has its own installation cells.

## Data and repository rules

Never commit raw/licensed audio, datasets, processed audio, features, caches, checkpoints, credentials, or tokens. Keep these on authorized local storage/private Drive. Do not force-add ignored files. Clear notebook outputs and review staged diffs before any separately authorized commit. `.gitignore` is a safeguard, not a complete secret scanner. Share private access with collaborators; do not make the repository public for Colab.

2A uses only the six assignment-provided BU/ISLE recordings, whose audio is licensed. TESS was acquired from Borealis DOI `10.5683/SP2/E8H2MF` (PROGRESS records CC BY-NC 4.0); retain attribution and check the downloaded license for your use. No audio redistribution is authorized.

## Status

[PLAN.md](PLAN.md) records intended work; [PROGRESS.md](PROGRESS.md) records observations and decisions. Existing classical recovery audits are available. Current neural jobs, text/fusion completion, final MFA validation/alignment, Praat inspection and reports remain **pending verification** in this documentation snapshot. No new completed results are claimed.

Preserve frozen splits and original results. Record revision, config, seed, input hashes and environment with each run. Avoid duplicating main-agent GPU/MFA jobs. Component READMEs contain run commands and checks.
