# PROGRESS.md: context-recovery log
Plan: [PLAN.md](PLAN.md) (approved 20 Nov 11:09). Deadline: 22 Nov 23:59 IST (target submit 22:00).

## Decisions
- TESS source: Borealis Dataverse doi:10.5683/SP2/E8H2MF (2800 files, same as Kaggle; license CC BY-NC 4.0).
- Never commit: raw/licensed audio, datasets, features, checkpoints, credentials (see .gitignore).
- GitHub repo stays PRIVATE until the user approves making it public.
- Envs: MFA → Miniforge `mfa` conda env (`%USERPROFILE%\miniforge3`). ML → `.venv` (uv, Py3.10, torch 2.5.1+cu121).
- Praat: `%USERPROFILE%\tools\praat\Praat.exe`.
- The 2A BU files are NIST SPHERE named .wav → convert to RIFF before MFA.

## Done
- [x] Phase 0: repo skeleton, .gitignore, md2pdf tool, PLAN.pdf

## Pending
- [ ] Phase 0: env verification, private GitHub repo + first push
- [ ] A1–A5 (2A) · B1–B8 (2B) · F

## Blockers
- none
