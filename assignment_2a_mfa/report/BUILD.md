# 2A report

Build both components with `python tools/build_reports.py` from the repository root. `report_2a.md` and `report_2a.pdf` use existing lexicon, strict validity CSV, six TextGrids and generated figures. No external 2A audio is loaded. Model provenance comes from the lexicon audit; MFA 3.4.4, acoustic archive 3.3.0 and successful validation (exit 0) are tool-observed execution facts, not user attestation. `work/validate_arpa/corpus/corpus.log` records 97.163 s total audio duration and 258.858 s full validation runtime. Neither is gold evaluation.

The report stays DRAFT until `outputs/evaluation/praat_observations.md` and an actual Praat screenshot under `outputs/` are supplied by the inspection owner. Generated alignment plots never substitute for Praat listening/observations. Physical Escape stopped UI; this generator never controls the desktop. `--final --allow-provisional-2a` can finalize a complete reconciled 2B report while retaining the 2A draft status. See the 2B report BUILD.md for strict final-run/collision requirements.

QA PNGs and local build environments are ignored. No licensed audio, caches, checkpoints or secrets may be committed. Build manifests record evidence hashes and pending requirements.
