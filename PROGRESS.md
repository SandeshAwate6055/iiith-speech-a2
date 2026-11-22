# PROGRESS.md: context-recovery log
Plan: [PLAN.md](PLAN.md) (approved 20 Nov 11:09). Deadline: 22 Nov 23:59 IST (target submit 22:00).

## Decisions
- TESS source: Borealis Dataverse doi:10.5683/SP2/E8H2MF (2800 files, same as Kaggle; license CC BY-NC 4.0).
- Never commit: raw/licensed audio, datasets, features, checkpoints, credentials (see .gitignore).
- GitHub repo stays PRIVATE until the user approves making it public.
- Envs (historical setup plan; final MFA uses micromamba): MFA → Miniforge `mfa` conda env (`%USERPROFILE%\miniforge3`). ML → `.venv` (uv, Py3.10, torch 2.5.1+cu121).
- Praat: `%USERPROFILE%\tools\praat\Praat.exe`.
- The 2A BU files are NIST SPHERE named .wav → convert to RIFF before MFA.

## Done
- [x] Phase 0: repo skeleton, .gitignore, md2pdf tool, PLAN.pdf, private GitHub repo (`SandeshAwate6055/iiith-speech-a2`), `.venv` (Py3.10 + torch 2.5.1+cu121 on RTX 2050)
- [x] Phase A1: 2A corpus prep (6 files -> RIFF 16k + normalized `.lab`, 6/6 tests pass)
- [x] Phase B1: TESS Borealis download (2800/2800), audit (`OA_bite_neutral` quirk handled, 0 dupes, 0 missing), Protocols A/B/C splits + 4/4 leakage tests pass
- [x] Phase B2: 16k preprocess, eGeMAPS (88-dim), WavLM-Base+ cached (`[13, T, 768]`, 5.21 GB fp16, 0 NaNs)
- [x] Phase B3: Classical baselines + `S*` (WavLM-Base+ + BiGRU + ASP) across A/B/C:
  - `S*`: Proto A = **99.96% ± 0.07%** F1 · Proto B (LOSO) = **56.23% ± 7.53%** F1 (60.71% UAR) · Proto C = **99.86% ± 0.13%** F1
  - `eGeMAPS+SVM`: Proto A = 98.43% F1 · Proto B = 12.45% F1 (18.04% UAR) · Proto C = 98.55% F1

## Current completion status (22 November, final recovery)
- [x] A2: custom lexicon, G2P, zero OOV and acoustic inventory validation.
- [x] A3: six primary TextGrids and strict structural/transcript checks.
- [x] A4: six-pair Praat visual inspection (full ISLE/zoomed BU), two genuine screenshots, scoped observations and alignment/error analysis. No manual gold/listening-based accuracy claimed.
- [x] B3-B5: all12 speech/text/concat/gated A/B/C headline runs completed and independently re-evaluated; only missing concat-C was trained after the crash.
- [x] B6: accuracy/UAR/F1 tables, all confusion matrices, within-fold temporal/contextual/fusion t-SNE and UMAP, per-emotion analysis, paired help/harm and five actual failures.
- [x] A5/B8: final PDFs,3 pages and6 pages; strict build manifests have no pending requirements and rendered pages were reviewed.
- [x] Final tests:19 passed; actual dependency versions and environment lock recorded. No completed models retrained.
- [ ] Owner approval and actual submission. Repository must remain PRIVATE. No email or public release has been performed.

## Remaining limitations (not missing P0 implementation)
- Historical text-training lineage was affected by concurrent writers. Final checkpoint-to-prediction mapping is re-evaluated and hash-linked; invalidated original P0 markers are not adopted.
- Gated C historical epoch curves are unavailable; final checkpoints/test inference are verified. Optional late fusion remains interrupted/excluded.
- Only two speakers; no population-level SER claim. No gold boundary MAE. Optional English-MFA outputs contain disclosed mismatch/spn issues.

## Recovery audit milestone (20 Nov 2026)
- Existing speech training is active (parent PowerShell PID 2736, Python 5660/19880); A/B/C are already queued. Protocol A fold_0 through fold_3 checkpoints observed. Do not duplicate training or edit its shared model/trainer files while the previous agent owns them.
- MFA 3.4.4 and Praat 7.0.03 version checks succeeded. Micromamba setup advanced to acoustic english_us_arpa download (installer parent PID 27936). Model/dictionary downloads are still owned by that process; A2 coverage and A3 alignment are not yet complete.
- Added independent scripts: mer.evaluation.audit_baselines and mer.evaluation.audit_feature_shift. Run from assignment_2b_emotion with PYTHONPATH=src. Outputs: Results/audits/baseline_recovery/. Original baseline results, feature caches and checkpoints preserved.
- Validated all 24 saved classical folds against prediction CSVs, all six summary means/population standard deviations, unique metadata/feature IDs, 88 finite features, label mapping and split disjointness. Four sampled eGeMAPS vectors reproduced from processed 16 kHz mono audio (allclose rtol=1e-4, atol=1e-5); this is a sample check, not full-cache re-extraction.
- Text Protocol C 0% reproduced in every fold. Every held-out true label is below its transcript's training-complement majority under train+val fitting. Correct the PLAN's scientific assertion: full-corpus P(y|text)=1/7 does not imply chance accuracy under every dependent finite partition. Train-only text C accuracy is 0.2143% mean. Preserve frozen splits and report this complementary-imbalance artifact.
- Classical baselines fit train+val, unlike neural train-only fits. Separate train-only/direct-SVM audit macro-F1 means: A 98.2880%, B 11.6766%, C 98.3649%. These are audit comparisons, not replacements for the original E1 rows.
- Legacy SVM uses probability argmax; direct predict differs and does not remove poor cross-speaker results. Train-only B maximum RBF similarity falls below .01 for 18.93% / 31.43% of test examples, consistent with feature shift but not proof of its sole causal role. No test-based tuning performed.
- A2 preparation: added scripts/audit_lexicon.py with strict missing-file, ARPAbet and corpus-OOV checks, including MFA probability-column parsing. Three new tests passed. Existing manual overrides: 9 unique words, zero syntax errors. Pronunciation accuracy, acoustic phone compatibility and full dictionary coverage remain unverified.
- Recovery scope is non-overlapping: only new audit scripts/tests/results plus this append. Next: wait for existing downloads/training; run full lexicon coverage + MFA validate, align six files, then Praat inspection; validate B3 and proceed to B4/B5 after GPU ownership clears. Repository privacy unchanged; no commits or pushes performed.
## P0 implementation milestone (20 Nov, continuation)
- A2 complete computationally: official MontrealCorpusTools/english_us_arpa HF snapshot 9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede, acoustic ARPA v3.3.0 (not stalled GitHub v3.0.0). 11 base OOV types received G2P candidates; corpus-scoped custom lexicon covers 145 types/248 pronunciations with zero OOV and every phone present in model inventory. Pronunciation source/review status recorded; no invented human verification.
- MFA validation completed successfully (97.163 s, 6 sound/text pairs, 2 speakers, no read/features/transcription/OOV errors). MFA's validator additionally trained its own small monophone diagnostic; this is not our primary alignment model.
- A3 complete: six primary TextGrids generated with official pretrained ARPA3.3.0 and fMLLR adaptation, beam100/retry400. Strict evaluation: all6 structurally valid, matching all transcript words (68/73/82/5/5/5), no gaps, overlaps, invalid intervals or spn. Six alignment figures rendered.
- A4 PARTIAL: automatic evaluation and plotted alignments available. Praat7.0.03 launched, but user physical Escape stopped computer-use before visual inspection. No further desktop interaction this turn. Praat screenshot/listening observations and any manually verified boundaries remain genuinely blocked; do not claim them complete.
- B3 complete: speech macro-F1 A99.96% +/-0.07pp, B56.23% +/-7.53pp, C99.86% +/-0.13pp (rounded summaries; final tables independently regenerate from prediction CSVs). Existing process completed without interruption.
- B4/B5 running sequentially via tools/continue_p0.ps1 (session80049): RoBERTa text A/B/C, then concat A/B/C, then gated attention A/B/C. Text A completed14.29%accuracy and4.26%macro-F1. Frozen-cache real-model CPU forward/backward checks passed for text and both fusion variants.
- New p0_train.py uses per-run OS file locks, input/source hashes, fold-completion markers and atomic histories; incomplete folds restart deterministically, completed folds are preserved. New fusion_concat.yaml. Original shared speech trainer/model files untouched.
- Evaluation modules now independently validate test IDs/labels/splits/probabilities, regenerate metrics and compare fusion help/harm; within-fold embedding plots avoid pooling incompatible fold coordinate spaces. Five actual baseline failures exported to Results/audits/evaluation_p0. All19 current tests passed.
- Private-repository Colab notebook and installation/run READMEs prepared. Colab is a fallback only; no credentials embedded and no external uploads performed. Reports are being generated from evidence; missing runs/Praat inspection must remain marked pending. No repository visibility changes, commits or pushes.

## Ownership collision discovered (20 Nov ~12:08 IST)
- Previous agent owner PID26864 launched a new legacy text/fusion queue PID10420 after its speech queue finished. This was absent during our initial process audit. It trained A/B/C text into the same directories as our independently queued p0_train, causing overwritten checkpoints/predictions and inconsistent observed summaries.
- Stopped ONLY our verified duplicate queue PID17664/22020 and its active p0_train PID12600/10004. Previous agent PID10420 and its children were left running. Our B5 queue did not start. See Results/audits/process_collision/incident.json.
- Treat text p0 provenance/complete markers as invalidated until reconciliation; do not resume those experiment names via p0_train. Current text A/B/C headline values must be recomputed from final previous-agent checkpoints into separate verification outputs once ownership clears. No scientific conclusions from overlapping-run provenance.
- Previous queue owns text A/B/C, gated fusion A/B/C, zero-text A/B tests, concat and late fusion A/B. Do not independently run these. Mandatory concat Protocol C remains to schedule after previous queue clears; previous queue also references configs/fusion_late.yaml (inspect when available).


## Current handoff state after ownership safeguard
- Completed this continuation: A2 custom lexicon/G2P/inventory validation; A3 all6 primary TextGrids; A4 automatic validity/duration checks and six annotated figures; B3 all protocols independently validated; evaluation infrastructure, five actual neural failure cases; private Colab notebook; setup/run documentation; evidence-derived draft PDFs.
- B3 independently regenerated means from predictions: macro-F1 A99.9642% (SD0.0716pp), B56.2345% (SD7.5295pp), C99.8570% (SD0.1336pp). Validation metadata in Results/audits/b3_validated/validation.json. Existing embedding identity remains inferred from legacy runner order, not independently proven by a historical sidecar.
- Report drafts: assignment_2a_mfa/report/report_2a.pdf (3 pages), assignment_2b_emotion/report/report_2b.pdf (6 pages). All pages rendered and visually reviewed; generator self-tests and refusal of incomplete final builds passed. These are explicitly DRAFT, not submission-ready.
- Clarification: 97.163 s is corpus AUDIO DURATION; full MFA validation runtime was258.858 s. No user attestation was used; values came from tool-observed MFA execution/logs.
- tools/build_reports.py --final fails closed on missing mandatory runs/plots/embeddings and unresolved ownership collision. Component report/BUILD.md documents exact evidence requirements. No manual gold or fabricated Praat screenshots.
- Added mer.evaluation.reconcile_results: after legacy queue10420 and all legacy/P0 training stop, re-evaluate all12 final checkpoint sets into Results/verified_p0, verify input/code/checkpoint hashes before/after and record checkpoint-linked verification.json files. Active-queue guard tested and refused before evaluation. It trains nothing and does not overwrite legacy results. It deliberately refuses missing sets; concat ProtocolC is not in the other agent's current queue and still needs scheduling after ownership is settled.
- Genuine blockers requiring coordination: user must stop the OTHER agent from scheduling new work (leave its existing queue10420 running), then confirm sole scheduling ownership. Async coordination request sent. The previous queue is preserved; our duplicate queue is stopped. Text P0 markers remain invalidated in the recorded incident.
- Praat UI was stopped by physical Escape. No further computer-use calls were made. Resume/permission to interact with Praat is needed in a later turn for actual inspection/screenshots; plotted observations alone do not fulfill that requirement.
- No credentials/raw audio/features/checkpoints were committed or uploaded. Repository visibility unchanged. No P1 work started while P0 remains incomplete.


## Final recovery (22 Nov ~12:55 IST)
- Sole-agent scheduling confirmed; no prior project jobs running. All47 checkpoint files loaded with finite tensors during read-only crash audit; existing results preserved.
- Reviewed225 permitted source/derived files committed and pushed to confirmed PRIVATE repository, backup commit c102351. Raw data, environments, model weights, caches and unrelated Compitator_ASS_2 excluded.
- Starting only missing concat-C baseline, followed sequentially by checkpoint-linked re-evaluation of12 headline runs into Results/verified_p0. No completed models will be retrained. Legacy PID guard now verifies process identity to avoid false blocking from reboot PID reuse.

- A4 visual inspection completed in Praat: all6 primary pairs opened; full ISLE and zoomed BU excerpts checked. Two actual editor screenshots and scoped observations saved. No invented listening-based judgments/gold boundaries, no alignment edits.
- Missing concat-C completed:99.71%accuracy,99.72%macro-F1; saved checkpoint re-evaluation now running into separate verified outputs.

- All12 headline checkpoint re-evaluations complete with exact before/after hashes and explicit embedding IDs; historical text-training collision is retained as a limitation, while final checkpoint-to-prediction correspondence is reconciled. Original per_fold outputs preserved.
- Reproducibility QA found provisional dependency pins differed from actual packages. Corrected component requirements to the installed environment (no upgrades); saved environment_versions.json and Windows lockfile.


## Final QA milestone (22 Nov)
- All12 headline verification records plus original results retained. Report2A and report2B both FINAL with0 pending evidence gates; all rendered pages reviewed, including corrected genuine Praat capture and detailed speaker-transfer/gate/zero-text discussion.
-20 completed evaluation rows include6 classical,12 neural headline and2 zero-text checks. Matched data-fit caveats and historical provenance limitations are explicit.52 local checkpoint files retained (48 headline plus4 unfinished late-fusion checkpoints).
- Required code layout, README commands and package versions checked. Final repository preservation includes the legitimate src/mer/data Python package; only actual dataset directories are excluded.
- Preparing a local submission package for owner review only. Private GitHub backup/final push is authorized; actual submission and any visibility change are not.
