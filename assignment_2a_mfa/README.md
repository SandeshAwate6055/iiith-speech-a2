# Assignment 2A: Forced Alignment Using Montreal Forced Aligner (MFA)

This directory contains the scripts, custom pronunciation dictionaries, generated TextGrids, visualizations, Praat inspection notes, and final report for **Assignment 2A**.

Run all commands from `assignment_2a_mfa/`. Only the six assignment-provided BU Radio and ISLE recordings are used. Raw/licensed audio files and MFA working caches are excluded from Git via `.gitignore`.

## 1. Environment Setup

Use the root Python 3.10 environment for helper/evaluation scripts, and install MFA 3.4.4 in a dedicated Conda or Micromamba environment:

```powershell
conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.4
conda activate mfa
mfa version
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

*(Note: If the GitHub release download stalls, `python scripts/prepare_models.py` packages the official `MontrealCorpusTools/english_us_arpa` v3.3.0 acoustic/dictionary/G2P models from Hugging Face revision `9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede`.)*

## 2. Prepare the Dataset

The raw input folder must contain `wav/*.wav` and matching transcripts under `transcripts/`. Because the three BU Radio `.wav` files are encoded with legacy **NIST SPHERE** (`NIST_1A`) headers, `prepare_corpus.py` decodes them via `libsndfile`, verifies 16 kHz mono, writes standard 16-bit PCM RIFF WAV files, normalizes transcripts into `.lab` files, and organizes speakers under `corpus/`:

```powershell
python scripts/prepare_corpus.py --raw "D:\path\to\provided_2a" --out corpus
python -m pytest tests -q
```

Verify that six WAV/`.lab` pairs and `corpus/corpus_report.json` are generated.

## 3. Pronunciation Dictionary & OOV Resolution (Extra Credit)

`prepare_lexicon.py` (or `build_dictionary.py` + `audit_lexicon.py`) combines the base `english_us_arpa` dictionary, G2P candidates (`dictionaries/g2p_oov.dict`), and curated domain overrides (`dictionaries/manual_overrides.dict`), checks phone compatibility against the acoustic model inventory, and writes `outputs/evaluation/lexicon_audit.json` and `lexicon_pronunciation_sources.csv`:

```powershell
python scripts/build_dictionary.py --base "D:\path\english_us_arpa.dict" --manual dictionaries/manual_overrides.dict --out dictionaries/custom_lexicon.dict
python scripts/audit_lexicon.py --dictionary dictionaries/custom_lexicon.dict --corpus corpus --out outputs/evaluation/lexicon_audit.json
```

- **Corpus coverage**: 145 unique word types, 248 pronunciation rows, 11 base OOV types resolved (`0` final OOVs), 100% phone compatibility with `english_us_arpa`.

## 4. Run Forced Alignment

Validate the corpus and run alignment with MFA:

```powershell
mfa validate corpus dictionaries/custom_lexicon.dict english_us_arpa
mfa align corpus dictionaries/custom_lexicon.dict english_us_arpa outputs/textgrids/english_us_arpa --beam 100 --retry_beam 400
```

Or execute the automated one-command pipeline runner:

```powershell
python scripts/run_alignment.py --mfa-env "$env:USERPROFILE/micromamba/root/envs/mfa" --micromamba-root "$env:USERPROFILE/micromamba/root"
```

### Generated TextGrid Outputs (`outputs/textgrids/`)
1. **`english_us_arpa/`**: Primary General American ARPAbet alignment (6/6 valid, 0 gaps/overlaps/`spn`).
2. **`adapted_arpa/`**: Speaker-adapted fMLLR alignment (`mfa adapt`).
3. **`english_mfa/`**: Multi-dialect IPA acoustic model comparison (discloses 1 transcript mismatch in `F2BJRLP2` and 9 `spn` intervals).
4. **`isle_probe/`**: L2 pronunciation variant probe dictionary on ISLE utterances.
5. **`ctc/`**: Neural CTC forced-alignment word-boundary baseline (`python scripts/ctc_baseline_align.py --corpus corpus --out outputs/textgrids/ctc`).

## 5. Inspection, Visualization & Final Report

Generate quantitative boundary evaluations and waveform/spectrogram/TextGrid figures:

```powershell
python scripts/evaluate_alignments.py --corpus corpus --tg_root outputs/textgrids --gold gold --out outputs/evaluation
python scripts/visualize.py --corpus corpus --tg_dir outputs/textgrids/english_us_arpa --out outputs/figures
```

- **Praat Inspection**: All six primary audio/TextGrid pairs were inspected in Praat 7.0.03 (using `scripts/open_in_praat.praat`). Genuine editor screenshots are saved in [`outputs/praat_screenshots/`](outputs/praat_screenshots/) and detailed boundary observations are recorded in [`outputs/evaluation/praat_observations.md`](outputs/evaluation/praat_observations.md).
- **Evaluation Summary**: [`outputs/evaluation/summary.md`](outputs/evaluation/summary.md) and [`outputs/evaluation/validity_and_durations.csv`](outputs/evaluation/validity_and_durations.csv).
- **Final PDF Report**: [`report/report_2a.pdf`](report/report_2a.pdf) (3 pages).
