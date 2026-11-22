# 2A: MFA forced alignment

Run commands from `assignment_2a_mfa`. Use only the six supplied BU/ISLE recordings. Never fetch outside 2A audio or commit raw/licensed audio, the prepared corpus, caches or credentials. Check main-process ownership before downloading models or running alignment.

## Environments and input

Use the root Python helper requirements for scripts. Install MFA separately using Miniforge/conda:

```powershell
conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.4
conda activate mfa
mfa version
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

Run Python scripts in the ML environment; run MFA commands in its conda environment. Input must contain `wav/*.wav` and matching transcript stems under `transcripts/`:

```powershell
python scripts/prepare_corpus.py --raw "D:\path\to\provided_2a" --out corpus
python -m pytest tests -q
```

Verify exactly six WAV/lab pairs and `corpus/corpus_report.json`. BU `.wav` files actually contain NIST SPHERE; libsndfile decodes them, then the script verifies 16 kHz, mixes to mono and writes RIFF PCM16. It normalizes transcripts and groups by speaker. It does not resample arbitrary outside recordings.

## Lexicon and alignment

Substitute the real downloaded dictionary `.dict` path from the MFA model cache below. Manual overrides require pronunciation review; passing syntax/OOV checks does not prove pronunciation accuracy or acoustic phone compatibility.

```powershell
python scripts/build_dictionary.py --base "D:\path\english_us_arpa.dict" --manual dictionaries/manual_overrides.dict --out dictionaries/custom_lexicon.dict
python scripts/audit_lexicon.py --dictionary dictionaries/custom_lexicon.dict --corpus corpus --out outputs/evaluation/lexicon_audit.json
mfa validate corpus dictionaries/custom_lexicon.dict english_us_arpa
mfa align corpus dictionaries/custom_lexicon.dict english_us_arpa outputs/textgrids/english_us_arpa --beam 100 --retry_beam 400
```

Stop on errors or missing/OOV inputs. Review/add pronunciations (optionally via `build_dictionary.py --g2p <generated.dict>`) before repeating validation. Claim zero OOV only after full coverage and MFA validation. Use a fresh output directory for reruns, and never clean another process's work directory. Alternative acoustic models require matching phone inventories and separate outputs; comparison outputs are retained separately; optional English-MFA mismatch/spn limitations are disclosed in the final report.

## Inspection and evidence

```powershell
python scripts/evaluate_alignments.py --corpus corpus --tg_root outputs/textgrids --gold gold --out outputs/evaluation
python scripts/visualize.py --corpus corpus --tg_dir outputs/textgrids/english_us_arpa --out outputs/figures
```

Open each corresponding audio and TextGrid in Praat, select both and choose View & Edit. Inspect words/phones against sound and spectrograms, especially proper nouns, initialisms, pauses and ISLE minimal pairs. Record skipped phones and timing offsets. Automatic validity/duration checks and cross-model agreement are proxies, not gold boundary accuracy. Verify all six output files; an empty folder does not establish success. Primary alignment, scoped Praat observations and final report are complete; see the final recovery section below.

Optional word-level CTC baseline in the torch/torchaudio 2.5.1 environment: `python scripts/ctc_baseline_align.py --corpus corpus --out outputs/textgrids/ctc`. This downloads pretrained weights, does not replace MFA phone alignment and must keep caches private. Review derived artifacts for licensing/privacy before separately authorized sharing.

## Verified local continuation

MFA 3.4.4 is installed with micromamba at `%USERPROFILE%/micromamba/root/envs/mfa`. Use `micromamba run -r <root> -n mfa mfa ...` so native DLL paths are activated. The original GitHub downloader stalled; the working source is the official `MontrealCorpusTools/english_us_arpa` Hugging Face snapshot `9c8d5d8e1dd5d9cf188a5571203de0e09ca6bede`, acoustic version 3.3.0. File hashes and model metadata are recorded in the private `work/models/hf_source.json`; final report records the source and revision. Acoustic/G2P archives are packaged from the official snapshot without altering model bytes.

`python scripts/prepare_lexicon.py` combines the base dictionary, `dictionaries/g2p_oov.dict`, and manual candidates into a corpus-scoped lexicon, checks every phone against the model inventory, and writes `outputs/evaluation/lexicon_pronunciation_sources.csv` and `lexicon_audit.json`. The current corpus has 145 word types, 248 pronunciations, 11 base OOV types and zero custom-lexicon OOVs. G2P command order in MFA 3.4 is `mfa g2p INPUT_PATH G2P_MODEL_PATH OUTPUT_PATH`; the input here is `work/oov_before.txt`. Pronunciation source labels are evidence of origin, not human-verification claims.

Computational one-command reuse (from this directory):

```powershell
python scripts/run_alignment.py --mfa-env "$env:USERPROFILE/micromamba/root/envs/mfa" --micromamba-root "$env:USERPROFILE/micromamba/root"
```

Provide `--raw <supplied-pairs-directory>` only if the six-pair corpus has not been prepared. Completed TextGrids are reused only when recorded input hashes still match. This command does not claim to automate the currently interrupted Praat inspection.


## Completed inspection and final report

All six primary pairs were opened in Praat 7.0.03 on22 November. Full ISLE views and zoomed BU excerpts were visually inspected, with genuine screenshots in `outputs/praat_screenshots/` and scoped notes in `outputs/evaluation/praat_observations.md`. No boundaries were edited and no listening-based phonetic gold set is claimed. The final report is `report/report_2a.pdf`. Optional English-MFA mismatch/spn issues are disclosed; the six primary ARPA outputs pass strict sequence/coverage checks.
