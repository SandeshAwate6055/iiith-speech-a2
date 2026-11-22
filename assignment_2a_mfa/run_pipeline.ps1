<#
.SYNOPSIS
  End-to-end automated Forced Alignment Pipeline (Assignment 2A).
.DESCRIPTION
  Runs: (1) audio conversion + transcript normalization, (2) MFA validate + G2P + custom lexicon,
  (3) multi-model MFA alignment (english_us_arpa, english_mfa, adapted), (4) wav2vec2 CTC baseline,
  (5) ISLE L2 minimal-pair pronunciation probe, (6) quantitative evaluation + spectrogram figures.
#>
param(
    [string]$RawDir = "..\Assignment_2\Ass_2\Assignment 2a",
    [string]$PyExe = "..\.venv\Scripts\python.exe",
    [string]$MambaExe = "$env:USERPROFILE\micromamba\Library\bin\micromamba.exe",
    [string]$MambaRoot = "$env:USERPROFILE\micromamba\root"
)
$ErrorActionPreference = "Continue"

function Run-MFA([string[]]$mfaArgs) {
    & $MambaExe run -r $MambaRoot -n mfa mfa @mfaArgs
    if ($LASTEXITCODE -ne 0) { throw "MFA command failed: mfa $($mfaArgs -join ' ')" }
}

Write-Host "=== 1. Prepare Corpus (NIST SPHERE -> RIFF 16k + Normalized .lab) ==="
& $PyExe scripts/prepare_corpus.py --raw $RawDir --out corpus

Write-Host "=== 2. Build Custom Dictionary (G2P + Manual Overrides) ==="
$baseArpa = "$env:USERPROFILE\Documents\MFA\pretrained_models\dictionary\english_us_arpa.dict"
Run-MFA @("g2p", "corpus", "english_us_arpa", "dictionaries/g2p_corpus.dict", "--clean", "--overwrite")
& $PyExe scripts/build_dictionary.py --base $baseArpa --g2p dictionaries/g2p_corpus.dict --manual dictionaries/manual_overrides.dict --out dictionaries/custom_lexicon.dict
Run-MFA @("validate", "corpus", "dictionaries/custom_lexicon.dict", "english_us_arpa", "--clean")

Write-Host "=== 3a. Align with Primary Model: english_us_arpa + custom_lexicon ==="
Run-MFA @("align", "corpus", "dictionaries/custom_lexicon.dict", "english_us_arpa", "outputs/textgrids/english_us_arpa", "--clean", "--overwrite", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid", "-j", "6")

Write-Host "=== 3b. Align with Comparison Model: english_mfa ==="
Run-MFA @("align", "corpus", "english_us_mfa", "english_mfa", "outputs/textgrids/english_mfa", "--clean", "--overwrite", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid", "-j", "6")

Write-Host "=== 3c. Adapt Acoustic Model (MAP) & Align ==="
Run-MFA @("adapt", "corpus", "dictionaries/custom_lexicon.dict", "english_us_arpa", "work/adapted_arpa.zip", "--clean", "--beam", "100", "--retry_beam", "400", "-j", "6")
Run-MFA @("align", "corpus", "dictionaries/custom_lexicon.dict", "work/adapted_arpa.zip", "outputs/textgrids/adapted_arpa", "--clean", "--overwrite", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid", "-j", "6")

Write-Host "=== 4. Modern E2E Baseline: wav2vec2 MMS_FA CTC ==="
& $PyExe scripts/ctc_baseline_align.py --corpus corpus --out outputs/textgrids/ctc

Write-Host "=== 5. ISLE L2 Minimal-Pair Pronunciation Probe ==="
& $PyExe scripts/l2_pronunciation_probe.py --mode make_dict --base_dict dictionaries/custom_lexicon.dict --out_dict dictionaries/isle_probe.dict
Run-MFA @("align", "corpus", "dictionaries/isle_probe.dict", "english_us_arpa", "outputs/textgrids/isle_probe", "--clean", "--overwrite", "--beam", "100", "--retry_beam", "400", "--output_format", "long_textgrid", "-j", "6")
& $PyExe scripts/l2_pronunciation_probe.py --mode analyze --tg_dir outputs/textgrids/isle_probe --out_md outputs/evaluation/l2_probe.md

Write-Host "=== 6. Evaluate & Visualize ==="
& $PyExe scripts/evaluate_alignments.py --corpus corpus --tg_root outputs/textgrids --gold gold --out outputs/evaluation
& $PyExe scripts/visualize.py --corpus corpus --tg_dir outputs/textgrids/english_us_arpa --out outputs/figures
Write-Host "=== 2A Pipeline Complete ==="
