"""Prepare the MFA corpus from the provided raw data.

- Reads audio with libsndfile (handles both NIST SPHERE `.wav` files from BU Radio and
  RIFF files with non-standard PAD chunks from ISLE), checks format, writes canonical
  RIFF PCM16 / 16 kHz / mono.
- Normalizes transcripts -> .lab (same basename), keeps original text in the report.
- Organizes per-speaker directories (enables MFA per-speaker fMLLR adaptation).
Usage: python scripts/prepare_corpus.py --raw <Assignment 2a dir> --out corpus
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from normalize_text import normalize

TARGET_SR = 16000


def speaker_of(stem: str) -> str:
    if stem.upper().startswith("F2B"):
        return "f2b"
    if stem.upper().startswith("ISLE_SESS"):
        return "isle_" + stem.split("_")[1].lower()
    return "unknown"


def header_format(path: Path) -> str:
    head = path.read_bytes()[:8]
    return "NIST_SPHERE" if head.startswith(b"NIST_1A") else ("RIFF" if head.startswith(b"RIFF") else "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="corpus")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    report = []
    for wav in sorted((raw / "wav").glob("*.wav")):
        stem = wav.stem
        txts = [p for p in (raw / "transcripts").iterdir() if p.stem == stem]
        assert len(txts) == 1, f"missing/duplicate transcript for {stem}"
        fmt = header_format(wav)
        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        assert sr == TARGET_SR, f"{stem}: unexpected sr {sr}"
        y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
        spk = speaker_of(stem)
        d = out / spk
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{stem}.wav"), y, TARGET_SR, subtype="PCM_16")
        orig = txts[0].read_text(encoding="utf-8", errors="replace")
        norm = normalize(orig)
        (d / f"{stem}.lab").write_text(norm + "\n", encoding="utf-8")
        report.append({
            "file": stem, "speaker": spk, "source_format": fmt, "sr": sr,
            "duration_s": round(len(y) / sr, 3), "peak": round(float(np.abs(y).max()), 4),
            "clipped_samples": int((np.abs(y) >= 0.999).sum()), "n_words": len(norm.split()),
            "original_text": " ".join(orig.split()), "normalized_text": norm,
        })
        print(f"{stem:38s} {fmt:12s} {len(y)/sr:6.2f}s words={len(norm.split())}")
    (out / "corpus_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
