"""Preprocess TESS audio: resample to 16 kHz mono, trim silence, peak-normalize to -1 dBFS.

Usage: python -m mer.features.preprocess_audio --meta data/metadata.csv --out data/processed_16k
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16000
PEAK_TARGET = 10 ** (-1.0 / 20.0)  # -1 dBFS (~0.89125)


def process_one(src: Path, dst: Path, top_db: float = 30.0, pad_ms: float = 50.0, normalize: bool = True):
    y, sr = sf.read(str(src), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    if top_db is not None and top_db > 0:
        _, (start, end) = librosa.effects.trim(y, top_db=top_db)
        pad = int(TARGET_SR * pad_ms / 1000.0)
        start = max(0, start - pad)
        end = min(len(y), end + pad)
        if end - start >= int(0.15 * TARGET_SR):  # guard against over-trimming
            y = y[start:end]
    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 1e-6:
            y = y * (PEAK_TARGET / peak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, TARGET_SR, subtype="PCM_16")
    return len(y) / TARGET_SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out", default="data/processed_16k")
    ap.add_argument("--top_db", type=float, default=30.0)
    ap.add_argument("--no_normalize", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.meta)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    durs = []
    proc_paths = []
    for row in tqdm(df.itertuples(), total=len(df), desc="preprocess_16k"):
        dst = out / f"{row.utt_id}.wav"
        if dst.exists():
            info = sf.info(str(dst))
            dur = info.frames / info.samplerate
        else:
            dur = process_one(Path(row.path), dst, top_db=a.top_db, normalize=not a.no_normalize)
        durs.append(round(dur, 4))
        proc_paths.append(str(dst.as_posix()))

    df["proc_path"] = proc_paths
    df["proc_duration_s"] = durs
    df.to_csv(a.meta, index=False)
    print(f"Preprocessed {len(df)} files -> {out} | dur min={min(durs):.2f}s med={np.median(durs):.2f}s max={max(durs):.2f}s")


if __name__ == "__main__":
    main()
