"""Generate deterministic, leakage-free evaluation splits for TESS.

Protocols:
  - A (Primary): 5-fold target-word-disjoint CV (128 train / 32 val / 40 test words per fold; both speakers).
  - B (Stress test): Leave-One-Speaker-Out (LOSO) + word-disjoint (2 folds: OAF->YAF, YAF->OAF;
                     128 train / 32 val words on train speaker, 40 disjoint test words on test speaker).
  - C (Reference): 5-fold random stratified CV (80/10/10-like: 4 folds split into 70/10 train/val, 20% test).
                   Explicitly flagged as optimistic due to speaker & target-word overlap.

Usage: python -m mer.data.make_splits --meta data/metadata.csv --out splits
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_protocol_a(df: pd.DataFrame, seed: int = 42, n_folds: int = 5):
    rng = np.random.default_rng(seed)
    words = np.array(sorted(df["word"].unique()))
    rng.shuffle(words)
    word_folds = np.array_split(words, n_folds)

    folds = []
    for k in range(n_folds):
        test_w = set(word_folds[k])
        val_w = set(word_folds[(k + 1) % n_folds][:32])
        train_w = set(words) - test_w - val_w
        folds.append({
            "fold": k,
            "train": df[df["word"].isin(train_w)]["utt_id"].tolist(),
            "val": df[df["word"].isin(val_w)]["utt_id"].tolist(),
            "test": df[df["word"].isin(test_w)]["utt_id"].tolist(),
            "train_words": sorted(train_w),
            "val_words": sorted(val_w),
            "test_words": sorted(test_w),
        })
    return {"protocol": "A_word_disjoint_5fold", "seed": seed, "folds": folds}


def make_protocol_b(df: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    words = np.array(sorted(df["word"].unique()))
    rng.shuffle(words)
    test_w = set(words[:40])
    val_w = set(words[40:72])
    train_w = set(words[72:])

    folds = []
    for k, (tr_spk, te_spk) in enumerate([("OAF", "YAF"), ("YAF", "OAF")]):
        folds.append({
            "fold": k,
            "train_speaker": tr_spk,
            "test_speaker": te_spk,
            "train": df[(df["speaker"] == tr_spk) & (df["word"].isin(train_w))]["utt_id"].tolist(),
            "val": df[(df["speaker"] == tr_spk) & (df["word"].isin(val_w))]["utt_id"].tolist(),
            "test": df[(df["speaker"] == te_spk) & (df["word"].isin(test_w))]["utt_id"].tolist(),
            "test_all_words": df[df["speaker"] == te_spk]["utt_id"].tolist(),
        })
    return {"protocol": "B_loso_word_disjoint", "seed": seed, "folds": folds}


def make_protocol_c(df: pd.DataFrame, seed: int = 42, n_folds: int = 5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    utt_ids = df["utt_id"].values
    labels = df["emotion_id"].values
    fold_indices = [te for _, te in skf.split(utt_ids, labels)]

    folds = []
    for k in range(n_folds):
        te_idx = fold_indices[k]
        # Take half of next fold as val (~10% total), rest as train (~70% total)
        next_idx = fold_indices[(k + 1) % n_folds]
        val_idx = next_idx[: len(next_idx) // 2]
        tr_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([te_idx, val_idx]))
        folds.append({
            "fold": k,
            "train": utt_ids[tr_idx].tolist(),
            "val": utt_ids[val_idx].tolist(),
            "test": utt_ids[te_idx].tolist(),
        })
    return {
        "protocol": "C_random_stratified_5fold",
        "warning": "Optimistic reference only: shares speakers and target words across train/test.",
        "seed": seed,
        "folds": folds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.meta)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in [("protocol_A.json", make_protocol_a),
                     ("protocol_B.json", make_protocol_b),
                     ("protocol_C.json", make_protocol_c)]:
        obj = fn(df, seed=a.seed)
        (out / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
        f0 = obj["folds"][0]
        print(f"{name}: {len(obj['folds'])} folds | fold0 train={len(f0['train'])} val={len(f0['val'])} test={len(f0['test'])}")


if __name__ == "__main__":
    main()
