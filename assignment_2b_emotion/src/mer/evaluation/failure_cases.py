"""Extract and visualize 5 real failure cases from test predictions (high-confidence & cross-speaker errors).

Plots spectrogram + F0 contour (parselmouth) + predicted probability distribution for each failure case,
and writes Results/failure_cases.md.
"""
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf

from mer.utils.common import EMOTIONS
from mer.evaluation.validate_evaluation import validate_run


def select_failures(pred_df: pd.DataFrame, k: int = 5):
    if pred_df.empty:
        return pred_df.copy()
    if k <= 0:
        return pred_df.iloc[:0].copy()
    err = pred_df[pred_df["y_true"] != pred_df["y_pred"]].copy()
    if len(err) == 0:
        return err
    # Sort by highest confidence (most confident mistakes), deduplicating by true_label where possible
    err = err.sort_values("confidence", ascending=False)
    picked = []
    seen_pairs = set()
    for _, r in err.iterrows():
        pair = (r["true_label"], r["pred_label"])
        if pair not in seen_pairs or len(err) <= k:
            picked.append(r)
            seen_pairs.add(pair)
        if len(picked) == k:
            break
    if len(picked) < min(k, len(err)):
        used = {r["utt_id"] for r in picked}
        for _, r in err.iterrows():
            if r["utt_id"] not in used:
                picked.append(r)
            if len(picked) == k:
                break
    return pd.DataFrame(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Results/per_fold")
    ap.add_argument("--run", default="speech_wavlm_bigru_asp_protoA")
    ap.add_argument("--loso_run", default="speech_wavlm_bigru_asp_protoB")
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out_md", default="Results/failure_cases.md")
    ap.add_argument("--out_plot", default="Results/plots/failure_cases.png")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta).set_index("utt_id")
    p_a = Path(a.root) / a.run / "predictions.csv"
    p_b = Path(a.root) / a.loso_run / "predictions.csv"
    df_a = validate_run(p_a.parent, a.meta)[0] if (p_a.parent / "summary.json").exists() and p_a.exists() else pd.DataFrame()
    df_b = validate_run(p_b.parent, a.meta)[0] if (p_b.parent / "summary.json").exists() and p_b.exists() else pd.DataFrame()

    # Pick up to 3 from Protocol A and 2 from Protocol B (LOSO) so both regimes are analyzed
    f_a = select_failures(df_a, k=3)
    if len(f_a):
        f_a["source"] = f"Protocol A ({a.run})"
    f_b = select_failures(df_b, k=max(0, 5 - len(f_a)))
    if len(f_b):
        f_b["source"] = f"Protocol B LOSO ({a.loso_run})"
    cases = pd.concat([f_a, f_b], ignore_index=True)
    if len(cases) == 0:
        print("No failure cases found.")
        return

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n), gridspec_kw={"width_ratios": [2.2, 1.0]})
    if n == 1:
        axes = axes[None, :]

    md_lines = [
        "# Error & Failure Case Analysis",
        "",
        "Selection criteria: highest-confidence misclassifications with diverse `(true, predicted)` pairs  -  up to 3 from Protocol A (word-disjoint) and enough from Protocol B (LOSO cross-speaker) to reach five when available.",
        "",
        "| # | Source | Utt ID | Speaker | Word | True | Predicted | Confidence | Duration (s) | Mean F0 (Hz) |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ]

    for idx, r in cases.iterrows():
        uid = r["utt_id"]
        wav_path = meta.loc[uid, "proc_path"] if "proc_path" in meta.columns else meta.loc[uid, "path"]
        y, sr = sf.read(wav_path, dtype="float32")
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch()
        f0_vals = pitch.selected_array["frequency"]
        f0_times = pitch.xs()
        f0_voiced = f0_vals[f0_vals > 0]
        mean_f0 = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0

        # Spectrogram + F0 overlay
        ax0 = axes[idx, 0]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=sr, hop_length=160, x_axis="time", y_axis="hz", ax=ax0, cmap="magma")
        ax0.set_ylim(0, 4000)
        ax0_r = ax0.twinx()
        f0_plot = np.where(f0_vals > 0, f0_vals, np.nan)
        ax0_r.plot(f0_times, f0_plot, color="cyan", lw=1.6, label="F0")
        ax0_r.set_ylim(50, 550)
        ax0_r.set_ylabel("F0 (Hz)", color="cyan", fontsize=8)
        ax0.set_title(f"#{idx+1} {uid} ({r['source']}): True={r['true_label']} -> Pred={r['pred_label']} (p={r['confidence']:.2f})", fontsize=9)

        # Probability bar
        ax1 = axes[idx, 1]
        probs = [r.get(f"prob_{e}", np.nan) for e in EMOTIONS]
        colors = ["#2ca02c" if e == r["true_label"] else ("#d62728" if e == r["pred_label"] else "#7f7f7f") for e in EMOTIONS]
        if np.isfinite(probs).all():
            ax1.barh(EMOTIONS, probs, color=colors)
        else:
            ax1.text(0.05, 0.5, "Class probabilities not saved", transform=ax1.transAxes)
        ax1.set_xlim(0, 1.05)
        ax1.set_xlabel("Probability", fontsize=8)
        ax1.tick_params(labelsize=8)

        md_lines.append(
            f"| {idx+1} | {r['source']} | `{uid}` | {r['speaker']} | {r['word']} | **{r['true_label']}** | **{r['pred_label']}** | {r['confidence']:.3f} | {len(y)/sr:.2f} | {mean_f0:.1f} |"
        )

    plt.tight_layout()
    Path(a.out_plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_plot, dpi=150)
    plt.close(fig)
    cases.to_csv(Path(a.out_md).with_suffix(".csv"), index=False)
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    md_lines += ["", "These are observed prediction errors. F0 and duration are descriptive; no causal explanation or listening judgment is inferred."]
    Path(a.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved {n} failure cases -> {a.out_md} and {a.out_plot}")


if __name__ == "__main__":
    main()
