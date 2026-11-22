"""Render spectrogram + waveform + word & phone alignment tiers for visual inspection."""
import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from praatio import textgrid


def plot_alignment(wav_path: Path, tg_path: Path, out_path: Path, t_start: float = 0.0, t_end: float | None = None):
    y, sr = sf.read(str(wav_path), dtype="float32")
    dur = len(y) / sr
    t_end = min(dur, t_end) if t_end else dur
    s0, s1 = int(t_start * sr), int(t_end * sr)
    y_seg = y[s0:s1]

    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    has_phones = "phones" in tg.tierNames

    fig, axes = plt.subplots(4 if has_phones else 3, 1, figsize=(12, 6.2 if has_phones else 5.0),
                             gridspec_kw={"height_ratios": [1.0, 2.2, 0.7, 0.7] if has_phones else [1.0, 2.2, 0.7]},
                             sharex=True)
    times = np.linspace(t_start, t_end, len(y_seg))
    axes[0].plot(times, y_seg, lw=0.6, color="#1f77b4")
    axes[0].set_ylabel("Amp")
    axes[0].set_title(f"{wav_path.stem} [{t_start:.2f}s - {t_end:.2f}s] ({tg_path.parent.parent.name})")

    S = librosa.amplitude_to_db(np.abs(librosa.stft(y_seg, n_fft=512, hop_length=160)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=160, x_axis="time", y_axis="hz", ax=axes[1], cmap="magma")
    # Shift x-axis to [t_start, t_end]
    im = axes[1].collections[0]
    axes[1].set_ylim(0, 5000)
    axes[1].set_ylabel("Hz")
    # Re-plot spectrogram with exact extent for shared x-axis
    axes[1].clear()
    axes[1].imshow(S, aspect="auto", origin="lower", extent=[t_start, t_end, 0, sr / 2], cmap="magma")
    axes[1].set_ylim(0, 5000)
    axes[1].set_ylabel("Freq (Hz)")

    def draw_tier(ax, tier_name, color):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(tier_name)
        for e in tg.getTier(tier_name).entries:
            if e.end < t_start or e.start > t_end:
                continue
            a = max(t_start, e.start)
            b = min(t_end, e.end)
            ax.axvline(a, color=color, lw=0.8, alpha=0.8)
            ax.axvline(b, color=color, lw=0.8, alpha=0.8)
            if (b - a) >= 0.025:
                ax.text((a + b) / 2, 0.5, e.label, ha="center", va="center", fontsize=8 if tier_name == "words" else 7,
                        rotation=0 if (b - a) > 0.06 else 90)
        # Overlay subtle word lines on spectrogram
        if tier_name == "words":
            for e in tg.getTier(tier_name).entries:
                if t_start <= e.start <= t_end:
                    axes[1].axvline(e.start, color="cyan", lw=0.6, alpha=0.5)

    draw_tier(axes[2], "words", "#2ca02c")
    if has_phones:
        draw_tier(axes[3], "phones", "#d62728")
        axes[3].set_xlabel("Time (s)")
    else:
        axes[2].set_xlabel("Time (s)")

    axes[0].set_xlim(t_start, t_end)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--tg_dir", default="outputs/textgrids/english_us_arpa")
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()

    out = Path(a.out)
    for tg_p in sorted(Path(a.tg_dir).rglob("*.TextGrid")):
        rel = tg_p.relative_to(a.tg_dir)
        wav_p = Path(a.corpus) / rel.with_suffix(".wav")
        if not wav_p.exists():
            continue
        # For long BU files (>10s), plot an informative 5s excerpt; for short ISLE files, plot full utterance
        info = sf.info(str(wav_p))
        dur = info.frames / info.samplerate
        if dur > 10.0:
            plot_alignment(wav_p, tg_p, out / f"{tg_p.stem}_excerpt.png", t_start=0.0, t_end=5.5)
        else:
            plot_alignment(wav_p, tg_p, out / f"{tg_p.stem}_full.png", t_start=0.0, t_end=dur)
    print(f"Saved alignment figures -> {out}")


if __name__ == "__main__":
    main()
