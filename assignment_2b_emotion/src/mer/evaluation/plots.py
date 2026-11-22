"""Generate publication-quality plots for Assignment 2B:
  1. Confusion matrices (row-normalized, summed over folds)
  2. Training & validation curves (loss + macro-F1)
  3. Comparative bar chart across Speech / Text / Fusion & Protocols A / B / C
  4. t-SNE and UMAP of learned representations (Temporal, Contextual, Fusion blocks) + Silhouette/kNN metrics
  5. Learned WavLM layer weights plot
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score

from mer.utils.common import EMOTIONS
from mer.evaluation.validate_evaluation import validate_run, completed_runs

PLOTS = Path("Results/plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_confusion(run_name: str, title: str, out_name: str):
    p = Path("Results/per_fold") / run_name / "predictions.csv"
    if not p.exists() or not (p.parent / "summary.json").exists():
        return
    df, _ = validate_run(p.parent)
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=list(range(len(EMOTIONS))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax, cbar_kws={"label": "Recall (%)"})
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=160)
    plt.close(fig)


def plot_training_curves(run_names: dict[str, str], out_name: str = "training_curves_protoA.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, run in run_names.items():
        hp = Path("Results/per_fold") / run / "train_history.json"
        if not hp.exists():
            continue
        hist = json.loads(hp.read_text(encoding="utf-8"))
        max_ep = max(len(v) for v in hist.values())
        tr_l, va_l, tr_f, va_f = [], [], [], []
        for ep in range(max_ep):
            rows = [v[ep] for v in hist.values() if len(v) > ep]
            tr_l.append(np.mean([r["train_loss"] for r in rows]))
            va_l.append(np.mean([r["val_loss"] for r in rows]))
            tr_f.append(np.mean([r["train_f1"] for r in rows]))
            va_f.append(np.mean([r["val_f1"] for r in rows]))
        eps = np.arange(1, max_ep + 1)
        axes[0].plot(eps, tr_l, "--", alpha=0.6, label=f"{label} train")
        axes[0].plot(eps, va_l, "-", lw=2, label=f"{label} val")
        axes[1].plot(eps, tr_f, "--", alpha=0.6, label=f"{label} train")
        axes[1].plot(eps, va_f, "-", lw=2, label=f"{label} val")
    axes[0].set_title("Loss (Mean across Folds)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title("Macro-F1 (Mean across Folds)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=160)
    plt.close(fig)


def plot_embeddings_panel(blocks: list[tuple[str, str]], method: str = "umap", out_name: str = "embeddings_umap_protoA.png", fold: int = 0):
    """blocks: list of (title, run_name) for Temporal, Contextual, Fusion."""
    fig, axes = plt.subplots(1, len(blocks), figsize=(5.2 * len(blocks), 4.6))
    if len(blocks) == 1:
        axes = [axes]
    palette = sns.color_palette("tab10", len(EMOTIONS))
    stats = {}

    for ax, (title, run) in zip(axes, blocks):
        rdir = Path("Results/per_fold") / run
        ep, pp = rdir / "embeddings.npy", rdir / "predictions.csv"
        if not (ep.exists() and pp.exists()):
            ax.set_title(f"{title} (missing)")
            continue
        df, validation = validate_run(rdir)
        X = np.load(ep, allow_pickle=False)
        mask = df["fold"].to_numpy() == fold
        if not mask.any():
            raise ValueError(f"Missing fold {fold} in {run}")
        X, df = X[mask], df.loc[mask].reset_index(drop=True)
        # Subsample only within this independently trained fold.
        rng = np.random.default_rng(42)
        if len(X) > 1400:
            idx = rng.choice(len(X), size=1400, replace=False)
            X_sub, df_sub = X[idx], df.iloc[idx].reset_index(drop=True)
        else:
            X_sub, df_sub = X, df

        y_emo = df_sub["y_true"].values
        y_spk = (df_sub["speaker"].values == "YAF").astype(int)
        sil_emo = float(silhouette_score(X_sub, y_emo)) if 1 < len(np.unique(y_emo)) < len(y_emo) else None
        sil_spk = float(silhouette_score(X_sub, y_spk)) if len(np.unique(y_spk)) > 1 else 0.0
        stats[title] = {"fold": fold, "n": len(X_sub), "silhouette_emotion": sil_emo,
                        "silhouette_speaker": sil_spk if len(np.unique(y_spk)) > 1 else None,
                        "correspondence": validation["embedding_validation"]}

        n_pca = min(50, X_sub.shape[1], X_sub.shape[0] - 1)
        Xp = PCA(n_components=n_pca, random_state=42).fit_transform(X_sub)
        if method == "tsne":
            Z = TSNE(n_components=2, perplexity=min(30, len(Xp)-1), init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)
        else:
            import umap
            Z = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=42).fit_transform(Xp)

        for ei, emo in enumerate(EMOTIONS):
            for spk, marker in [("OAF", "o"), ("YAF", "^")]:
                m = (df_sub["y_true"].values == ei) & (df_sub["speaker"].values == spk)
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.72, color=palette[ei],
                               marker=marker, label=emo if not any(t.get_label() == emo for t in ax.collections) else None)
        ax.set_title(f"{title} / fold {fold} (n={len(X_sub)})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(EMOTIONS), frameon=False, fontsize=9)
    fig.suptitle(f"Within-fold Representations ({method.upper()}, fold {fold}; axes independent) — Circle=OAF (64y), Triangle=YAF (26y)", fontsize=11, y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(PLOTS / out_name, dpi=170)
    plt.close(fig)
    (PLOTS / f"{Path(out_name).stem}_metrics.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"Saved {out_name} | {stats}")


def plot_layer_weights(run_name: str, out_name: str = "wavlm_layer_weights.png"):
    sp = Path("Results/per_fold") / run_name / "summary.json"
    if not sp.exists():
        return
    d = json.loads(sp.read_text(encoding="utf-8"))
    lw = d.get("layer_weights_mean")
    if not lw:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xs = np.arange(len(lw))
    ax.bar(xs, lw, color="#4c72b0")
    ax.set_xticks(xs)
    ax.set_xticklabels(["CNN"] + [f"L{i}" for i in range(1, len(lw))])
    ax.set_ylabel("Learned Softmax Weight")
    ax.set_xlabel("WavLM-Base+ Layer")
    ax.set_title("Learned Layer Importance (Mean across Folds)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=160)
    plt.close(fig)


def plot_model_comparison(out_name: str = "model_comparison_protocols.png"):
    rows = []
    for run in completed_runs():
        _, d = validate_run(run)
        rows.append({"run": run.name, "protocol": d["protocol"], "f1": d["macro_f1_mean"] * 100, "std": d["macro_f1_std"] * 100})
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(rows)*0.8), 4.8))
    ax.bar(np.arange(len(rows)), [r["f1"] for r in rows],
           yerr=[r["std"] for r in rows], capsize=4, color="#66c2a5")
    ax.set_xticks(np.arange(len(rows)), [r["run"] for r in rows])
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.set_title("Macro-F1: fold mean +/- population fold SD (not confidence intervals)")
    plt.xticks(rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS / out_name, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_model_comparison()
    plot_confusion("speech_wavlm_bigru_asp_protoA", "Speech (WavLM+BiGRU+ASP) — Protocol A", "cm_speech_protoA.png")
    plot_confusion("speech_wavlm_bigru_asp_protoB", "Speech (WavLM+BiGRU+ASP) — Protocol B (LOSO)", "cm_speech_protoB.png")
    plot_confusion("text_roberta_top2ft_protoA", "Text (RoBERTa) — Protocol A", "cm_text_protoA.png")
    plot_confusion("fusion_gated_xattn_protoA", "Fusion (Gated X-Attn) — Protocol A", "cm_fusion_protoA.png")
    plot_confusion("fusion_gated_xattn_protoB", "Fusion (Gated X-Attn) — Protocol B (LOSO)", "cm_fusion_protoB.png")
    plot_training_curves({
        "Speech (S*)": "speech_wavlm_bigru_asp_protoA",
        "Text (T*)": "text_roberta_top2ft_protoA",
        "Fusion (F*)": "fusion_gated_xattn_protoA",
    })
    plot_layer_weights("speech_wavlm_bigru_asp_protoA")
    blocks = [
        ("Temporal Block (Speech)", "speech_wavlm_bigru_asp_protoA"),
        ("Contextual Block (Text)", "text_roberta_top2ft_protoA"),
        ("Fusion Block (Multimodal)", "fusion_gated_xattn_protoA"),
    ]
    plot_embeddings_panel(blocks, method="umap", out_name="embeddings_umap_protoA.png")
    plot_embeddings_panel(blocks, method="tsne", out_name="embeddings_tsne_protoA.png")
