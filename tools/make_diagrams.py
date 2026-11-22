"""Render block-style architecture diagrams (PNG) used by the READMEs and PDF reports.

Every box mirrors the implemented code (src/mer/models/*.py, assignment_2a_mfa/scripts/*).
Usage: python tools/make_diagrams.py   (needs matplotlib)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
C = dict(inp="#dbeafe", pre="#e0e7ff", feat="#fef3c7", seq="#dcfce7", fuse="#fce7f3",
         head="#fee2e2", out="#f3f4f6", ext="#ede9fe", note="#ffffff")
EDGE = "#334155"


def new(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub="", kind="pre", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C[kind], ec=EDGE, lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#0f172a")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 1.6, color="#334155", linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color="#0f172a")


def arrow(ax, p, q, label="", rad=0.0, fs=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#1e293b",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, label, ha="center", fontsize=fs, color="#7c2d12",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def save(fig, name, title):
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 2A pipeline
def fig_2a():
    fig, ax = new(13.2, 6.15)
    box(ax, 0.2, 4.9, 2.4, 1.1, "Audio (6 files)", "3 BU Radio (NIST SPHERE)\n3 ISLE learner (RIFF)", "inp")
    box(ax, 0.2, 3.4, 2.4, 1.1, "Transcripts (6)", "raw .txt / .TXT", "inp")
    box(ax, 3.2, 4.9, 2.6, 1.1, "Audio conversion", "libsndfile decode, mono,\n16 kHz PCM16 RIFF, per-speaker dirs", "pre")
    box(ax, 3.2, 3.4, 2.6, 1.1, "Transcript normalisation", "normalize_text.py -> .lab\n(lower-case, punctuation removed)", "pre")
    box(ax, 6.4, 4.9, 2.7, 1.1, "Base lexicon", "english_us_arpa.dict (v3.3.0)", "feat")
    box(ax, 6.4, 3.4, 2.7, 1.1, "OOV handling", "11 OOV types: MFA G2P +\nmanual overrides (9 words)", "feat")
    box(ax, 9.7, 4.15, 3.3, 1.3, "custom_lexicon.dict", "145 word types / 248 pronunciations\n0 OOV, phones match model inventory", "feat")
    box(ax, 3.2, 1.55, 2.9, 1.15, "audit_lexicon.py + mfa validate", "syntax / coverage / phone checks", "seq")
    box(ax, 6.7, 1.55, 3.0, 1.15, "MFA 3.4.4 forced alignment", "english_us_arpa 3.3.0 GMM-HMM\nbeam 100, retry-beam 400, fMLLR", "fuse")
    box(ax, 10.2, 1.55, 2.8, 1.15, "TextGrids", "tier 1 words, tier 2 phones\n6 primary files", "head")
    box(ax, 0.2, 0.05, 3.0, 1.0, "Praat inspection", "6 pairs opened, 2 screenshots", "out")
    box(ax, 3.6, 0.05, 3.0, 1.0, "Structural evaluation", "gaps / overlaps / spn / word sequence", "out")
    box(ax, 7.0, 0.05, 2.9, 1.0, "Visualisation", "waveform + spectrogram + tiers", "out")
    box(ax, 10.2, 0.05, 2.8, 1.0, "Extra-credit systems", "adapted_arpa, english_mfa,\nisle_probe, CTC (MMS_FA)", "ext")
    arrow(ax, (2.6, 5.45), (3.2, 5.45)); arrow(ax, (2.6, 3.95), (3.2, 3.95))
    arrow(ax, (5.8, 3.95), (6.4, 3.95)); arrow(ax, (7.75, 4.9), (7.75, 4.5))
    arrow(ax, (9.1, 4.0), (9.7, 4.6)); arrow(ax, (9.1, 5.45), (9.7, 5.1))
    arrow(ax, (4.5, 3.4), (4.5, 2.7), "vocabulary")
    arrow(ax, (11.3, 4.15), (8.9, 2.7), "dictionary", rad=0.0)
    arrow(ax, (4.5, 4.9), (7.6, 2.7), "16 kHz wav", rad=-0.15)
    arrow(ax, (6.1, 2.1), (6.7, 2.1)); arrow(ax, (9.7, 2.1), (10.2, 2.1))
    arrow(ax, (11.6, 1.55), (11.6, 1.05)); arrow(ax, (10.9, 1.55), (8.5, 1.05)); arrow(ax, (10.4, 1.55), (5.1, 1.05), rad=0.0)
    arrow(ax, (10.2, 1.7), (1.8, 1.05), rad=0.0)
    save(fig, "arch_2a_pipeline.png", "Assignment 2A - MFA forced-alignment pipeline (as implemented)")


# ---------------------------------------------------------------- 2B overview
def fig_2b_overview():
    fig, ax = new(13.2, 5.6)
    box(ax, 0.2, 3.9, 2.2, 1.0, "Speech input", "TESS wav, 24.4 kHz,\n2,800 clips", "inp")
    box(ax, 0.2, 0.7, 2.2, 1.0, "Text input", '"Say the word <w>"\n200 target words', "inp")
    box(ax, 2.9, 3.9, 2.3, 1.0, "Block 1: preprocessing", "16 kHz mono, trim 30 dB,\npeak -1 dBFS", "pre")
    box(ax, 2.9, 0.7, 2.3, 1.0, "Block 1: preprocessing", "RoBERTa BPE tokens\nmax length 32", "pre")
    box(ax, 5.7, 3.9, 2.4, 1.0, "Block 2: WavLM-Base+", "frozen, cached [13, T, 768]", "feat")
    box(ax, 5.7, 0.7, 2.4, 1.0, "Block 2: RoBERTa-base", "token states [N, 768]", "feat")
    box(ax, 8.6, 3.9, 2.0, 1.0, "Block 3: temporal", "layer-sum, BiGRU, ASP\n-> [512]", "seq")
    box(ax, 8.6, 0.7, 2.0, 1.0, "Block 3: contextual", "top-2 layers tuned,\nattn-pool -> [256]", "seq")
    box(ax, 8.1, 2.2, 3.0, 1.2, "Block 4: fusion", "concat | bidirectional gated\ncross-attention (4 heads)", "fuse")
    box(ax, 11.5, 2.2, 1.6, 1.2, "Block 5: head", "LN-MLP -> 7 logits", "head")
    box(ax, 11.5, 0.5, 1.6, 0.9, "Emotion", "7 classes", "out")
    for a, b in [((2.4, 4.4), (2.9, 4.4)), ((5.2, 4.4), (5.7, 4.4)), ((8.1, 4.4), (8.6, 4.4)),
                 ((2.4, 1.2), (2.9, 1.2)), ((5.2, 1.2), (5.7, 1.2)), ((8.1, 1.2), (8.6, 1.2)),
                 ((10.6, 4.1), (10.2, 3.4)), ((10.6, 1.5), (10.2, 2.2)), ((11.1, 2.8), (11.5, 2.8)),
                 ((12.3, 2.2), (12.3, 1.4))]:
        arrow(ax, a, b)
    ax.text(6.6, 5.35, "Speech branch", fontsize=9, fontweight="bold", color="#166534", ha="center")
    ax.text(6.6, 0.25, "Text branch", fontsize=9, fontweight="bold", color="#166534", ha="center")
    ax.text(5.0, 2.75, "Frame sequences [T, 256] and token sequences [N, 256]\nare exchanged only in gated cross-attention fusion.",
            fontsize=7.5, color="#475569", ha="center")
    save(fig, "arch_2b_overview.png", "Assignment 2B - five functional blocks (speech, text and fusion)")


# ---------------------------------------------------------------- speech
def fig_speech():
    fig, ax = new(14.7, 2.9)
    items = [("Cached WavLM-Base+", "[B, 13, T, 768]\n(50 frames/s)", "inp"),
             ("LayerWeightedSum", "softmax over 13 layers\n-> [B, T, 768]", "feat"),
             ("SpecAugment", "train only, p = 0.3\ntime + feature mask", "pre"),
             ("Projection", "Linear 768->256\nLayerNorm, Dropout 0.3", "feat"),
             ("BiGRU", "1 layer, 128 per direction\n-> [B, T, 256]", "seq"),
             ("Attentive Stats Pool", "mean + std (attn-weighted)\n-> [B, 512]", "seq"),
             ("MLPHead", "LN, Drop, 512->256, GELU,\nDrop, 256->7", "head")]
    w, g = 1.78, 0.28
    for i, (t, s, k) in enumerate(items):
        x = 0.15 + i * (w + g)
        box(ax, x, 0.9, w, 1.4, t, s, k, fs=7.4)
        if i:
            arrow(ax, (x - g, 1.6), (x, 1.6))
    ax.text(0.15 + 6 * (w + g) + w / 2, 0.6, "logits [B, 7]", ha="center", fontsize=8, fontweight="bold")
    ax.text(7.0, 0.15, "temporal embedding = ASP output [B, 512] (block-3 visualisation); loss: cross-entropy, label smoothing 0.1; AdamW lr 1e-3",
            ha="center", fontsize=7.5, color="#475569")
    save(fig, "arch_2b_speech.png", "Speech model: WavLM-BiGRU-ASP (SpeechEmotionModel)")


# ---------------------------------------------------------------- text
def fig_text():
    fig, ax = new(12.8, 2.9)
    items = [("Transcript", '"say the word <w>"', "inp"),
             ("RoBERTa tokenizer", "FacebookAI/roberta-base\n[B, N<=32]", "pre"),
             ("RoBERTa encoder", "12 layers; top 2 fine-tuned\n(lr 2e-5), rest frozen\n[B, N, 768]", "feat"),
             ("Projection", "Linear 768->256\nLayerNorm, Dropout 0.3", "feat"),
             ("Masked AttnPool", "tanh bottleneck 128\n-> [B, 256]", "seq"),
             ("MLPHead", "256->256->7", "head")]
    w, g = 1.7, 0.32
    for i, (t, s, k) in enumerate(items):
        x = 0.15 + i * (w + g)
        box(ax, x, 0.8, w, 1.5, t, s, k, fs=7.8)
        if i:
            arrow(ax, (x - g, 1.55), (x, 1.55))
    ax.text(6.4, 0.3, "contextual embedding = pooled [B, 256] (block-3 visualisation); head lr 1e-3, encoder lr 2e-5",
            ha="center", fontsize=7.5, color="#475569")
    save(fig, "arch_2b_text.png", "Text model: RoBERTa + attention pooling (TextEmotionModel)")


# ---------------------------------------------------------------- fusion
def fig_fusion():
    fig, ax = new(13.6, 7.4)
    box(ax, 0.2, 5.6, 2.3, 1.0, "Speech frames", "BiGRU output [B, T, 256]", "seq")
    box(ax, 0.2, 2.9, 2.3, 1.0, "Text tokens", "projected states [B, N, 256]", "seq")
    box(ax, 3.0, 5.6, 1.9, 1.0, "s_frame_proj", "Linear 256->256", "feat")
    box(ax, 3.0, 2.9, 1.9, 1.0, "t_tok_proj", "Linear 256->256", "feat")
    box(ax, 5.4, 5.7, 2.6, 1.3, "Speech -> Text attention", "MultiheadAttention, 4 heads\nQ = speech, K,V = text", "fuse")
    box(ax, 5.4, 2.5, 2.6, 1.3, "Text -> Speech attention", "MultiheadAttention, 4 heads\nQ = text, K,V = speech", "fuse")
    box(ax, 8.4, 5.7, 1.9, 1.3, "Residual + LayerNorm", "s_ctx [B, T, 256]", "pre")
    box(ax, 8.4, 2.5, 1.9, 1.3, "Residual + LayerNorm", "t_ctx [B, N, 256]", "pre")
    box(ax, 10.7, 5.7, 1.9, 1.3, "AttnPool", "hs_x [B, 256]", "seq")
    box(ax, 10.7, 2.5, 1.9, 1.3, "AttnPool", "ht_x [B, 256]", "seq")
    box(ax, 10.2, 4.05, 2.8, 1.1, "Modality dropout (p=0.2)", "train only; never drops both", "ext")
    box(ax, 5.4, 4.05, 4.2, 1.1, "Vector gate", "z = sigmoid(Linear 512->256([hs_x ; ht_x]))\nfused = z * hs_x + (1 - z) * ht_x", "fuse")
    box(ax, 2.9, 4.05, 2.1, 1.1, "MLPHead", "LN, Drop, 256->256,\nGELU, Drop, 256->7", "head")
    box(ax, 0.2, 4.05, 2.2, 1.1, "logits [B, 7]", "fused embedding [B, 256]", "out")
    # arrows
    arrow(ax, (2.5, 6.1), (3.0, 6.1)); arrow(ax, (2.5, 3.4), (3.0, 3.4))
    arrow(ax, (4.9, 6.35), (5.4, 6.35)); arrow(ax, (4.9, 3.15), (5.4, 3.15))
    arrow(ax, (8.0, 6.35), (8.4, 6.35)); arrow(ax, (8.0, 3.15), (8.4, 3.15))
    arrow(ax, (10.3, 6.35), (10.7, 6.35)); arrow(ax, (10.3, 3.15), (10.7, 3.15))
    arrow(ax, (11.65, 5.7), (11.65, 5.15)); arrow(ax, (11.65, 3.8), (11.65, 4.05), rad=0.0)
    arrow(ax, (10.2, 4.6), (9.6, 4.6)); arrow(ax, (5.4, 4.6), (5.0, 4.6)); arrow(ax, (2.9, 4.6), (2.4, 4.6))
    ax.text(6.5, 0.95, "Each attention block takes its Query from one stream and Key/Value from the other\n(text padding / speech padding masks applied). The gated fused embedding [B, 256]\nis the block-4 embedding used for t-SNE/UMAP.", fontsize=7.8, color="#475569", ha="center")
    ax.text(6.5, 0.25, "Concat variant (fusion_concat.yaml): speech ASP [B,512] -> Linear 512->256 + LN + GELU; text pool [B,256] -> Linear 256->256 + LN + GELU;\nconcat [B,512] -> Linear 512->256 + LN + GELU -> MLPHead. Both branches are otherwise identical to the speech and text models.",
            fontsize=7.3, color="#475569", ha="center")
    save(fig, "arch_2b_fusion.png", "Fusion model: bidirectional gated cross-attention (MultimodalEmotionModel, fusion_type=gated_xattn)")


if __name__ == "__main__":
    fig_2a(); fig_2b_overview(); fig_speech(); fig_text(); fig_fusion()
