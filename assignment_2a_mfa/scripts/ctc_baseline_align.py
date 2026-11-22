"""Modern E2E CTC forced alignment baseline using torchaudio MMS_FA (wav2vec2-based, Pratap et al. 2023).

Produces word-level TextGrids in outputs/textgrids/ctc/ for quantitative boundary comparison against MFA
(cf. Rousso et al., Interspeech 2024: 'Tradition or Innovation').
Usage: python scripts/ctc_baseline_align.py --corpus corpus --out outputs/textgrids/ctc
"""
import argparse
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from praatio import textgrid


def align_one(wav_path: Path, lab_path: Path, bundle, model, tokenizer, aligner, device):
    y, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == bundle.sample_rate
    dur = len(y) / sr
    waveform = torch.from_numpy(y).unsqueeze(0).to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)

    raw_words = lab_path.read_text(encoding="utf-8").strip().split()
    # MMS_FA tokenizer expects lowercase a-z + apostrophe; map each word safely
    clean_words = [re.sub(r"[^a-z']", "", w.lower()) for w in raw_words]
    clean_words = [w if w else "a" for w in clean_words]

    tokens = tokenizer(clean_words)
    token_spans = aligner(emission[0], tokens)
    ratio = len(y) / emission.shape[1] / sr

    entries = []
    for orig_w, spans in zip(raw_words, token_spans):
        if not spans:
            continue
        t0 = max(0.0, float(spans[0].start) * ratio)
        t1 = min(dur, float(spans[-1].end) * ratio)
        if t1 > t0:
            entries.append((t0, t1, orig_w))

    tg = textgrid.Textgrid(minTimestamp=0.0, maxTimestamp=dur)
    tier = textgrid.IntervalTier("words", entries, minT=0.0, maxT=dur)
    tg.addTier(tier)
    return tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="outputs/textgrids/ctc")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    out_root = Path(a.out)
    for wav in sorted(Path(a.corpus).rglob("*.wav")):
        lab = wav.with_suffix(".lab")
        if not lab.exists():
            continue
        rel = wav.relative_to(a.corpus).with_suffix(".TextGrid")
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tg = align_one(wav, lab, bundle, model, tokenizer, aligner, device)
        tg.save(str(dst), format="long_textgrid", includeBlankSpaces=True)
        print(f"CTC aligned -> {dst} ({len(tg.getTier('words').entries)} words)")


if __name__ == "__main__":
    main()
