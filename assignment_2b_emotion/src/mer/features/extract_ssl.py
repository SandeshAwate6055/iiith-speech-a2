"""Extract and cache frozen SSL speech representations (all hidden layers, fp16).

Saves per-utterance tensors of shape [L, T, D] (float16) to data/features/<tag>/<utt_id>.pt.
Usage: python -m mer.features.extract_ssl --model microsoft/wavlm-base-plus --tag wavlm_base_plus
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--tag", default="wavlm_base_plus")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out) / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.meta)
    path_col = "proc_path" if "proc_path" in df.columns else "path"

    fe = AutoFeatureExtractor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()

    n_layers = None
    hidden_dim = None
    nan_count = 0

    rows = list(df.itertuples())
    for i in tqdm(range(0, len(rows), a.batch_size), desc=f"extract_{a.tag}"):
        batch = rows[i : i + a.batch_size]
        todo = [r for r in batch if not (out_dir / f"{r.utt_id}.pt").exists()]
        if not todo:
            continue
        wavs, lens = [], []
        for r in todo:
            y, sr = sf.read(str(getattr(r, path_col)), dtype="float32")
            assert sr == 16000, f"Expected 16kHz, got {sr} for {r.utt_id}"
            wavs.append(y)
            lens.append(len(y))
        inputs = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of L tensors [B, T, D]
        hs = torch.stack(out.hidden_states, dim=1)  # [B, L, T, D]
        n_layers, hidden_dim = hs.shape[1], hs.shape[3]

        # Compute valid output lengths using model's conv feature extractor formula
        if hasattr(model, "_get_feat_extract_output_lengths"):
            feat_lens = model._get_feat_extract_output_lengths(torch.tensor(lens)).tolist()
        else:
            feat_lens = [hs.shape[2]] * len(todo)

        for b_idx, r in enumerate(todo):
            t_valid = max(1, min(int(feat_lens[b_idx]), hs.shape[2]))
            tensor = hs[b_idx, :, :t_valid, :].to(torch.float16).cpu().contiguous()
            if torch.isnan(tensor).any():
                nan_count += 1
            torch.save(tensor, out_dir / f"{r.utt_id}.pt")

    manifest = {
        "model": a.model, "tag": a.tag, "n_files": len(df),
        "n_layers": n_layers, "hidden_dim": hidden_dim, "nan_files": nan_count,
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"Done {a.tag}: {len(df)} files, shape=[{n_layers}, T, {hidden_dim}], nans={nan_count}")


if __name__ == "__main__":
    main()
