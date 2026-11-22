"""PyTorch Datasets and dynamic padding collate functions for speech, text, and multimodal pipelines."""
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

_SSL_CACHE: dict[str, torch.Tensor] = {}


def _load_ssl_cached(path: Path) -> torch.Tensor:
    key = str(path)
    t = _SSL_CACHE.get(key)
    if t is None:
        t = torch.load(path, map_location="cpu", weights_only=True)  # keep float16 in RAM
        _SSL_CACHE[key] = t
    return t.float()


class TESSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        utt_ids: list[str],
        mode: str = "speech",
        ssl_dir: str | Path | None = None,
        tokenizer_name: str = "roberta-base",
        align_spans_path: str | Path | None = None,
    ):
        self.mode = mode
        sub = df.set_index("utt_id").loc[utt_ids].reset_index()
        self.utt_ids = sub["utt_id"].tolist()
        self.labels = sub["emotion_id"].astype(int).tolist()
        self.transcripts = sub["transcript"].tolist()
        self.speakers = sub["speaker"].tolist()
        self.words = sub["word"].tolist()
        self.ssl_dir = Path(ssl_dir) if ssl_dir else None

        self.encodings = None
        if mode in ("text", "fusion"):
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            self.encodings = tok(self.transcripts, padding=True, truncation=True, max_length=32, return_tensors="pt")

        self.spans = None
        if align_spans_path and Path(align_spans_path).exists():
            self.spans = json.loads(Path(align_spans_path).read_text(encoding="utf-8"))

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        item = {
            "utt_id": uid,
            "label": self.labels[idx],
            "speaker": self.speakers[idx],
            "word": self.words[idx],
        }
        if self.mode in ("speech", "fusion"):
            feat = _load_ssl_cached(self.ssl_dir / f"{uid}.pt")
            item["ssl_feats"] = feat  # [L, T, D]
            if self.spans and uid in self.spans:
                # [start_s, end_s] of target word at 50 Hz frame rate (20ms/frame)
                t_len = feat.shape[1]
                s0, s1 = self.spans[uid]
                f0 = max(0, min(t_len - 1, int(round(s0 * 50.0))))
                f1 = max(f0 + 1, min(t_len, int(round(s1 * 50.0))))
                tmask = torch.zeros(t_len, dtype=torch.bool)
                tmask[f0:f1] = True
                item["target_mask"] = tmask
        if self.mode in ("text", "fusion"):
            item["input_ids"] = self.encodings["input_ids"][idx]
            item["attention_mask"] = self.encodings["attention_mask"][idx]
        return item


def collate_tess(batch: list[dict]) -> dict:
    out = {
        "utt_id": [b["utt_id"] for b in batch],
        "speaker": [b["speaker"] for b in batch],
        "word": [b["word"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if "ssl_feats" in batch[0]:
        L, _, D = batch[0]["ssl_feats"].shape
        max_t = max(b["ssl_feats"].shape[1] for b in batch)
        feats = torch.zeros((len(batch), L, max_t, D), dtype=torch.float32)
        mask = torch.zeros((len(batch), max_t), dtype=torch.bool)
        for i, b in enumerate(batch):
            t = b["ssl_feats"].shape[1]
            feats[i, :, :t, :] = b["ssl_feats"]
            mask[i, :t] = True
        out["ssl_feats"] = feats
        out["speech_mask"] = mask
        if "target_mask" in batch[0]:
            tmask = torch.zeros((len(batch), max_t), dtype=torch.bool)
            for i, b in enumerate(batch):
                t = b["target_mask"].shape[0]
                tmask[i, :t] = b["target_mask"]
            out["target_mask"] = tmask
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    return out
