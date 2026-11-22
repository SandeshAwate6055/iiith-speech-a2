"""Config-driven train & test runner shared by speech_pipeline, text_pipeline, and fusion_pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mer.data.datasets import TESSDataset, collate_tess
from mer.models.fusion import MultimodalEmotionModel
from mer.models.speech import SpeechEmotionModel
from mer.models.text import TextEmotionModel
from mer.training.trainer import evaluate_loader, train_fold
from mer.utils.common import load_config, load_split, set_seed


def build_model(mode: str, cfg: dict):
    mcfg = cfg["model"]
    if mode == "speech":
        return SpeechEmotionModel(**mcfg)
    if mode == "text":
        return TextEmotionModel(**mcfg)
    if mode == "fusion":
        return MultimodalEmotionModel(**mcfg)
    raise ValueError(f"Unknown mode: {mode}")


def _ssl_dir_from_cfg(mode: str, cfg: dict):
    if mode == "text":
        return None
    tag = cfg["data"].get("ssl_tag", "wavlm_base_plus")
    return Path(cfg["data"].get("features_dir", "data/features")) / tag


def _tok_from_cfg(mode: str, cfg: dict):
    if mode == "speech":
        return "roberta-base"
    if mode == "text":
        return cfg["model"].get("model_name", "roberta-base")
    return cfg["model"]["text_cfg"].get("model_name", "roberta-base")


def run_train(mode: str, config_path: str, protocol: str, run_name: str | None = None, folds_override: list[int] | None = None):
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["data"].get("meta", "data/metadata.csv"))
    sp = load_split(protocol, cfg["data"].get("splits_dir", "splits"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = run_name or f"{cfg.get('name', mode)}_proto{protocol.upper()}"
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints")) / exp_name
    out_dir = Path(cfg.get("out_dir", "Results/per_fold")) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.json").write_text(json.dumps({"mode": mode, "protocol": protocol, "cfg": cfg}, indent=1))

    ssl_dir = _ssl_dir_from_cfg(mode, cfg)
    tok_name = _tok_from_cfg(mode, cfg)
    align_spans = cfg["data"].get("align_spans")
    bs = int(cfg["train"].get("batch_size", 32))

    histories = {}
    for f in sp["folds"]:
        k = f["fold"]
        if folds_override is not None and k not in folds_override:
            continue
        set_seed(int(cfg["train"].get("seed", 42)) + k)
        tr_ds = TESSDataset(df, f["train"], mode=mode, ssl_dir=ssl_dir, tokenizer_name=tok_name, align_spans_path=align_spans)
        va_ds = TESSDataset(df, f["val"], mode=mode, ssl_dir=ssl_dir, tokenizer_name=tok_name, align_spans_path=align_spans)
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_tess, num_workers=0)
        va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, collate_fn=collate_tess, num_workers=0)

        model = build_model(mode, cfg)
        ckpt_path = ckpt_dir / f"fold_{k}.pt"
        _, hist, best_ep, best_f1 = train_fold(model, tr_loader, va_loader, mode, cfg["train"], ckpt_path, device)
        histories[f"fold_{k}"] = hist
        print(f"[{exp_name}] fold {k}: best_ep={best_ep} val_f1={best_f1:.4f}")

    (out_dir / "train_history.json").write_text(json.dumps(histories, indent=1), encoding="utf-8")
    return exp_name


def run_test(mode: str, config_path: str, protocol: str, run_name: str | None = None,
             zero_text: bool = False, zero_speech: bool = False, save_tag: str | None = None):
    cfg = load_config(config_path)
    df = pd.read_csv(cfg["data"].get("meta", "data/metadata.csv"))
    sp = load_split(protocol, cfg["data"].get("splits_dir", "splits"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = run_name or f"{cfg.get('name', mode)}_proto{protocol.upper()}"
    ckpt_dir = Path(cfg.get("ckpt_dir", "checkpoints")) / exp_name
    out_dir = Path(cfg.get("out_dir", "Results/per_fold")) / (save_tag or exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ssl_dir = _ssl_dir_from_cfg(mode, cfg)
    tok_name = _tok_from_cfg(mode, cfg)
    align_spans = cfg["data"].get("align_spans")
    bs = int(cfg["train"].get("batch_size", 32))

    fold_metrics, all_preds, all_embs = [], [], []
    layer_weights_folds = []
    extra = {"zero_text": zero_text, "zero_speech": zero_speech} if mode == "fusion" else {}

    for f in sp["folds"]:
        k = f["fold"]
        ckpt_path = ckpt_dir / f"fold_{k}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        te_ds = TESSDataset(df, f["test"], mode=mode, ssl_dir=ssl_dir, tokenizer_name=tok_name, align_spans_path=align_spans)
        te_loader = DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=collate_tess, num_workers=0)

        model = build_model(mode, cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        m, pdf, emb = evaluate_loader(model, te_loader, mode, device, **extra)
        m["fold"] = k
        fold_metrics.append(m)
        pdf.insert(0, "fold", k)
        all_preds.append(pdf)
        if emb is not None:
            all_embs.append(emb)
        # Extract learned SSL layer weights if speech/fusion
        lw = None
        if mode == "speech" and hasattr(model, "branch"):
            lw = torch.softmax(model.branch.layer_sum.weights.detach().cpu(), dim=0).numpy().tolist()
        elif mode == "fusion" and hasattr(model, "speech"):
            lw = torch.softmax(model.speech.layer_sum.weights.detach().cpu(), dim=0).numpy().tolist()
        if lw is not None:
            layer_weights_folds.append(lw)

    summary = {
        "model": cfg.get("name", mode),
        "modality": mode,
        "protocol": protocol.upper(),
        "accuracy_mean": round(float(np.mean([m["accuracy"] for m in fold_metrics])), 4),
        "accuracy_std": round(float(np.std([m["accuracy"] for m in fold_metrics])), 4),
        "uar_mean": round(float(np.mean([m["uar"] for m in fold_metrics])), 4),
        "uar_std": round(float(np.std([m["uar"] for m in fold_metrics])), 4),
        "macro_f1_mean": round(float(np.mean([m["macro_f1"] for m in fold_metrics])), 4),
        "macro_f1_std": round(float(np.std([m["macro_f1"] for m in fold_metrics])), 4),
        "folds": fold_metrics,
    }
    if any("gate_mean" in m for m in fold_metrics):
        summary["gate_mean"] = round(float(np.mean([m["gate_mean"] for m in fold_metrics])), 4)
        summary["gate_std"] = round(float(np.mean([m["gate_std"] for m in fold_metrics])), 4)
    if layer_weights_folds:
        summary["layer_weights_mean"] = [round(float(x), 4) for x in np.mean(layer_weights_folds, axis=0)]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    if all_embs:
        np.save(out_dir / "embeddings.npy", np.concatenate(all_embs, axis=0))

    print(
        f"[TEST {save_tag or exp_name}] acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} | "
        f"UAR={summary['uar_mean']:.4f}±{summary['uar_std']:.4f} | "
        f"macro_f1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}"
        + (f" | gate={summary['gate_mean']:.3f}" if "gate_mean" in summary else "")
    )
    return summary


def cli(mode: str, action: str, default_config: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--protocol", default="A", choices=["A", "B", "C", "a", "b", "c"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--zero_text", action="store_true")
    ap.add_argument("--zero_speech", action="store_true")
    ap.add_argument("--save_tag", default=None)
    a = ap.parse_args()
    if action == "train":
        run_train(mode, a.config, a.protocol.upper(), run_name=a.name)
    else:
        run_test(mode, a.config, a.protocol.upper(), run_name=a.name,
                 zero_text=a.zero_text, zero_speech=a.zero_speech, save_tag=a.save_tag)
