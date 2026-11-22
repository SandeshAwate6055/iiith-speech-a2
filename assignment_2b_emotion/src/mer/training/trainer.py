"""Shared training & evaluation engine for Speech, Text, and Multimodal pipelines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mer.evaluation.metrics import compute_metrics
from mer.utils.common import EMOTIONS, set_seed


def _forward_batch(model, batch, mode: str, device: torch.device, **extra):
    if mode == "speech":
        return model(batch["ssl_feats"].to(device), batch["speech_mask"].to(device))
    if mode == "text":
        return model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    tmask = batch["target_mask"].to(device) if "target_mask" in batch else None
    return model(
        batch["ssl_feats"].to(device),
        batch["speech_mask"].to(device),
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        target_mask=tmask,
        **extra,
    )


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, mode: str, device: torch.device, **extra):
    model.eval()
    ys, preds, probs_all, uids, spks, words = [], [], [], [], [], []
    embs_primary, gates = [], []
    total_loss, n_items = 0.0, 0
    emb_key = {"speech": "temporal_emb", "text": "contextual_emb", "fusion": "fusion_emb"}[mode]

    for batch in loader:
        y = batch["label"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = _forward_batch(model, batch, mode, device, **extra)
            loss = F.cross_entropy(out["logits"], y)
        probs = F.softmax(out["logits"].float(), dim=-1)
        total_loss += float(loss.item()) * y.size(0)
        n_items += y.size(0)
        ys.append(y.cpu().numpy())
        preds.append(probs.argmax(dim=-1).cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        uids.extend(batch["utt_id"])
        spks.extend(batch["speaker"])
        words.extend(batch["word"])
        if emb_key in out and out[emb_key] is not None:
            embs_primary.append(out[emb_key].float().cpu().numpy())
        if out.get("gate") is not None:
            gates.append(out["gate"].float().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    probs_mat = np.concatenate(probs_all)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = round(total_loss / max(1, n_items), 5)

    pred_df = pd.DataFrame({
        "utt_id": uids, "speaker": spks, "word": words,
        "y_true": y_true, "y_pred": y_pred,
        "true_label": [EMOTIONS[i] for i in y_true],
        "pred_label": [EMOTIONS[i] for i in y_pred],
        "confidence": probs_mat.max(axis=1),
    })
    for i, e in enumerate(EMOTIONS):
        pred_df[f"prob_{e}"] = probs_mat[:, i]
    if gates:
        g = np.concatenate(gates).squeeze(-1)
        pred_df["gate"] = g
        metrics["gate_mean"] = round(float(g.mean()), 4)
        metrics["gate_std"] = round(float(g.std()), 4)

    emb_arr = np.concatenate(embs_primary) if embs_primary else None
    return metrics, pred_df, emb_arr


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    cfg: dict,
    ckpt_path: Path,
    device: torch.device,
):
    set_seed(cfg.get("seed", 42))
    model = model.to(device)

    # Separate LR for pretrained transformer encoder vs heads
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder." in name:
            enc_params.append(p)
        else:
            head_params.append(p)
    param_groups = [{"params": head_params, "lr": float(cfg.get("lr", 1e-3))}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": float(cfg.get("encoder_lr", 2e-5))})

    opt = torch.optim.AdamW(param_groups, weight_decay=float(cfg.get("weight_decay", 1e-2)))
    epochs = int(cfg.get("epochs", 40))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    label_smoothing = float(cfg.get("label_smoothing", 0.1))
    patience = int(cfg.get("patience", 8))

    best_f1 = -1.0
    wait = 0
    history = []
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        tr_y, tr_p = [], []
        for batch in train_loader:
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = _forward_batch(model, batch, mode, device)
                loss = F.cross_entropy(out["logits"], y, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss.item()) * y.size(0)
            n_tr += y.size(0)
            tr_y.append(y.cpu().numpy())
            tr_p.append(out["logits"].detach().argmax(dim=-1).cpu().numpy())
        sched.step()

        tr_m = compute_metrics(np.concatenate(tr_y), np.concatenate(tr_p))
        val_m, _, _ = evaluate_loader(model, val_loader, mode, device)
        row = {
            "epoch": ep,
            "train_loss": round(tr_loss / max(1, n_tr), 5),
            "train_f1": tr_m["macro_f1"],
            "val_loss": val_m["loss"],
            "val_f1": val_m["macro_f1"],
            "val_acc": val_m["accuracy"],
        }
        history.append(row)

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            wait = 0
            torch.save({"model": model.state_dict(), "val_f1": best_f1, "epoch": ep}, ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                break

    # Reload best
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model, history, state["epoch"], best_f1
