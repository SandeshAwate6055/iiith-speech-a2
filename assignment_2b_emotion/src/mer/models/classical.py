"""Classical ML baselines: eGeMAPS + RBF-SVM (speech) and TF-IDF + LogisticRegression (text).

Usage:
  python -m mer.models.classical --modality speech --protocol A
  python -m mer.models.classical --modality text --protocol A
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mer.evaluation.metrics import compute_metrics
from mer.utils.common import EMOTIONS, load_split


def run_classical(modality: str, protocol: str, meta_path="data/metadata.csv",
                  egemaps_path="data/features/egemaps.csv", out_dir="Results/per_fold"):
    df = pd.read_csv(meta_path).set_index("utt_id")
    sp = load_split(protocol)
    eg_df = pd.read_csv(egemaps_path).set_index("utt_id") if modality == "speech" else None

    tag = f"classical_{modality}_proto{protocol.upper()}"
    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    all_preds = []
    for f in sp["folds"]:
        k = f["fold"]
        tr_ids = f["train"] + f["val"]
        te_ids = f["test"]
        y_tr = df.loc[tr_ids, "emotion_id"].values
        y_te = df.loc[te_ids, "emotion_id"].values

        if modality == "speech":
            X_tr = eg_df.loc[tr_ids].values
            X_te = eg_df.loc[te_ids].values
            clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C=10.0, kernel="rbf", probability=True, random_state=42))])
        else:
            X_tr = df.loc[tr_ids, "transcript"].values
            X_te = df.loc[te_ids, "transcript"].values
            clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("lr", LogisticRegression(max_iter=1000, random_state=42))])

        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        y_pred = probs.argmax(axis=1)
        m = compute_metrics(y_te, y_pred)
        m["fold"] = k
        fold_metrics.append(m)

        pdf = pd.DataFrame({
            "fold": k, "utt_id": te_ids,
            "speaker": df.loc[te_ids, "speaker"].values,
            "word": df.loc[te_ids, "word"].values,
            "y_true": y_te, "y_pred": y_pred,
            "true_label": [EMOTIONS[i] for i in y_te],
            "pred_label": [EMOTIONS[i] for i in y_pred],
            "confidence": probs.max(axis=1),
        })
        all_preds.append(pdf)

    summary = {
        "model": "eGeMAPS+SVM" if modality == "speech" else "TFIDF+LogReg",
        "modality": modality,
        "protocol": protocol.upper(),
        "accuracy_mean": round(float(np.mean([m["accuracy"] for m in fold_metrics])), 4),
        "accuracy_std": round(float(np.std([m["accuracy"] for m in fold_metrics])), 4),
        "uar_mean": round(float(np.mean([m["uar"] for m in fold_metrics])), 4),
        "uar_std": round(float(np.std([m["uar"] for m in fold_metrics])), 4),
        "macro_f1_mean": round(float(np.mean([m["macro_f1"] for m in fold_metrics])), 4),
        "macro_f1_std": round(float(np.std([m["macro_f1"] for m in fold_metrics])), 4),
        "folds": fold_metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    pd.concat(all_preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    print(f"[{tag}] acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} | macro_f1={summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["speech", "text"], required=True)
    ap.add_argument("--protocol", default="A")
    a = ap.parse_args()
    run_classical(a.modality, a.protocol)
