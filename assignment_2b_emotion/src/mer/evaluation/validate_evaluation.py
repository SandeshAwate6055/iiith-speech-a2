"""Read-only validation of completed evaluation artifacts; no training imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mer.evaluation.metrics import compute_metrics
from mer.utils.common import EMOTIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_predictions(df):
    require(not df.empty, "Empty predictions")
    require(set(["fold", "utt_id", "y_true", "y_pred"]).issubset(df), "Missing prediction columns")
    require(not df[["fold", "utt_id", "y_true", "y_pred"]].isna().any().any(), "Null prediction keys/labels")
    require(not df.duplicated(["fold", "utt_id"]).any(), "Duplicate fold/utterance keys")
    for col in ["fold", "y_true", "y_pred"]:
        require(np.isfinite(df[col]).all() and (df[col] == df[col].astype(int)).all(), f"Invalid {col}")
    for col, label in [("y_true", "true_label"), ("y_pred", "pred_label")]:
        require(df[col].isin(range(len(EMOTIONS))).all(), f"Out-of-range {col}")
        if label in df:
            require((df[label] == df[col].map(dict(enumerate(EMOTIONS)))).all(), f"Wrong {label}")
    if "confidence" in df:
        require(np.isfinite(df.confidence).all() and df.confidence.between(0, 1).all(), "Invalid confidence")
    cols = [f"prob_{e}" for e in EMOTIONS]
    if any(c in df for c in cols):
        require(all(c in df for c in cols), "Incomplete probabilities")
        x = df[cols].to_numpy()
        require(np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all(), "Invalid probabilities")
        require(np.allclose(x.sum(1), 1, atol=1e-5), "Probabilities do not sum to one")
        require(np.array_equal(x.argmax(1), df.y_pred), "Probability/prediction mismatch")
        if "confidence" in df:
            require(np.allclose(x.max(1), df.confidence, atol=1e-6), "Confidence mismatch")


def validate_run(run_dir, meta_path="data/metadata.csv", splits_dir="splits"):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    df = pd.read_csv(run_dir / "predictions.csv")
    validate_predictions(df)
    meta = pd.read_csv(meta_path)
    require(not meta.utt_id.duplicated().any(), "Duplicate metadata IDs")
    meta = meta.set_index("utt_id")
    require(df.utt_id.isin(meta.index).all(), "Unknown prediction IDs")
    aligned = meta.loc[df.utt_id]
    require(np.array_equal(df.y_true, aligned.emotion_id), "Metadata label mismatch")
    for col in ["speaker", "word"]:
        if col in df:
            require(np.array_equal(df[col], aligned[col]), f"Metadata {col} mismatch")
    protocol = summary["protocol"].upper()
    split_path = Path(splits_dir) / f"protocol_{protocol}.json"
    folds = json.loads(split_path.read_text(encoding="utf-8"))["folds"]
    require(set(df.fold) == {f["fold"] for f in folds}, "Incomplete or extra folds")
    order = []
    metrics = []
    for f in folds:
        sets = [set(f[k]) for k in ["train", "val", "test"]]
        require(all(len(s) == len(f[k]) for s, k in zip(sets, ["train", "val", "test"])), "Duplicate split IDs")
        require(all(s <= set(meta.index) for s in sets), "Unknown split IDs")
        require(not any(sets[i] & sets[j] for i, j in [(0,1),(0,2),(1,2)]), "Split overlap")
        if protocol in ["A", "B"]:
            words = [set(meta.loc[list(s), "word"]) for s in sets]
            require(not any(words[i] & words[j] for i,j in [(0,1),(0,2),(1,2)]), "Word leakage")
        if protocol == "B":
            require(not set(meta.loc[list(sets[0] | sets[1]), "speaker"]) & set(meta.loc[list(sets[2]), "speaker"]), "Speaker leakage")
        sub = df[df.fold == f["fold"]]
        require(set(sub.utt_id) == sets[2], f"Test ID mismatch fold {f['fold']}")
        order.extend((f["fold"], uid) for uid in f["test"])
        metrics.append(dict(fold=f["fold"], **compute_metrics(sub.y_true.to_numpy(), sub.y_pred.to_numpy())))
    report = dict(summary, folds=metrics, n_folds=len(metrics))
    for metric in ["accuracy", "uar", "macro_f1"]:
        values = [m[metric] for m in metrics]
        report[metric + "_mean"] = float(np.mean(values))
        report[metric + "_std"] = float(np.std(values))
    report["source_sha256"] = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in ["summary.json", "predictions.csv"]}
    report["embedding_validation"] = "absent"
    if (run_dir / "embeddings.npy").exists():
        x = np.load(run_dir / "embeddings.npy", allow_pickle=False)
        require(x.ndim == 2 and len(x) == len(df) and np.isfinite(x).all(), "Invalid embedding shape/values")
        sidecar = run_dir / "embedding_ids.csv"
        if sidecar.exists():
            ids = pd.read_csv(sidecar)
            require(list(zip(ids.fold, ids.utt_id)) == list(zip(df.fold, df.utt_id)), "Embedding ID order mismatch")
            report["embedding_validation"] = "explicit ID sidecar"
        else:
            require(list(zip(df.fold, df.utt_id)) == order, "Legacy embedding order differs from runner split order")
            report["embedding_validation"] = "legacy runner split order; row identity inferred, not independently proven"
    return df, report


def completed_runs(root="Results/per_fold"):
    for run in sorted(Path(root).iterdir()):
        if run.is_dir() and (run / "summary.json").exists() and (run / "predictions.csv").exists():
            yield run
