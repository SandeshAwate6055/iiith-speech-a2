"""Evaluation metrics: accuracy, UAR (macro recall), macro P/R/F1, per-class metrics, confusion matrix."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from mer.utils.common import EMOTIONS, NUM_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0
    )
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    return {
        "accuracy": round(acc, 5),
        "uar": round(float(r_m), 5),
        "macro_precision": round(float(p_m), 5),
        "macro_recall": round(float(r_m), 5),
        "macro_f1": round(float(f1_m), 5),
        "per_class": {
            EMOTIONS[i]: {
                "precision": round(float(p_c[i]), 4),
                "recall": round(float(r_c[i]), 4),
                "f1": round(float(f1_c[i]), 4),
                "support": int(sup_c[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }
