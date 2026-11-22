"""Paired statistical significance test (McNemar's exact/chi2 test) between two runs on the same test set."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from mer.evaluation.validate_evaluation import validate_predictions, require, validate_run
from mer.utils.common import EMOTIONS


def pair_predictions(pred_a, pred_b):
    for df in [pred_a, pred_b]:
        validate_predictions(df)
    keys = ["fold", "utt_id"]
    require(set(map(tuple, pred_a[keys].to_numpy())) == set(map(tuple, pred_b[keys].to_numpy())), "Paired test sets differ")
    m = pred_a.merge(pred_b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    require((m.y_true_a == m.y_true_b).all(), "Paired truth labels differ")
    m["y_true"] = m.y_true_a
    return m


def fusion_help(speech, fusion):
    """Paired corrections and regressions, by fold and emotion."""
    m = pair_predictions(speech, fusion)
    rows = []
    for (fold, label), sub in m.groupby(["fold", "y_true"]):
        a = sub.y_pred_a == sub.y_true
        b = sub.y_pred_b == sub.y_true
        rows.append(dict(fold=int(fold), emotion=EMOTIONS[int(label)], n=len(sub),
                         speech_correct=int(a.sum()), fusion_correct=int(b.sum()),
                         helped=int((~a & b).sum()), harmed=int((a & ~b).sum()),
                         accuracy_delta=float(b.mean()-a.mean())))
    return pd.DataFrame(rows)


def mcnemar_test(pred_a: pd.DataFrame, pred_b: pd.DataFrame):
    m = pair_predictions(pred_a, pred_b)
    a_ok = m["y_pred_a"] == m["y_true"]
    b_ok = m["y_pred_b"] == m["y_true"]
    n01 = int((~a_ok & b_ok).sum())  # B wins
    n10 = int((a_ok & ~b_ok).sum())  # A wins
    n_disc = n01 + n10
    p_val = float(binomtest(min(n01, n10), n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    return {
        "caveat": "Utterance-level exploratory exact McNemar; ignores word/speaker dependence and shared CV training. No population-level significance claim.",
        "n_samples": int(len(m)),
        "acc_a": round(float(a_ok.mean()), 5),
        "acc_b": round(float(b_ok.mean()), 5),
        "a_wins_b_loses": n10,
        "b_wins_a_loses": n01,
        "p_value": round(p_val, 6),
        "significant_0.05": p_val < 0.05,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_emotion_out", default=None, help="Run A=speech, run B=fusion")
    a = ap.parse_args()
    pa, da = validate_run(Path("Results/per_fold") / a.run_a)
    pb, db = validate_run(Path("Results/per_fold") / a.run_b)
    require(da["protocol"] == db["protocol"], "Protocols differ")
    res = {"run_a": a.run_a, "run_b": a.run_b, **mcnemar_test(pa, pb)}
    if a.per_emotion_out:
        fusion_help(pa, pb).to_csv(a.per_emotion_out, index=False)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
