"""Regenerate global tables from validated, complete test prediction sets."""
import argparse
import json
from pathlib import Path
import pandas as pd
from mer.evaluation.validate_evaluation import completed_runs, validate_run


def aggregate(root="Results/per_fold", out="Results"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, emotion_rows, reports = [], [], {}
    for run in completed_runs(root):
        _, d = validate_run(run)
        reports[run.name] = d
        row = {k: d[k] for k in ["model", "modality", "protocol", "n_folds"]}
        row["run"] = run.name
        for metric in ["accuracy", "uar", "macro_f1"]:
            row[metric] = f"{100*d[metric+'_mean']:.2f} +/- {100*d[metric+'_std']:.2f}"
            row[metric+"_mean"] = d[metric+"_mean"]
            row[metric+"_std"] = d[metric+"_std"]
        rows.append(row)
        for fold in d["folds"]:
            for emotion, values in fold["per_class"].items():
                emotion_rows.append(dict(run=run.name, protocol=d["protocol"], modality=d["modality"], fold=fold["fold"], emotion=emotion, **values))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(out / "accuracy_tables.csv", index=False)
    pd.DataFrame(emotion_rows).to_csv(out / "per_emotion_validated.csv", index=False)
    (out / "evaluation_validation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    cols = ["run", "modality", "protocol", "n_folds", "accuracy", "uar", "macro_f1"]
    lines = ["# Prediction-derived evaluation", "", "Percentages: unweighted fold mean +/- population fold SD (ddof=0), not confidence intervals. Only complete validated runs are included. No universal macro-F1 chance threshold is assumed.", "", "| " + " | ".join(cols) + " |", "|" + "---|"*len(cols)]
    lines += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for row in rows]
    (out / "accuracy_tables.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Regenerated {len(df)} completed runs in {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Results/per_fold")
    ap.add_argument("--out", default="Results")
    a = ap.parse_args()
    aggregate(a.root, a.out)

if __name__ == "__main__":
    main()
