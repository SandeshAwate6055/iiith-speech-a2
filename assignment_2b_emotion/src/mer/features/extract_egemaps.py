"""Extract eGeMAPS v02 (88-dim utterance functionals) using opensmile.

Usage: python -m mer.features.extract_egemaps --meta data/metadata.csv --out data/features/egemaps.csv
"""
import argparse
from pathlib import Path

import opensmile
import pandas as pd
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out", default="data/features/egemaps.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.meta)
    path_col = "proc_path" if "proc_path" in df.columns else "path"
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feats = []
    utt_ids = []
    for r in tqdm(df.itertuples(), total=len(df), desc="egemaps"):
        f = smile.process_file(str(getattr(r, path_col)))
        feats.append(f.iloc[0].values)
        utt_ids.append(r.utt_id)

    cols = list(smile.feature_names)
    out_df = pd.DataFrame(feats, columns=cols)
    out_df.insert(0, "utt_id", utt_ids)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved eGeMAPS {out_df.shape} -> {out} (nans={int(out_df.isna().sum().sum())})")


if __name__ == "__main__":
    main()
