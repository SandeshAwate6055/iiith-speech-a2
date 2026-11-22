"""Verify dataset completeness and zero data leakage across all split protocols."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata.csv"
SPLITS = ROOT / "splits"


def _load():
    df = pd.read_csv(META).set_index("utt_id")
    return df


def test_metadata_completeness():
    df = _load()
    assert len(df) == 2800
    assert df["word"].nunique() == 200
    assert df["speaker"].nunique() == 2
    assert df["emotion"].nunique() == 7
    assert (df["emotion"].value_counts() == 400).all()


def test_protocol_a_no_word_leakage():
    df = _load()
    sp = json.loads((SPLITS / "protocol_A.json").read_text())
    all_test = []
    for f in sp["folds"]:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        w_tr = set(df.loc[list(tr), "word"])
        w_va = set(df.loc[list(va), "word"])
        w_te = set(df.loc[list(te), "word"])
        assert w_tr.isdisjoint(w_va) and w_tr.isdisjoint(w_te) and w_va.isdisjoint(w_te)
        # Balanced emotions in test
        assert (df.loc[list(te), "emotion"].value_counts() == 80).all()
        all_test.extend(te)
    assert len(all_test) == 2800 and len(set(all_test)) == 2800


def test_protocol_b_no_speaker_or_word_leakage():
    df = _load()
    sp = json.loads((SPLITS / "protocol_B.json").read_text())
    for f in sp["folds"]:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        s_tr = set(df.loc[list(tr), "speaker"]) | set(df.loc[list(va), "speaker"])
        s_te = set(df.loc[list(te), "speaker"])
        assert s_tr.isdisjoint(s_te)
        w_tr = set(df.loc[list(tr), "word"]) | set(df.loc[list(va), "word"])
        w_te = set(df.loc[list(te), "word"])
        assert w_tr.isdisjoint(w_te)
        assert (df.loc[list(te), "emotion"].value_counts() == 40).all()


def test_protocol_c_disjoint_utts():
    sp = json.loads((SPLITS / "protocol_C.json").read_text())
    all_test = []
    for f in sp["folds"]:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
        all_test.extend(te)
    assert len(all_test) == 2800 and len(set(all_test)) == 2800
