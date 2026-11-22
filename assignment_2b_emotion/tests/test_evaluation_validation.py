import pandas as pd
import pytest
from mer.evaluation.significance import mcnemar_test, fusion_help
from mer.evaluation.validate_evaluation import validate_predictions


def predictions():
    return pd.DataFrame(dict(fold=[0,0], utt_id=["a","b"], y_true=[0,1], y_pred=[0,0]))


def test_pairing():
    a = predictions()
    b = a.iloc[::-1].copy()
    b.loc[b.utt_id == "b", "y_pred"] = 1
    assert mcnemar_test(a,b)["b_wins_a_loses"] == 1
    assert fusion_help(a,b).helped.sum() == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "truth"])
def test_reject_pairs(change):
    a = predictions()
    b = a.copy()
    if change == "missing": b = b.iloc[:1]
    if change == "duplicate": b = pd.concat([b,b.iloc[:1]])
    if change == "truth": b.loc[0,"y_true"] = 2
    with pytest.raises(ValueError): mcnemar_test(a,b)


def test_probability_mismatch():
    from mer.utils.common import EMOTIONS
    a = predictions()
    for e in EMOTIONS: a["prob_"+e] = 0.
    a["prob_fear"] = 1.
    with pytest.raises(ValueError): validate_predictions(a)
