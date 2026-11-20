import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_text import normalize  # noqa: E402


def test_broken_number_hyphen():
    assert normalize("In nineteen seventy- six, Democratic") == "in nineteen seventy six democratic"


def test_initialism_possessive():
    assert normalize("the S.J.C.'s current leader") == "the sjc's current leader"


def test_acronym_possessive_and_punct():
    assert normalize("As WBUR's\n\nMargo Melnicove reports,") == "as wbur's margo melnicove reports"


def test_word_hyphen_kept():
    assert normalize("to de-politicize judicial") == "to de-politicize judicial"


def test_uppercase_isle():
    assert normalize("I SAID WHITE NOT BAIT\n") == "i said white not bait"


def test_quotes_and_colon():
    assert normalize("Wanted:  Chief Justice 'quoted' word.") == "wanted chief justice quoted word"
