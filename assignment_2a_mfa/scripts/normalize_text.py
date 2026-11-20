"""Transcript normalization for MFA (.lab files).

Rules (each is unit-tested in tests/test_normalize.py):
 1. Collapse all whitespace/newlines.
 2. Re-join broken hyphenation "seventy- six" -> "seventy six" (number compounds become separate
    words, which are all in the pronunciation dictionary).
 3. Initialisms with periods / possessives become a single token: "S.J.C.'s" -> "sjc's".
    (Their pronunciation comes from dictionaries/manual_overrides.dict.)
 4. Strip punctuation except word-internal apostrophes and hyphens; lowercase.
"""
import re

NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}


def normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    # rule 2: "seventy- six" / "seventy-six" between number words -> space
    def _num(m):
        a, b = m.group(1), m.group(2)
        return f"{a} {b}" if a.lower() in NUMBER_WORDS and b.lower() in NUMBER_WORDS else f"{a}-{b}"
    t = re.sub(r"\b([A-Za-z]+)-\s*([A-Za-z]+)\b", _num, t)
    # rule 3: dotted initialisms (optionally possessive): S.J.C.'s -> SJC's
    t = re.sub(r"\b((?:[A-Za-z]\.){2,})('s)?", lambda m: m.group(1).replace(".", "") + (m.group(2) or ""), t)
    # rule 4: punctuation
    t = re.sub(r"[^\w\s'\-]", " ", t)          # drop . , : ; ! ? " etc.
    t = re.sub(r"(?<!\w)['\-]|['\-](?!\w)", " ", t)  # apostrophes/hyphens not word-internal
    return re.sub(r"\s+", " ", t).strip().lower()


if __name__ == "__main__":
    import sys
    print(normalize(open(sys.argv[1], encoding="utf-8").read()))
