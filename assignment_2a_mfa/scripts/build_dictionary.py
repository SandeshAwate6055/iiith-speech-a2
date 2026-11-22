"""Build a custom MFA pronunciation dictionary combining base english_us_arpa (preserving
pronunciation/silence probabilities), G2P predictions, and hand-verified manual overrides.

Usage: python scripts/build_dictionary.py --base <base_dict_path> --g2p <g2p_dict_path>
       --manual dictionaries/manual_overrides.dict --out dictionaries/custom_lexicon.dict
"""
import argparse
from collections import defaultdict
from pathlib import Path


def read_simple_dict(path: Path) -> dict[str, list[str]]:
    entries = defaultdict(list)
    if not path or not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        word, pron = parts[0].lower(), " ".join(parts[1].split())
        if pron not in entries[word]:
            entries[word].append(pron)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--g2p", default="")
    ap.add_argument("--manual", default="dictionaries/manual_overrides.dict")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="dictionaries/custom_lexicon.dict")
    ap.add_argument("--report", default="outputs/evaluation/oov_report.md")
    a = ap.parse_args()

    base_lines_by_word = defaultdict(list)
    has_probs = False
    for line in Path(a.base).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) >= 6:
            has_probs = True
        w = cols[0].strip().lower()
        base_lines_by_word[w].append(line)

    g2p = read_simple_dict(Path(a.g2p)) if a.g2p else {}
    manual = read_simple_dict(Path(a.manual))

    corpus_words = sorted({w for p in Path(a.corpus).rglob("*.lab") for w in p.read_text(encoding="utf-8").split()})
    oov_before = [w for w in corpus_words if w not in base_lines_by_word]

    def fmt(w: str, pron: str) -> str:
        return f"{w}\t0.99\t0.50\t1.00\t1.00\t{pron}" if has_probs else f"{w}\t{pron}"

    merged = defaultdict(list, {k: list(v) for k, v in base_lines_by_word.items()})
    for w, prons in g2p.items():
        if w not in merged and w not in manual:
            merged[w] = [fmt(w, p) for p in prons]
    for w, prons in manual.items():
        merged[w] = [fmt(w, p) for p in prons]

    oov_after = [w for w in corpus_words if w not in merged]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_lines = [ln for w in sorted(merged.keys()) for ln in merged[w]]
    out.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rlines = [
        "# Custom Pronunciation Dictionary & OOV Resolution Report",
        "",
        f"- **Corpus unique word types:** {len(corpus_words)}",
        f"- **OOV before custom lexicon (base `english_us_arpa`):** {len(oov_before)} (`{', '.join(oov_before)}`)",
        f"- **OOV after custom lexicon (G2P + manual overrides):** {len(oov_after)}",
        "",
        "| OOV Word | G2P Prediction | Final Custom Dictionary Pronunciation(s) | Source |",
        "|---|---|---|---|",
    ]
    for w in oov_before:
        g = "; ".join(g2p.get(w, ["—"]))
        fin = "; ".join(manual[w]) if w in manual else g
        src = "Manual override (verified)" if w in manual else "MFA G2P"
        rlines.append(f"| `{w}` | `{g}` | `{fin}` | {src} |")
    rep.write_text("\n".join(rlines) + "\n", encoding="utf-8")
    print(f"Custom dict -> {out} ({len(merged)} words) | OOVs: {len(oov_before)} -> {len(oov_after)}")


if __name__ == "__main__":
    main()
