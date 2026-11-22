"""ISLE L2 Minimal-Pair Pronunciation Assessment Probe (Extra Credit / Research Extension).

The 3 ISLE utterances are non-native learners reading minimal-pair contrasts:
  01: I SAID WHITE NOT BAIT   (contrast: /w/ vs /v/ L1 German transfer; /ay/ vs /ey/)
  02: I SAID NEW NOT NO       (contrast: /uw/ vs /ow/; yod /y uw/)
  03: I SAID BAD NOT BED      (contrast: /ae/ vs /eh/, classic German/Italian L1 confusion)

Method: construct a probe dictionary where each target word has BOTH canonical and
confusable-minimal-pair pronunciation variants. MFA's Viterbi forced alignment selects
the acoustically most likely pronunciation variant per token, revealing whether the
learner's production aligned closer to the canonical phone or the L1-transferred variant.
"""
import argparse
from pathlib import Path

from praatio import textgrid

PROBE_VARIANTS = {
    "white": ["W AY1 T", "V AY1 T", "W EY1 T"],
    "bait":  ["B EY1 T", "B AY1 T", "B EH1 T"],
    "new":   ["N UW1", "N Y UW1", "N OW1"],
    "no":    ["N OW1", "N UW1"],
    "bad":   ["B AE1 D", "B EH1 D", "B AE1 T", "B EH1 T"],
    "bed":   ["B EH1 D", "B AE1 D", "B EH1 T"],
}


def make_probe_dict(base_dict: Path, out_dict: Path):
    lines = base_dict.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = []
    for ln in lines:
        parts = ln.strip().split(None, 1)
        if parts and parts[0].lower() in PROBE_VARIANTS:
            continue
        kept.append(ln)
    has_probs = any(len(ln.split("\t")) >= 6 for ln in lines[:10])
    for w, prons in PROBE_VARIANTS.items():
        for p in prons:
            kept.append(f"{w}\t0.99\t0.50\t1.00\t1.00\t{p}" if has_probs else f"{w}\t{p}")
    out_dict.parent.mkdir(parents=True, exist_ok=True)
    out_dict.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Wrote probe dict -> {out_dict} ({len(PROBE_VARIANTS)} target words with variants)")


def analyze_probe(tg_dir: Path, out_md: Path):
    rows = []
    for tg_p in sorted(tg_dir.rglob("ISLE_*.TextGrid")):
        tg = textgrid.openTextgrid(str(tg_p), includeEmptyIntervals=False)
        words = tg.getTier("words").entries
        phones = tg.getTier("phones").entries
        for w in words:
            wl = w.label.strip().lower()
            if wl not in PROBE_VARIANTS:
                continue
            # Collect phones overlapping this word interval
            w_phones = [p.label.strip() for p in phones if (p.start + p.end) / 2 >= w.start and (p.start + p.end) / 2 <= w.end]
            aligned_pron = " ".join(w_phones)
            canonical = PROBE_VARIANTS[wl][0]
            rows.append({
                "file": tg_p.stem, "word": wl, "start_s": round(w.start, 3), "end_s": round(w.end, 3),
                "canonical": canonical, "mfa_selected": aligned_pron,
                "matches_canonical": (aligned_pron == canonical),
            })
    lines = [
        "# ISLE L2 Minimal-Pair Pronunciation Probe",
        "",
        "MFA forced alignment with a multi-variant probe dictionary allowing canonical vs. L1-confusable realizations:",
        "",
        "| File | Word | Span (s) | Canonical (ARPAbet) | MFA Selected Phones | Matches Canonical? |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['file']}` | **{r['word']}** | {r['start_s']:.3f}–{r['end_s']:.3f} | `{r['canonical']}` | `{r['mfa_selected']}` | {'Yes' if r['matches_canonical'] else '**No (variant selected)**'} |"
        )
    lines += [
        "",
        "*Caveat: GMM-HMM acoustic likelihoods provide an unsupervised pronunciation cue rather than a clinical/human phonetic transcription; all findings should be verified against the spectrogram in Praat.*",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote L2 probe report -> {out_md} ({len(rows)} target tokens analyzed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["make_dict", "analyze"], required=True)
    ap.add_argument("--base_dict", default="dictionaries/custom_lexicon.dict")
    ap.add_argument("--out_dict", default="dictionaries/isle_probe.dict")
    ap.add_argument("--tg_dir", default="outputs/textgrids/isle_probe")
    ap.add_argument("--out_md", default="outputs/evaluation/l2_probe.md")
    a = ap.parse_args()
    if a.mode == "make_dict":
        make_probe_dict(Path(a.base_dict), Path(a.out_dict))
    else:
        analyze_probe(Path(a.tg_dir), Path(a.out_md))


if __name__ == "__main__":
    main()
