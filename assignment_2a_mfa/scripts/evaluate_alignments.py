"""Quantitative validation and cross-system evaluation of forced-alignment TextGrids.

Checks:
  1. Structural validity: words & phones tiers, [0, dur] coverage, no gaps/overlaps, word-sequence match.
  2. Duration plausibility: phone duration summary, 30ms floor (squeezed phones), >300ms outliers.
  3. Cross-system word-boundary agreement: mean/median |Î”|, % within 20ms & 50ms.
  4. Optional gold-standard evaluation against gold/*.TextGrid.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from praatio import textgrid

SIL_LABELS = {"", "sil", "sp", "spn"}


def non_empty_intervals(tg, tier_name: str):
    if tier_name not in tg.tierNames:
        return []
    return [(e.start, e.end, e.label.strip().lower()) for e in tg.getTier(tier_name).entries if e.label.strip().lower() not in SIL_LABELS]


def validate_tg(tg_path: Path, wav_path: Path, lab_path: Path):
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)
    info = sf.info(str(wav_path))
    dur = info.frames / info.samplerate
    expected_words = lab_path.read_text(encoding="utf-8").strip().lower().split()

    has_words = "words" in tg.tierNames
    has_phones = "phones" in tg.tierNames
    words = non_empty_intervals(tg, "words")
    phones = non_empty_intervals(tg, "phones")

    # Include empty intervals: silence is part of tier coverage.
    overlaps = 0
    gaps = 0
    invalid_intervals = 0
    coverage_ok = abs(tg.maxTimestamp - dur) <= 0.001
    for tn in [t for t in ("words", "phones") if t in tg.tierNames]:
        ents = tg.getTier(tn).entries
        coverage_ok = coverage_ok and bool(ents) and abs(ents[0].start) <= 0.001 and abs(ents[-1].end - dur) <= 0.001
        invalid_intervals += sum(e.start < -1e-4 or e.end > dur + 0.001 or e.end <= e.start for e in ents)
        for a, b in zip(ents, ents[1:]):
            if b.start > a.end + 1e-4:
                gaps += 1
            if b.start < a.end - 1e-4:
                overlaps += 1

    word_seq = [w[2] for w in words]
    exact_match = (word_seq == expected_words)
    n_spn = sum(1 for e in (tg.getTier("phones").entries if has_phones else []) if e.label.strip().lower() == "spn")
    phone_durs_ms = [(p[1] - p[0]) * 1000.0 for p in phones]
    n_floor_30 = sum(1 for d in phone_durs_ms if d <= 31.0)
    n_long_300 = sum(1 for d in phone_durs_ms if d >= 300.0)

    return {
        "file": tg_path.stem,
        "duration_s": round(dur, 3),
        "tg_max_s": round(tg.maxTimestamp, 3),
        "has_words": has_words,
        "has_phones": has_phones,
        "overlaps": overlaps,
        "gaps": gaps,
        "invalid_intervals": invalid_intervals,
        "coverage_ok": bool(coverage_ok),
        "structurally_valid": bool(has_words and has_phones and exact_match and coverage_ok and overlaps == 0 and gaps == 0 and invalid_intervals == 0),
        "expected_words": len(expected_words),
        "aligned_words": len(words),
        "word_seq_match": exact_match,
        "n_phones": len(phones),
        "n_spn_oov": n_spn,
        "phone_dur_mean_ms": round(float(np.mean(phone_durs_ms)), 1) if phone_durs_ms else 0.0,
        "phone_dur_med_ms": round(float(np.median(phone_durs_ms)), 1) if phone_durs_ms else 0.0,
        "phones_at_30ms_floor": n_floor_30,
        "phones_over_300ms": n_long_300,
    }


def compare_word_boundaries(dir_a: Path, dir_b: Path):
    diffs_ms = []
    per_file = []
    for tga_p in sorted(dir_a.rglob("*.TextGrid")):
        rel = tga_p.relative_to(dir_a)
        tgb_p = dir_b / rel
        if not tgb_p.exists():
            continue
        wa = non_empty_intervals(textgrid.openTextgrid(str(tga_p), includeEmptyIntervals=False), "words")
        wb = non_empty_intervals(textgrid.openTextgrid(str(tgb_p), includeEmptyIntervals=False), "words")
        if [w[2] for w in wa] != [w[2] for w in wb]:
            continue
        f_diffs = []
        for (sa, ea, la), (sb, eb, lb) in zip(wa, wb):
            f_diffs.append(abs(sa - sb) * 1000.0)
            f_diffs.append(abs(ea - eb) * 1000.0)
        diffs_ms.extend(f_diffs)
        per_file.append({
            "file": tga_p.stem,
            "n_boundaries": len(f_diffs),
            "mean_ms": round(float(np.mean(f_diffs)), 2),
            "median_ms": round(float(np.median(f_diffs)), 2),
            "pct_le_20ms": round(float(np.mean(np.array(f_diffs) <= 20.0) * 100), 1),
            "pct_le_50ms": round(float(np.mean(np.array(f_diffs) <= 50.0) * 100), 1),
        })
    arr = np.array(diffs_ms) if diffs_ms else np.array([0.0])
    return {
        "n_boundaries": int(len(diffs_ms)),
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "pct_le_20ms": round(float((arr <= 20.0).mean() * 100), 1),
        "pct_le_50ms": round(float((arr <= 50.0).mean() * 100), 1),
        "per_file": per_file,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--tg_root", default="outputs/textgrids")
    ap.add_argument("--gold", default="gold")
    ap.add_argument("--out", default="outputs/evaluation")
    a = ap.parse_args()

    corpus = Path(a.corpus)
    tg_root = Path(a.tg_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    systems = sorted([d.name for d in tg_root.iterdir() if d.is_dir()])
    val_rows = []
    for sys_name in systems:
        for tg_p in sorted((tg_root / sys_name).rglob("*.TextGrid")):
            rel = tg_p.relative_to(tg_root / sys_name)
            wav_p = corpus / rel.with_suffix(".wav")
            lab_p = corpus / rel.with_suffix(".lab")
            if wav_p.exists() and lab_p.exists():
                r = validate_tg(tg_p, wav_p, lab_p)
                r["system"] = sys_name
                val_rows.append(r)

    if not val_rows:
        raise ValueError("No TextGrids paired with corpus audio/transcripts")
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(out / "validity_and_durations.csv", index=False)

    # Cross-system word boundary agreement vs primary (english_us_arpa)
    primary = "english_us_arpa" if "english_us_arpa" in systems else systems[0]
    agree_rows = []
    for sys_name in systems:
        if sys_name == primary:
            continue
        res = compare_word_boundaries(tg_root / primary, tg_root / sys_name)
        agree_rows.append({"pair": f"{primary} vs {sys_name}", **{k: v for k, v in res.items() if k != "per_file"}})
    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(out / "cross_system_agreement.csv", index=False)

    # Optional gold evaluation
    gold_dir = Path(a.gold)
    gold_rows = []
    if gold_dir.exists() and list(gold_dir.rglob("*.TextGrid")):
        for sys_name in systems:
            res = compare_word_boundaries(gold_dir, tg_root / sys_name)
            gold_rows.append({"system": sys_name, **{k: v for k, v in res.items() if k != "per_file"}})
        pd.DataFrame(gold_rows).to_csv(out / "gold_evaluation.csv", index=False)

    # Markdown summary
    lines = ["# Assignment 2A: Alignment Validation & Evaluation Summary", "", "## 1. Validity & Phone Duration Summary"]
    lines.append(val_df.to_markdown(index=False))
    if len(agree_df):
        lines += ["", f"## 2. Cross-System Word-Boundary Agreement (vs `{primary}`)", agree_df.to_markdown(index=False)]
    if gold_rows:
        lines += ["", "## 3. Manual Gold-Set Word-Boundary Error", pd.DataFrame(gold_rows).to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation complete -> {out} ({len(systems)} systems: {systems})")


if __name__ == "__main__":
    main()
