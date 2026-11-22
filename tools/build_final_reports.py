"""Build the final research-style PDF reports for Assignment 2A and 2B.

Evidence gate: all numbers come from files already in the repository (verified_p0 predictions,
accuracy tables, alignment evaluation). `build_reports.py` is used only as the read-only evidence
validator (metadata/split/leakage/checkpoint-hash checks); nothing is trained or re-evaluated.

    python tools/build_final_reports.py --verification-root assignment_2b_emotion/Results/verified_p0

Requires reportlab, pymupdf, Pillow, numpy, PyYAML (see assignment_2b_emotion/report build env).
"""
import argparse, csv, hashlib, json, re, sys, tempfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape as esc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_reports as br  # evidence validation only
import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
A2, B2 = ROOT / "assignment_2a_mfa", ROOT / "assignment_2b_emotion"
REPO = "https://github.com/SandeshAwate6055/iiith-speech-a2"
TMP = Path(tempfile.mkdtemp())
NAVY, GREY = colors.HexColor("#1a3d6d"), colors.HexColor("#475569")
LABELS = br.LABELS


def styles():
    s = getSampleStyleSheet()
    def add(name, **kw):
        s.add(ParagraphStyle(name=name, **kw))
    add("body", fontName="Helvetica", fontSize=9, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5)
    add("bul", parent=s["body"], leftIndent=14, bulletIndent=3, spaceAfter=2.5)
    add("sec1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=14, spaceAfter=7)
    add("sec2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=4)
    add("cap", fontName="Helvetica-Oblique", fontSize=7.8, leading=10, textColor=GREY, alignment=TA_CENTER, spaceAfter=9)
    add("tcap", fontName="Helvetica-Bold", fontSize=7.8, leading=10, textColor=GREY, spaceBefore=4, spaceAfter=3)
    add("cell", fontName="Helvetica", fontSize=7.4, leading=9.2)
    add("cellb", fontName="Helvetica-Bold", fontSize=7.4, leading=9.2)
    add("ctitle", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8)
    add("csub", fontName="Helvetica", fontSize=12, leading=16, textColor=GREY, alignment=TA_CENTER, spaceAfter=6)
    add("cmeta", fontName="Helvetica", fontSize=9.5, leading=14, alignment=TA_CENTER, spaceAfter=2)
    add("code", fontName="Courier", fontSize=6.9, leading=8.6, backColor=colors.HexColor("#f1f5f9"), borderPadding=4,
        leftIndent=4, spaceAfter=7)
    add("ref", parent=s["body"], leftIndent=16, firstLineIndent=-16, fontSize=8.2, leading=11, alignment=0, spaceAfter=3)
    return s


ST = styles()


class Doc(BaseDocTemplate):
    def __init__(self, path, title, footer):
        super().__init__(str(path), pagesize=A4, leftMargin=54, rightMargin=54, topMargin=56, bottomMargin=56,
                         title=title, author="IIITH Speech Processing Assignment 2", invariant=1)
        self.footer = footer
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="cover", frames=[fr]), PageTemplate(id="body", frames=[fr], onPage=self._deco)])

    def _deco(self, c, d):
        c.saveState(); c.setStrokeColor(colors.HexColor("#c3ceda")); c.line(54, 44, 541, 44)
        c.setFont("Helvetica", 7.2); c.setFillColor(GREY)
        c.drawString(54, 32, self.footer); c.drawRightString(541, 32, f"Page {d.page}"); c.restoreState()

    def afterFlowable(self, fl):
        if hasattr(fl, "_toc"):
            self.notify("TOCEntry", (fl._toc[0], fl._toc[1], self.page))


def prep(path):
    path = Path(path)
    if path.stat().st_size < 380_000:
        return path
    im = PILImage.open(path).convert("RGB")
    if im.width > 1700:
        im = im.resize((1700, int(im.height * 1700 / im.width)), PILImage.LANCZOS)
    out = TMP / (hashlib.md5(str(path).encode()).hexdigest() + ".jpg")
    im.save(out, "JPEG", quality=87)
    return out


class Rep:
    def __init__(self, path, title, footer):
        self.path, self.title, self.footer = Path(path), title, footer
        self.story, self.nf, self.nt = [], 0, 0

    def h1(self, t):
        p = Paragraph(esc(t), ST["sec1"]); p._toc = (0, t); self.story.append(p)

    def h2(self, t):
        p = Paragraph(esc(t), ST["sec2"]); p._toc = (1, t); self.story.append(p)

    def p(self, t, style="body"):
        self.story.append(Paragraph(t, ST[style]))

    def bullets(self, items):
        for t in items:
            self.story.append(Paragraph(t, ST["bul"], bulletText="\u2022"))

    def code(self, t):
        self.story.append(Preformatted(t.strip("\n"), ST["code"]))

    def table(self, caption, head, rows, colw=None, bold_first=False):
        self.nt += 1
        cap = Paragraph(f"Table {self.nt}. {caption}", ST["tcap"])
        data = [[Paragraph(f"<b>{esc(str(h))}</b>", ST["cell"]) for h in head]]
        for r in rows:
            data.append([Paragraph(esc(str(c)) if not str(c).startswith("<") else str(c), ST["cellb" if (bold_first and i == 0) else "cell"])
                         for i, c in enumerate(r)])
        w = self.doc_width()
        cw = [w * x / sum(colw) for x in colw] if colw else [w / len(head)] * len(head)
        t = Table(data, colWidths=cw, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5edf5")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c3d0")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6)]))
        self.story += [KeepTogether([cap, t]), Spacer(1, 7)]

    def doc_width(self):
        return A4[0] - 108

    def fig(self, path, caption, width=470, maxh=330):
        p = prep(path)
        w0, h0 = PILImage.open(p).size
        sc = min(width / w0, maxh / h0)
        self.nf += 1
        self.story.append(KeepTogether([Image(str(p), w0 * sc, h0 * sc), Paragraph(f"Figure {self.nf}. {caption}", ST["cap"])]))

    def pb(self):
        self.story.append(PageBreak())

    def build(self, cover):
        toc = TableOfContents(); toc.dotsMinLevel = 0
        toc.levelStyles = [ParagraphStyle("t0", fontName="Helvetica-Bold", fontSize=9.5, leading=14, leftIndent=0, firstLineIndent=0),
                           ParagraphStyle("t1", fontName="Helvetica", fontSize=8.6, leading=12, leftIndent=16)]
        story = list(cover) + [NextPageTemplate("body"), PageBreak(),
                               Paragraph("Contents", ST["sec1"]), toc, PageBreak()] + self.story
        doc = Doc(self.path, self.title, self.footer)
        doc.multiBuild(story)


def cover(title, subtitle, code, abstract):
    items = [Spacer(1, 90), Paragraph("IIIT Hyderabad - Speech Processing Lab", ST["csub"]), Spacer(1, 20),
             Paragraph(title, ST["ctitle"]), Paragraph(subtitle, ST["csub"]), Spacer(1, 26),
             Paragraph(f"<b>Assignment {code}</b> | Final technical report", ST["cmeta"]),
             Paragraph("Author: <i>[name - to be filled by the student]</i> &nbsp;|&nbsp; Roll number: <i>[to be filled]</i>", ST["cmeta"]),
             Paragraph("Date: 22 November 2026 &nbsp;|&nbsp; Deadline: 22 November 2026, 11:59 PM IST", ST["cmeta"]),
             Paragraph(f"Source code, results and figures: <font color='#1a3d6d'>{REPO}</font>", ST["cmeta"]), Spacer(1, 30),
             Paragraph("<b>Abstract</b>", ST["cmeta"]), Paragraph(abstract, ST["body"])]
    return items


# ----------------------------------------------------------------------------- helpers for data
def read_tg(path):
    tiers, name, cur = {}, None, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("name ="):
            name = s.split('"')[1]; tiers[name] = []
        elif s.startswith("intervals ["):
            cur = {}
        elif cur is not None and s.startswith("xmin ="):
            cur["xmin"] = float(s.split("=")[1])
        elif cur is not None and s.startswith("xmax ="):
            cur["xmax"] = float(s.split("=")[1])
        elif cur is not None and s.startswith("text ="):
            cur["text"] = s.split('"')[1] if '"' in s else ""
            tiers[name].append((cur["xmin"], cur["xmax"], cur["text"])); cur = None
    return tiers


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


# ============================================================================= 2A
def build_2a(ev, outdir):
    E = A2 / "outputs/evaluation"
    corpus = json.loads((A2 / "corpus/corpus_report.json").read_text(encoding="utf-8"))
    val = [r for r in br.csvrows(E / "validity_and_durations.csv")]
    prim = [r for r in val if r["system"] == "english_us_arpa"]
    mfa = [r for r in val if r["system"] == "english_mfa"]
    cross = br.csvrows(E / "cross_system_agreement.csv")
    audit = br.js(E / "lexicon_audit.json")
    src = br.csvrows(E / "lexicon_pronunciation_sources.csv")
    srcc = Counter(r["source"] for r in src)
    prov = br.js(E / "alignment_run_provenance.json")
    n_words = sum(int(r["aligned_words"]) for r in prim); n_ph = sum(int(r["n_phones"]) for r in prim)
    dur = sum(float(r["duration_s"]) for r in prim)
    short = sum(int(r["phones_at_30ms_floor"]) for r in prim); long_ = sum(int(r["phones_over_300ms"]) for r in prim)
    R = Rep(outdir / "report_2a.pdf", "Assignment 2A - Forced alignment with MFA",
            "IIITH Speech Processing | Assignment 2A | Forced alignment with the Montreal Forced Aligner")
    abstract = (f"We align six recordings (three Boston University Radio News excerpts and three ISLE non-native learner utterances; "
                f"{dur:.1f} s, {n_words} words, {n_ph} phones) with the Montreal Forced Aligner 3.4.4 and the pretrained english_us_arpa 3.3.0 "
                "acoustic model. The pipeline converts NIST SPHERE audio to 16 kHz PCM16, normalises transcripts, builds a corpus-scoped "
                "pronunciation lexicon (11 out-of-vocabulary word types resolved with MFA G2P and manual overrides; 0 OOV remaining), runs "
                "forced alignment with beam 100 / retry-beam 400, and inspects every TextGrid structurally, visually (Praat 7.0.03) and against "
                "four alternative aligners (MAP-adapted model, english_mfa, an L2 pronunciation-variant probe and a wav2vec2 CTC baseline). "
                "No manually annotated gold boundaries exist for these recordings, so boundary accuracy is not claimed; we report validity, "
                "plausibility and cross-system agreement instead.")
    cov = cover("Forced Alignment with the Montreal Forced Aligner", "Word and phoneme TextGrids for native broadcast and non-native learner speech",
                "2A", abstract)

    R.h1("1  Introduction and objectives")
    R.p("Forced alignment finds the time boundaries of known words and phones in an utterance given its transcript. It is the first step of "
        "most phonetic analyses and of many data-preparation pipelines for speech technology. Assignment 2A asks for a complete, reproducible "
        "Montreal Forced Aligner (MFA) pipeline on six supplied recordings, producing word- and phoneme-level TextGrids that are checked in Praat, "
        "together with a documented pronunciation dictionary and discussion of the alignment quality.")
    R.bullets(["<b>Mandatory:</b> install MFA; prepare audio/transcripts; construct or extend a pronunciation dictionary (handling OOV words); "
               "align; export TextGrids; inspect them in Praat; visualise and explain the alignment.",
               "<b>Extra credit pursued:</b> G2P-assisted lexicon with a coverage audit, speaker-adaptive/MAP-adapted model, a second MFA model "
               "(english_mfa, IPA), an ISLE minimal-pair pronunciation probe and a neural CTC forced-alignment baseline.",
               "<b>Scope statement:</b> only the six supplied recordings are used; there is no reference (gold) alignment, hence no boundary "
               "error in milliseconds against ground truth is reported."])
    R.fig(ROOT / "docs/figures/arch_2a_pipeline.png", "Implemented 2A pipeline. Boxes correspond to scripts in assignment_2a_mfa/scripts/ and "
          "outputs under assignment_2a_mfa/outputs/.", 480, 330)

    R.h1("2  Data and preparation")
    R.h2("2.1  Corpus")
    rows = [[r["file"].replace("ISLE_SESS0131_BLOCKD02_", "ISLE ").replace("_sprt1", ""), r["source_format"], f"{r['duration_s']:.3f}",
             str(r["n_words"]), f"{r['peak']:.3f}", str(r["clipped_samples"])] for r in corpus]
    R.table("Supplied recordings after inspection (assignment_2a_mfa/corpus/corpus_report.json).",
            ["Recording", "Source container", "Duration (s)", "Words", "Peak (full scale)", "Clipped samples"], rows, [2.2, 1.6, 1.2, 0.9, 1.3, 1.3])
    R.p("The three BU recordings (F2BJRLP1-3) are single-speaker radio-news excerpts; the three ISLE files contain a German-accented learner "
        "reading minimal-pair carrier sentences (<i>I said white not bait</i>, <i>I said new not no</i>, <i>I said bad not bed</i>). "
        "The BU files carry a <b>NIST SPHERE</b> header although they are named .wav, which MFA's reader rejects; the ISLE files are RIFF with "
        "non-standard padding chunks.")
    R.h2("2.2  Audio conversion")
    R.p("<font face='Courier'>scripts/prepare_corpus.py</font> reads every file with libsndfile (which understands both containers), asserts the "
        "16 kHz sampling rate, down-mixes to mono and writes canonical RIFF PCM16 into one directory per speaker "
        "(<font face='Courier'>corpus/f2b</font>, <font face='Courier'>corpus/isle_sess0131</font>). Per-speaker directories matter because "
        "MFA estimates speaker-adaptive (fMLLR) transforms per directory. No sample was clipped and no resampling was necessary.")
    R.h2("2.3  Transcript normalisation")
    orig, norm = corpus[0]["original_text"], corpus[0]["normalized_text"]
    R.p("<font face='Courier'>scripts/normalize_text.py</font> lower-cases, removes punctuation, repairs line-wrap hyphenation "
        "(<i>seventy- six</i> becomes <i>seventy six</i>), keeps word-internal hyphens and clitic apostrophes (<i>de-politicize</i>, "
        "<i>sjc's</i>, <i>wbur's</i>) and collapses initialisms such as S.J.C.'s into a single token that the lexicon maps to its spelled-out "
        "pronunciation. Six unit tests in <font face='Courier'>tests/test_normalize.py</font> pin these rules. Example (F2BJRLP1, first sentence):")
    R.code(f"original  : {orig[:92]}...\nnormalized: {norm[:92]}...")

    R.h1("3  Pronunciation lexicon")
    R.p("MFA needs a pronunciation for every transcript word. We start from the official english_us_arpa dictionary (ARPAbet with stress "
        "digits, 3.3.0) and extend it only where the corpus requires it. The pinned source is the Hugging Face snapshot "
        f"<font face='Courier'>MontrealCorpusTools/english_us_arpa@{audit['model_revision'][:12]}</font> (acoustic 3.3.0), packaged by "
        "<font face='Courier'>scripts/prepare_models.py</font> without modifying model bytes.")
    R.bullets([f"The corpus has <b>{audit['dictionary_words']}</b> word types. <b>{len(audit['base_oovs'])}</b> are absent from the base dictionary: "
               + ", ".join(f"<i>{esc(w)}</i>" for w in audit["base_oovs"]) + ".",
               "<b>G2P:</b> the pretrained MFA g2p model proposes candidate pronunciations for every OOV (dictionaries/g2p_oov.dict). Candidates "
               "are noisy for names and initialisms (e.g. <i>wbur's</i> -> B ER0 Z).",
               "<b>Manual overrides</b> (dictionaries/manual_overrides.dict, 9 words, 12 rows) replace implausible candidates for proper "
               "nouns and the spelled-out initialisms and add a second variant for ambiguous names (<i>dukakis</i>, <i>melnicove</i>, <i>de-politicize</i>).",
               f"The merged <font face='Courier'>custom_lexicon.dict</font> has <b>{sum(srcc.values())}</b> pronunciation rows "
               f"({srcc['base']} base, {srcc['manual']} manual, {srcc['g2p']} G2P) and <b>{audit['oov_types']} OOV</b> types; "
               "<font face='Courier'>scripts/audit_lexicon.py</font> verifies ARPAbet syntax, per-file coverage and that every phone exists in "
               "the acoustic model inventory (acoustic_phones_compatible = true)."])
    R.table("Hand-curated pronunciations (assignment_2a_mfa/dictionaries/manual_overrides.dict).",
            ["Word", "Pronunciation(s) used", "Rationale"],
            [["sjc's", "EH1 S JH EY1 S IY1 Z", "initialism read letter by letter, possessive Z"],
             ["wbur's", "D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R Z", "radio call sign spelled out (double-u-b-u-r)"],
             ["dukakis", "D UW0 K AA1 K IH0 S | D AH0 K AA1 K IH0 S", "two plausible initial vowels"],
             ["melnicove", "M EH1 L N IH0 K OW2 V | M EH1 L N AH0 K OW2 V", "name; reduced / full second vowel"],
             ["maffy", "M AE1 F IY0", "name; G2P output accepted after review"],
             ["hennessy / hennessy's", "HH EH1 N AH0 S IY0 (Z)", "name and possessive"],
             ["de-politicize", "D IY2 P AH0 L IH1 T AH0 S AY2 Z | D IH0 P ...", "hyphenated compound, two prefix variants"],
             ["judgeships", "JH AH1 JH SH IH2 P S", "compound plural missing from base"]], [1.5, 3.3, 3.0])
    R.p("<b>Limitation.</b> Syntax, coverage and inventory checks do not prove that a pronunciation is correct for the speaker; the rows are "
        "candidates with source labels (lexicon_pronunciation_sources.csv), not human-verified gold. The audit's own limitation field says so.")

    R.h1("4  Forced alignment with MFA")
    R.h2("4.1  Acoustic model and alignment procedure")
    R.p("MFA wraps Kaldi. english_us_arpa 3.3.0 is a GMM-HMM model (metadata: 289,667 training utterances, 2,445 speakers) with 13 MFCCs, deltas, "
        "CMVN, an LDA transform and speaker adaptation (fMLLR; <font face='Courier'>uses_speaker_adaptation = true</font>). Given a transcript, "
        "MFA composes the lexicon (word -> phones), the HMM topology and the audio features into a decoding graph and runs Viterbi alignment inside a "
        "beam; with speaker adaptation the first pass supplies per-speaker transforms for a second, adapted pass [1][2]. Words that cannot be aligned "
        "within the beam are retried with a wider beam. The default beams (10 / 40) are too tight for 25-30 s broadcast excerpts, so "
        "<font face='Courier'>configs/pipeline.yaml</font> sets <b>beam 100 / retry-beam 400</b>.")
    R.code("mfa validate corpus dictionaries/custom_lexicon.dict english_us_arpa\n"
           "mfa align corpus dictionaries/custom_lexicon.dict work/models/hf_arpa.zip outputs/textgrids/english_us_arpa \\\n"
           "    --config_path configs/pipeline.yaml --no_tokenization --num_jobs 2")
    R.p(f"The run is hash-locked: <font face='Courier'>alignment_run_provenance.json</font> records SHA-256 digests of all 12 inputs plus the "
        f"lexicon, config and model archive (MFA {prov['mfa_version']}), and of the six output TextGrids. MFA validation completed with exit code 0 and "
        "no read, feature, transcript or OOV errors on 97.163 s of audio (258.858 s total runtime).")
    R.h2("4.2  TextGrid structure and examples")
    R.p("Each output has two interval tiers: <b>words</b> (transcript words plus empty intervals for silence) and <b>phones</b> (ARPAbet with "
        "stress digits). Tables 3 and 4 show the complete primary output for ISLE 01, whose boundaries are discussed in Section 7.")
    tg = read_tg(A2 / "outputs/textgrids/english_us_arpa/isle_sess0131/ISLE_SESS0131_BLOCKD02_01_sprt1.TextGrid")
    ph = tg["phones"]; wd = tg["words"]
    R.table("ISLE 01 (<i>I said white not bait</i>): words tier.", ["#", "start (s)", "end (s)", "dur (ms)", "word"],
            [[i + 1, f"{a:.3f}", f"{b:.3f}", f"{1000 * (b - a):.0f}", t or "(silence)"] for i, (a, b, t) in enumerate(wd)], [0.5, 1, 1, 1, 2])
    R.table("ISLE 01: phones tier (ARPAbet).", ["#", "start (s)", "end (s)", "dur (ms)", "phone"],
            [[i + 1, f"{a:.3f}", f"{b:.3f}", f"{1000 * (b - a):.0f}", t or "(silence)"] for i, (a, b, t) in enumerate(ph)], [0.5, 1, 1, 1, 2])
    wanted = {"dukakis", "hennessy", "sjc's", "wbur's", "melnicove", "maffy"}
    rows = []
    for stem in ["F2BJRLP1", "F2BJRLP2", "F2BJRLP3"]:
        t = read_tg(A2 / f"outputs/textgrids/english_us_arpa/f2b/{stem}.TextGrid")
        for a, b, w in t["words"]:
            if w in wanted and w not in [r[1] for r in rows if r[0] == stem]:
                p = " ".join(x for s, e, x in t["phones"] if s >= a - 1e-6 and e <= b + 1e-6 and x)
                rows.append([stem, w, f"{a:.2f}-{b:.2f}", p])
    R.table("Proper nouns and initialisms in the BU recordings (first occurrence per file).", ["File", "Word", "Span (s)", "Aligned phones"], rows, [1.2, 1.1, 1.3, 4])

    R.h1("5  Structural evaluation")
    R.p("Because no gold alignment is available, <font face='Courier'>scripts/evaluate_alignments.py</font> checks that each TextGrid is a valid, "
        "gap-free tiling of the recording, that the word sequence equals the normalised transcript and that no OOV/<i>spn</i> placeholder interval was "
        "produced; it also summarises phone durations as plausibility flags.")
    rows = [[r["file"].replace("ISLE_SESS0131_BLOCKD02_", "ISLE ").replace("_sprt1", ""), r["duration_s"], r["aligned_words"] + "/" + r["expected_words"],
             r["n_phones"], r["gaps"] + " / " + r["overlaps"], r["n_spn_oov"], r["phone_dur_mean_ms"], r["phone_dur_med_ms"],
             r["phones_at_30ms_floor"], r["phones_over_300ms"]] for r in prim]
    rows.append(["<b>Total</b>", f"{dur:.2f}", f"{n_words}/{n_words}", n_ph, "0 / 0", "0", "-", "-", short, long_])
    R.table("Primary system (english_us_arpa + custom lexicon): validity and duration statistics.",
            ["File", "Dur (s)", "Words aligned", "Phones", "Gaps / overl.", "spn", "Mean ph (ms)", "Median ph (ms)", "Phones at 30 ms", "Phones > 300 ms"],
            rows, [1.5, .8, 1.1, .8, 1, .6, 1, 1, 1, 1])
    R.p(f"All six outputs are structurally valid and the aligned word sequences match the transcripts (68/73/82/5/5/5 words). "
        f"{short} intervals sit exactly at the 30 ms minimum duration of the 3-state HMM topology (26-30 per BU file, none in ISLE) and {long_} phone "
        "exceeds 300 ms (ISLE 02). These are review candidates, not confirmed errors: very short function-word phones in fast read speech legitimately "
        "approach the topological floor.")

    R.h1("6  Extra-credit alignment systems")
    R.p("Four additional alignments were produced on the same corpus (outputs/textgrids/*) to study sensitivity to the acoustic model and to test "
        "pronunciation variants. Agreement is measured as the absolute difference of word boundaries to the primary system (all 476 or, for "
        "english_mfa, 330 comparable boundaries).")
    R.table("Word-boundary disagreement with the primary english_us_arpa alignment (cross_system_agreement.csv).",
            ["System", "Boundaries", "Mean (ms)", "Median (ms)", "P90 (ms)", "<= 20 ms (%)", "<= 50 ms (%)"],
            [[r["pair"].replace("english_us_arpa vs ", ""), r["n_boundaries"], r["mean_ms"], r["median_ms"], r["p90_ms"], r["pct_le_20ms"], r["pct_le_50ms"]] for r in cross],
            [2, 1, 1, 1, 1, 1.1, 1.1])
    R.bullets(["<b>adapted_arpa</b> - the model is MAP-adapted to the six recordings (<font face='Courier'>mfa adapt</font>, archive kept locally). "
               "Word boundaries differ by 0.08 ms on average: with six short recordings adaptation barely moves the Gaussians, so the result is "
               "essentially the primary alignment.",
               "<b>isle_probe</b> - same model with a probe dictionary that offers several variants for the six contrast words; boundaries equal the primary "
               "system (0.08 ms) as expected when only the choice of pronunciation variant changes (Section 6.1).",
               f"<b>english_mfa</b> (IPA model, 330 comparable boundaries) differs by 30.4 ms on average (median 10 ms; 86.7% within 20 ms). "
               f"It is the least reliable system here: F2BJRLP2 yields {mfa[1]['aligned_words']} words instead of {mfa[1]['expected_words']} (transcript mismatch) and "
               f"{sum(int(r['n_spn_oov']) for r in mfa)} spn intervals appear across the corpus, because it uses MFA's stock IPA dictionary without our overrides.",
               "<b>CTC baseline</b> - torchaudio MMS_FA (wav2vec2, [3]) produces word-level boundaries only. It disagrees by 36.7 ms on average "
               "(median 29.8 ms; 33.2% within 20 ms, 75.4% within 50 ms), in line with the frame-resolution and peaky-output behaviour of CTC aligners [4]. "
               "Its tiers lack phones, so it is not a substitute for the MFA outputs."])
    R.h2("6.1  ISLE minimal-pair pronunciation probe")
    l2 = [l for l in (E / "l2_probe.md").read_text(encoding="utf-8").splitlines() if l.startswith("| `ISLE")]
    rows = []
    for l in l2:
        c = [x.strip() for x in l.strip("|").split("|")]
        rows.append([c[0].replace("`", "").replace("ISLE_SESS0131_BLOCKD02_", "ISLE ").replace("_sprt1", ""), c[1].replace("*", ""), c[2],
                     c[3].replace("`", ""), c[4].replace("`", ""), c[5].replace("*", "")])
    R.table("Pronunciation variants selected by Viterbi alignment (l2_probe.md).", ["File", "Word", "Span (s)", "Canonical", "Selected", "Canonical?"], rows,
            [1.1, .8, 1.2, 1.4, 1.4, 1.6])
    R.p("For <i>new</i> the aligner prefers the yod variant N Y UW1 and for <i>bad</i> it selects B EH1 T (vowel and final devoicing of the German-accented "
        "learner), the two cases where the canonical form was not chosen. This is an unsupervised <i>likelihood</i> cue only; it was not verified by "
        "listening and must not be read as a phonetic transcription.")

    R.h1("7  Praat inspection and observations")
    R.p("All six primary sound/TextGrid pairs were opened in Praat 7.0.03 (<font face='Courier'>scripts/open_in_praat.praat</font>). Each ISLE file was viewed over its "
        "full duration; each BU file was viewed in full and zoomed to approximately its first 5.5 s, because Praat suppresses the spectrogram for long "
        "windows. This is a visual boundary inspection, not a listening-based transcription; no boundary or label was edited.")
    R.fig(A2 / "outputs/praat_screenshots/praat_isle01.jpg", "Praat editor screenshot, ISLE 01 (genuine capture): waveform, spectrogram, words and phones tiers.", 440, 300)
    R.fig(A2 / "outputs/praat_screenshots/praat_isle02.jpg", "Praat editor screenshot, ISLE 02 (genuine capture); the long final OW1 interval of <i>no</i> is discussed below.", 440, 300)
    obs = [l[2:] for l in (E / "praat_observations.md").read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
    R.bullets([esc(o) for o in obs])
    R.fig(A2 / "outputs/figures/ISLE_SESS0131_BLOCKD02_01_sprt1_full.png",
          "Generated waveform / spectrogram / alignment plot for ISLE 01 (scripts/visualize.py). Distinct from the Praat screenshots above.", 470, 300)
    R.fig(A2 / "outputs/figures/F2BJRLP1_excerpt.png", "Generated excerpt for F2BJRLP1 (first seconds of the broadcast).", 470, 300)

    R.h1("8  Discussion")
    R.bullets(["<b>Pipeline robustness.</b> The only non-trivial data issue was the SPHERE header; once audio is converted, MFA aligned all six files on the first run "
               "with zero OOV and zero <i>spn</i> intervals.",
               "<b>Lexicon quality dominates.</b> Names and initialisms (<i>Dukakis, Melnicove, WBUR's, S.J.C.'s</i>) are the typical failure points of forced alignment; "
               "supplying reviewed pronunciations avoided OOV placeholders. Table 5 shows each is aligned with an explicit phone sequence.",
               "<b>Native vs learner speech.</b> BU phones average about 73 ms (many at the 30 ms HMM floor), ISLE phones about 128-138 ms with long silences: slow, "
               "careful carrier-sentence speech is easier to align in time but its pronunciation can deviate from the canonical dictionary, as the probe shows.",
               "<b>Model agreement.</b> Two independent acoustic frameworks (GMM-HMM and a CTC network) agree within 50 ms for about 75% of word boundaries, "
               "which supports plausibility but is no substitute for reference annotations."])
    R.h2("8.1  Limitations and honest scope")
    R.bullets(["No manual gold boundaries and no listening test: boundary error (ms) against ground truth is <b>not</b> reported.",
               "Pronunciations are candidates; manual overrides were chosen by the authors for the supplied text and are not independently verified.",
               "Six recordings from two speaker groups cannot support population-level claims; english_mfa mismatches are a limitation of its stock IPA dictionary.",
               "Praat observations are visual; very short phones (<= 31 ms) are duration flags, not proof of missing phones."])

    R.h1("9  Reproducibility")
    R.p("Commands are given in <font face='Courier'>assignment_2a_mfa/README.md</font>. The raw audio is licensed and is not redistributed; run the pipeline against the "
        "supplied files:")
    R.code("conda create -n mfa -c conda-forge montreal-forced-aligner=3.4.4\n"
           "python scripts/prepare_models.py                       # pinned HF snapshot, no audio\n"
           "python scripts/prepare_corpus.py --raw <assignment_2a_dir> --out corpus\n"
           "python scripts/run_alignment.py --mfa-env <env> --micromamba-root <root>\n"
           "python scripts/evaluate_alignments.py && python scripts/visualize.py\n"
           "python -m pytest tests -q                              # 9 tests")
    R.h1("10  Conclusion")
    R.p("The pipeline turns six heterogeneous recordings into valid word- and phone-level TextGrids with a fully covered, audited lexicon, and documents the alignment "
        "quality with structural checks, genuine Praat inspection and four alternative systems. The main finding is that careful lexicon construction and correct audio "
        "conversion, not model choice, decide whether alignment succeeds on this data. The main limitation is the lack of gold boundaries.")
    R.h1("References")
    for r in ["[1] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, M. Sonderegger. Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. Interspeech 2017.",
              "[2] D. Povey et al. The Kaldi Speech Recognition Toolkit. IEEE ASRU Workshop, 2011.",
              "[3] V. Pratap et al. Scaling Speech Technology to 1,000+ Languages (MMS). arXiv:2305.13516, 2023.",
              "[4] A. Rousso et al. Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment. Interspeech 2024.",
              "[5] M. Ostendorf, P. Price, S. Shattuck-Hufnagel. The Boston University Radio News Corpus. Tech. Rep. ECS-95-001, Boston University, 1995.",
              "[6] W. Menzel et al. The ISLE corpus of non-native spoken English. LREC 2000.",
              "[7] P. Boersma, D. Weenink. Praat: doing phonetics by computer (version 7.0.03). http://www.praat.org"]:
        R.p(esc(r), "ref")
    R.build(cov)
    return R


# ============================================================================= 2B
def build_2b(ev, good, status, failures, outdir, vroot):
    P = B2 / "Results/plots"
    acc = {r["run"]: r for r in br.csvrows(B2 / "Results/accuracy_tables.csv")}
    cfgs = {n: yaml.safe_load((B2 / f"configs/{n}.yaml").read_text(encoding="utf-8")) for n in ["speech", "text", "fusion_concat", "fusion"]}
    emb = br.js(P / "embeddings_umap_protoA_metrics.json")
    R = Rep(outdir / "report_2b.pdf", "Assignment 2B - Multimodal emotion recognition",
            "IIITH Speech Processing | Assignment 2B | Speech, text and fusion emotion recognition on TESS")
    f = lambda run, m="macro_f1": f"{pct(good[run][m + '_mean'])} +/- {pct(good[run][m + '_std'])}"
    abstract = ("We build and evaluate speech, text and multimodal emotion classifiers on the Toronto Emotional Speech Set (TESS: 2,800 acted clips, 7 emotions, "
                "2 speakers, 200 carrier words) under three frozen evaluation protocols - word-disjoint (A), speaker-transfer (B) and random-stratified (C). "
                "The speech model (frozen WavLM-Base+, learnable layer sum, BiGRU, attentive statistics pooling) reaches "
                f"{pct(good['speech_wavlm_bigru_asp_protoA']['macro_f1_mean'])}% macro-F1 on A but only {pct(good['speech_wavlm_bigru_asp_protoB']['macro_f1_mean'])}% on the "
                "speaker-transfer protocol B. Text carries no label information in TESS (every transcript occurs with all seven emotions), so the RoBERTa model stays at chance, "
                "and neither concatenation nor bidirectional gated cross-attention fusion improves on speech alone. We report per-emotion recall, paired help/harm counts, "
                "a zero-text intervention, three-block embedding projections and five prediction-backed failure cases, and disclose a historical text-training provenance "
                "limitation.")
    cov = cover("Multimodal Emotion Recognition", "Speech, text and fusion pipelines on TESS with three evaluation protocols", "2B", abstract)

    R.h1("1  Introduction and objectives")
    R.p("Assignment 2B asks for three emotion-recognition pipelines - speech-only, text-only and speech+text fusion - built from five functional blocks "
        "(preprocessing, feature extraction, temporal/contextual modelling, fusion, classifier), trained with explicit train/test scripts, compared with accuracy "
        "tables, plots and an error analysis. We additionally add classical baselines, two fusion mechanisms, a modality ablation and a leakage-controlled evaluation design.")
    R.bullets(["<b>Primary question.</b> How much of the emotion signal in TESS is acoustic, how much textual, and does fusion help?",
               "<b>Methodological aim.</b> Avoid optimistic estimates: word-disjoint folds (A) and an unseen-speaker, unseen-word protocol (B) accompany the conventional random split (C).",
               "<b>Deliverables.</b> Six entry points (models/{speech,text,fusion}_pipeline/{train,test}.py), configs, frozen splits, verified results, plots, tests."])
    R.fig(ROOT / "docs/figures/arch_2b_overview.png", "The five functional blocks of the implemented system.", 480, 250)

    R.h1("2  Dataset and audit")
    R.p("TESS [1] contains 2,800 utterances: two female actresses (OAF, older; YAF, younger) speak 200 target words in the carrier phrase "
        "<i>Say the word ...</i> in seven emotions (angry, disgust, fear, happy, neutral, sad, pleasant surprise), 400 clips per emotion and 1,400 per speaker. "
        "The audit (Results/data_audit.md) found no duplicate files (MD5) or utterance IDs and no incomplete (speaker, word) pair; sampling rates are 24,414 Hz "
        "(2,799 files) and 96,000 Hz (one file); durations range from 1.25 to 2.99 s (median 2.05 s).")
    aud = [l for l in (B2 / "Results/data_audit.md").read_text(encoding="utf-8").splitlines() if re.match(r"\| (angry|disgust|fear|happy|neutral|sad|surprise) \|", l)]
    rows = []
    for l in aud:
        c = [x.strip().replace("\u00c2\u00b1", "+/-").replace("\u00b1", "+/-") for x in l.strip("|").split("|")]
        rows.append(c)
    R.table("Raw acoustic summary per emotion (mean +/- SD over 400 clips).", ["Emotion", "Duration (s)", "RMS (dBFS)", "Peak"], rows, [1, 1.5, 1.5, 1.5])
    R.p("Duration and loudness differ systematically across emotions (e.g. disgust and sad are the longest, neutral is the quietest, angry the loudest) and across speakers "
        "(YAF is about 6.5 dB louder than OAF). These are low-level cues a network can exploit, which motivates peak normalisation and the speaker-transfer protocol.")
    R.fig(P / "data_audit_distributions.png", "Dataset audit distributions (duration and level by emotion and speaker).", 440, 260)
    R.p("<b>Text construction.</b> The transcript is the carrier phrase with the target word (<i>say the word &lt;w&gt;</i>); the emotion/speaker suffix of the filename is never "
        "passed to any model. Consequently P(emotion | text) is exactly 1/7 for all 200 transcripts: text alone cannot predict emotion in the full corpus.")

    R.h1("3  Evaluation protocols")
    R.table("Frozen split manifests (splits/protocol_{A,B,C}.json). Counts are utterances per fold.",
            ["Protocol", "Design", "Folds", "Train / val / test", "Leakage controls"],
            [["A", "Word-disjoint, both speakers in every split", "5", "1,792 / 448 / 560", "No word shared between train, val, test (tested)"],
             ["B", "Speaker transfer: train/val on one speaker, test on the other, test words unseen", "2", "896 / 224 / 280", "No speaker or word overlap between train+val and test (tested)"],
             ["C", "Random stratified (optimistic reference)", "5", "1,960 / 280 / 560", "Utterances disjoint; words and speakers shared"]], [0.9, 2.8, 0.6, 1.4, 2.6])
    R.p("Every protocol keeps test sets disjoint across folds (A and C cover all 2,800 utterances exactly once). Models are trained on the training IDs; the validation set "
        "only selects the checkpoint (best validation macro-F1) and stops training (patience 6-8). <b>No test-set tuning was performed.</b> We report accuracy, "
        "unweighted average recall (UAR) and macro-F1 as the unweighted mean over folds with population standard deviation (ddof = 0); these are not confidence intervals. "
        "Because two speakers cannot establish speaker-independent behaviour in general, protocol B is a stress test, not a population estimate. "
        "Splits, label mappings and leakage are verified by tests/test_splits.py (4 tests) and, again, by the report's evidence gate before this document was built.")

    R.h1("4  Methodology")
    R.h2("4.1  Block overview")
    R.table("Mapping of the five required functional blocks to the implementation.",
            ["Block", "Speech", "Text", "Fusion"],
            [["1 Preprocessing", "16 kHz mono, trim 30 dB with 50 ms padding, peak -1 dBFS", "carrier phrase, RoBERTa BPE, max 32 tokens", "-"],
             ["2 Feature extraction", "frozen WavLM-Base+ hidden states [13, T, 768] (cached); eGeMAPS-88 for the baseline", "RoBERTa-base token states [N, 768]", "-"],
             ["3 Temporal / contextual", "layer-weighted sum, projection 768->256, BiGRU (2x128), attentive statistics pooling -> 512", "top-2 layers fine-tuned, projection 768->256, masked attention pooling -> 256", "-"],
             ["4 Fusion", "-", "-", "concatenation or bidirectional gated cross-attention (4 heads) -> 256"],
             ["5 Classifier", "LayerNorm-MLP 512->256->7", "LayerNorm-MLP 256->256->7", "LayerNorm-MLP 256->256->7"]], [1.2, 3, 2.6, 2.4])
    R.h2("4.2  Preprocessing and speech features")
    R.p("Audio is converted to 16 kHz mono, leading/trailing silence is trimmed at 30 dB below peak with 50 ms padding and the peak is normalised to -1 dBFS "
        "(mer.features.preprocess_audio). WavLM-Base+ [2] is run once and all 13 hidden states (CNN output plus 12 transformer layers) are cached in fp16 "
        "([13, T, 768], T about 50 frames per second); caching makes cross-validated training feasible on a 4 GB GPU, at the price of not fine-tuning the encoder. "
        "A separate 88-dimensional eGeMAPS functional vector [3] feeds the classical baseline.")
    R.h2("4.3  Speech model: WavLM-BiGRU-ASP")
    R.fig(ROOT / "docs/figures/arch_2b_speech.png", "Speech model (src/mer/models/speech.py). B = batch, T = frames.", 480, 150)
    R.p("A softmax-normalised learnable vector over the 13 layers [4] produces a task-specific mixture of low-level (acoustic) and high-level (phonetic) representations; "
        "its learned weights are almost uniform (0.072-0.083; layer-weight figure in Section 5.4), so no single layer dominates. SpecAugment-style time/feature masking [5] (p = 0.3, training only) "
        "regularises the sequence before a 768->256 projection. A single-layer bidirectional GRU [6] (128 units per direction, packed sequences) models temporal dynamics and "
        "attentive statistics pooling [7] returns the attention-weighted mean and standard deviation (512-d), capturing how the emotion cue is distributed and how variable it is. "
        "A LayerNorm-MLP (512-256-7, GELU, dropout 0.3) classifies.")
    R.h2("4.4  Text model: RoBERTa with attention pooling")
    R.fig(ROOT / "docs/figures/arch_2b_text.png", "Text model (src/mer/models/text.py). N = token count (at most 32).", 480, 150)
    R.p("RoBERTa-base [8] encodes the carrier phrase; all parameters are frozen except the top two transformer layers (encoder learning rate 2e-5, head 1e-3). "
        "Token states are projected to 256-d and pooled with masked attention; a 256-256-7 MLP classifies. Because the transcript is identical across emotions, "
        "this model is a <i>control</i>: any accuracy above 1/7 would indicate leakage.")
    R.h2("4.5  Fusion models")
    R.fig(ROOT / "docs/figures/arch_2b_fusion.png", "Bidirectional gated cross-attention fusion (src/mer/models/fusion.py) and the concatenation variant.", 480, 400)
    R.p("<b>Concatenation</b> projects the 512-d speech embedding and the 256-d text embedding to 256-d, concatenates them (512-d) and applies Linear-LayerNorm-GELU "
        "before the classifier; it isolates the effect of simply adding text. <b>Gated cross-attention</b> follows multimodal-transformer and gated-unit ideas [9][10]: speech "
        "frames [T, 256] attend to text tokens [N, 256] and vice versa (two 4-head MultiheadAttention blocks with residual LayerNorm and padding masks), each context sequence is "
        "attention-pooled to 256-d, and a learned vector gate z = sigmoid(W[h<sub>s</sub>; h<sub>t</sub>]) mixes them: fused = z*h<sub>s</sub> + (1-z)*h<sub>t</sub>. "
        "Modality dropout (p = 0.2; speech or text is zeroed for a sample, never both) is applied in training. A <i>zero-text</i> test-time intervention replaces the text "
        "representations by zeros to probe reliance on text. Gate values are descriptive, not causal contribution measures.")
    R.h2("4.6  Baselines")
    R.p("<b>eGeMAPS + SVM</b> (StandardScaler, RBF SVM with C = 10) and <b>TF-IDF + logistic regression</b> (1-2 grams) are trained on train+validation IDs of each fold "
        "(mer.models.classical). They therefore see more data than the neural models, which fit on train only; the original rows are reported as produced and the fitting "
        "difference is a stated caveat.")
    R.h2("4.7  Training configuration")
    rows = []
    for n, c in cfgs.items():
        t = c["train"]
        rows.append([n, t["batch_size"], t["epochs"], t["patience"], t["lr"], t.get("encoder_lr", "-"), t["weight_decay"], t["label_smoothing"]])
    R.table("Training hyper-parameters read from configs/*.yaml (AdamW, cosine schedule, gradient clipping 1.0, mixed precision, seed 42).",
            ["Config", "Batch", "Max epochs", "Patience", "Head LR", "Encoder LR", "Weight decay", "Label smoothing"], rows, [1.6, .7, 1, .8, .8, 1, 1, 1.1])
    R.p("Checkpoint selection uses validation macro-F1. Observed per-fold training lengths range from 7 to 18 epochs (train_history.json) because early stopping triggers quickly on this easy task.")

    R.h1("5  Results")
    R.h2("5.1  Headline comparison")
    nm = [("Speech", "speech_wavlm_bigru_asp"), ("Text", "text_roberta_top2ft"), ("Concat fusion", "fusion_concat"), ("Gated fusion", "fusion_gated_xattn")]
    rows = []
    for lab, stem in nm:
        for p in "ABC":
            r = stem + "_proto" + p
            rows.append([lab, p, f(r, "accuracy"), f(r, "uar"), f(r)])
    for r in ["classical_speech_protoA", "classical_speech_protoB", "classical_speech_protoC", "classical_text_protoA", "classical_text_protoB", "classical_text_protoC"]:
        a = acc[r]
        rows.append([("eGeMAPS+SVM" if "speech" in r else "TF-IDF+LR"), r[-1], a["accuracy"].replace("+/-", "+/-"), a["uar"], a["macro_f1"]])
    R.table("Test performance (%) as fold mean +/- population SD. Neural rows: checkpoint-linked re-evaluation (Results/verified_p0). Classical rows: train+validation fit.",
            ["System", "Protocol", "Accuracy", "UAR", "Macro-F1"], rows, [1.8, .8, 1.6, 1.6, 1.6])
    R.fig(P / "model_comparison_protocols.png", "Macro-F1 of all systems on protocols A, B and C (generated from saved predictions).", 440, 250)
    R.p("<b>Reading the table.</b> (i) On A and C every acoustic model is near ceiling (>= 98.4% macro-F1), so these protocols hardly discriminate between systems; the "
        "eGeMAPS+SVM baseline is already 98.4%. (ii) Protocol B separates the models: speech 56.2%, gated fusion 46.4%, concatenation 34.4%, eGeMAPS+SVM only 12.5% macro-F1 "
        "(chance is 14.3% accuracy and 3.57% macro-F1 for a constant predictor). (iii) Text is at chance everywhere; for the TF-IDF baseline accuracy on C is 0.00% "
        "(Section 8.2).")
    R.h2("5.2  Confusion matrices")
    R.fig(P / "cm_speech_protoA.png", "Speech, Protocol A (word-disjoint): near-perfect diagonal.", 330, 260)
    R.fig(P / "cm_speech_protoB.png", "Speech, Protocol B (speaker transfer): happy is never recovered; fear is perfect.", 330, 260)
    R.fig(P / "cm_text_protoA.png", "Text, Protocol A: predictions collapse onto a single class, as expected for label-free text.", 330, 260)
    R.fig(P / "cm_fusion_protoA.png", "Gated fusion, Protocol A.", 330, 260)
    R.fig(P / "cm_fusion_protoB.png", "Gated fusion, Protocol B: different errors from speech alone.", 330, 260)
    R.h2("5.3  Per-emotion recall")
    rows = []
    for i, e in enumerate(LABELS):
        rows.append([e] + [pct(good[stem + "_protoB"]["recalls"][i], 1) for _, stem in nm] + [pct(good["speech_wavlm_bigru_asp_protoA"]["recalls"][i], 2)])
    R.table("Mean per-emotion recall (%) on Protocol B (speaker transfer); last column: speech on Protocol A.",
            ["Emotion", "Speech B", "Text B", "Concat B", "Gated B", "Speech A"], rows, [1.2, 1, 1, 1, 1, 1])
    R.p("Under speaker transfer the speech model recognises <i>fear</i> perfectly (100%) but <i>happy</i> never (0%) and <i>angry</i> in only 42.5% of cases. In Protocol A all "
        "emotions are at least 99.75% (easiest: angry 100.00%, hardest: surprise 99.75%; ties exist). Gated fusion lifts <i>neutral</i> (+26.25 points) and <i>surprise</i> (+12.50) "
        "but loses <i>angry</i> (-38.75), <i>fear</i> (-27.50) and <i>sad</i> (-35.00) relative to speech alone, which is consistent with arousal cues that do not transfer between the "
        "two actresses (YAF is louder than OAF, Section 2). The causes are hypotheses; no controlled or listening experiment was run.")
    R.h2("5.4  Does fusion help? Paired analysis and ablation")
    rows = []
    for p in "ABC":
        for lab, stem in nm[2:]:
            z = br.js(vroot / f"{stem}_proto{p}_paired.json")
            sp = good["speech_wavlm_bigru_asp_proto" + p]; fu = good[stem + "_proto" + p]
            rows.append([p, lab, f"{100 * (fu['macro_f1_mean'] - sp['macro_f1_mean']):+.2f}", z["b_wins_a_loses"], z["a_wins_b_loses"], f"{z['p_value']:.2g}"])
    R.table("Fusion versus speech-only on identical test utterances: macro-F1 difference (points), utterances fixed (fusion right, speech wrong), broken (speech right, fusion wrong) and exact McNemar p-value (exploratory).",
            ["Protocol", "Fusion", "Delta macro-F1", "Helped", "Harmed", "p (McNemar)"], rows, [.9, 1.5, 1.3, .9, .9, 1.2])
    R.p("In no protocol does fusion improve macro-F1: deltas are -0.04 to -0.18 points on A and C and -9.84 (gated) / -21.79 (concat) points on B, where gated fusion repairs 45 "
        "utterances but breaks 93. The McNemar statistics ignore dependence between utterances sharing a word or speaker and between cross-validation folds, so they are exploratory only. "
        "Since P(emotion | text) is uniform, the text branch can only contribute speaker- or word-specific noise; fusion cannot add label information here.")
    sg = br.js(vroot / "fusion_gated_xattn_protoA/summary.json"); zA = br.js(vroot / "fusion_gated_xattn_protoA_zerotext/summary.json"); zB = br.js(vroot / "fusion_gated_xattn_protoB_zerotext/summary.json")
    gm = [br.js(vroot / f"fusion_gated_xattn_proto{p}/summary.json")["gate_mean"] for p in "ABC"]
    R.bullets([f"<b>Gate statistics:</b> mean gate 0.508 / 0.506 / 0.508 on A / B / C (value near 1 = speech-weighted). The gate stays close to 0.5, consistent with both streams already being cross-attended; it is descriptive and not a causal modality attribution.",
               f"<b>Zero-text intervention</b> on the gated model: A {pct(zA['accuracy_mean'])}% accuracy / {pct(zA['macro_f1_mean'])}% macro-F1 (vs {pct(sg['accuracy_mean'])}% / {pct(sg['macro_f1_mean'])}% with text); "
               f"B {pct(zB['accuracy_mean'])}% / {pct(zB['macro_f1_mean'])}% (vs 52.14% / 46.40%). Removing the text path does not restore speech-only accuracy on A, "
               "so modality dropout did not make the fusion model robust to a missing modality; the intervention is applied to intermediate representations of a model trained with text, not to a retrained speech model. "
               "On B, zeroing text slightly <i>raises</i> accuracy, again indicating that text adds noise."])
    R.fig(P / "training_curves_protoA.png", "Validation curves of the verified runs (Protocol A). Gated Protocol C has no saved history; none was fabricated.", 440, 250)
    R.fig(P / "wavlm_layer_weights.png", "Learned softmax weights over the 13 WavLM layers (mean over Protocol A folds): near-uniform.", 400, 220)

    R.h1("6  Learned-representation analysis")
    R.p("Block-3/4 embeddings of fold 0 of Protocol A (560 test utterances) are projected with t-SNE and UMAP for the temporal (speech ASP, 512-d), contextual (text pool, 256-d) and "
        "fused (gated, 256-d) blocks. Each panel is a separate within-fold projection with its own axes. Silhouette scores computed on the embeddings "
        "(Results/plots/*_metrics.json) quantify what the maps show:")
    R.table("Fold-0 embedding structure (n = 560 each).", ["Block", "Silhouette by emotion", "Silhouette by speaker"],
            [[k, f"{v['silhouette_emotion']:.3f}", f"{v['silhouette_speaker']:.3f}"] for k, v in emb.items()], [2, 1.6, 1.6])
    R.fig(P / "embeddings_umap_protoA.png", "UMAP of temporal, contextual and fused embeddings (Protocol A, fold 0).", 480, 190)
    R.fig(P / "embeddings_tsne_protoA.png", "t-SNE of the same three embeddings.", 480, 190)
    R.p("The speech embedding clusters by emotion (silhouette 0.49) and not by speaker (0.02); the text embedding shows no structure for either (-0.01), confirming that RoBERTa "
        "sees no emotional signal; the fused embedding is the most compact (0.78) because the gate lets the head operate on a mixed 256-d space. These are descriptive diagnostics: "
        "2-D separation is not independent evidence of predictive performance, and silhouette values from spaces of different dimensionality should not be ranked strictly.")

    R.h1("7  Failure analysis")
    rows = [[f"{x['run'].replace('_proto', ' / ')} fold {x['fold']}", x["utt_id"], f"{LABELS[int(x['y_true'])]} -> {LABELS[int(x['y_pred'])]}", f"{float(x['confidence']):.3f}"] for x in failures]
    R.table("Five prediction-backed errors (Results/failure_cases.csv); each row was matched to the saved predictions.", ["Run", "Utterance", "True -> predicted", "Confidence"], rows, [3, 1.8, 1.8, 1])
    R.fig(P / "failure_cases.png", "Waveform, spectrogram, F0 contour and class probabilities for the five failures.", 480, 400)
    R.p("The selection favours confident errors (0.57-0.83 probability) and diverse (true, predicted) pairs; it is not an unbiased error sample. Four of five cases come from the "
        "speaker-transfer protocol and show the high-arousal confusions of Section 5.3: <i>happy</i> predicted as <i>angry</i> or <i>surprise</i>, <i>angry</i> as <i>fear</i>, "
        "and <i>neutral</i> as <i>angry</i>. The single Protocol-A error (surprise -> happy, OAF <i>make</i>) has the lowest confidence (0.57). Mean F0 and duration are shown in "
        "Results/failure_cases.md as descriptive values; explanations such as acting style are hypotheses that would require listening or controlled tests.")

    R.h1("8  Discussion")
    R.h2("8.1  Speech carries the signal; the protocol decides how much is recoverable")
    R.p("The 99.9% scores of A and C overstate generalisation: both speakers appear in training, and a simple 88-feature SVM already reaches 98.4%. The unseen-speaker, unseen-word protocol B "
        "drops the best model to 60.7% accuracy (56.2% macro-F1), with a 100%/0% recall split between fear and happy. Any claim about emotion recognition from TESS must therefore "
        "state the protocol, and two speakers cannot support population-level conclusions.")
    R.h2("8.2  Why text is at chance and why a classical text model scores 0%")
    R.p("Every transcript occurs exactly once per emotion and speaker, so a text-only model cannot beat 1/7 in expectation. Under random stratified splits (C), however, a held-out "
        "utterance's emotion is, by construction, under-represented among the training utterances with the same transcript; a TF-IDF/logistic model trained on train+validation therefore predicts "
        "the complement's majority class and is systematically <i>wrong</i>: 0.00% accuracy in every fold, which was reproduced independently. This complementary-imbalance artefact is "
        "not evidence of emotion understanding or of swapped labels; the neural text model, trained on train only, scores 14.25%.")
    R.h2("8.3  Why fusion does not help")
    R.p("Fusion helps when the second modality carries complementary label information. Here text has none, so the fusion model can only learn speaker/word shortcuts; on B it "
        "generalises worse than speech alone. This is a property of the dataset, not a general verdict on multimodal fusion: on corpora with real semantic content text would matter.")
    R.h1("9  Limitations and provenance disclosure")
    R.bullets(["<b>Two speakers, acted speech, single language.</b> No population-level claim; B is a two-fold stress test.",
               "<b>Fitting budgets differ.</b> Classical baselines fit train+validation, neural models train only; original rows are retained and not tuned.",
               "<b>Text-training provenance.</b> Historical text/fusion training was affected by concurrent writers into shared checkpoint folders. Original completion markers are not used. "
               "Final numbers come from re-evaluating the final checkpoints into a separate verification directory; per-run records bind checkpoint, prediction and embedding SHA-256 hashes. "
               "This proves which weights produced the predictions, not the full training lineage.",
               "<b>Missing items.</b> Gated Protocol C has no saved epoch history; the optional late-fusion run was interrupted and is excluded; no listening tests; failure-case causes are hypotheses.",
               "<b>Statistics.</b> McNemar tests ignore dependence between utterances and folds; fold standard deviations are descriptive."])
    R.h1("10  Reproducibility")
    R.p("Environment and commands are in <font face='Courier'>assignment_2b_emotion/README.md</font>; exact package versions are pinned in requirements.txt, environment_versions.json "
        "and requirements-lock-windows.txt. TESS (CC BY-NC 4.0), cached features and checkpoints are not stored in Git. Fresh runs should use a new experiment name (<font face='Courier'>--name</font>).")
    R.code("cd assignment_2b_emotion; $env:PYTHONPATH = (Resolve-Path src).Path\n"
           "python -m mer.data.download_tess --out data/raw/TESS\n"
           "python -m mer.data.prepare_tess --raw data/raw/TESS --out data/metadata.csv\n"
           "python -m mer.features.extract_ssl --meta data/metadata.csv \\\n"
           "    --model microsoft/wavlm-base-plus --tag wavlm_base_plus --out data/features\n"
           "python models/speech_pipeline/train.py --config configs/speech.yaml --protocol A\n"
           "python models/fusion_pipeline/train.py --config configs/fusion.yaml --protocol A\n"
           "python -m pytest tests -q                                        # 10 tests")
    R.h1("11  Conclusion")
    R.p("A frozen-WavLM speech model with attentive statistics pooling is an excellent emotion classifier on TESS when speakers are seen, and a mediocre one (56% macro-F1) when they are not. "
        "Text is uninformative by construction, so bidirectional gated cross-attention and concatenation fusion do not improve over speech and hurt under speaker transfer. The evaluation "
        "design - three frozen protocols, leakage tests, paired analysis and checkpoint-linked verification - matters as much as the architecture, and the honest limitations above "
        "bound what the numbers can claim.")
    R.h1("References")
    for r in ["[1] M. K. Pichora-Fuller, K. Dupuis. Toronto emotional speech set (TESS). Borealis, 2020. https://doi.org/10.5683/SP2/E8H2MF",
              "[2] S. Chen et al. WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE JSTSP 2022. arXiv:2110.13900.",
              "[3] F. Eyben et al. The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for voice research and affective computing. IEEE Trans. Affective Computing 2016.",
              "[4] S.-w. Yang et al. SUPERB: Speech processing Universal PERformance Benchmark. Interspeech 2021.",
              "[5] D. S. Park et al. SpecAugment: A simple data augmentation method for automatic speech recognition. Interspeech 2019.",
              "[6] K. Cho et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014.",
              "[7] K. Okabe, T. Koshinaka, K. Shinoda. Attentive statistics pooling for deep speaker embedding. Interspeech 2018. arXiv:1803.10963.",
              "[8] Y. Liu et al. RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692, 2019.",
              "[9] Y.-H. H. Tsai et al. Multimodal Transformer for unaligned multimodal language sequences. ACL 2019. arXiv:1906.00295.",
              "[10] J. Arevalo et al. Gated multimodal units for information fusion. ICLR Workshop 2017. arXiv:1702.01992.",
              "[11] L. McInnes, J. Healy, J. Melville. UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426, 2018.",
              "[12] L. van der Maaten, G. Hinton. Visualizing data using t-SNE. JMLR 9, 2008.",
              "[13] I. Loshchilov, F. Hutter. Decoupled weight decay regularization (AdamW). ICLR 2019."]:
        R.p(esc(r), "ref")
    R.build(cov)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verification-root", type=Path, default=B2 / "Results/verified_p0")
    a = ap.parse_args()
    ev = br.Evidence(ROOT); ev.verification_root = a.verification_root.resolve()
    br.a_pages(ev)
    good, status, failures = br.collect_b(ev)
    if ev.pending:
        print("EVIDENCE GATE FAILED:\n" + "\n".join("- " + x for x in ev.pending), file=sys.stderr); return 2
    build_2a(ev, A2 / "report")
    build_2b(ev, good, status, failures, B2 / "report", ev.verification_root)
    import pymupdf
    for p in [A2 / "report/report_2a.pdf", B2 / "report/report_2b.pdf"]:
        d = pymupdf.open(p); print(p.name, len(d), "pages", round(p.stat().st_size / 1e6, 2), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
