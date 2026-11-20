"""Minimal Markdown -> PDF converter (PyMuPDF Story + python-markdown).

Usage: python tools/md2pdf.py INPUT.md [OUTPUT.pdf]
Images referenced relative to the markdown file are embedded. Mermaid blocks are kept as code.
"""
import sys
from pathlib import Path

import markdown
import pymupdf

CSS = """
body { font-family: sans-serif; font-size: 9pt; line-height: 1.35; }
h1 { font-size: 16pt; color: #1a3d6d; } h2 { font-size: 13pt; color: #1a3d6d; margin-top: 10pt; }
h3 { font-size: 11pt; color: #234; } h4 { font-size: 10pt; }
table { border-collapse: collapse; margin: 4pt 0; } th, td { border: 0.5pt solid #999; padding: 2pt 3pt; font-size: 7.5pt; vertical-align: top; }
th { background-color: #e8eef7; }
code { font-family: monospace; font-size: 7.5pt; background-color: #f3f3f3; }
pre { font-family: monospace; font-size: 7pt; background-color: #f5f5f5; padding: 4pt; }
blockquote { color: #444; border-left: 2pt solid #aac; padding-left: 6pt; }
img { width: 100%; }
"""


def convert(md_path: Path, pdf_path: Path):
    html = markdown.markdown(md_path.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    story = pymupdf.Story(html=html, user_css=CSS, archive=pymupdf.Archive(str(md_path.parent)))
    writer = pymupdf.DocumentWriter(str(pdf_path))
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (36, 40, -36, -40)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    convert(src, dst)
