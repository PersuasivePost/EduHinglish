"""
Re-extract clean, readable text from the new-syllabus NCERT chapter PDFs.

The old cleaned_text.txt files interleave the margin column ("Think It Over"
questions, "Meet a Scientist" boxes) and figure labels with the main body text.
This script uses word positions and font sizes from pdfplumber to:
  - keep main body text (11pt) in reading order and re-join justified lines,
  - keep headings (bold, larger than body) on their own lines,
  - collect sidebar boxes (10pt) as separate [Box] paragraphs after the page's text,
  - drop figure labels, captions, running headers/footers and print-shop marks.

Usage:
    python scripts/clean_ncert_text.py                 # all chapters
    python scripts/clean_ncert_text.py --subject ss    # only Social Science
    python scripts/clean_ncert_text.py --only class9_bio_ch02
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pdfplumber

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# processed folder name -> source PDF
SS_PDFS = {
    **{f"ss9_ch{n:02d}": f"class 9 ss/class 9 ss ch {n}.pdf" for n in range(1, 10)},
    **{f"ss10_his_ch{n:02d}": f"class 10 ss/class 10 hist ch {n}.pdf" for n in range(1, 6)},
    **{f"ss10_geo_ch{n:02d}": f"class 10 ss/class 10 geo ch {n}.pdf" for n in range(1, 8)},
    **{f"ss10_civics_ch{n:02d}": f"class 10 ss/class 10 civics ch {n}.pdf" for n in range(1, 6)},
    **{f"ss10_eco_ch{n:02d}": f"class 10 ss/class 10 eco ch {n}.pdf" for n in range(1, 6)},
}
BIO_PDFS = {
    "class9_bio_ch02":  "class 9 biology/class9_ch02.pdf",
    "class9_bio_ch03":  "class 9 biology/class9_ch03.pdf",
    "class9_bio_ch11":  "class 9 biology/class9_ch11.pdf",
    "class9_bio_ch12":  "class 9 biology/class9_ch12.pdf",
    "class10_bio_ch05": "class 10 biology/class10_ch05.pdf",
    "class10_bio_ch06": "class 10 biology/class10_ch06.pdf",
    "class10_bio_ch07": "class 10 biology/class10_ch07.pdf",
    "class10_bio_ch08": "class 10 biology/class10_ch08.pdf",
    "class10_bio_ch13": "class 10 biology/class10_ch13.pdf",
}
CHAPTER_PDFS = {**BIO_PDFS, **SS_PDFS}

HEADER_ZONE = 0.085  # fraction of page height: running chapter title / page number
FOOTER_ZONE = 0.92   # fraction of page height: print-shop line (CChhaapptteerr--0022..iinndddd ...)
LINE_TOL    = 3.0    # words within this vertical distance are on the same line
WORD_GAP    = 30     # max horizontal gap between words of the same line segment
SKIP_FONTS  = ("Wingdings", "Symbol", "ZapfDingbats")
NOT_A_HEADING = re.compile(r"^(\d+|Chapter|Grade \d+|Curiosity)$")
ENDS_SENTENCE = re.compile(r"[.?!:)\"'”’]$")


def body_font_size(pdf) -> float:
    """Most common font size by word count; 11pt in the new books, 10.5pt in the older Class 10 ones."""
    sizes = Counter()
    for page in pdf.pages[1:7]:  # a few body pages is enough; skip the title page
        sizes.update(round(w["size"], 1) for w in page.extract_words(extra_attrs=["size"]))
        page.close()  # pdfplumber caches parsed pages; free them to keep memory flat
    return sizes.most_common(1)[0][0]


def classify(word: dict, body: float, page_height: float, box_depth: float = 1.4) -> str | None:
    size, font = word["size"], word["fontname"]
    if any(f in font for f in SKIP_FONTS):
        return None
    if word["top"] > FOOTER_ZONE * page_height or (word["top"] < HEADER_ZONE * page_height and size < 20):
        return None
    if size >= body * 2.5 and len(word["text"]) == 1 and word["text"].isalpha():
        return "dropcap"
    if body + 0.9 <= size < 40:
        return "heading"
    if "Bold" in font and word["text"].isupper() and len(word["text"]) > 1 and body - 3 <= size < body - 0.45:
        return "smallcap"  # rest of a small-caps heading: 'D' (12pt) + 'EVELOPMENT' (8.4pt)
    if abs(size - body) < 0.45:
        return "body"
    if body - box_depth <= size <= body - 0.45:
        return "box"
    return None  # figure labels, captions, rotated text


def collapse_overprint(token: str) -> str:
    """Older books fake bold by printing each glyph k times: 'HHHHHEEEEE' -> 'HE', '88..22' -> '8.2'."""
    if len(token) in (4, 5) and len(set(token)) == 1:
        return token[0]  # a single overprinted letter of a letter-spaced title ('UUUUU')
    for k in (5, 4, 3, 2):
        if len(token) >= 2 * k and len(token) % k == 0 and token == "".join(c * k for c in token[::k]):
            return token[::k]
    return token


def dedupe_words(text: str) -> str:
    """'Activity Activity Activity 8.2' -> 'Activity 8.2' (overprinted words also repeat whole)."""
    out = []
    for t in text.split():
        if not out or t != out[-1]:
            out.append(t)
    return " ".join(out)


def column_starts(words: list[dict], page_width: float) -> list[float]:
    """Left edges of side columns (e.g. History 'New words' glossaries set at body size).
    Many lines start at the same x there, which justified body text never does by chance."""
    counts = Counter(round(w["x0"] / 2) * 2 for w in words if w["x0"] > page_width * 0.45)
    return [x for x, n in counts.items() if n >= 5]


def line_segments(words: list[dict], split_at: list[float] = ()) -> list[dict]:
    """Group words of one class into horizontal line segments."""
    # Cluster into lines first: bold words sit ~1.5pt lower than regular ones on the same line.
    lines = []
    for w in sorted(words, key=lambda w: w["top"]):
        w["text"] = collapse_overprint(w["text"])
        if lines and abs(w["top"] - lines[-1][0]["top"]) < LINE_TOL + (1.5 if w["size"] < 9.5 else 0):
            lines[-1].append(w)
        else:
            lines.append([w])

    segments = []
    for line in lines:
        current = None
        for w in sorted(line, key=lambda w: w["x0"]):
            glued = current and len(current["text"].split()[-1]) == 1 and current["text"][-1].isupper() \
                and w["x0"] - current["x1"] < 1.5  # small-caps initial
            starts_column = any(abs(w["x0"] - x) < 2.5 for x in split_at) and w["x0"] - current["x1"] > 6 \
                if current else False
            if current and -3 <= w["x0"] - current["x1"] < WORD_GAP and not starts_column:
                current["text"] += ("" if glued or w["text"] in ",.;:)" else " ") + w["text"]
                current["x1"] = w["x1"]
            else:
                current = {"top": line[0]["top"], "x0": w["x0"], "x1": w["x1"], "text": w["text"]}
                segments.append(current)
    return segments


def blocks(segments: list[dict], size: float) -> list[list[dict]]:
    """Group line segments into column blocks (similar x-start, small vertical gaps)."""
    result = []
    for s in sorted(segments, key=lambda s: (s["top"], s["x0"])):
        for b in result:
            last = b[-1]
            same_column = abs(b[0]["x0"] - s["x0"]) < 25 or (last["x0"] <= s["x0"] <= last["x1"])
            if same_column and 0 < s["top"] - last["top"] < size * 2.2:
                b.append(s)
                break
        else:
            result.append([s])
    return result


def join_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if text.endswith("-") and not text.endswith(" -"):
            text = text[:-1] + line if line[:1].islower() else text + line
        else:
            text = f"{text} {line}" if text else line
    return re.sub(r"\s+", " ", text).strip()


MIRRORED = re.compile(r"retpahC|NOITCES|tnirpeR")  # rotated side tabs read back-to-front
SYMBOLS = {"\uf050": "✓", "\uf04f": "✗"}  # Wingdings2 tick / cross used in NCERT tables


def render_table(rows: list[list]) -> str | None:
    cells = [[re.sub(r"\s+", " ", "".join(SYMBOLS.get(ch, ch) for ch in (c or ""))).strip() for c in row] for row in rows]
    cells = [[c for c in row if c] for row in cells]
    cells = [row for row in cells if row]
    flat = [c for row in cells for c in row]
    if len(cells) < 2 or max(len(r) for r in cells) < 2 or sum(len(c) for c in flat) / len(flat) < 2:
        return None  # a framed figure or box (or letters of a figure label), not a real table
    return " / ".join(" | ".join(row) for row in cells)


def in_any(word: dict, bboxes: list[tuple]) -> bool:
    return any(x0 - 2 <= word["x0"] and word["x1"] <= x1 + 2 and top - 2 <= word["top"] and word["bottom"] <= bottom + 2
               for x0, top, x1, bottom in bboxes)


CAPTION = re.compile(r"^(Fig\.?|Figure|Source\s*[A-Z]?\s*[:\-–]|Map\s+\d|Table\s+\d)", re.I)


RUNNING_HEAD = re.compile(r"Understanding Society|India and Beyond|Beyond\b|Grade 9|Part 1")  # Class 9 SS footers
CHAPTER_FOOTER = re.compile(r"^\d+ – [^.]{3,80} \d+$")                                       # "1 – Understanding ... 3"


def is_page_furniture(text: str) -> bool:
    return CHAPTER_FOOTER.match(text) is not None or (len(text.split()) <= 12 and RUNNING_HEAD.search(text) is not None)


def extract_page(page, body: float, box_depth: float = 1.4,
                 side_columns: bool = False) -> tuple[list[tuple[float, str, str]], list[str]]:
    tables, table_boxes = [], []
    for t in page.find_tables():
        rendered = render_table(t.extract())
        if rendered and not MIRRORED.search(rendered):
            tables.append(f"[Table] {rendered}")
            table_boxes.append(t.bbox)

    words = [w for w in page.extract_words(extra_attrs=["size", "fontname"], x_tolerance=1.5)
             if not in_any(w, table_boxes)]
    by_class = {"body": [], "heading": [], "box": [], "dropcap": [], "smallcap": []}
    for w in words:
        cls = classify(w, body, page.height, box_depth)
        if cls:
            by_class[cls].append(w)

    items = []  # (top, kind, text) for body + headings, in reading order
    # Only the Class 10 History book keeps glossaries in a body-sized side column; elsewhere,
    # lines that restart at one x are text wrapping around a figure and must stay in the body.
    side_columns = column_starts(by_class["body"], page.width) if side_columns else []
    body_segments = line_segments(by_class["body"], side_columns)
    side_boxes = []
    for b in blocks(body_segments, body):
        text = join_lines([s["text"] for s in b])
        if side_columns and any(abs(b[0]["x0"] - x) < 2.5 for x in side_columns):
            side_boxes.append(text)  # glossary / margin note in a side column
        else:
            items.append((b[0]["top"], "body", text))
    # Older books open a chapter with a large drop-cap letter ("H" + "ow do we tell ...").
    for cap in by_class["dropcap"]:
        for i, (top, kind, text) in enumerate(items):
            if kind == "body" and abs(top - cap["top"]) < body * 3 and text[:1].islower():
                items[i] = (top, kind, cap["text"] + text)
                break

    # Headings outside the main body column are margin labels ("Think It Over",
    # "Ready to Go Beyond", ...) and only interrupt the body flow, so drop them.
    if body_segments:
        x0s = sorted(s["x0"] for s in body_segments)
        x1s = sorted(s["x1"] for s in body_segments)
        col_x0, col_x1 = x0s[len(x0s) // 4] - 45, x1s[3 * len(x1s) // 4]
    else:
        col_x0, col_x1 = 0, page.width
    for s in line_segments(by_class["heading"] + by_class["smallcap"]):
        text = dedupe_words(s["text"].strip())
        too_short = not any(len(t) >= 3 and t.isalpha() for t in text.split())  # decorative glyph runs
        if too_short or NOT_A_HEADING.match(text) or not (col_x0 <= s["x0"] < col_x1):
            continue
        items.append((s["top"], "heading", text))
    items.sort(key=lambda t: t[0])

    boxes = [join_lines([s["text"] for s in b]) for b in blocks(line_segments(by_class["box"]), body - 1)]
    boxes = [b for b in merge_boxes(side_boxes + boxes)
             if len(b.split()) >= 4 and not CAPTION.match(b) and not is_page_furniture(b)]
    return items, boxes + tables


def merge_headings(items: list[tuple[float, str, str]]) -> list[tuple[str, str]]:
    """Join heading lines that wrap onto the next line (vertically adjacent)."""
    merged, last_top = [], None
    for top, kind, text in items:
        if kind == "heading" and merged and merged[-1][0] == "heading" and top - last_top < 35:
            merged[-1] = ("heading", dedupe_words(f"{merged[-1][1]} {text}"))
        else:
            merged.append((kind, text))
        last_top = top
    return merged


def merge_boxes(boxes: list[str]) -> list[str]:
    """Join box fragments where a sentence runs on into the next fragment."""
    merged = []
    for b in boxes:
        if merged and not ENDS_SENTENCE.search(merged[-1]) and b[:1].islower():
            merged[-1] = join_lines([merged[-1], b])
        else:
            merged.append(b)
    return merged


def manual_page(manual_dir: Path | None, index: int) -> list[tuple[str, str]] | None:
    """A few pages are scanned images with no text layer; their hand transcription lives in
    data/processed/<chapter>/page_<index>.txt and replaces the (empty) extraction."""
    path = manual_dir / f"page_{index}.txt" if manual_dir else None
    if not path or not path.exists():
        return None
    items = []
    for para in path.read_text(encoding="utf-8").split("\n\n"):
        para = para.strip()
        if para.startswith("## "):
            items.append(("heading", para[3:]))
        elif para:
            items.append(("body", para))
    return items


def extract_chapter(pdf_path: Path, manual_dir: Path | None = None, box_depth: float = 1.4,
                    side_columns: bool = False) -> str:
    paragraphs, pending_boxes = [], []
    with pdfplumber.open(pdf_path) as pdf:
        body = body_font_size(pdf)
        for index, page in enumerate(pdf.pages):
            manual = manual_page(manual_dir, index)
            if manual is not None:
                items, boxes = [], []
                for kind, text in manual:
                    if text.startswith("[Box] "):
                        boxes.append(text[6:])
                    else:
                        items.append((0, kind, text))
            else:
                items, boxes = extract_page(page, body, box_depth, side_columns)
            page.close()
            for kind, text in merge_headings(items):
                if kind == "heading":
                    paragraphs.extend(b if b.startswith("[Table]") else f"[Box] {b}" for b in pending_boxes)
                    pending_boxes = []
                    paragraphs.append(f"## {text}")
                elif paragraphs and not paragraphs[-1].startswith(("## ", "[Box]")) \
                        and not ENDS_SENTENCE.search(paragraphs[-1]):
                    paragraphs[-1] = join_lines([paragraphs[-1], text])  # continues across block/page
                else:
                    paragraphs.append(text)
            pending_boxes.extend(boxes)
    paragraphs.extend(b if b.startswith("[Table]") else f"[Box] {b}" for b in pending_boxes)
    return "\n\n".join(paragraphs) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="process a single processed-folder name")
    parser.add_argument("--subject", choices=["bio", "ss"], help="process only one subject's chapters")
    args = parser.parse_args()

    chapters = {"bio": BIO_PDFS, "ss": SS_PDFS}.get(args.subject, CHAPTER_PDFS)
    for name, rel_pdf in chapters.items():
        if args.only and name != args.only:
            continue
        # Social Science books set primary sources and activity boxes in 9pt, so look deeper for boxes.
        box_depth = 2.6 if name in SS_PDFS else 1.4
        text = extract_chapter(RAW_DIR / rel_pdf, PROCESSED_DIR / name, box_depth,
                               side_columns=name.startswith(("ss10_his", "ss10_geo")))
        out = PROCESSED_DIR / name / "cleaned_text.txt"
        out.write_text(text, encoding="utf-8")
        print(f"[OK] {name}: {len(text.split())} words -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
