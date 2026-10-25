"""Tables and figures straight from the PDF layout with PyMuPDF (no AI).

Tables: `find_tables` boxes that have a "Table N" caption next to them (that filter removes the
formulas and plots PyMuPDF also boxes); merged header cells are carried across, rows that PyMuPDF
packed into multi-line cells are split, and every row is written out with its column names.
Figures: the vector drawings and images next to a "Figure N" caption, rendered to PNG.
"""
import re
import unicodedata

import pymupdf

from . import config

# Some templates localize the label ("Tabela IV", "Figura 2")
TABLE_CAP = re.compile(r"^\s*(table|tab\.|tabela|tabla)\s*([IVX]+|\d+|[A-Z]\d+)\s*[:.|]?", re.I)
FIG_CAP = re.compile(r"^\s*(figure|fig\.|figura)\s*(\d+|[A-Z]\d+)\s*[:.|]?", re.I)
NUMERIC = re.compile(r"^[\s\d.,±%()+\-–−*/x×]+$")
CAPTION_GAP = 80     # max points between a table/figure and its caption
FIGURE_DPI = 150
FIGURE_KINDS = [("diagram", r"architecture|framework|overview|pipeline|illustration|schematic|structure|model"),
                ("plot", r"performance|comparison|results?|accuracy|error|loss|curve|vs\.?|versus|effect|impact"),
                ("example", r"visuali[sz]ation|example|case|map|heat ?map|sample")]


def _txt(s) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def _captions(page, pattern) -> list[tuple[pymupdf.Rect, str, str]]:
    """(rect, number, caption text) for text blocks starting with a caption label."""
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = _txt(text)
        m = pattern.match(text)
        if m:
            words = " ".join(text.split())
            out.append((pymupdf.Rect(x0, y0, x1, y1), m.group(2), " ".join(words.split()[:80])))
    return out


def _overlaps(rect, caption_rect) -> bool:
    """Horizontal overlap of at least 30% of the narrower of the two."""
    return min(rect.x1, caption_rect.x1) - max(rect.x0, caption_rect.x0) >= 0.3 * min(rect.width, caption_rect.width)


def _near(rect, caption_rect) -> float | None:
    """Vertical gap between an object and a caption that overlaps it horizontally, else None."""
    if not _overlaps(rect, caption_rect):
        return None
    gap = max(caption_rect.y0 - rect.y1, rect.y0 - caption_rect.y1)
    return gap if gap <= CAPTION_GAP else None


# ---------------------------------------------------------------- tables
def _header_rows(rows: list[list]) -> int:
    """Leading rows that hold no numeric cell are header rows; a table without numbers has one."""
    for n, row in enumerate(rows):
        if any(NUMERIC.match(c) for c in row if c):
            return max(1, n)
    return 1


def _numeric_share(rows: list[list]) -> float:
    cells = [c for row in rows for c in row if c]
    return sum(bool(NUMERIC.match(c)) for c in cells) / len(cells) if cells else 0.0


def _filled_share(rows: list[list]) -> float:
    """Share of non-empty cells; a plot boxed by the line detector is mostly empty cells."""
    cells = [c for row in rows for c in row]
    return sum(bool(c) for c in cells) / len(cells) if cells else 0.0


def _is_prose(lines: list[str]) -> bool:
    """A body-text paragraph (long lines), as opposed to table cells or labels."""
    return len(lines) >= 2 and sum(len(l.split()) >= 8 for l in lines) >= 0.6 * len(lines)


def _text_table(page, caption_rect):
    """Table without ruling lines next to its caption: PyMuPDF's text strategy inside the area
    between the caption and the next body paragraph (below first, then above)."""
    blocks = [(pymupdf.Rect(b[:4]), _txt(b[4]).split("\n")) for b in page.get_text("blocks")]
    found = []  # (gap to caption, rect, rows); the caption belongs to the nearer table
    for below in (True, False):
        if below:
            stops = [r.y0 for r, lines in blocks if r.y0 >= caption_rect.y1 and _overlaps(r, caption_rect)
                     and (_is_prose(lines) or TABLE_CAP.match(lines[0]) or FIG_CAP.match(lines[0]))]
            clip = pymupdf.Rect(page.rect.x0 + 20, caption_rect.y1, page.rect.x1 - 20,
                                min(stops + [caption_rect.y1 + 320, page.rect.y1]))
        else:
            stops = [r.y1 for r, lines in blocks if r.y1 <= caption_rect.y0 and _overlaps(r, caption_rect)
                     and (_is_prose(lines) or TABLE_CAP.match(lines[0]) or FIG_CAP.match(lines[0]))]
            clip = pymupdf.Rect(page.rect.x0 + 20, max(stops + [caption_rect.y0 - 320, page.rect.y0]),
                                page.rect.x1 - 20, caption_rect.y0)
        if clip.height < 20:
            continue
        raw, rect = _words_table(page, clip)
        if raw and len(raw) >= 2 and len(raw[0]) >= 2 and _numeric_share(raw[1:]) >= 0.2:
            found.append((rect.y0 - caption_rect.y1 if below else caption_rect.y0 - rect.y1, rect, raw))
    if not found:
        return None, None
    _, rect, raw = min(found, key=lambda f: f[0])
    return rect, raw


COLUMN_GAP = 6  # points; words closer than this belong to the same cell


def _words_table(page, clip):
    """Rows from word baselines, columns from the x-gaps that no data row crosses."""
    words = [w for w in page.get_text("words", clip=clip) if w[4].strip()]
    if not words:
        return None, None
    rows: list[list] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        mid = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - mid) < 3:
            rows[-1][1].append(w)
        else:
            rows.append([mid, [w]])
    lines = [sorted(ws, key=lambda w: w[0]) for _, ws in rows]
    data = [ws for ws in lines if any(NUMERIC.match(_txt(w[4])) for w in ws)] or lines
    spans: list[list[float]] = []
    for x0, x1 in sorted((w[0], w[2]) for ws in data for w in ws):
        if spans and x0 - spans[-1][1] < COLUMN_GAP:
            spans[-1][1] = max(spans[-1][1], x1)
        else:
            spans.append([x0, x1])
    if len(spans) < 2:
        return None, None

    def column(w) -> int:
        mid = (w[0] + w[2]) / 2
        return min(range(len(spans)), key=lambda i: 0 if spans[i][0] <= mid <= spans[i][1]
                   else min(abs(mid - spans[i][0]), abs(mid - spans[i][1])))

    table = []
    for ws in lines:
        cells = [""] * len(spans)
        for w in ws:
            i = column(w)
            cells[i] = f"{cells[i]} {_txt(w[4])}".strip()
        table.append(cells)
    rect = pymupdf.Rect(min(w[0] for w in words), min(w[1] for w in words),
                        max(w[2] for w in words), max(w[3] for w in words))
    return table, rect


def _column_names(header: list[list]) -> list[str]:
    """One name per column; a None cell in a header row is part of the merged cell to its left."""
    names = [[] for _ in header[0]]
    for row in header:
        last = ""
        for i, cell in enumerate(row):
            cell = _txt(cell).replace("\n", " ") if cell is not None else last
            last = cell
            if cell and (not names[i] or names[i][-1] != cell):
                names[i].append(cell)
    return [" / ".join(n) or f"col{i + 1}" for i, n in enumerate(names)]


def _split_rows(rows: list[list]) -> tuple[list[list[str]], bool]:
    """PyMuPDF packs rows without ruling lines into one row of multi-line cells: split them.
    Returns rows and whether a split was uneven (cells with different line counts)."""
    out, uneven = [], False
    for row in rows:
        lines = [_txt(c).split("\n") if c else [] for c in row]
        k = max((len(cell) for cell in lines), default=1)
        if k <= 1:
            out.append([cell[0] if cell else "" for cell in lines])
            continue
        for i, cell in enumerate(lines):
            if len(cell) != k:
                # one value spanning the rows ("Tyson"), or a wrapped one ("RICH" "MOND")
                uneven |= len(cell) > 1
                lines[i] = [" ".join(cell)] * k
        out += [[cell[j] for cell in lines] for j in range(k)]
    # A merged first-column cell (dataset name) applies to the rows below it
    for prev, row in zip(out, out[1:]):
        if row and not row[0] and any(row[1:]):
            row[0] = prev[0]
    return out, uneven


def _table(raw: list[list], num: str, pno: int, caption: str, rect, method: str) -> dict:
    h = _header_rows(raw)
    body, uneven = _split_rows(raw[h:])
    return {"table_no": num, "page": pno, "caption": caption, "columns": _column_names(raw[:h]),
            "rows": [r for r in body if any(r)], "bbox": list(rect), "method": method,
            "suspicious": int(uneven or method == "text")}


def extract_tables(doc) -> list[dict]:
    """Ruled tables (PyMuPDF line strategy) matched to their caption; captions with no ruled table
    get a text-strategy table; ruled tables without a caption are kept only if they hold numbers."""
    tables, unnamed = [], 0
    for pno, page in enumerate(doc, start=1):
        caps = _captions(page, TABLE_CAP)
        used = set()
        for t in page.find_tables().tables:
            rect, raw = pymupdf.Rect(t.bbox), t.extract()
            if len(raw) < 2 or t.col_count < 2:
                continue
            near = [(g, num, cap) for crect, num, cap in caps if (g := _near(rect, crect)) is not None]
            if near:
                _, num, caption = min(near)
                used.add(num)
            elif len(raw) >= 3 and _numeric_share(raw[1:]) >= 0.3 and _filled_share(raw) >= 0.6:
                unnamed += 1  # caption not found as its own text block, but clearly a data table
                num, caption = f"?{unnamed}", f"Unlabelled table on page {pno}"
            else:
                continue  # a formula or a plot boxed by the line detector
            tables.append(_table(raw, num, pno, caption, rect, "lines"))
        for crect, num, caption in caps:
            if num in used or any(t["table_no"] == num for t in tables):
                continue
            rect, raw = _text_table(page, crect)
            if rect is not None:
                tables.append(_table(raw, num, pno, caption, rect, "text"))
    return tables


def table_paragraphs(tables: list[dict]) -> list[dict]:
    """One paragraph per row with its column names, so any chunk of the table reads on its own."""
    paragraphs = []
    for t in tables:
        head = f"Table {t['table_no']}"
        short = t["caption"][:150]
        lines = [f"{head} (page {t['page']}): {t['caption']}"]
        for row in t["rows"]:
            pairs = [f"{c}: {v}" for c, v in zip(t["columns"], row) if v]
            lines.append(f"{head} | " + " | ".join(pairs))
        text = "\n".join(lines)
        paragraphs.append({"section": "tables", "heading": f"{head}: {short}", "text": text})
    return paragraphs


# ---------------------------------------------------------------- figures
def figure_kind(caption: str) -> str:
    for kind, rx in FIGURE_KINDS:
        if re.search(rx, caption, re.I):
            return kind
    return "other"


def _figure_region(page, caption_rect):
    """Union of drawing clusters and images just above (else below) the caption."""
    graphics = [pymupdf.Rect(r) for r in page.cluster_drawings()]
    graphics += [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    graphics = [g for g in graphics if g.width * g.height > 400]
    for above in (True, False):
        side = [g for g in graphics if (g.y1 <= caption_rect.y0 + 2 if above else g.y0 >= caption_rect.y1 - 2)
                and _near(g, caption_rect) is not None]
        if not side:
            continue
        side.sort(key=lambda g: -g.y1 if above else g.y0)
        region = pymupdf.Rect(side[0])
        for g in side[1:]:  # grow while the next graphic touches the region
            gap = region.y0 - g.y1 if above else g.y0 - region.y1
            if gap < 30:
                region |= g
        return region
    return None


def extract_figures(doc, paper_id: str, render: bool = True) -> list[dict]:
    out_dir = config.FIGURE_DIR / paper_id.replace("/", "_")
    figures, seen = [], set()
    for pno, page in enumerate(doc, start=1):
        for crect, num, caption in _captions(page, FIG_CAP):
            if num in seen:
                continue  # "Figure 3" mentioned again at a block start in the running text
            region = _figure_region(page, crect)
            if region is None:
                continue
            seen.add(num)
            path = out_dir / f"fig{num}.png"
            if render and not path.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                clip = (region | crect) + (-4, -4, 4, 4)  # caption included: the image explains itself
                page.get_pixmap(clip=clip & page.rect, dpi=FIGURE_DPI).save(str(path))
            figures.append({"fig_no": num, "page": pno, "caption": caption, "kind": figure_kind(caption),
                            "image_path": str(path), "bbox": list(region)})
    return figures


FIG_REF = re.compile(r"\bfig(?:ure|s?\.)?s?\s*(\d+)", re.I)


def figure_refs(text: str) -> set[str]:
    """Figure numbers a passage refers to ("Figure 2", "Fig. 3", "Figs. 2")."""
    return set(FIG_REF.findall(text))
