"""PDF -> paragraphs tagged with canonical section and subsection heading.

Uses GROBID when a server is running, otherwise PyMuPDF + header heuristics.
Result: {"paragraphs": [{"section", "heading", "text"}], "sections": {section: text}, "code_url", "parser"}
"""
import bisect
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
import requests

from . import config, objects

SECTIONS = ["abstract", "introduction", "related_work", "methods", "experiments", "results",
            "discussion", "limitations", "conclusion"]
# Nothing is dropped: everything outside the main sections gets one of these labels instead
EXTRA_SECTIONS = ["tables", "figures", "appendix", "references", "footnotes", "front", "other"]

# Order matters: first matching rule wins.
RULES = [
    # anchored: a table header like "Category References" must not end the paper
    ("STOP", r"^\W*(reference|bibliograph|acknowledg|appendi|supplementar|checklist)"),
    ("abstract", r"^abstract"),
    ("introduction", r"introduction|motivation|overview$"),
    ("related_work", r"related|background|prior work|previous work|literature"),
    ("limitations", r"limitation|broader impact|societal impact|ethic|threats to validity"),
    ("conclusion", r"conclu|future work|summary|closing remarks"),
    ("discussion", r"discussion"),
    ("experiments", r"experiment|evaluation|empirical|benchmark"),
    ("methods", r"problem (setup|set-up|setting|formulation|definition|statement)|preliminar|notation"),
    ("experiments", r"setup|set-up|implementation|dataset"),
    ("results", r"result|ablation|analysis|comparison|performance|case stud"),
    ("methods", r"method|approach|propos|model|architecture|framework|algorithm|formulation|"
                r"preliminar|design|problem|technique|our |learning|training|network|notation|theory"),
]
_rules = [(name, re.compile(rx, re.I)) for name, rx in RULES]

# "3 Model", "3.1 Encoder", "3.1. Encoder", "1.Introduction", "III. METHOD", "III.METHODOLOGY", "A. Setup"
NUM_HEAD = re.compile(r"^((?P<num>\d{1,2})(?P<sub>(\.\d+)*)(\.\s*|\s+)|(?P<roman>[IVX]{1,4})\.\s*|(?P<letter>[A-H])(\.\s*|\s+))"
                      r"(?P<title>[A-Z][^.]{1,80})$")
NUM_ONLY = re.compile(r"^(\d{1,2}(\.\d+)*\.?|[IVX]{1,4}\.?|[A-H]\.?)$")
PLAIN_HEAD = re.compile(r"^(abstract|introduction|related work|background|method(s|ology)?|approach|experiments?|"
                        r"results|discussion|limitations?|conclusions?|references|acknowledge?ments?|appendix)\b.{0,40}$", re.I)
GITHUB = re.compile(r"https?://github\.com/[\w\-]+/[\w\-.]+", re.I)
LIMIT_SENT = re.compile(r"limitation|shortcoming|fail(s|ed)? to|does not (handle|scale|generalize)|"
                        r"future work|we leave|remains? (an )?open|drawback", re.I)
SENT_END = re.compile(r"[.!?:]\)?$")
# A PDF line counts as lost (and is rescued into "other") when more than this share of its words is missing
RESCUE_MISSING = 0.2


def back_section(title: str) -> str:
    """Section for text after a STOP header: the reference list, acknowledgements, or appendix."""
    t = title.strip().lower()
    if re.search(r"reference|bibliograph", t):
        return "references"
    if re.search(r"acknowledg", t):
        return "front"
    return "appendix"


def classify(title: str) -> str | None:
    t = title.strip().lower()
    for name, rx in _rules:
        if rx.search(t):
            return name
    return None


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class Assigner:
    """Tracks the canonical section and the human-readable heading of the running text."""

    def __init__(self, started: bool = False):
        self.current = None        # canonical section of current top-level header
        self.top_heading = ""
        self.heading = ""
        self.after_refs = False
        self.seen_experiments = False
        # Before the abstract/introduction, "headers" are title-page noise (e.g. "1 School of ...")
        self.started = started

    def header(self, title: str, top_level: bool, label: str = "") -> None:
        label = label or title
        kind = classify(title)
        if not self.started:
            if kind not in ("abstract", "introduction"):
                return
            self.started = True
        if kind == "STOP" or self.after_refs:
            # Back matter is kept, labelled: references, acknowledgements (front), appendix
            self.after_refs = True
            self.current = (back_section(title) if kind == "STOP" else
                            "limitations" if kind == "limitations" else "appendix")
            self.top_heading = self.heading = label
            return
        if not top_level:
            if kind == "limitations":
                self.current = "limitations"
            self.heading = f"{self.top_heading} > {label}" if self.top_heading else label
            return
        if kind == "methods" and self.seen_experiments:
            return  # e.g. a "Method" table header inside the experiments; papers don't go back
        if kind is None:
            kind = "results" if self.seen_experiments else "methods"
        if kind in ("experiments", "results"):
            self.seen_experiments = True
        self.current = kind
        self.top_heading = self.heading = label


def merge_broken(paragraphs: list[dict]) -> list[dict]:
    """Join paragraphs split by a column or page break (previous one ends mid-sentence), and attach
    fragments (formula lines, labels, axis ticks) to the paragraph before them - never drop text."""
    out: list[dict] = []
    for p in paragraphs:
        if not p["text"]:
            continue
        prev = out[-1] if out else None
        same = prev and prev["section"] == p["section"] and prev["heading"] == p["heading"]
        fragment = len(p["text"].split()) < 8 and not SENT_END.search(p["text"])
        if same and (fragment or (not SENT_END.search(prev["text"]) and p["text"][:1].islower())):
            prev["text"] += " " + p["text"]
        else:
            out.append(dict(p))
    return out


def finish(paragraphs: list[dict], code_url: str | None, parser: str) -> dict:
    paragraphs = merge_broken(paragraphs)
    sections: dict[str, list[str]] = {}
    for p in paragraphs:
        sections.setdefault(p["section"], []).append(p["text"])
    if "limitations" not in sections:
        found = []
        for p in paragraphs:
            if p["section"] in ("discussion", "conclusion", "results"):
                found += [s for s in re.split(r"(?<=[.!?])\s+", p["text"])
                          if LIMIT_SENT.search(s) and 30 < len(s) < 600]
        if found:
            text = " ".join(found)
            paragraphs.append({"section": "limitations", "heading": "Limitations (extracted sentences)", "text": text})
            sections["limitations"] = [text]
    out = {k: "\n\n".join(v) for k, v in sections.items()}
    return {"paragraphs": paragraphs, "sections": out, "code_url": code_url, "parser": parser}


# ---------------------------------------------------------------- PyMuPDF
SPACE_GAP = 0.15  # gap between glyphs (in font sizes) that counts as a word space


def _line_text(spans) -> str:
    """Rebuild a line from glyph positions. Some PDFs (tight LaTeX justification) store no space
    characters at all, so a space is inserted wherever two glyphs are visibly apart."""
    out, prev_x1 = [], None
    for s in spans:
        for ch in s["chars"]:
            c, (x0, _, x1, _) = ch["c"], ch["bbox"]
            if prev_x1 is not None and c != " " and out and out[-1] != " " and x0 - prev_x1 > SPACE_GAP * s["size"]:
                out.append(" ")
            out.append(c)
            prev_x1 = x1
    return unicodedata.normalize("NFKC", "".join(out)).strip()  # NFKC also expands ligatures like "ï¬"


def _page_blocks(doc) -> list[list[list[tuple]]]:
    """Per page, its text blocks; each block = [(text, size, bold)] lines."""
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    pages = []
    for page in doc:
        blocks = []
        for block in page.get_text("rawdict", flags=flags)["blocks"]:
            lines = []
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if "".join(c["c"] for c in s["chars"]).strip()]
                if not spans:
                    continue
                text = _line_text(spans)
                size = round(max(s["size"] for s in spans), 1)
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                lines.append((text, size, bold))
            if lines:
                blocks.append(lines)
        pages.append(blocks)
    return pages


def _blocks(doc):
    """[(lines)] per text block, all pages; each line = (text, size, bold)."""
    return [b for page in _page_blocks(doc) for b in page]


def _merge_number_lines(block):
    """'3' + 'Proposed Model' on separate lines (NeurIPS/AAAI layout) -> '3 Proposed Model'."""
    out, i = [], 0
    while i < len(block):
        text, size, bold = block[i]
        if NUM_ONLY.match(text) and i + 1 < len(block) and block[i + 1][0][:1].isupper():
            nxt = block[i + 1]
            out.append((f"{text} {nxt[0]}", max(size, nxt[1]), bold and nxt[2]))
            i += 2
            continue
        out.append(block[i])
        i += 1
    return out


def _numbered_header(text: str, emphasized: bool, alone: bool):
    """Return (title, top_level) if the line is a numbered section header, else None."""
    m = NUM_HEAD.match(text)
    if not m or len(text) > 90 or text.endswith((",", ";", ":")):
        return None
    if m.group("num") and int(m.group("num")) > 15:
        return None
    title = m.group("title")
    if len(title.split()) > 12 or len(re.findall(r"[A-Za-z]", title)) < 4:
        return None
    # Bold/large, ALL CAPS (IEEE small caps), or a short line standing alone in its block (Elsevier)
    if not (emphasized or title.isupper() or (alone and len(text) < 60)):
        return None
    if m.group("letter") and not emphasized:
        return None  # "A. Smith ..." in running text
    top = bool((m.group("num") and not m.group("sub")) or m.group("roman"))
    if m.group("num"):
        number = int(m.group("num"))
    elif m.group("roman"):
        number = ROMAN.get(m.group("roman"))
        if number is None:
            return None
    else:
        number = None  # lettered (appendix-style) subsection
    return title, top, number


ROMAN = {r: i for i, r in enumerate(["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                                     "XI", "XII", "XIII", "XIV", "XV"], start=1)}


def parse_pymupdf(path: str) -> dict:
    doc = pymupdf.open(path)
    page_blocks = _page_blocks(doc)
    # Full-text layer: every line of every page, nothing filtered
    block_lines = [[[t for t, _, _ in b] for b in page] for page in page_blocks]
    blocks = [_merge_number_lines(b) for page in page_blocks for b in page]
    all_lines = [l for b in blocks for l in b]
    sizes = Counter()
    for text, size, _ in all_lines:
        sizes[size] += len(text)
    body = sizes.most_common(1)[0][0] if sizes else 10
    # Running headers/footers: identical short lines on many pages
    repeats = Counter(t for t, _, _ in all_lines if len(t) < 80)
    n_pages = len(doc)

    paragraphs: list[dict] = []
    a = Assigner()
    last_top = 0  # section numbers must increase (1, 2, 3 ...); "3.x" only under section 3
    for block in blocks:
        buf: list[str] = []

        def flush():
            if buf:  # text before the abstract (title, authors, affiliations) is "front"
                paragraphs.append({"section": a.current or "front", "heading": a.heading,
                                   "text": clean("\n".join(buf))})
            buf.clear()

        for text, size, bold in block:
            if text.isdigit() or (n_pages > 3 and repeats[text] > n_pages / 2):
                continue
            emphasized = bold or size > body + 0.4
            head = _numbered_header(text, emphasized, alone=len(block) == 1)
            if head:
                title, top, number = head
                if top and not (last_top < number <= last_top + 3):
                    head = None  # formula/figure label like "2 STAST D", or a list item
                elif not top and number is not None and number != last_top:
                    head = None
            if head:
                flush()
                a.header(title, top, label=text)
                if top and a.started:  # title-page noise before the introduction doesn't count
                    last_top = number
                continue
            if PLAIN_HEAD.match(text) and (emphasized or text.isupper()) and len(text) < 60:
                flush()
                a.header(text, True, label=text.title() if text.isupper() else text)
                continue
            if re.match(r"^abstract\W", text, re.I) and a.current is None and not a.after_refs:
                a.current, a.heading, a.started = "abstract", "Abstract", True
                text = re.sub(r"^abstract\W+", "", text, flags=re.I)
            buf.append(text)
        flush()

    full_text = "\n".join(t for t, _, _ in all_lines)
    pre_refs = re.split(r"\n(references|bibliography)\n", full_text, flags=re.I)[0]
    code = GITHUB.search(pre_refs)
    result = finish(paragraphs, code.group(0).rstrip(".") if code else None, "pymupdf")
    result["page_blocks"] = block_lines
    return result


def parse_plain(path: str) -> dict:
    """Last resort when no headers are found: every text block is a paragraph of section 'body'."""
    paragraphs = [{"section": "body", "heading": "", "text": clean("\n".join(t for t, _, _ in b))}
                  for b in _blocks(pymupdf.open(path))]
    return finish(paragraphs, None, "plain")


# ---------------------------------------------------------------- GROBID
TEI = {"t": "http://www.tei-c.org/ns/1.0"}


def grobid_alive() -> bool:
    try:
        return requests.get(f"{config.GROBID_URL}/api/isalive", timeout=2).ok
    except requests.RequestException:
        return False


SUB_HEAD = re.compile(r"^([A-H]\.|\d{1,2}\))\s*")   # IEEE "A. Setup", "1) Baselines"
ROMAN_HEAD = re.compile(r"^[IVX]{1,4}\.\s*")          # IEEE "III. METHODOLOGY"
APPENDIX_HEAD = re.compile(r"^[A-H](\.\d+)*\s+\S")      # "A RELATION ...", "A.1 NODE ..." (not IEEE "A. X")


def _grobid_level(text: str, n: str) -> tuple[str, bool, str]:
    """(title, top_level, label) for a GROBID head. GROBID gives n="3.1" for most papers but leaves IEEE
    numbering inside the text; an unnumbered head that is no known section name ("Model", "Accuracy",
    stray formula text) is a subsection or a table header, never a new top-level section."""
    if re.match(r"^[A-H](\.|$)", n):
        return "Appendix", True, f"{n} {text}"  # GROBID numbers appendix sections A, A.1, B
    if n:
        return text, "." not in n, f"{n} {text}"
    if ROMAN_HEAD.match(text):
        return ROMAN_HEAD.sub("", text), True, text
    if SUB_HEAD.match(text):
        return SUB_HEAD.sub("", text), False, text
    return text, classify(text) is not None, text


def parse_grobid(path: str) -> dict:
    from lxml import etree

    # GROBID output is kept next to the PDFs, so a later --reparse needs no GROBID server
    tei = config.TEI_DIR / (Path(path).stem + ".tei.xml")
    if not tei.exists():
        with open(path, "rb") as f:
            r = requests.post(f"{config.GROBID_URL}/api/processFulltextDocument",
                              files={"input": f}, data={"consolidateHeader": "0"}, timeout=180)
        r.raise_for_status()
        tei.parent.mkdir(parents=True, exist_ok=True)
        tei.write_bytes(r.content)
    root = etree.fromstring(tei.read_bytes())
    x = lambda el, path: el.xpath(path, namespaces=TEI)  # noqa: E731
    paragraphs: list[dict] = []

    # Title page: title, authors, affiliations, keywords
    front = [_text(t) for t in x(root, "//t:teiHeader//t:titleStmt/t:title")]
    front += [_text(a) for a in x(root, "//t:teiHeader//t:sourceDesc//t:author")]
    front += [_text(k) for k in x(root, "//t:profileDesc/t:textClass//t:term")]
    if any(front):
        paragraphs.append({"section": "front", "heading": "Title page", "text": "\n".join(f for f in front if f)})
    for p in x(root, "//t:profileDesc/t:abstract//t:p"):
        paragraphs.append({"section": "abstract", "heading": "Abstract", "text": _text(p)})

    a = Assigner(started=True)  # GROBID already removed the title page
    # back/div/div: GROBID sometimes nests a real section inside <div type="acknowledgement">
    for div in x(root, "//t:text/t:body/t:div | //t:text/t:back/t:div | //t:text/t:back/t:div/t:div"):
        head = div.find("t:head", TEI)
        back = next((el for el in (div.getparent(), div.getparent().getparent()) if el.tag.endswith("back")), None)
        if back is not None:  # a headless back div must not continue the conclusion
            outer = div if div.getparent() is back else div.getparent()
            a.after_refs, a.current = True, BACK_TYPES.get(outer.get("type"), "appendix")
            a.top_heading = a.heading = outer.get("type") or "Appendix"
        if head is not None and head.text:
            text, n = head.text.strip(), (head.get("n") or "").strip(".")
            if not n and a.current == "conclusion" and APPENDIX_HEAD.match(text):
                n = text.split()[0]  # "A RELATION TO ...", "A.1 NODE ..." after the conclusion
            a.header(*_grobid_level(text, n))
        elif a.current is None and not a.after_refs and not a.top_heading:
            a.header("Introduction", True)  # GROBID often drops the first heading of the body
        for el in div:
            tag = etree.QName(el).localname
            if tag == "p":
                text = _text(el, "")
            elif tag == "formula":
                label = el.find("t:label", TEI)
                body = _text(el, "")
                text = f"[Eq. {_text(label)}] {body}" if label is not None else f"[Eq.] {body}"
            elif tag == "list":
                text = "\n".join(_text(i, "") for i in el.findall("t:item", TEI))
            else:
                continue  # head, nested div (visited by the xpath above)
            paragraphs.append({"section": a.current or "other", "heading": a.heading, "text": text})

    for fig in x(root, "//t:figure[@type='table']"):
        head = clean(" ".join(x(fig, "t:head//text() | t:figDesc//text()")))
        rows = [" | ".join(" ".join(c.itertext()).strip() for c in x(row, "t:cell")) for row in x(fig, ".//t:row")]
        if rows or head:
            paragraphs.append({"section": "tables", "heading": head[:120], "text": head + "\n" + "\n".join(rows)})
    for fig in x(root, "//t:figure[not(@type='table')]"):
        label = fig.find("t:label", TEI)
        name = f"Figure {_text(label)}" if label is not None else _text(fig.find("t:head", TEI)) or "Figure"
        text = _text(fig.find("t:figDesc", TEI), "")
        if text:
            text = text if LABEL_NO.match(text) else f"{name}: {text}"  # figDesc often repeats the label
            paragraphs.append({"section": "figures", "heading": name, "text": text})
    for note in x(root, "//t:note[@place='foot']"):
        paragraphs.append({"section": "footnotes", "heading": "Footnotes", "text": _text(note, "")})
    for ref in x(root, "//t:back//t:listBibl/t:biblStruct"):
        paragraphs.append({"section": "references", "heading": "References", "text": _text(ref)})

    full = " ".join(x(root, "//t:text/t:body//text()"))
    code = GITHUB.search(full)
    return finish(paragraphs, code.group(0).rstrip(".") if code else None, "grobid")


BACK_TYPES = {"acknowledgement": "front", "funding": "front", "availability": "front",
              "annex": "appendix", "references": "references"}


def _text(el, sep: str = " ") -> str:
    """All text inside an XML element. sep=" " for field lists (authors, bibliography entries, where
    "".join would glue "Smith" "2019"); sep="" for running text, which carries its own spacing."""
    if el is None:
        return ""
    return clean(sep.join(el.itertext()) if not sep else sep.join(t.strip() for t in el.itertext() if t.strip()))


_grobid = None


def reset_grobid():
    """Forget the cached GROBID availability (after starting or stopping the server)."""
    global _grobid
    _grobid = None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fill_missing_sections(g: dict, m: dict) -> dict:
    """GROBID sometimes drops a heading (often "Related Work"), so its text lands in the previous
    section, or a conclusion gets tagged as acknowledgement. Where PyMuPDF found a section GROBID
    lacks, relabel those GROBID paragraphs from PyMuPDF's labels, located by matching paragraph text."""
    missing = {k for k, v in m["sections"].items()
               if k not in g["sections"] and (k in SECTIONS[1:] or k == "appendix") and k != "limitations"
               and len(v.split()) > 50}
    if not missing:
        return g
    stream, starts = "", []
    for p in m["paragraphs"]:
        starts.append(len(stream))
        stream += _norm(p["text"])
    def pymupdf_label(p: dict) -> dict | None:
        key = _norm(p["text"])[:60]
        if len(key) >= 30 and (i := stream.find(key)) >= 0:
            src = m["paragraphs"][bisect.bisect_right(starts, i) - 1]
            if src["section"] in missing:
                return {**p, "section": src["section"], "heading": src["heading"]}
        return None

    paragraphs = []
    for p in g["paragraphs"]:
        if p["heading"] == "Limitations (extracted sentences)":
            continue  # finish() re-extracts it
        if p["section"] in SECTIONS[1:] or p["section"] in ("appendix", "other"):
            p = pymupdf_label(p) or p
        paragraphs.append(p)
    return finish(paragraphs, g["code_url"], "grobid+pymupdf")


def parse_pdf(path: str) -> dict:
    global _grobid
    if _grobid is None:
        _grobid = grobid_alive()
    result = None
    if _grobid or (config.TEI_DIR / (Path(path).stem + ".tei.xml")).exists():
        try:
            result = parse_grobid(path)
        except Exception:
            result = None
    fallback = parse_pymupdf(path)
    if result and result["sections"].get("methods"):
        result = _fill_missing_sections(result, fallback)
    elif not result or fallback["sections"].get("methods"):
        result = fallback
    if not any(k in result["sections"] for k in SECTIONS[1:]):
        result = parse_plain(path)  # header detection failed: keep the text searchable
    if result is not fallback and "references" in fallback["sections"]:
        # GROBID splits references into fields and loses some (months, pages); keep the printed text
        result = _replace(result, "references", [p for p in fallback["paragraphs"] if p["section"] == "references"])
    result = _add_objects(result, path)
    result = _rescue_missing(result, fallback["page_blocks"])
    result["pages"] = ["\n".join(clean("\n".join(lines)) for lines in blocks) for blocks in fallback["page_blocks"]]
    return result


def _replace(result: dict, section: str, paragraphs: list[dict]) -> dict:
    """Swap one section's paragraphs (the derived limitation sentences are rebuilt by finish)."""
    kept = [p for p in result["paragraphs"]
            if p["section"] != section and p["heading"] != "Limitations (extracted sentences)"]
    return {**result, **finish(kept + paragraphs, result["code_url"], result["parser"])}


LABEL_NO = re.compile(r"^\s*(?:table|tab\.|tabela|tabla|figure|fig\.|figura)\s*([IVX]+|\d+|[A-Z]\d+)", re.I)


def _add_objects(result: dict, path: str) -> dict:
    """Tables and figures from the PDF layout. PyMuPDF tables replace GROBID's text tables of the same
    number (GROBID's are kept for numbers PyMuPDF missed); GROBID's figure captions are cleaner, so
    they are used for the figure records when both found the same figure."""
    doc = pymupdf.open(path)
    tables = objects.extract_tables(doc)
    figures = objects.extract_figures(doc, Path(path).stem)

    def number(p):
        m = LABEL_NO.match(p["heading"])
        return m.group(1) if m else None

    have = {t["table_no"] for t in tables}
    grobid_tables = [p for p in result["paragraphs"] if p["section"] == "tables" and number(p) not in have]
    result = _replace(result, "tables", grobid_tables + objects.table_paragraphs(tables))

    captions = {number(p): p["text"] for p in result["paragraphs"] if p["section"] == "figures"}
    for f in figures:
        f["caption"] = captions.get(f["fig_no"]) or f["caption"]
    extra = [{"section": "figures", "heading": f"Figure {f['fig_no']}", "text": f["caption"]}
             for f in figures if f["fig_no"] not in captions]
    if extra:
        result = _replace(result, "_none", extra)
    result["tables"], result["figures"] = tables, figures
    return result


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", unicodedata.normalize("NFKC", text).lower())


def coverage(paragraphs: list[dict], pages: list[str]) -> float:
    """Share of the PDF's words (pages layer) that are present in the structured paragraphs."""
    stored = Counter(_words(" ".join(p["text"] for p in paragraphs)))
    full = Counter(_words("\n".join(pages)))
    n = sum(full.values())
    return sum(min(c, stored[k]) for k, c in full.items()) / n if n else 1.0


def _rescue_missing(result: dict, page_blocks: list[list[list[str]]]) -> dict:
    """Safety net for zero loss: every PDF line whose words mostly did not make it into the structured
    paragraphs (text inside figures, references GROBID could not parse, odd layouts) is added as
    section "other"; consecutive missing lines of a block stay together, tagged with their page."""
    available = Counter(_words(" ".join(p["text"] for p in result["paragraphs"])))
    rescued = []

    def missing_line(line: str) -> bool:
        words = _words(line)
        missing = 0
        for w in words:
            if available[w] > 0:
                available[w] -= 1
            else:
                missing += 1
        return bool(words) and missing / len(words) > RESCUE_MISSING

    for page_no, blocks in enumerate(page_blocks, start=1):
        for lines in blocks:
            run: list[str] = []
            for line in lines + [""]:  # "" flushes the last run
                if line and missing_line(line):
                    run.append(line)
                elif run:
                    rescued.append({"section": "other", "heading": f"Page {page_no} (text outside the parsed structure)",
                                    "text": clean("\n".join(run))})
                    run = []
    if not rescued:
        return result
    out = finish([p for p in result["paragraphs"] if p["heading"] != "Limitations (extracted sentences)"]
                 + rescued, result["code_url"], result["parser"])
    return {**result, **out}
