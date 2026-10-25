"""Ingest downloaded PDFs: parse into paragraphs, build passages (reading units) and small chunks
(search units), embed chunks into Chroma, index keywords in SQLite FTS5."""
import json
import re
import sys
from pathlib import Path

from . import config, dedup, grobid
from .db import connect
from .models import (approach_collection, chunk_collection, embed_chunks, embed_papers,
                     paper_collection)
from .objects import figure_refs
from .parse import coverage, parse_pdf


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _split_long(text: str, size: int) -> list[str]:
    """Split one over-long paragraph into ~size-word pieces at sentence boundaries."""
    pieces, cur, n = [], [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        w = len(sent.split())
        if cur and n + w > size:
            pieces.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def pack(paragraphs: list[str], size: int) -> list[str]:
    """Group consecutive paragraphs into units of at most ~size words, never cutting a paragraph
    unless the paragraph alone is longer than 1.5 x size."""
    units, cur, n = [], [], 0
    for para in paragraphs:
        parts = _split_long(para, size) if len(para.split()) > 1.5 * size else [para]
        for part in parts:
            w = len(part.split())
            if cur and n + w > size:
                units.append("\n\n".join(cur))
                cur, n = [], 0
            cur.append(part)
            n += w
    if cur:
        units.append("\n\n".join(cur))
    return units


def build_passages(paragraphs: list[dict]) -> list[dict]:
    """Consecutive paragraphs with the same section + heading form a subsection -> one or more passages,
    each with its small search chunks."""
    groups: list[dict] = []
    for p in paragraphs:
        if groups and groups[-1]["section"] == p["section"] and groups[-1]["heading"] == p["heading"]:
            groups[-1]["paras"].append(p["text"])
        else:
            groups.append({"section": p["section"], "heading": p["heading"], "paras": [p["text"]]})
    passages = []
    for g in groups:
        for text in pack(g["paras"], config.PASSAGE_WORDS):
            passages.append({"section": g["section"], "heading": g["heading"], "text": text,
                             "chunks": pack(text.split("\n\n"), config.CHUNK_WORDS)})
    return passages


def remove_from_index(conn, pid: str, keep_approach: bool = False):
    """keep_approach: re-parsing a paper keeps its card, so its approach vector stays valid."""
    chunk_collection().delete(where={"paper_id": pid})
    paper_collection().delete(ids=[pid])
    if not keep_approach:
        approach_collection().delete(ids=[pid])
    for table in ("chunks", "chunks_fts", "passages", "pages", "pages_fts", "paper_tables", "figures", "figure_refs"):
        conn.execute(f"DELETE FROM {table} WHERE paper_id=?", (pid,))


def index_paper(conn, paper: dict, parsed: dict) -> int:
    pid = paper["paper_id"]
    remove_from_index(conn, pid, keep_approach=True)
    conn.execute("DELETE FROM sections WHERE paper_id=?", (pid,))
    for name, text in parsed["sections"].items():
        conn.execute("INSERT INTO sections VALUES (?,?,?)", (pid, name, text))
    for page_no, text in enumerate(parsed.get("pages", []), start=1):
        conn.execute("INSERT INTO pages VALUES (?,?,?)", (pid, page_no, text))
        conn.execute("INSERT INTO pages_fts (text, paper_id, page) VALUES (?,?,?)", (text, pid, page_no))
    for t in parsed.get("tables", []):
        conn.execute("INSERT INTO paper_tables VALUES (?,?,?,?,?,?,?,?)",
                     (pid, t["table_no"], t["page"], t["caption"], json.dumps(t["columns"], ensure_ascii=False),
                      json.dumps(t["rows"], ensure_ascii=False), t["method"], t["suspicious"]))
    for f in parsed.get("figures", []):
        conn.execute("INSERT INTO figures VALUES (?,?,?,?,?,?)",
                     (pid, f["fig_no"], f["page"], f["caption"], f["kind"], f["image_path"]))
    if parsed.get("pages"):
        conn.execute("UPDATE papers SET coverage=? WHERE paper_id=?",
                     (round(coverage(parsed["paragraphs"], parsed["pages"]), 4), pid))

    ids, embed_texts, metas = [], [], []
    for pos, ps in enumerate(build_passages(parsed["paragraphs"])):
        psg_id = f"{pid}::{pos}"
        conn.execute("INSERT INTO passages VALUES (?,?,?,?,?,?)",
                     (psg_id, pid, ps["section"], ps["heading"], pos, ps["text"]))
        if ps["section"] not in ("figures", "references"):  # a caption is not a reference to its figure
            for fig_no in figure_refs(ps["text"]):
                conn.execute("INSERT OR IGNORE INTO figure_refs VALUES (?,?,?)", (psg_id, pid, fig_no))
        for j, chunk in enumerate(ps["chunks"]):
            cid = f"{psg_id}::{j}"
            ids.append(cid)
            embed_texts.append(f"{paper['title']} | {ps['section']} | {ps['heading']}: {chunk}")
            metas.append({
                "paper_id": pid, "passage_id": psg_id, "section": ps["section"],
                "year": paper["year"] or 0, "citation_count": paper["citation_count"] or 0,
            })
            conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                         (cid, pid, psg_id, ps["section"], ps["heading"], chunk))
            conn.execute("INSERT INTO chunks_fts (text, heading, section, paper_id, chunk_id) VALUES (?,?,?,?,?)",
                         (chunk, ps["heading"], ps["section"], pid, cid))
    if ids:
        chunk_collection().add(ids=ids, embeddings=embed_chunks(embed_texts).tolist(), metadatas=metas)

    abstract = parsed["sections"].get("abstract") or paper.get("abstract") or ""
    vec = embed_papers([f"{paper['title']}[SEP]{abstract}"])[0]
    paper_collection().add(ids=[pid], embeddings=[vec.tolist()],
                           metadatas=[{"title": paper["title"], "year": paper["year"] or 0}])
    return len(ids)


def ingest(limit: int | None = None, reparse: bool = False) -> dict:
    conn = connect()
    merged = dedup.run(conn)
    status = "('downloaded','ingested')" if reparse else "('downloaded')"
    papers = [dict(r) for r in conn.execute(
        f"SELECT * FROM papers WHERE status IN {status} AND pdf_path IS NOT NULL ORDER BY added_at")]
    if limit:
        papers = papers[:limit]
    if not papers:
        return {"ingested": 0, "failed": 0, "chunks": 0, "duplicates_merged": merged}
    if all((config.TEI_DIR / f"{Path(p['pdf_path']).stem}.tei.xml").exists() for p in papers):
        done, failed, n_chunks = _ingest_papers(conn, papers)  # GROBID output already saved
    else:
        with grobid.session():
            done, failed, n_chunks = _ingest_papers(conn, papers)
    return {"ingested": done, "failed": failed, "chunks": n_chunks, "duplicates_merged": merged}


def _ingest_papers(conn, papers: list[dict]) -> tuple[int, int, int]:
    done = failed = n_chunks = 0
    for i, p in enumerate(papers):
        log(f"[{i + 1}/{len(papers)}] {p['paper_id']} {p['title'][:70]}")
        try:
            parsed = parse_pdf(p["pdf_path"])
        except Exception as e:
            log(f"  parse failed: {e}")
            conn.execute("UPDATE papers SET status='failed' WHERE paper_id=?", (p["paper_id"],))
            conn.commit()
            failed += 1
            continue
        if p.get("abstract"):
            # arXiv abstract is always clean; replace whatever the PDF parser found
            parsed["paragraphs"] = [x for x in parsed["paragraphs"] if x["section"] != "abstract"]
            parsed["paragraphs"].insert(0, {"section": "abstract", "heading": "Abstract", "text": p["abstract"]})
            parsed["sections"]["abstract"] = p["abstract"]
        n_chunks += index_paper(conn, p, parsed)
        conn.execute("UPDATE papers SET status='ingested', code_url=COALESCE(code_url, ?) WHERE paper_id=?",
                     (parsed["code_url"], p["paper_id"]))
        conn.commit()
        cov = conn.execute("SELECT coverage FROM papers WHERE paper_id=?", (p["paper_id"],)).fetchone()[0] or 0
        if cov < MIN_COVERAGE:
            log(f"  WARNING: only {cov:.1%} of the PDF text is in the structured index (see `audit {p['paper_id']}`)")
        done += 1
    return done, failed, n_chunks


MIN_COVERAGE = 0.97


def audit(paper_id: str | None = None) -> list[dict]:
    """Per paper: share of the PDF text in the structured index, tables that need checking,
    figures nobody refers to, and references to figures that were not found."""
    conn = connect()
    sql = "SELECT paper_id, title, coverage FROM papers WHERE status='ingested'"
    papers = conn.execute(sql + (" AND paper_id=?" if paper_id else "") + " ORDER BY coverage",
                          (paper_id,) if paper_id else ()).fetchall()
    out = []
    for p in papers:
        pid = p["paper_id"]
        figs = {r[0] for r in conn.execute("SELECT fig_no FROM figures WHERE paper_id=?", (pid,))}
        refs = {r[0] for r in conn.execute("SELECT DISTINCT fig_no FROM figure_refs WHERE paper_id=?", (pid,))}
        out.append({
            "paper_id": pid, "title": p["title"], "coverage": p["coverage"],
            "below_target": (p["coverage"] or 0) < MIN_COVERAGE,
            "tables": conn.execute("SELECT COUNT(*) FROM paper_tables WHERE paper_id=?", (pid,)).fetchone()[0],
            "suspicious_tables": [r[0] for r in conn.execute(
                "SELECT table_no FROM paper_tables WHERE paper_id=? AND suspicious=1", (pid,))],
            "figures": len(figs),
            "figures_never_referenced": sorted(figs - refs),
            "referenced_figures_not_found": sorted(refs - figs),
        })
    return out


def stats() -> dict:
    conn = connect()
    by_status = {r[0]: r[1] for r in conn.execute("SELECT status, COUNT(*) FROM papers GROUP BY status")}
    ingested = by_status.get("ingested", 0) or 1
    by_section = {r[0]: r[1] for r in conn.execute(
        "SELECT s.section, COUNT(*) FROM sections s JOIN papers p USING(paper_id) "
        "WHERE p.status='ingested' AND length(s.text) > 200 GROUP BY s.section")}
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "papers_by_status": by_status,
        "papers_with_section_pct": {k: round(100 * v / ingested) for k, v in sorted(by_section.items())},
        "passages": one("SELECT COUNT(*) FROM passages"),
        "chunks": one("SELECT COUNT(*) FROM chunks"),
        "cards": one("SELECT COUNT(*) FROM cards"),
        "result_rows": one("SELECT COUNT(*) FROM results"),
        "unverified_result_rows": one("SELECT COUNT(*) FROM results WHERE verified = 0"),
        "citation_edges": one("SELECT COUNT(*) FROM citations"),
        "with_code": one("SELECT COUNT(*) FROM papers WHERE code_url IS NOT NULL AND status='ingested'"),
        "coverage_mean": one("SELECT ROUND(AVG(coverage), 3) FROM papers WHERE status='ingested'"),
        "coverage_min": one("SELECT ROUND(MIN(coverage), 3) FROM papers WHERE status='ingested'"),
        "tables": one("SELECT COUNT(*) FROM paper_tables"),
        "figures": one("SELECT COUNT(*) FROM figures"),
    }
