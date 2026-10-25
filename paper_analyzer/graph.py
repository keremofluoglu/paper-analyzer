"""Citation graph queries and method trends."""
import re

from .db import connect, rows
from .fetch import s2_citations
from .search import topic_paper_ids


def foundational_papers(topic: str = "", k: int = 15) -> list[dict]:
    """Papers most cited by the corpus (or by the topic's papers). in_corpus=False means not downloaded yet."""
    conn = connect()
    sql = ("SELECT c.cited_s2, COUNT(*) AS cited_by_corpus FROM citations c "
           "JOIN papers p ON p.s2_id = c.citing_s2 WHERE p.status = 'ingested'")
    params: list = []
    if topic:
        ids = topic_paper_ids(topic)
        if not ids:
            return []
        sql += f" AND p.paper_id IN ({','.join('?' * len(ids))})"
        params += ids
    sql += " GROUP BY c.cited_s2 ORDER BY cited_by_corpus DESC LIMIT ?"
    params.append(k)
    out = []
    for r in conn.execute(sql, params):
        meta = conn.execute("SELECT * FROM s2_papers WHERE s2_id=?", (r["cited_s2"],)).fetchone()
        local = conn.execute("SELECT paper_id FROM papers WHERE s2_id=? AND status='ingested'",
                             (r["cited_s2"],)).fetchone()
        out.append({
            "title": meta["title"] if meta else None, "year": meta["year"] if meta else None,
            "arxiv_id": meta["arxiv_id"] if meta else None,
            "total_citations": meta["citation_count"] if meta else None,
            "cited_by_corpus": r["cited_by_corpus"], "in_corpus": bool(local),
        })
    return out


def follow_ups(paper_id: str, external: bool = False, k: int = 20) -> list[dict]:
    """Newer papers that cite this one. external=True also asks Semantic Scholar (outside the corpus)."""
    conn = connect()
    p = conn.execute("SELECT s2_id FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if not p or not p["s2_id"]:
        return []
    out = rows(conn, """SELECT p.paper_id, p.title, p.year, p.citation_count, 1 AS in_corpus
                        FROM citations c JOIN papers p ON p.s2_id = c.citing_s2
                        WHERE c.cited_s2 = ? AND p.status='ingested'
                        ORDER BY p.year DESC, p.citation_count DESC""", (p["s2_id"],))
    if external:
        have = {r["title"] for r in out}
        for c in s2_citations(p["s2_id"], limit=200):
            if c.get("title") not in have:
                out.append({"paper_id": (c.get("externalIds") or {}).get("ArXiv"), "title": c.get("title"),
                            "year": c.get("year"), "citation_count": c.get("citationCount"), "in_corpus": 0})
        out.sort(key=lambda r: ((r["year"] or 0), r["citation_count"] or 0), reverse=True)
    return out[:k]


def method_trends(terms: list[str], topic: str = "", section: str = "methods") -> dict:
    """Papers per year whose `section` mentions each term. Returns {term: {year: count}}."""
    conn = connect()
    restrict, params = "", []
    if topic:
        ids = topic_paper_ids(topic, n_chunks=300)
        restrict = f" AND p.paper_id IN ({','.join('?' * len(ids))})" if ids else " AND 0"
        params = ids
    out = {}
    for term in terms:
        phrase = " ".join(re.findall(r"[\w\-]+", term))
        q = f'"{phrase}"'
        counts = conn.execute(
            f"""SELECT p.year, COUNT(DISTINCT p.paper_id) FROM chunks_fts f JOIN papers p ON p.paper_id = f.paper_id
                WHERE chunks_fts MATCH ? AND f.section = ? AND p.status='ingested' {restrict}
                GROUP BY p.year ORDER BY p.year""", [q, section] + params).fetchall()
        out[term] = {y: c for y, c in counts}
    totals = conn.execute(f"SELECT year, COUNT(*) FROM papers p WHERE status='ingested' {restrict} GROUP BY year",
                          params).fetchall()
    out["_papers_per_year"] = {y: c for y, c in totals}
    return out


def corpus_graph(limit: int = 400) -> tuple[list[dict], list[tuple[str, str]]]:
    """In-corpus citation graph for the dashboard (most cited papers only)."""
    conn = connect()
    nodes = rows(conn, "SELECT paper_id, s2_id, title, year, citation_count FROM papers "
                       "WHERE status='ingested' AND s2_id IS NOT NULL ORDER BY citation_count DESC LIMIT ?", (limit,))
    by_s2 = {n["s2_id"]: n["paper_id"] for n in nodes}
    marks = ",".join("?" * len(by_s2))
    edges = [(by_s2[a], by_s2[b]) for a, b in conn.execute(
        f"SELECT citing_s2, cited_s2 FROM citations WHERE citing_s2 IN ({marks}) AND cited_s2 IN ({marks})",
        list(by_s2) * 2)] if by_s2 else []
    return nodes, edges
