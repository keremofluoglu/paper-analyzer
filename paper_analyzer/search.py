"""Retrieval.

Passages: small chunks are found by vector (Chroma) + keyword (SQLite FTS5 BM25) search, fused with
reciprocal rank fusion, reranked by a cross-encoder, then expanded to their parent passage (whole
subsection) and grouped by paper.
Approaches: one vector per paper card ("method + key idea"), for "which approaches exist for X".
"""
import json
import math
import re

from . import config
from .db import connect
from .models import approach_collection, chunk_collection, embed_query, paper_collection, reranker


def _fts_query(q: str) -> str:
    terms = [t for t in re.findall(r"[\w\-]+", q.lower()) if len(t) > 2]
    return " OR ".join(f'"{t}"' for t in terms) or '""'


def vector_hits(query: str, sections: list[str] | None, n: int, year_min: int | None) -> list[str]:
    col = chunk_collection()
    if col.count() == 0:
        return []
    where = []
    if sections:
        where.append({"section": {"$in": sections}})
    if year_min:
        where.append({"year": {"$gte": year_min}})
    w = None if not where else where[0] if len(where) == 1 else {"$and": where}
    res = col.query(query_embeddings=[embed_query(query).tolist()], n_results=n, where=w)
    return res["ids"][0] if res["ids"] else []


def keyword_hits(conn, query: str, sections: list[str] | None, n: int, year_min: int | None) -> list[str]:
    sql = ("SELECT f.chunk_id FROM chunks_fts f JOIN papers p ON p.paper_id = f.paper_id "
           "WHERE chunks_fts MATCH ?")
    params: list = [_fts_query(query)]
    if sections:
        sql += f" AND f.section IN ({','.join('?' * len(sections))})"
        params += sections
    if year_min:
        sql += " AND p.year >= ?"
        params.append(year_min)
    sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(n)
    return [r[0] for r in conn.execute(sql, params)]


def _quality_boost(citations: int, venue: str) -> float:
    v = (venue or "").lower()
    return config.CITATION_BOOST * math.log1p(citations or 0) + \
        (config.VENUE_BOOST if any(w in v for w in config.VENUE_WHITELIST) else 0)


def _window(passage: str, match: str, chars: int) -> str:
    """The passage, cut to `chars` characters around the matching chunk (0 = whole passage)."""
    if chars <= 0 or len(passage) <= chars:
        return passage
    i = max(0, passage.find(match[:80]))
    start = max(0, min(i - chars // 5, len(passage) - chars))
    return ("..." if start else "") + passage[start:start + chars] + ("..." if start + chars < len(passage) else "")


def search(query: str, sections: list[str] | str | None = "methods", k: int = 6, per_paper: int = None,
           year_min: int | None = None, mode: str = "hybrid", chars: int = 3000) -> list[dict]:
    """Returns up to k papers, each with its best `per_paper` passages.
    mode: 'hybrid' (default) or 'vector' (plain vector search, no keywords or rerank; for comparison)."""
    per_paper = per_paper or config.PASSAGES_PER_PAPER
    if isinstance(sections, str):
        sections = [sections]
    conn = connect()
    n = config.CANDIDATES * 2
    vec = vector_hits(query, sections, n, year_min)
    if mode == "vector":
        ranked = vec
    else:
        kw = keyword_hits(conn, query, sections, n, year_min)
        fused: dict[str, float] = {}
        for lst in (vec, kw):
            for rank, cid in enumerate(lst):
                fused[cid] = fused.get(cid, 0) + 1 / (config.RRF_K + rank + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)[:n]
    if not ranked:
        return []

    marks = ",".join("?" * len(ranked))
    rows = {r["chunk_id"]: dict(r) for r in conn.execute(
        f"""SELECT c.chunk_id, c.paper_id, c.passage_id, c.section, c.heading, c.text,
                   p.title, p.year, p.venue, p.citation_count, p.url, p.code_url
            FROM chunks c JOIN papers p USING(paper_id) WHERE c.chunk_id IN ({marks})""", ranked)}
    hits = [rows[c] for c in ranked if c in rows]
    if mode == "vector":
        for i, h in enumerate(hits):
            h["score"] = -i
    else:
        scores = reranker().predict([(query, f"{h['heading']}: {h['text']}") for h in hits],
                                    batch_size=16, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["score"] = float(s) + _quality_boost(h["citation_count"], h["venue"])

    # chunk -> passage (best chunk wins) -> paper
    best_by_passage: dict[str, dict] = {}
    for h in hits:
        if h["passage_id"] not in best_by_passage or h["score"] > best_by_passage[h["passage_id"]]["score"]:
            best_by_passage[h["passage_id"]] = h
    papers: dict[str, dict] = {}
    for h in sorted(best_by_passage.values(), key=lambda h: h["score"], reverse=True):
        p = papers.setdefault(h["paper_id"], {
            "paper_id": h["paper_id"], "title": h["title"], "year": h["year"], "venue": h["venue"],
            "citation_count": h["citation_count"], "code_url": h["code_url"], "url": h["url"],
            "score": round(h["score"], 3), "passages": []})
        if len(p["passages"]) < per_paper:
            p["passages"].append(h)
    out = list(papers.values())[:k]

    ids = [ps["passage_id"] for p in out for ps in p["passages"]]
    full = {r[0]: r[1] for r in conn.execute(
        f"SELECT passage_id, text FROM passages WHERE passage_id IN ({','.join('?' * len(ids))})", ids)}
    figs = passage_figures(conn, ids)
    for p in out:
        p["passages"] = [{"passage_id": h["passage_id"], "section": h["section"], "heading": h["heading"],
                          "score": round(h["score"], 3), "text": _window(full[h["passage_id"]], h["text"], chars),
                          "figures": figs.get(h["passage_id"], [])}
                         for h in p["passages"]]
    return out


def passage_figures(conn, passage_ids: list[str]) -> dict[str, list[dict]]:
    """Figures each passage refers to ("as shown in Figure 2"), with their caption and PNG path."""
    if not passage_ids:
        return {}
    out: dict[str, list[dict]] = {}
    for r in conn.execute(
            f"""SELECT r.passage_id, f.fig_no, f.kind, f.caption, f.image_path FROM figure_refs r
                JOIN figures f ON f.paper_id = r.paper_id AND f.fig_no = r.fig_no
                WHERE r.passage_id IN ({','.join('?' * len(passage_ids))})""", passage_ids):
        out.setdefault(r["passage_id"], []).append(
            {"fig_no": r["fig_no"], "kind": r["kind"], "caption": r["caption"], "image_path": r["image_path"]})
    return out


def get_passage(passage_id: str, neighbors: int = 0) -> list[dict]:
    """A passage in full, optionally with the passages right before/after it in the paper."""
    conn = connect()
    row = conn.execute("SELECT paper_id, position FROM passages WHERE passage_id=?", (passage_id,)).fetchone()
    if not row:
        return []
    out = [dict(r) for r in conn.execute(
        "SELECT passage_id, section, heading, text FROM passages WHERE paper_id=? AND position BETWEEN ? AND ? "
        "ORDER BY position", (row["paper_id"], row["position"] - neighbors, row["position"] + neighbors))]
    figs = passage_figures(conn, [p["passage_id"] for p in out])
    for p in out:
        p["figures"] = figs.get(p["passage_id"], [])
    return out


# ---------------------------------------------------------------- approaches (from paper cards)
def approach_text(title: str, card: dict) -> str:
    return (f"{title}. Method: {card['method']} Key idea: {card['key_idea']} "
            f"Problem: {card['problem']} Results: {card['main_results']}")


def index_approach(paper_id: str, title: str, year: int, card: dict):
    from .models import embed_chunks
    text = approach_text(title, card)
    approach_collection().upsert(ids=[paper_id], documents=[text],
                                 embeddings=embed_chunks([text]).tolist(),
                                 metadatas=[{"title": title, "year": year or 0}])


def search_approaches(query: str, k: int = 10, task: str = "", data_type: str = "", model_family: str = "",
                      year_min: int | None = None) -> list[dict]:
    """Which approaches (paper methods) fit the query. Tag filters are substring matches on card tags."""
    col = approach_collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[embed_query(query).tolist()], n_results=min(100, col.count()),
                    where={"year": {"$gte": year_min}} if year_min else None)
    conn = connect()
    cands = []
    for pid, doc in zip(res["ids"][0], res["documents"][0]):
        row = conn.execute("SELECT p.paper_id, p.title, p.year, p.venue, p.citation_count, p.code_url, c.tags "
                           "FROM papers p JOIN cards c USING(paper_id) WHERE p.paper_id=?", (pid,)).fetchone()
        if not row:
            continue
        tags = json.loads(row["tags"] or "{}")
        wanted = (("tasks", task), ("data_types", data_type), ("model_families", model_family))
        if any(val and not any(val.lower() in t.lower() for t in tags.get(key, [])) for key, val in wanted):
            continue
        cands.append({**dict(row), "tags": tags, "approach": doc})
    if not cands:
        return []
    scores = reranker().predict([(query, c["approach"]) for c in cands], batch_size=16, show_progress_bar=False)
    for c, s in zip(cands, scores):
        c["score"] = round(float(s) + _quality_boost(c["citation_count"], c["venue"]), 3)
    return sorted(cands, key=lambda c: c["score"], reverse=True)[:k]


# ---------------------------------------------------------------- paper level
def similar_papers(paper_id: str, k: int = 10) -> list[dict]:
    col = paper_collection()
    got = col.get(ids=[paper_id], include=["embeddings"])
    if not got["ids"]:
        return []
    res = col.query(query_embeddings=[got["embeddings"][0]], n_results=k + 1)
    conn = connect()
    out = []
    for pid, dist in zip(res["ids"][0], res["distances"][0]):
        if pid == paper_id:
            continue
        p = conn.execute("SELECT paper_id, title, year, venue, citation_count, url FROM papers WHERE paper_id=?",
                         (pid,)).fetchone()
        if p:
            out.append({**dict(p), "similarity": round(1 - dist, 3)})
    return out[:k]


def topic_paper_ids(query: str, n_chunks: int = 100) -> list[str]:
    """Papers relevant to a topic (cheap: vector search over abstracts + intros + methods, no rerank)."""
    ids = vector_hits(query, ["abstract", "introduction", "methods"], n_chunks, None)
    return list(dict.fromkeys(cid.split("::")[0] for cid in ids))
