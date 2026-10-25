"""Merge duplicate papers (same Semantic Scholar id or same normalized title, e.g. two arXiv entries)."""
import re


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _rank(p: dict) -> tuple:
    # Prefer already-ingested, then a known peer-reviewed venue, then more citations
    return (p["status"] == "ingested", bool(p["venue"]), p["citation_count"] or 0)


def run(conn) -> int:
    papers = [dict(r) for r in conn.execute(
        "SELECT paper_id, s2_id, title, venue, citation_count, status FROM papers "
        "WHERE status NOT IN ('duplicate','failed')")]
    groups: dict[str, list[dict]] = {}
    for p in papers:
        key = p["s2_id"] or "t:" + norm_title(p["title"])
        groups.setdefault(key, []).append(p)
    by_title: dict[str, list[dict]] = {}
    for p in papers:
        by_title.setdefault(norm_title(p["title"]), []).append(p)

    merged = 0
    seen = set()
    for group in list(groups.values()) + list(by_title.values()):
        group = [p for p in group if p["paper_id"] not in seen]
        if len(group) < 2:
            continue
        group.sort(key=_rank, reverse=True)
        keep, *dups = group
        for d in dups:
            seen.add(d["paper_id"])
            if d["status"] == "ingested":
                from .store import remove_from_index
                remove_from_index(conn, d["paper_id"])
            conn.execute("UPDATE papers SET status='duplicate', duplicate_of=? WHERE paper_id=?",
                         (keep["paper_id"], d["paper_id"]))
            conn.execute(
                "UPDATE papers SET venue=COALESCE(NULLIF(venue,''), ?), "
                "citation_count=MAX(citation_count, ?) WHERE paper_id=?",
                (d["venue"], d["citation_count"] or 0, keep["paper_id"]))
            merged += 1
    conn.commit()
    return merged
