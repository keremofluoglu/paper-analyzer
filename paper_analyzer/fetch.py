"""Find and download papers: arXiv keyword search, Semantic Scholar metadata + citation expansion, PDF download."""
import json
import re
import sys
import time
from datetime import datetime, timezone

import arxiv
import requests

from . import config
from .db import connect

S2 = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,year,venue,citationCount"
_session = requests.Session()
_session.headers["User-Agent"] = "paper-analyzer/0.1 (personal research tool)"
if config.S2_API_KEY:
    _session.headers["x-api-key"] = config.S2_API_KEY
_last_s2 = 0.0


def log(*a):
    print(*a, file=sys.stderr, flush=True)  # stdout is reserved for the MCP protocol


def clean_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def pdf_filename(paper_id: str) -> str:
    return paper_id.replace("/", "_") + ".pdf"


# ---------------------------------------------------------------- Semantic Scholar
def s2_request(method: str, path: str, **kw):
    global _last_s2
    for attempt in range(6):
        wait = config.S2_DELAY - (time.time() - _last_s2)
        if wait > 0:
            time.sleep(wait)
        _last_s2 = time.time()
        try:
            r = _session.request(method, S2 + path, timeout=30, **kw)
        except requests.RequestException as e:
            log(f"  S2 network error: {e}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    log(f"  S2 gave up on {path}")
    return None


def s2_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """arXiv id -> S2 record."""
    out = {}
    for i in range(0, len(arxiv_ids), 400):
        part = arxiv_ids[i:i + 400]
        data = s2_request("POST", "/paper/batch", params={"fields": S2_FIELDS},
                          json={"ids": [f"arXiv:{a}" for a in part]}) or []
        for aid, rec in zip(part, data):
            if rec:
                out[aid] = rec
    return out


def s2_references(s2_id: str) -> list[dict]:
    data = s2_request("GET", f"/paper/{s2_id}/references", params={"fields": S2_FIELDS, "limit": 1000})
    return [d["citedPaper"] for d in (data or {}).get("data", []) if d.get("citedPaper", {}).get("paperId")]


def s2_citations(s2_id: str, limit: int = 100) -> list[dict]:
    data = s2_request("GET", f"/paper/{s2_id}/citations", params={"fields": S2_FIELDS, "limit": limit})
    return [d["citingPaper"] for d in (data or {}).get("data", []) if d.get("citingPaper", {}).get("paperId")]


def remember_s2(conn, rec: dict):
    arx = (rec.get("externalIds") or {}).get("ArXiv")
    conn.execute(
        "INSERT OR REPLACE INTO s2_papers VALUES (?,?,?,?,?,?)",
        (rec["paperId"], arx, rec.get("title"), rec.get("year"), rec.get("venue"), rec.get("citationCount") or 0),
    )


def passes_quality(rec: dict) -> bool:
    cites = rec.get("citationCount") or 0
    year = rec.get("year") or datetime.now().year
    age = max(1, datetime.now().year - year + 1)
    return cites >= config.MIN_CITATIONS_TOTAL or cites / age >= config.MIN_CITATIONS_PER_YEAR


# ---------------------------------------------------------------- arXiv
_arxiv = arxiv.Client(page_size=100, delay_seconds=config.ARXIV_DELAY, num_retries=5)


def build_query(q: str) -> str:
    if ":" in q:  # user wrote raw arXiv syntax
        return q
    words = [w for w in re.findall(r"[\w\-]+", q) if len(w) > 1]
    return " AND ".join(f"all:{w}" for w in words)


def arxiv_search(query: str, n: int, newest_first: bool = False) -> list[arxiv.Result]:
    s = arxiv.Search(
        query=build_query(query),
        max_results=n,
        sort_by=arxiv.SortCriterion.SubmittedDate if newest_first else arxiv.SortCriterion.Relevance,
    )
    return list(_arxiv.results(s))


def arxiv_by_ids(ids: list[str]) -> list[arxiv.Result]:
    out = []
    for i in range(0, len(ids), 100):
        out += list(_arxiv.results(arxiv.Search(id_list=ids[i:i + 100], max_results=len(ids[i:i + 100]))))
    return out


def upsert_paper(conn, r: arxiv.Result, source: str):
    pid = clean_id(r.get_short_id())
    conn.execute(
        """INSERT INTO papers (paper_id, doi, title, authors, year, published, venue, url, abstract, source)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(paper_id) DO NOTHING""",
        (pid, r.doi, " ".join(r.title.split()), ", ".join(a.name for a in r.authors[:20]),
         r.published.year, r.published.isoformat(), r.journal_ref, f"https://arxiv.org/abs/{pid}",
         " ".join(r.summary.split()), source),
    )
    return pid


def attach_s2(conn, pids: list[str]):
    """Fill s2 id / venue / citation counts and store each paper's reference edges."""
    recs = s2_batch(pids)
    for pid, rec in recs.items():
        remember_s2(conn, rec)
        conn.execute(
            """UPDATE papers SET s2_id=?, citation_count=?, venue=COALESCE(NULLIF(?, ''), venue)
               WHERE paper_id=?""",
            (rec["paperId"], rec.get("citationCount") or 0, rec.get("venue"), pid),
        )
    conn.commit()
    return recs


def store_references(conn, s2_id: str) -> list[dict]:
    refs = s2_references(s2_id)
    for ref in refs:
        remember_s2(conn, ref)
        conn.execute("INSERT OR IGNORE INTO citations VALUES (?,?)", (s2_id, ref["paperId"]))
    conn.commit()
    return refs


# ---------------------------------------------------------------- download
def find_code_url(pid: str) -> str | None:
    try:
        r = _session.get(f"https://huggingface.co/api/papers/{pid}", timeout=15)
        if r.ok:
            return r.json().get("githubRepo")
    except (requests.RequestException, ValueError):
        pass
    return None


def download(conn, pid: str) -> bool:
    path = config.PDF_DIR / pdf_filename(pid)
    if not path.exists():
        try:
            r = _session.get(f"https://export.arxiv.org/pdf/{pid}", timeout=60)
            if not r.ok or not r.content.startswith(b"%PDF"):
                r = _session.get(f"https://arxiv.org/pdf/{pid}", timeout=60)
            if not r.ok or not r.content.startswith(b"%PDF"):
                log(f"  download failed {pid}: HTTP {r.status_code}")
                conn.execute("UPDATE papers SET status='failed' WHERE paper_id=?", (pid,))
                return False
            path.write_bytes(r.content)
            time.sleep(1)
        except requests.RequestException as e:
            log(f"  download failed {pid}: {e}")
            return False
    code = find_code_url(pid)
    conn.execute(
        "UPDATE papers SET pdf_path=?, status='downloaded', code_url=COALESCE(?, code_url) WHERE paper_id=? AND status IN ('new','failed')",
        (str(path), code, pid),
    )
    conn.commit()
    return True


# ---------------------------------------------------------------- topics
def load_topics() -> dict:
    path = config.ws().topics_path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_topic(query: str, n: int, expand: bool):
    topics = load_topics()
    old = topics.get(query, {})
    topics[query] = {"n": old.get("n", n), "expand": old.get("expand", expand),
                     "last_run": datetime.now(timezone.utc).isoformat()}
    config.ws().topics_path.write_text(json.dumps(topics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- main entry
def fetch(query: str, n: int = 100, expand_citations: bool = True, since: datetime | None = None,
          save: bool = True, max_expand: int = config.EXPAND_PER_QUERY) -> dict:
    """Search arXiv, optionally expand with highly cited references, download all PDFs."""
    conn = connect()
    log(f"Searching arXiv: {query!r} (top {n})")
    found = arxiv_search(query, n, newest_first=since is not None)
    if since:
        found = [r for r in found if r.published > since]
    seeds = [upsert_paper(conn, r, "search") for r in found]
    conn.commit()
    log(f"  {len(seeds)} papers from search")

    recs = attach_s2(conn, seeds)
    added = []
    if expand_citations:
        votes: dict[str, int] = {}
        meta: dict[str, dict] = {}
        for i, pid in enumerate(seeds):
            rec = recs.get(pid)
            if not rec:
                continue
            log(f"  references {i + 1}/{len(seeds)}")
            for ref in store_references(conn, rec["paperId"]):
                arx = (ref.get("externalIds") or {}).get("ArXiv")
                if arx and passes_quality(ref):
                    votes[arx] = votes.get(arx, 0) + 1
                    meta[arx] = ref
        known = {r["paper_id"] for r in conn.execute("SELECT paper_id FROM papers")}
        ranked = sorted((a for a in votes if a not in known),
                        key=lambda a: (votes[a], meta[a].get("citationCount") or 0), reverse=True)
        picks = ranked[:max_expand]
        log(f"  {len(picks)} highly cited referenced papers added")
        for r in arxiv_by_ids(picks):
            added.append(upsert_paper(conn, r, "citation"))
        conn.commit()
        attach_s2(conn, added)
        for pid in added:
            rec = conn.execute("SELECT s2_id FROM papers WHERE paper_id=?", (pid,)).fetchone()
            if rec and rec["s2_id"]:
                store_references(conn, rec["s2_id"])
    else:
        for pid in seeds:
            rec = recs.get(pid)
            if rec:
                store_references(conn, rec["paperId"])

    todo = [r["paper_id"] for r in conn.execute("SELECT paper_id FROM papers WHERE status IN ('new','failed')")]
    ok = 0
    for i, pid in enumerate(todo):
        log(f"  download {i + 1}/{len(todo)} {pid}")
        ok += download(conn, pid)
    if save:
        save_topic(query, n, expand_citations)
    return {"query": query, "from_search": len(seeds), "from_citations": len(added), "downloaded": ok}
