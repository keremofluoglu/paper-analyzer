"""Structured paper cards (problem, method, datasets, results, limitations, tags...) and benchmark result rows.

Cards need an LLM to read the paper. Two ways:
  * From a Claude Code chat (free): `paper.py uncarded` + `paper.py save-card` (see AI_INSTRUCTIONS.md).
  * Claude Message Batches API (paid, unattended): `paper.py cards-api`, needs ANTHROPIC_API_KEY.

A card only helps *find* papers; the original text stays in the database. Every saved card is checked
against the paper: result numbers not printed in the paper are marked unverified, and evidence quotes
not found verbatim are counted.
"""
import json
import re
import sys
import time

from . import config
from .db import connect, rows

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "dataset": {"type": "string"},
        "metric": {"type": "string"},
        "value": {"type": "number", "description": "Exactly as printed in the paper (e.g. 85.3, not 0.853)"},
        "method_name": {"type": "string"},
        "is_proposed": {"type": "boolean"},
        "higher_is_better": {"type": "boolean"},
    },
    "required": ["task", "dataset", "metric", "value", "method_name", "is_proposed", "higher_is_better"],
    "additionalProperties": False,
}

STR_LIST = {"type": "array", "items": {"type": "string"}}
EVIDENCE_FIELDS = ["method", "key_idea", "main_results", "limitations"]

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string", "description": "What problem the paper solves, 1-2 sentences"},
        "method": {"type": "string", "description": "Name of the proposed method and how it works, 2-4 sentences"},
        "key_idea": {"type": "string", "description": "The single core insight that makes it work"},
        "datasets": STR_LIST,
        "metrics": STR_LIST,
        "main_results": {"type": "string", "description": "Headline numbers vs strongest baseline"},
        "limitations": {"type": "string", "description": "Weaknesses stated by the authors or evident from "
                        "experiments, including conditions where the method does not apply"},
        "future_work": {"type": "string"},
        "tags": {
            "type": "object",
            "properties": {
                "tasks": {**STR_LIST, "description": "e.g. time series forecasting, node classification"},
                "data_types": {**STR_LIST, "description": "e.g. graph, image, text, time series, tabular"},
                "model_families": {**STR_LIST, "description": "e.g. transformer, GNN, CNN, diffusion, RNN"},
            },
            "required": ["tasks", "data_types", "model_families"],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "object",
            "description": "For each field, one sentence copied word for word from the paper that supports it",
            "properties": {f: {"type": "string"} for f in EVIDENCE_FIELDS},
            "required": EVIDENCE_FIELDS,
            "additionalProperties": False,
        },
        "results": {"type": "array", "items": RESULT_SCHEMA,
                    "description": "Main benchmark numbers from result tables: the proposed method and the best baselines"},
    },
    "required": ["problem", "method", "key_idea", "datasets", "metrics", "main_results",
                 "limitations", "future_work", "tags", "evidence", "results"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "Fill a structured card for this research paper using only what the paper says. "
    "Use dataset and metric names exactly as written in the paper. In `results`, give the main "
    "benchmark numbers from the results tables for the proposed method (is_proposed=true) and the "
    "strongest baselines (is_proposed=false), with values exactly as printed; use an empty list if "
    "there are no numeric results. Tables extracted from PDFs can have shifted columns: only report a "
    "number if you are sure which method, dataset and metric it belongs to. In `evidence`, copy one "
    "sentence verbatim from the paper for each field. Keep conditions and caveats in `limitations`. "
    "Write 'not stated' for text fields the paper does not cover."
)

MAX_CONTEXT_CHARS = 400_000  # ~100k tokens: whole papers fit; only giant appendices get cut


CONTEXT_SKIP = ("references", "front", "footnotes", "other")  # not needed to understand the method


def paper_context(pid: str) -> str:
    """The whole paper (every passage with its heading), not a truncated excerpt."""
    conn = connect()
    p = conn.execute("SELECT title, year, venue FROM papers WHERE paper_id=?", (pid,)).fetchone()
    parts = [f"Title: {p['title']} ({p['year']}, {p['venue'] or 'arXiv'})"]
    last = None
    for r in conn.execute("SELECT section, heading, text FROM passages WHERE paper_id=? AND section NOT IN "
                          f"({','.join('?' * len(CONTEXT_SKIP))}) ORDER BY position", (pid, *CONTEXT_SKIP)):
        head = f"{r['section']} | {r['heading']}" if r["heading"] else r["section"]
        parts.append((f"## {head}\n" if head != last else "") + r["text"])
        last = head
    text = "\n\n".join(parts)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n[... rest of the paper cut for length]"
    return text


def next_uncarded(n: int = 5) -> list[dict]:
    conn = connect()
    ids = [r[0] for r in conn.execute(
        "SELECT paper_id FROM papers WHERE status='ingested' AND paper_id NOT IN (SELECT paper_id FROM cards) "
        "ORDER BY citation_count DESC LIMIT ?", (n,))]
    return [{"paper_id": pid, "content": paper_context(pid)} for pid in ids]


# ---------------------------------------------------------------- verification against the paper text
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _numbers(text: str) -> set[float]:
    return {round(float(x), 6) for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text)}


def number_in_paper(value: float, numbers: set[float]) -> bool:
    """True if the value (or the same value as a percentage / fraction) is printed in the paper."""
    return any(round(c, 6) in numbers for c in (value, value * 100, value / 100))


def quote_in_paper(quote: str, norm_text: str) -> bool:
    q = _norm(quote)
    return len(q) >= 20 and (q in norm_text or q[:80] in norm_text)


def save_card(pid: str, card: dict, source: str = "claude-code") -> dict:
    missing = [k for k in CARD_SCHEMA["required"] if k not in card]
    if missing:
        raise ValueError(f"card is missing fields: {missing}")
    conn = connect()
    paper = conn.execute("SELECT title, year FROM papers WHERE paper_id=?", (pid,)).fetchone()
    if not paper:
        raise ValueError(f"unknown paper_id {pid}")
    # Full text (every page) plus the structured text: numbers in formulas, captions, appendix tables count
    full = "\n".join(r[0] for r in conn.execute(
        "SELECT text FROM pages WHERE paper_id=? UNION ALL SELECT text FROM sections WHERE paper_id=?", (pid, pid)))
    numbers, norm_text = _numbers(full), _norm(full)

    evidence = card.get("evidence") or {}
    quotes = [q for q in evidence.values() if q]
    found = sum(quote_in_paper(q, norm_text) for q in quotes)
    conn.execute("DELETE FROM cards WHERE paper_id=?", (pid,))
    conn.execute("DELETE FROM results WHERE paper_id=?", (pid,))
    conn.execute(
        "INSERT INTO cards (paper_id, problem, method, key_idea, datasets, metrics, main_results, limitations, "
        "future_work, tags, evidence, evidence_found, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, card["problem"], card["method"], card["key_idea"], json.dumps(card["datasets"]),
         json.dumps(card["metrics"]), card["main_results"], card["limitations"], card["future_work"],
         json.dumps(card.get("tags") or {}), json.dumps(evidence), found, source))
    saved, unverified = 0, []
    for r in card.get("results") or []:
        try:
            value = float(r["value"])
        except (TypeError, ValueError, KeyError):
            continue
        ok = number_in_paper(value, numbers)
        conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                     (pid, r.get("task", ""), r.get("dataset", ""), r.get("metric", ""), value,
                      r.get("method_name", ""), int(bool(r.get("is_proposed"))),
                      int(bool(r.get("higher_is_better", True))), int(ok)))
        saved += 1
        if not ok:
            unverified.append(f"{r.get('method_name')} / {r.get('dataset')} / {r.get('metric')} = {value}")
    conn.commit()

    from .search import index_approach
    index_approach(pid, paper["title"], paper["year"], card)
    return {"paper_id": pid, "saved": True, "result_rows": saved,
            "unverified_numbers": unverified,
            "evidence_quotes_found": f"{found}/{len(quotes)}"}


def get_card(pid: str) -> dict | None:
    conn = connect()
    c = conn.execute("SELECT * FROM cards WHERE paper_id=?", (pid,)).fetchone()
    if not c:
        return None
    card = dict(c)
    for f in ("datasets", "metrics"):
        card[f] = json.loads(card[f] or "[]")
    for f in ("tags", "evidence"):
        card[f] = json.loads(card[f] or "{}")
    card["results"] = rows(conn, "SELECT task, dataset, metric, value, method_name, is_proposed, verified "
                                 "FROM results WHERE paper_id=?", (pid,))
    return card


def compare_results(dataset: str = "", metric: str = "", task: str = "", limit: int = 50) -> list[dict]:
    conn = connect()
    sql = ("SELECT r.dataset, r.metric, r.method_name, r.value, r.is_proposed, r.higher_is_better, r.verified, "
           "r.task, p.paper_id, p.title, p.year FROM results r JOIN papers p USING(paper_id) WHERE 1=1")
    params = []
    for col, val in (("dataset", dataset), ("metric", metric), ("task", task)):
        if val:
            sql += f" AND r.{col} LIKE ?"
            params.append(f"%{val}%")
    sql += " ORDER BY r.dataset, r.metric, CASE WHEN r.higher_is_better THEN -r.value ELSE r.value END LIMIT ?"
    params.append(limit)
    return rows(conn, sql, params)


# ---------------------------------------------------------------- Claude API (optional)


def submit_batch(limit: int = 1000) -> str:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    papers = next_uncarded(limit)
    if not papers:
        return ""
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[
        Request(
            custom_id=p["paper_id"].replace("/", "_").replace(".", "-"),
            params=MessageCreateParamsNonStreaming(
                model=config.CARD_MODEL,
                max_tokens=8000,
                system=INSTRUCTIONS,
                messages=[{"role": "user", "content": p["content"]}],
                output_config={"format": {"type": "json_schema", "schema": CARD_SCHEMA}},
            ),
        )
        for p in papers
    ])
    ids = {p["paper_id"].replace("/", "_").replace(".", "-"): p["paper_id"] for p in papers}
    batch_file = config.ws().batch_file
    pending = json.loads(batch_file.read_text()) if batch_file.exists() else {}
    pending[batch.id] = ids
    batch_file.write_text(json.dumps(pending, indent=1))
    print(f"Submitted batch {batch.id} with {len(papers)} papers", file=sys.stderr)
    return batch.id


def collect_batches(wait: bool = False) -> dict:
    import anthropic

    batch_file = config.ws().batch_file
    if not batch_file.exists():
        return {"saved": 0, "pending": 0}
    client = anthropic.Anthropic()
    pending = json.loads(batch_file.read_text())
    saved = failed = 0
    for batch_id in list(pending):
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended" or not wait:
                break
            print(f"{batch_id}: {batch.request_counts.processing} still processing", file=sys.stderr)
            time.sleep(60)
        if batch.processing_status != "ended":
            continue
        id_map = pending[batch_id]
        for result in client.messages.batches.results(batch_id):
            pid = id_map.get(result.custom_id)
            if result.result.type != "succeeded" or not pid:
                failed += 1
                continue
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                save_card(pid, json.loads(text), source=config.CARD_MODEL)
                saved += 1
            except (ValueError, json.JSONDecodeError) as e:
                print(f"  {pid}: {e}", file=sys.stderr)
                failed += 1
        del pending[batch_id]
    batch_file.write_text(json.dumps(pending, indent=1))
    return {"saved": saved, "failed": failed, "pending": len(pending)}
