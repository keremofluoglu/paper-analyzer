"""Command line for people and AI assistants. Run through paper.py (see AI_INSTRUCTIONS.md)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import typer

from . import config

app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode=None)

# Commands that work without an active workspace
NO_WORKSPACE = {"init", "workspaces", "download-models", "ui"}


@app.callback()
def main(ctx: typer.Context,
         workspace: str = typer.Option(None, "--workspace", "-w",
                                       help="Workspace to use (default: the .paper-workspace file in this folder)")):
    """Paper Analyzer. Each literature review lives in its own workspace."""
    try:
        if workspace:
            config.select_workspace(workspace)
        if ctx.invoked_subcommand not in NO_WORKSPACE:
            config.ws()
    except config.WorkspaceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise typer.Exit(2)


def show(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def print_papers(papers: list[dict]):
    """Search results grouped by paper, each with its best passages."""
    if not papers:
        print("No results. The corpus may not cover this topic yet (see `stats`, `fetch`).")
    for i, p in enumerate(papers):
        code = f" | code={p['code_url']}" if p.get("code_url") else ""
        print(f"\n=== [{i + 1}] {p['paper_id']} | {p['title']} | {p['year']} | {p['venue'] or 'arXiv'} | "
              f"cites={p['citation_count']}{code}")
        for ps in p["passages"]:
            print(f"\n--- passage {ps['passage_id']} | {ps['section']} | {ps['heading'] or '-'} | score={ps['score']}")
            print(ps["text"])
            print_figures(ps.get("figures"))


def print_figures(figures: list[dict] | None):
    for f in figures or []:
        print(f"[figure {f['fig_no']} ({f['kind']}): {f['caption'][:160]} -> {f['image_path']}]")


def _sections(section: str):
    """"all" = the paper's own text; references, title page, footnotes and rescued fragments by name."""
    from .parse import SECTIONS
    if section == "all":
        return SECTIONS + ["tables", "figures", "appendix"]
    return [s.strip() for s in section.split(",")]


# ---------------------------------------------------------------- asking questions
@app.command()
def search(query: str, section: str = "methods", k: int = 6, year_min: int = 0,
           per_paper: int = typer.Option(config.PASSAGES_PER_PAPER, help="Passages shown per paper"),
           chars: int = typer.Option(3000, help="Max characters per passage (cut around the match), 0 = full"),
           mode: str = typer.Option("hybrid", help="hybrid or vector")):
    """Find the K best papers for QUERY, each with its best passages (whole subsections).
    SECTION: methods (default), abstract, introduction, related_work, experiments, results,
    discussion, limitations, conclusion, tables, figures, appendix, or all (= those);
    references, front (title page), footnotes, other (rescued text) by name. Comma separated works too."""
    from .search import search as do_search
    print_papers(do_search(query, _sections(section), k=k, per_paper=per_paper,
                           year_min=year_min or None, mode=mode, chars=chars))


@app.command()
def limitations(query: str, k: int = 6, chars: int = 3000):
    """Where approaches fail: searches limitations, discussion and conclusion sections."""
    from .search import search as do_search
    print_papers(do_search(query, ["limitations", "discussion", "conclusion"], k=k, chars=chars))


@app.command()
def approaches(query: str, k: int = 10, task: str = "", data_type: str = "", model_family: str = "",
               year_min: int = 0):
    """Which methods exist for QUERY: one result per paper, from paper cards (only carded papers).
    Optional tag filters, e.g. --model-family transformer --data-type graph."""
    from .search import search_approaches
    res = search_approaches(query, k, task, data_type, model_family, year_min or None)
    if not res:
        print("No matching approaches. Only papers with cards are in this index (see `cards-status`).")
    for i, a in enumerate(res):
        code = f" | code={a['code_url']}" if a.get("code_url") else ""
        print(f"\n[{i + 1}] {a['paper_id']} | {a['title']} | {a['year']} | cites={a['citation_count']} | "
              f"score={a['score']}{code}")
        print(f"tags: {json.dumps(a['tags'], ensure_ascii=False)}")
        print(a["approach"])


@app.command()
def passage(passage_id: str, neighbors: int = typer.Option(0, help="Also show N passages before and after")):
    """Read a passage in full (ids come from `search`), optionally with its surroundings."""
    from .search import get_passage
    for ps in get_passage(passage_id, neighbors):
        print(f"\n--- {ps['passage_id']} | {ps['section']} | {ps['heading'] or '-'}\n{ps['text']}")
        print_figures(ps["figures"])


@app.command()
def paper(paper_id: str, sections: str = "abstract,methods,results,limitations,conclusion",
          chars: int = typer.Option(6000, help="Max characters per section, 0 = full")):
    """One paper: metadata, card, sections (--sections all for everything), its tables (cell by cell)
    and figures (caption + PNG path; open the PNG to look at it)."""
    from .cards import get_card
    from .db import connect, rows
    conn = connect()
    p = conn.execute("SELECT paper_id, title, authors, year, venue, citation_count, url, code_url, coverage "
                     "FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if not p:
        print(f"paper {paper_id} not found")
        raise typer.Exit(1)
    wanted = _sections(sections)
    secs = {r["section"]: (r["text"] if chars <= 0 else r["text"][:chars])
            for r in conn.execute("SELECT section, text FROM sections WHERE paper_id=?", (paper_id,))
            if r["section"] in wanted}
    tables = [{**t, "columns": json.loads(t["columns"]), "rows": json.loads(t["rows"])} for t in rows(
        conn, "SELECT table_no, page, caption, columns, rows, method, suspicious FROM paper_tables WHERE paper_id=?",
        (paper_id,))]
    figures = rows(conn, "SELECT fig_no, page, kind, caption, image_path FROM figures WHERE paper_id=?", (paper_id,))
    show({**dict(p), "card": get_card(paper_id), "sections": secs, "tables": tables, "figures": figures})


@app.command()
def figure(paper_id: str, fig_no: str):
    """Caption, kind and PNG path of one figure (open the PNG to look at it), and the passages citing it."""
    from .db import connect, rows
    conn = connect()
    f = conn.execute("SELECT fig_no, page, kind, caption, image_path FROM figures WHERE paper_id=? AND fig_no=?",
                     (paper_id, fig_no)).fetchone()
    if not f:
        print(f"figure {fig_no} of {paper_id} not found (see `paper {paper_id}` for its figures)")
        raise typer.Exit(1)
    cited = rows(conn, "SELECT r.passage_id, p.section, p.heading FROM figure_refs r "
                       "JOIN passages p USING(passage_id) WHERE r.paper_id=? AND r.fig_no=?", (paper_id, fig_no))
    show({**dict(f), "cited_in": cited})


@app.command()
def audit(paper_id: str = typer.Argument(None)):
    """Storage check per paper: share of the PDF text in the structured index (target >= 0.97),
    tables to double-check, figures nobody refers to, references to figures that were not found."""
    from .store import audit as do_audit
    show(do_audit(paper_id))


@app.command()
def fulltext(query: str, k: int = 10):
    """Keyword search over the complete page text of every paper (nothing filtered), with page numbers."""
    from .db import connect
    from .search import _fts_query
    sql = ("SELECT f.paper_id, f.page, p.title, snippet(pages_fts, 0, '[', ']', ' ... ', 24) AS hit "
           "FROM pages_fts f JOIN papers p ON p.paper_id = f.paper_id WHERE pages_fts MATCH ? "
           "ORDER BY bm25(pages_fts) LIMIT ?")
    for r in connect().execute(sql, (_fts_query(query), k)):
        print(f"{r['paper_id']} p.{r['page']} | {r['title'][:60]} | {r['hit']}")


@app.command("list")
def list_papers(title: str = "", year_min: int = 0, has_card: bool = False, limit: int = 50):
    """List papers, most cited first."""
    from .db import connect, rows
    sql = "SELECT paper_id, title, year, venue, citation_count FROM papers WHERE status='ingested'"
    params: list = []
    if title:
        sql += " AND title LIKE ?"
        params.append(f"%{title}%")
    if year_min:
        sql += " AND year >= ?"
        params.append(year_min)
    if has_card:
        sql += " AND paper_id IN (SELECT paper_id FROM cards)"
    sql += " ORDER BY citation_count DESC LIMIT ?"
    params.append(limit)
    for r in rows(connect(), sql, params):
        print(f"{r['paper_id']} | {r['title']} | {r['year']} | {r['venue'] or 'arXiv'} | cites={r['citation_count']}")


@app.command()
def similar(paper_id: str, k: int = 10):
    """Papers most similar to PAPER_ID."""
    from .search import similar_papers
    show(similar_papers(paper_id, k))


@app.command()
def compare(dataset: str = "", metric: str = "", task: str = "", limit: int = 50):
    """Benchmark leaderboard from paper cards (substring filters). Best first."""
    from .cards import compare_results
    show(compare_results(dataset, metric, task, limit))


@app.command()
def foundational(topic: str = "", k: int = 15):
    """Papers the corpus cites most (optionally only papers about TOPIC)."""
    from .graph import foundational_papers
    show(foundational_papers(topic, k))


@app.command()
def followups(paper_id: str, external: bool = False, k: int = 20):
    """Newer papers citing PAPER_ID. --external also asks Semantic Scholar."""
    from .graph import follow_ups
    show(follow_ups(paper_id, external, k))


@app.command()
def trends(terms: str, topic: str = ""):
    """Papers per year mentioning each comma-separated term in their methods section."""
    from .graph import method_trends
    show(method_trends([t.strip() for t in terms.split(",") if t.strip()], topic))


@app.command("my-project")
def my_project():
    """The user's own project description (my_project.md in the workspace)."""
    path = config.ws().my_project_path
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    print(text or f"(empty - the user can describe their project in {path})")


# ---------------------------------------------------------------- paper cards
@app.command()
def uncarded(n: int = 3):
    """Next papers without a card, with their text, for an AI to read and then `save-card`."""
    from .cards import next_uncarded
    papers = next_uncarded(n)
    if not papers:
        print("All ingested papers have cards.")
    for p in papers:
        print(f"\n===== paper_id: {p['paper_id']} =====\n{p['content']}")


@app.command("card-schema")
def card_schema():
    """JSON schema that save-card expects."""
    from .cards import CARD_SCHEMA
    show(CARD_SCHEMA)


@app.command("save-card")
def save_card(paper_id: str, file: Path = typer.Option(None, help="JSON file; omit to read JSON from stdin")):
    """Save a card (JSON matching card-schema) for PAPER_ID."""
    from .cards import save_card as do_save
    raw = file.read_text(encoding="utf-8") if file else sys.stdin.read()
    show(do_save(paper_id, json.loads(raw)))


@app.command()
def card(paper_id: str):
    """Show the card of PAPER_ID."""
    from .cards import get_card
    show(get_card(paper_id) or {"error": "no card yet"})


@app.command("cards-status")
def cards_status():
    """How many papers have cards, and how trustworthy they look."""
    from .db import connect
    conn = connect()
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    show({"papers": one("SELECT COUNT(*) FROM papers WHERE status='ingested'"),
          "with_card": one("SELECT COUNT(*) FROM cards"),
          "result_rows": one("SELECT COUNT(*) FROM results"),
          "unverified_result_rows": one("SELECT COUNT(*) FROM results WHERE verified = 0"),
          "cards_with_all_evidence_found": one("SELECT COUNT(*) FROM cards WHERE evidence_found >= 4")})


@app.command("cards-api")
def cards_api(limit: int = 1000, wait: bool = False):
    """Make cards with the Claude Message Batches API (needs ANTHROPIC_API_KEY, costs money).
    First run submits; run again (or --wait) to collect."""
    from .cards import collect_batches, submit_batch
    batch_file = config.ws().batch_file
    if batch_file.exists() and json.loads(batch_file.read_text()):
        show(collect_batches(wait))
    else:
        submit_batch(limit)
        if wait:
            show(collect_batches(wait=True))


# ---------------------------------------------------------------- building the corpus
@app.command()
def fetch(query: str, n: int = 100, expand_citations: bool = typer.Option(True, "--expand-citations/--no-expand"),
          ingest_after: bool = typer.Option(True, "--ingest/--no-ingest"),
          max_expand: int = typer.Option(config.EXPAND_PER_QUERY, help="Max referenced papers to add")):
    """Search arXiv for QUERY, add highly cited referenced papers, download PDFs, then ingest. Slow."""
    from .fetch import fetch as do_fetch
    show(do_fetch(query, n, expand_citations, max_expand=max_expand))
    if ingest_after:
        from .store import ingest as do_ingest
        show(do_ingest())


@app.command()
def ingest(limit: int = 0, reparse: bool = False):
    """Parse, chunk and embed downloaded PDFs. --reparse re-processes already ingested papers."""
    from .store import ingest as do_ingest
    show(do_ingest(limit or None, reparse))


@app.command()
def update(n: int = 200):
    """Fetch new papers for every saved topic since its last run, then ingest."""
    from .update import update as do_update
    show(do_update(n))


@app.command()
def stats():
    """Corpus statistics and saved topics."""
    from .fetch import load_topics
    from .store import stats as do_stats
    show({**do_stats(), "topics": list(load_topics())})


@app.command("download-models")
def download_models():
    """Download the three local AI models (~2 GB, one time). Safe to re-run; resumes after drops."""
    from .models import local_model
    for repo in (config.CHUNK_MODEL, config.PAPER_MODEL, config.RERANK_MODEL):
        print(f"{repo}: {local_model(repo)}")


@app.command()
def dedup():
    """Merge duplicate papers."""
    from .db import connect
    from .dedup import run
    print(f"merged {run(connect())} duplicates")


@app.command("schedule-update")
def schedule_update(day: str = "MON", time: str = "09:00"):
    """Create a weekly Windows scheduled task that runs `update` for the current workspace."""
    name = config.ws().name
    cmd = f'"{sys.executable}" "{config.ROOT / "paper.py"}" --workspace {name} update'
    subprocess.run(["schtasks", "/Create", "/F", "/SC", "WEEKLY", "/D", day, "/ST", time,
                    "/TN", f"PaperAnalyzerUpdate-{name}", "/TR", cmd], check=False)


@app.command()
def ui(port: int = 8501):
    """Open the Paper Analyzer UI in the browser (Ctrl+C here stops it)."""
    env = dict(os.environ)
    found = config.find_marker()
    if found and "PAPER_WORKSPACE" not in env:
        env["PAPER_WORKSPACE"] = found[0]  # preselect this folder's workspace
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "dashboard" / "app.py"),
                    "--server.port", str(port), "--browser.gatherUsageStats", "false"], env=env)


# ---------------------------------------------------------------- workspaces
@app.command()
def init(name: str):
    """Create workspace NAME (if new) and link the current folder to it with a .paper-workspace file."""
    try:
        w = config.create_workspace(name)
    except config.WorkspaceError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise typer.Exit(2)
    marker = Path.cwd() / config.MARKER
    old = config.find_marker()
    if old and old[1] == marker and old[0] != name:
        print(f"ERROR: {marker} already links this folder to workspace {old[0]!r}; delete it first", file=sys.stderr)
        raise typer.Exit(2)
    marker.write_text(name + "\n", encoding="utf-8")
    show({"workspace": name, "path": str(w.path), "marker": str(marker)})


@app.command()
def workspaces():
    """All workspaces with their paper counts."""
    import sqlite3
    out = []
    for name in config.list_workspaces():
        db = config.Workspace(name).db_path
        n = 0
        if db.exists():
            with sqlite3.connect(db) as c:
                n = c.execute("SELECT COUNT(*) FROM papers WHERE status='ingested'").fetchone()[0]
        out.append({"workspace": name, "papers": n})
    show(out)


@app.command()
def workspace():
    """The active workspace and where it is stored."""
    w = config.ws()
    show({"workspace": w.name, "path": str(w.path), "marker": str((config.find_marker() or (None, None))[1])})


if __name__ == "__main__":
    app()
