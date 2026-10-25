"""Paper Analyzer UI: `paper.py ui` (or `streamlit run dashboard/app.py`).

One workspace (literature review) at a time, chosen in the sidebar. Collection runs the same
`paper.py fetch` command as the CLI, as a background process whose log is shown live.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from paper_analyzer import config  # noqa: E402

st.set_page_config(page_title="Paper Analyzer", layout="wide")


# ---------------------------------------------------------------- workspace (sidebar)
def default_workspace(names: list[str]) -> int:
    wanted = os.environ.get("PAPER_WORKSPACE") or (config.find_marker() or (None,))[0]
    return names.index(wanted) if wanted in names else 0


with st.sidebar:
    st.header("Workspace")
    names = config.list_workspaces()
    if names:
        ws_name = st.selectbox("Literature review", names, index=default_workspace(names),
                               label_visibility="collapsed")
    else:
        ws_name = None
        st.info("No workspace yet. Create one below.")
    with st.form("new_ws", clear_on_submit=True):
        new = st.text_input("New workspace", placeholder="e.g. traffic-forecasting")
        if st.form_submit_button("Create") and new:
            try:
                config.create_workspace(new.strip().lower())
                st.success(f"Created {new}")
                st.rerun()
            except config.WorkspaceError as e:
                st.error(str(e))
    st.caption("Link a project folder to a workspace with `paper.py init <name>` in that folder; "
               "Claude in that folder then uses this workspace.")

if not ws_name:
    st.stop()
WS = config.select_workspace(ws_name)

from paper_analyzer.cards import compare_results, get_card  # noqa: E402
from paper_analyzer.db import connect, rows  # noqa: E402

conn = connect()


@st.cache_data(ttl=30)
def papers_df(ws: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT p.paper_id, p.title, p.year, p.venue, p.citation_count, p.coverage, p.code_url, p.source, "
        "(c.paper_id IS NOT NULL) AS has_card FROM papers p LEFT JOIN cards c USING(paper_id) "
        "WHERE p.status='ingested' ORDER BY p.citation_count DESC", conn)


df = papers_df(ws_name)
st.title(f"Paper Analyzer · {ws_name}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers", len(df))
c2.metric("With card", int(df["has_card"].sum()) if len(df) else 0)
c3.metric("Tables", conn.execute("SELECT COUNT(*) FROM paper_tables").fetchone()[0])
c4.metric("Figures", conn.execute("SELECT COUNT(*) FROM figures").fetchone()[0])
c5.metric("Text coverage", f"{df['coverage'].mean():.1%}" if len(df) and df["coverage"].notna().any() else "-")

tab_search, tab_papers, tab_paper, tab_collect, tab_board, tab_trend, tab_graph, tab_map = st.tabs(
    ["Search", "Papers", "Paper", "Collect", "Leaderboard", "Trends", "Citation graph", "Method map"])


def unique(names: list[str]) -> list[str]:
    """Table column names as dataframe headers must be unique: "MAE", "MAE (2)"."""
    out = []
    for n in names:
        out.append(n if n not in out else f"{n} ({sum(o.split(' (')[0] == n for o in out) + 1})")
    return out


def show_figures(figs: list[dict], width: int = 360):
    for f in figs:
        if Path(f["image_path"]).exists():
            st.image(f["image_path"], caption=f["caption"][:200], width=width)
        else:
            st.caption(f"Figure {f['fig_no']}: {f['caption'][:200]} (image missing)")


# ---------------------------------------------------------------- search
with tab_search:
    from paper_analyzer.parse import EXTRA_SECTIONS, SECTIONS
    q = st.text_input("Question or topic", placeholder="e.g. capture spatial dependency between road sensors")
    s1, s2, s3 = st.columns([3, 1, 1])
    secs = s1.multiselect("Sections", SECTIONS + EXTRA_SECTIONS, default=["methods"])
    k = s2.number_input("Papers", 1, 30, 6)
    mode = s3.selectbox("Mode", ["hybrid", "vector"], help="hybrid = vector + keywords + reranker")
    if q and len(df):
        from paper_analyzer.search import search
        with st.spinner("Searching (the first search loads the models)..."):
            results = search(q, secs or None, k=int(k), mode=mode, chars=0)
        if not results:
            st.info("No results. The workspace may not cover this topic yet (see Collect).")
        for i, p in enumerate(results, start=1):
            st.markdown(f"### {i}. {p['title']}")
            st.caption(f"{p['paper_id']} · {p['year']} · {p['venue'] or 'arXiv'} · {p['citation_count']} citations"
                       + (f" · [code]({p['code_url']})" if p.get("code_url") else ""))
            for ps in p["passages"]:
                with st.expander(f"{ps['section']} · {ps['heading'] or '-'} · score {ps['score']}", expanded=True):
                    st.write(ps["text"])
                    show_figures(ps["figures"])

# ---------------------------------------------------------------- papers
with tab_papers:
    if len(df):
        f1, f2 = st.columns([3, 1])
        title_q = f1.text_input("Filter title")
        yr = f2.number_input("Year from", 1950, 2100, int(df["year"].min() or 1990))
        view = df[(df["year"] >= yr) & df["title"].str.contains(title_q, case=False, regex=False)]
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={"code_url": st.column_config.LinkColumn("code"),
                                    "coverage": st.column_config.ProgressColumn("coverage", min_value=0, max_value=1)})
    else:
        st.info("No papers yet. Start a collection in the Collect tab.")

# ---------------------------------------------------------------- one paper
with tab_paper:
    if len(df):
        titles = df.set_index("paper_id")["title"]
        pid = st.selectbox("Paper", df["paper_id"], format_func=lambda x: f"{x} - {titles[x]}")
        meta = conn.execute("SELECT authors, url, coverage FROM papers WHERE paper_id=?", (pid,)).fetchone()
        st.caption(f"{meta['authors'] or ''} · [{meta['url']}]({meta['url']}) · text coverage {meta['coverage'] or 0:.1%}")
        card = get_card(pid)
        if card:
            with st.expander("Card", expanded=True):
                for field in ("problem", "method", "key_idea", "main_results", "limitations", "future_work"):
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {card[field]}")
                st.markdown(f"**Datasets:** {', '.join(card['datasets'])}  \n**Metrics:** {', '.join(card['metrics'])}")
                if card["results"]:
                    st.caption("verified = 0: the number was not found in the paper text")
                    st.dataframe(pd.DataFrame(card["results"]), hide_index=True)
        else:
            st.info("No card yet. Ask Claude in your project folder: 'make paper cards for the next 10 papers'.")

        figs = rows(conn, "SELECT fig_no, page, kind, caption, image_path FROM figures WHERE paper_id=?", (pid,))
        tabs = rows(conn, "SELECT table_no, page, caption, columns, rows, method, suspicious FROM paper_tables "
                          "WHERE paper_id=?", (pid,))
        t_text, t_tables, t_figs = st.tabs(["Text", f"Tables ({len(tabs)})", f"Figures ({len(figs)})"])
        with t_text:
            for r in conn.execute("SELECT section, heading, text FROM passages WHERE paper_id=? ORDER BY position", (pid,)):
                with st.expander(f"{r['section']} | {r['heading'] or '-'} ({len(r['text'].split())} words)"):
                    st.write(r["text"])
        with t_tables:
            for t in tabs:
                note = " · ⚠ check against the PDF" if t["suspicious"] else ""
                st.markdown(f"**Table {t['table_no']}** (page {t['page']}){note}  \n{t['caption'][:300]}")
                cols = unique(json.loads(t["columns"]))
                body = [(r + [""] * len(cols))[:len(cols)] for r in json.loads(t["rows"])]
                st.dataframe(pd.DataFrame(body, columns=cols), hide_index=True)
        with t_figs:
            show_figures(figs, width=520)

# ---------------------------------------------------------------- collect
STATE = WS.path / "collect.json"
LOG = WS.path / "collect.log"


def collect_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def start_collect(query: str, n: int, expand: bool, max_expand: int):
    cmd = [sys.executable, str(ROOT / "paper.py"), "--workspace", ws_name, "fetch", query, "--n", str(n),
           "--max-expand", str(max_expand), "--expand-citations" if expand else "--no-expand"]
    # The wrapper writes __EXIT__ <code> when fetch + ingest finish, so the status survives page reloads
    wrapper = [sys.executable, "-u", "-c",
               "import subprocess, sys; r = subprocess.run(sys.argv[1:]); print('__EXIT__', r.returncode)", *cmd]
    log = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(wrapper, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT),
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    STATE.write_text(json.dumps({"pid": proc.pid, "query": query, "started": time.time()}), encoding="utf-8")


with tab_collect:
    st.markdown("Search arXiv for a topic, add the highly cited papers they reference, download, and index them. "
                "Slow: arXiv allows one request every 3 s; PDFs are ~2 MB each.")

    @st.fragment(run_every=3)
    def collect_status():
        state = collect_state()
        log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
        running = bool(state) and "__EXIT__" not in log
        if state:
            done = "__EXIT__ 0" in log
            label = "running" if running else ("finished" if done else "stopped with an error")
            st.markdown(f"**Last collection:** `{state['query']}` · {label}")
            lines = [l for l in log.splitlines() if l.strip() and "Warning" not in l]
            prog = [re.search(r"(download|references) (\d+)/(\d+)|^\[(\d+)/(\d+)\]", l) for l in lines]
            prog = [m for m in prog if m]
            if prog and running:
                m = prog[-1]
                i, n = (int(m.group(2)), int(m.group(3))) if m.group(2) else (int(m.group(4)), int(m.group(5)))
                step = m.group(1) or "indexing"
                st.progress(i / max(n, 1), text=f"{step} {i}/{n}")
            st.code("\n".join(lines[-15:]) or "(starting...)", language=None)
            if running and st.button("Stop collection"):
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(state["pid"])], capture_output=True)
                LOG.write_text(log + "\n__EXIT__ stopped\n", encoding="utf-8")
        return running

    running = collect_status()
    with st.form("collect"):
        topic = st.text_input("Topic keywords", placeholder="e.g. graph neural network traffic forecasting")
        a, b, c = st.columns(3)
        n = a.number_input("Papers from search", 5, 500, 50)
        expand = b.checkbox("Add highly cited references", value=True)
        max_expand = c.number_input("Max referenced papers", 0, 500, 50)
        if st.form_submit_button("Start collection", disabled=running) and topic:
            start_collect(topic.strip(), int(n), expand, int(max_expand))
            st.rerun()
    topics = json.loads(WS.topics_path.read_text(encoding="utf-8")) if WS.topics_path.exists() else {}
    if topics:
        st.caption("Saved topics (re-run with `paper.py update`): " + " · ".join(topics))

# ---------------------------------------------------------------- leaderboard
with tab_board:
    datasets = [r[0] for r in conn.execute("SELECT dataset FROM results GROUP BY dataset ORDER BY COUNT(*) DESC")]
    if datasets:
        ds = st.selectbox("Dataset", datasets)
        res = pd.DataFrame(compare_results(dataset=ds, limit=500))
        res = res[res["dataset"] == ds]
        metric = st.selectbox("Metric", sorted(res["metric"].unique()))
        board = res[res["metric"] == metric].sort_values("value", ascending=not bool(res["higher_is_better"].iloc[0]))
        st.dataframe(board[["method_name", "value", "is_proposed", "verified", "title", "year", "paper_id"]],
                     hide_index=True, width="stretch")
    else:
        st.info("No benchmark results yet. They come from paper cards.")

# ---------------------------------------------------------------- trends
with tab_trend:
    terms = st.text_input("Method terms (comma separated)", "transformer, LSTM, graph neural network")
    topic = st.text_input("Restrict to topic (optional)")
    if st.button("Show trend"):
        from paper_analyzer.graph import method_trends
        data = method_trends([t.strip() for t in terms.split(",") if t.strip()], topic)
        totals = data.pop("_papers_per_year")
        trend = [{"year": y, "term": t, "papers": c, "share": c / max(1, totals.get(y, 1))}
                 for t, yc in data.items() for y, c in yc.items()]
        if trend:
            share = st.toggle("Show as share of papers that year")
            st.plotly_chart(px.line(pd.DataFrame(trend), x="year", y="share" if share else "papers",
                                    color="term", markers=True), width="stretch")
        else:
            st.info("No mentions found.")

# ---------------------------------------------------------------- citation graph
with tab_graph:
    n_nodes = st.slider("Most cited papers to show", 20, 400, 120)
    if st.button("Draw graph"):
        from pyvis.network import Network

        from paper_analyzer.graph import corpus_graph
        nodes, edges = corpus_graph(n_nodes)
        net = Network(height="700px", width="100%", directed=True, bgcolor="#ffffff")
        for p in nodes:
            net.add_node(p["paper_id"], label=p["title"][:40], title=f"{p['title']} ({p['year']})",
                         size=5 + min(40, (p["citation_count"] or 0) ** 0.4))
        for a_, b_ in edges:
            net.add_edge(a_, b_)
        net.barnes_hut()
        st.components.v1.html(net.generate_html(), height=720)
        st.caption(f"{len(nodes)} papers, {len(edges)} citations inside the workspace. Arrow = cites.")

# ---------------------------------------------------------------- method map
with tab_map:
    if st.button("Build method map"):
        import numpy as np
        import umap

        from paper_analyzer.models import paper_collection
        got = paper_collection().get(include=["embeddings", "metadatas"])
        if len(got["ids"]) < 10:
            st.info("Need at least 10 papers.")
        else:
            xy = umap.UMAP(n_neighbors=min(15, len(got["ids"]) - 1), min_dist=0.1,
                           metric="cosine").fit_transform(np.array(got["embeddings"]))
            m = pd.DataFrame({"paper_id": got["ids"], "x": xy[:, 0], "y": xy[:, 1],
                              "title": [x["title"] for x in got["metadatas"]],
                              "year": [x["year"] for x in got["metadatas"]]})
            st.plotly_chart(px.scatter(m, x="x", y="y", color="year", hover_data=["title", "paper_id"], height=700),
                            width="stretch")
            st.caption("Each dot is a paper; nearby papers have similar topics and methods (SPECTER2).")
