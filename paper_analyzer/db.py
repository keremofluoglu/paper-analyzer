"""SQLite catalog: papers, sections, chunks (with FTS5 keyword index), cards, results, citation graph."""
import sqlite3

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,          -- arXiv id without version, e.g. 2106.09685
    s2_id TEXT,
    doi TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    year INTEGER,
    published TEXT,
    venue TEXT,
    citation_count INTEGER DEFAULT 0,
    url TEXT,
    code_url TEXT,
    pdf_path TEXT,
    abstract TEXT,
    source TEXT,                        -- 'search' or 'citation'
    status TEXT DEFAULT 'new',          -- new | downloaded | ingested | failed | duplicate
    duplicate_of TEXT,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_papers_s2 ON papers(s2_id);

CREATE TABLE IF NOT EXISTS sections (
    paper_id TEXT,
    section TEXT,
    text TEXT,
    PRIMARY KEY (paper_id, section)
);

-- "Small-to-big": small chunks (~200 words, paragraph-aligned) are searched,
-- their parent passage (a whole subsection, up to ~1200 words) is what gets read.
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    paper_id TEXT,
    section TEXT,
    heading TEXT,
    position INTEGER,
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_passages_paper ON passages(paper_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT,
    passage_id TEXT,
    section TEXT,
    heading TEXT,
    text TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, section UNINDEXED, paper_id UNINDEXED, chunk_id UNINDEXED,
    tokenize = 'porter unicode61'
);

-- Full-text layer: every page's text as PyMuPDF reads it (nothing filtered; for keyword search,
-- quotes and number checks)
CREATE TABLE IF NOT EXISTS pages (
    paper_id TEXT,
    page INTEGER,
    text TEXT,
    PRIMARY KEY (paper_id, page)
);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    text, paper_id UNINDEXED, page UNINDEXED, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS paper_tables (
    paper_id TEXT,
    table_no TEXT,        -- as printed ("2", "IV"); "?1" = no caption found
    page INTEGER,
    caption TEXT,
    columns TEXT,         -- JSON list of column names (merged header cells joined with " / ")
    rows TEXT,            -- JSON list of rows (lists of cell strings)
    method TEXT,          -- 'lines' (ruled table) or 'text' (column gaps; check against the PDF)
    suspicious INTEGER    -- 1 = layout was ambiguous; verify numbers against the page text
);
CREATE INDEX IF NOT EXISTS idx_tables_paper ON paper_tables(paper_id);

CREATE TABLE IF NOT EXISTS figures (
    paper_id TEXT,
    fig_no TEXT,
    page INTEGER,
    caption TEXT,
    kind TEXT,            -- diagram | plot | example | other (from caption words)
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_figures_paper ON figures(paper_id);

CREATE TABLE IF NOT EXISTS figure_refs (   -- which passages mention which figure
    passage_id TEXT,
    paper_id TEXT,
    fig_no TEXT,
    PRIMARY KEY (passage_id, fig_no)
);

CREATE TABLE IF NOT EXISTS cards (
    paper_id TEXT PRIMARY KEY,
    problem TEXT,
    method TEXT,
    key_idea TEXT,
    datasets TEXT,       -- JSON list
    metrics TEXT,        -- JSON list
    main_results TEXT,
    limitations TEXT,
    future_work TEXT,
    tags TEXT,           -- JSON: {"tasks": [...], "data_types": [...], "model_families": [...]}
    evidence TEXT,       -- JSON: {field: exact quote from the paper}
    evidence_found INTEGER,  -- how many evidence quotes were found verbatim in the paper text
    source TEXT,         -- 'claude-code' or model id
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results (
    paper_id TEXT,
    task TEXT,
    dataset TEXT,
    metric TEXT,
    value REAL,
    method_name TEXT,
    is_proposed INTEGER,  -- 1 = the paper's own method, 0 = a baseline it reports
    higher_is_better INTEGER,
    verified INTEGER      -- 1 = the number appears in the paper text, 0 = not found (treat with care)
);
CREATE INDEX IF NOT EXISTS idx_results_ds ON results(dataset);

CREATE TABLE IF NOT EXISTS s2_papers (     -- metadata for every Semantic Scholar paper seen
    s2_id TEXT PRIMARY KEY,
    arxiv_id TEXT,
    title TEXT,
    year INTEGER,
    venue TEXT,
    citation_count INTEGER
);

CREATE TABLE IF NOT EXISTS citations (
    citing_s2 TEXT,
    cited_s2 TEXT,
    PRIMARY KEY (citing_s2, cited_s2)
);
CREATE INDEX IF NOT EXISTS idx_cit_cited ON citations(cited_s2);
"""


def connect() -> sqlite3.Connection:
    """The active workspace's database (see config.ws)."""
    conn = sqlite3.connect(config.ws().db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    if "coverage" not in cols:  # added with the zero-loss storage
        conn.execute("ALTER TABLE papers ADD COLUMN coverage REAL")
    return conn


def rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
