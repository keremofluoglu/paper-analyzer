# Paper Analyzer

Local vector database of CS/AI papers. Not auto-loaded anywhere: the user points a Claude Code session
at this folder, and that session follows `AI_INSTRUCTIONS.md` (the usage contract for other AIs;
keep it in sync with `paper_analyzer/cli.py`).

## Layout
- `paper.py` - entry point, works from any cwd; wraps `paper_analyzer/cli.py` (typer)
- `paper_analyzer/` - fetch (arXiv + Semantic Scholar), parse (GROBID or PyMuPDF), objects (tables,
  figures), store (Chroma + SQLite FTS5), search (hybrid + rerank), cards, graph, dedup, update, models
- `library/` - shared per-paper files, downloaded once: `pdfs/`, `tei/` (GROBID output, so `--reparse`
  needs no GROBID), `figures/<paper_id>/fig<n>.png`
- `workspaces/<name>/` - one per literature review: papers.db, chroma/, topics.json, my_project.md,
  reports/, collect.log. Chosen by `--workspace` > `PAPER_WORKSPACE` > `.paper-workspace` file in the
  cwd or a parent (written by `paper.py init <name>`); no silent default (`config.ws()`).
- `dashboard/app.py` - Streamlit UI (`paper.py ui` or `Paper Analyzer.bat`); its Collect tab runs
  `paper.py fetch` as a background process
- GROBID: `paper_analyzer/grobid.py` starts `lfoppiano/grobid` with Docker inside the WSL distro "Ubuntu-24.04"
  only for the duration of `ingest`, then stops it (and terminates the distro if ingest started it) to free RAM.
  No WSL/Docker = PyMuPDF fallback. `GROBID_AUTO=0` disables; a GROBID already running at `GROBID_URL` is used as is.

## Storage model (zero loss)
- Three layers: PDF (kept) / `pages` table = full page text (100%) / structured passages (`papers.coverage`
  = share of page words present, target >= 0.97; `paper.py audit`).
- parse order: GROBID TEI, PyMuPDF labels fill sections GROBID lost, else PyMuPDF; references from the
  printed text; PyMuPDF tables (`find_tables` + caption match, word-gap columns for unruled tables)
  replace GROBID's; any PDF line still missing is rescued into section `other` (`RESCUE_MISSING`).
- Nothing is dropped: back matter -> references/appendix/front, fragments attach to the previous paragraph.
- passages: a subsection (<= PASSAGE_WORDS) = reading unit; chunks: ~CHUNK_WORDS paragraph-aligned = search unit
- search: vector + FTS5 over chunks -> RRF -> cross-encoder rerank -> best chunk per passage -> grouped by paper;
  passages carry the figures they cite (`figure_refs`)
- cards: full-paper context; on save, result numbers and evidence quotes are checked against the page text;
  each card is embedded into the `approaches` collection (one vector per method) with tags for filtering
- No generative AI in the storage pipeline: text is never rewritten.

Logs go to stderr; stdout is command output. The torch CUDA wheel lives in C:\PythonWheels\torch
(the user's system Python keeps its CPU torch; never modify it).
