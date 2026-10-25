# Paper Tool: instructions for AI assistants

You were pointed here by the user. This folder holds a local research-paper database (CS/AI papers
from arXiv). Use it to answer the user's research questions from real papers instead of from memory.

## Workspaces: one per literature review

Papers are kept in **workspaces** so different reviews never mix. The user's project folder (your
working directory) is linked to its workspace by a `.paper-workspace` file, and every command run from
that folder uses it automatically.

- `paper workspace` shows the active one. If it reports "No workspace selected", ask the user which
  review this folder belongs to, then run `paper init <name>` in the project folder (creates the
  workspace if new and writes `.paper-workspace`). Names: lowercase, digits, `-`, `_`.
- `paper workspaces` lists all of them with paper counts.
- `--workspace <name>` (before the command) overrides the folder's workspace for one call.
  Never write into another review's workspace unless the user asks.

## What is stored (nothing is thrown away)

- **Sections**: abstract, introduction, related_work, methods, experiments, results, discussion,
  limitations, conclusion, plus tables, figures (captions), appendix, references, footnotes,
  front (title page, acknowledgements) and other (text the parser could not place, tagged with its page).
- **Passages** = whole subsections (e.g. "3.2 Temporal Attention Module"), up to ~1200 words. This is
  what you read. Search matches ~200-word chunks inside passages and returns the passage around the
  match, grouped by paper.
- **Tables** cell by cell with column names; every row is also searchable as text
  ("Table 2 | Model: GCN | Cora / Accuracy: 81.5"). `suspicious` tables had an ambiguous layout:
  check their numbers in the page text before quoting.
- **Figures** as PNG images with caption and kind (diagram / plot / example). Passages that mention a
  figure list it under the text: `[figure 2 (diagram): caption -> <png path>]`.
- **Full page text** of every paper (`paper fulltext`), for exact wording, page numbers and anything
  the sections do not show.
- **Cards** (optional, per paper): structured summary + tags + benchmark numbers. A card only helps
  *find* papers. Base advice on the passages, not on card text alone.

## How to run it

Every command has this form. Run it **from the user's project folder** (so its workspace is used):

Bash:
```bash
"/c/Projects/paper-analyzer/.venv/Scripts/python.exe" "/c/Projects/paper-analyzer/paper.py" <command> [options]
```
PowerShell:
```powershell
& "C:\Projects\paper-analyzer\.venv\Scripts\python.exe" "C:\Projects\paper-analyzer\paper.py" <command> [options]
```
Below, `paper <command>` is short for that full line. `paper <command> --help` shows all options.
Each search call loads the models first (~10-30 s). That is normal. Log messages go to stderr; ignore them.

Start with `paper workspace` and `paper stats`.

## Commands

| Command | What it does |
|---|---|
| `paper stats` | Paper counts, section coverage, text coverage, saved topics |
| `paper my-project` | The user's own project description for this workspace; read it before advising |
| `paper search "query"` | Best papers + their best passages (+ figures they mention). Default section `methods`. `--section results`, `--section limitations,discussion`, `--section tables`, `--section all`. `--k 10` papers, `--per-paper 3`, `--year-min 2020`, `--chars 0` (full passages) |
| `paper passage <passage_id> --neighbors 1` | Read a passage in full, plus the passages around it |
| `paper fulltext "words"` | Keyword search over the complete page text, with page numbers |
| `paper approaches "query"` | One line per *method* that fits the query (carded papers only). Filters: `--model-family transformer`, `--data-type graph`, `--task forecasting` |
| `paper limitations "query"` | Where approaches fail (limitations/discussion/conclusion) |
| `paper paper <paper_id>` | Metadata, card, sections, all tables (cell by cell) and figures (caption + PNG). `--sections all --chars 0` for all text |
| `paper figure <paper_id> <n>` | One figure: caption, kind, PNG path, and the passages that cite it |
| `paper list --title "word"` | List papers (most cited first) |
| `paper similar <paper_id>` | Most similar papers |
| `paper compare --dataset X --metric Y` | Benchmark leaderboard from cards. `verified=0` rows were not found in the paper text: double-check before quoting |
| `paper card <paper_id>` | The card: problem, method, key idea, tags, evidence quotes, results |
| `paper foundational --topic "X"` | The papers the workspace cites most: the field's foundations |
| `paper followups <paper_id>` | Newer papers that cite it (`--external` asks Semantic Scholar too) |
| `paper trends "transformer, LSTM"` | Papers per year mentioning each term in their methods |
| `paper audit [paper_id]` | Storage check: text coverage (target >= 0.97), suspicious tables, unmatched figures |
| `paper cards-status` | How many papers have cards |
| `paper fetch "keywords" --n 50` | Download + index new papers into this workspace. **Slow (10-30 min). Ask the user first.** |
| `paper update` | Fetch new papers for all saved topics of this workspace. Ask first. |
| `paper ui` | Opens the user's browser UI (search, papers, tables, figures, collection) |

## Looking at figures

When a passage you rely on cites a figure that matters for the answer (an architecture diagram, a
results plot), open its PNG (the path printed after the passage) with your file/image reading tool
and look at it. Numbers read off a plot are approximate: say "read from Figure n, approximate" and
never mix them with numbers from tables or text.

## Answering "is there a better way to do X?"

1. `paper my-project` to learn the user's current approach (if it is empty, ask the user briefly).
2. If cards exist: `paper approaches "<problem>"` for an overview of candidate methods.
3. `paper search` on `methods` with 2-4 different phrasings (technical terms, synonyms, known method
   names). Use `paper passage <id> --neighbors 1` when a passage is cut or needs context.
   `paper limitations` for known failure modes. Look at the architecture figure of the strongest candidates.
4. For the strongest candidates: `paper search "<method> results" --section tables,results` or
   `paper compare` for numbers, `paper followups <id>` to check whether a newer paper improved on it.
5. Answer with a table: approach | key idea | reported results | pros / cons | code | paper_id + title.
   Then give a concrete recommendation relative to the user's current approach.

Rules:
- Cite paper_id and title for every claim. Only quote numbers that the tool printed, and say so when
  a number is marked unverified or comes from a suspicious table.
- If searches return little or nothing relevant, say the workspace does not cover it and offer
  `paper fetch "<keywords>"`.
- Chat replies in the user's language. **Written reports are always in English**, whatever language
  the chat is in (paper terms stay exact).

## Making paper cards (when the user asks, e.g. "make cards for 10 papers")

Cards power `approaches`, `compare` and `card`. For each batch:
1. `paper uncarded --n 2` prints the full text of papers without cards (tables and appendix included).
2. Read each paper completely and write a JSON card matching `paper card-schema`:
   - dataset and metric names exactly as in the paper;
   - `results`: numbers exactly as printed in the result tables, for the proposed method
     (`is_proposed: true`) and the strongest baselines. Skip a number if you are not sure which
     method/dataset/metric it belongs to (`paper paper <id>` shows the tables cell by cell);
   - `evidence`: one sentence copied word for word from the paper for method, key_idea,
     main_results, limitations;
   - `limitations`: keep conditions and caveats ("only for static graphs").
3. Save it by piping the JSON in:
   ```bash
   cat <<'EOF' | "/c/Projects/paper-analyzer/.venv/Scripts/python.exe" "/c/Projects/paper-analyzer/paper.py" save-card <paper_id>
   {"problem": "...", "method": "...", "key_idea": "...", "datasets": ["..."], "metrics": ["..."],
    "main_results": "...", "limitations": "...", "future_work": "...",
    "tags": {"tasks": ["..."], "data_types": ["..."], "model_families": ["..."]},
    "evidence": {"method": "...", "key_idea": "...", "main_results": "...", "limitations": "..."},
    "results": [{"task": "...", "dataset": "...", "metric": "...", "value": 0.0, "method_name": "...",
                 "is_proposed": true, "higher_is_better": true}]}
   EOF
   ```
   (Or write the JSON to a file and use `save-card <paper_id> --file card.json`.)
4. The reply lists `unverified_numbers` and `evidence_quotes_found` (checked against the full page
   text). If numbers are unverified or quotes were not found, re-check them and save the card again.
5. Repeat until the requested number is done.

## Do not
- Do not edit files in `C:\Projects\paper-analyzer` or delete anything in `library/` or `workspaces/`.
  Only use the commands above (`init` may write `.paper-workspace` into the user's project folder).
- Do not run `fetch`, `update`, `ingest` or `cards-api` without the user's OK (slow, or costs money).
