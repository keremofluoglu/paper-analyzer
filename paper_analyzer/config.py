"""Central settings. All paths are absolute so paper.py works from any working directory."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Model downloads: plain HTTP (resumable) instead of hf-xet, which stalls on unstable connections
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

ROOT = Path(__file__).resolve().parent.parent

# Shared library: files that belong to a paper, not to a review (each paper is downloaded once)
LIBRARY = ROOT / "library"
PDF_DIR = LIBRARY / "pdfs"          # original papers, kept permanently
TEI_DIR = LIBRARY / "tei"           # GROBID output per paper
FIGURE_DIR = LIBRARY / "figures"    # <paper_id>/fig<n>.png
for _d in (PDF_DIR, TEI_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Workspaces: one per literature review (own paper list, index, cards, reports)
WORKSPACES = ROOT / "workspaces"
MARKER = ".paper-workspace"         # file in the user's project folder naming its workspace
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


class WorkspaceError(Exception):
    pass


@dataclass(frozen=True)
class Workspace:
    name: str

    @property
    def path(self) -> Path:
        return WORKSPACES / self.name

    db_path = property(lambda self: self.path / "papers.db")
    chroma_dir = property(lambda self: self.path / "chroma")
    topics_path = property(lambda self: self.path / "topics.json")
    my_project_path = property(lambda self: self.path / "my_project.md")
    reports_dir = property(lambda self: self.path / "reports")
    batch_file = property(lambda self: self.path / "card_batches.json")


_current: Workspace | None = None


def list_workspaces() -> list[str]:
    return sorted(p.name for p in WORKSPACES.glob("*") if p.is_dir())


def create_workspace(name: str) -> Workspace:
    if not NAME_RE.match(name):
        raise WorkspaceError(f"invalid workspace name {name!r}: use lowercase letters, digits, - and _")
    w = Workspace(name)
    for d in (w.path, w.chroma_dir, w.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not w.my_project_path.exists():
        w.my_project_path.write_text("", encoding="utf-8")
    return w


def select_workspace(name: str) -> Workspace:
    global _current
    if name not in list_workspaces():
        raise WorkspaceError(f"workspace {name!r} does not exist. Existing: {', '.join(list_workspaces()) or 'none'}. "
                             f"Create it with: paper.py init {name}")
    _current = Workspace(name)
    return _current


def find_marker(start: Path | None = None) -> tuple[str, Path] | None:
    """The workspace named by a .paper-workspace file in START (default: cwd) or a parent folder."""
    for folder in [start or Path.cwd(), *(start or Path.cwd()).parents]:
        marker = folder / MARKER
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip(), marker
    return None


def ws() -> Workspace:
    """The active workspace: --workspace option > PAPER_WORKSPACE env > .paper-workspace marker.
    Never a silent default, so nothing is ever written into the wrong literature review."""
    if _current:
        return _current
    if os.environ.get("PAPER_WORKSPACE"):
        return select_workspace(os.environ["PAPER_WORKSPACE"])
    found = find_marker()
    if found:
        return select_workspace(found[0])
    raise WorkspaceError(
        "No workspace selected. In the project folder run `paper.py init <name>` once (creates a "
        f"{MARKER} file there), or pass --workspace <name>. Existing: {', '.join(list_workspaces()) or 'none'}")

# Models (all run locally; downloaded from Hugging Face on first use)
CHUNK_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PAPER_MODEL = "allenai/specter2_base"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Chunking (words, ~1.3 tokens per word). Small chunks are searched; passages are returned.
CHUNK_WORDS = 200       # search unit, built from whole paragraphs
PASSAGE_WORDS = 1200    # reading unit: a subsection, split at paragraph boundaries if longer
PASSAGES_PER_PAPER = 2  # search returns this many passages per paper

# Search
CANDIDATES = 50          # per retriever before fusion
RRF_K = 60
CITATION_BOOST = 0.05    # weight of log(1 + citations) added to rerank score
VENUE_BOOST = 0.3        # bonus for papers from a whitelisted venue
VENUE_WHITELIST = [
    "neurips", "neural information processing", "icml", "iclr", "cvpr", "iccv", "eccv",
    "acl", "emnlp", "naacl", "aaai", "ijcai", "kdd", "www", "sigir", "jmlr", "tpami",
    "transactions on pattern analysis", "nature", "science",
]

# Collection / quality filter
MIN_CITATIONS_PER_YEAR = 5      # for papers pulled in via citation expansion
MIN_CITATIONS_TOTAL = 50        # alternative: enough total citations
EXPAND_PER_QUERY = 50           # max referenced papers added per fetch
ARXIV_DELAY = 3.0
S2_DELAY = 1.1
S2_API_KEY = os.environ.get("S2_API_KEY")  # optional, gives higher rate limits

GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
# ingest starts GROBID in this WSL distro (Docker inside Ubuntu) and stops it afterwards; "0" disables
GROBID_AUTO = os.environ.get("GROBID_AUTO", "1") != "0"
GROBID_WSL_DISTRO = os.environ.get("GROBID_WSL_DISTRO", "Ubuntu-24.04")
GROBID_IMAGE = "lfoppiano/grobid:0.8.1"
GROBID_CONTAINER = "paper-grobid"

# Paper cards via Claude API (optional; default is filling cards from a Claude Code chat)
CARD_MODEL = os.environ.get("PAPER_CARD_MODEL", "claude-haiku-4-5")
