"""Lazy-loaded local models and the Chroma client (loaded once per process)."""
import sys
import time
from functools import lru_cache

from . import config

SKIP_FILES = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite"]


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(None)
def local_model(repo: str) -> str:
    """Download a model once (resuming after dropped connections) and return its local folder."""
    from pathlib import Path

    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    # Weights only appear in the snapshot once fully downloaded, so a snapshot with weights is usable
    # offline. (snapshot_download(local_files_only=True) would refuse it, because the files we skip
    # on purpose - onnx, pytorch_model.bin - count as missing.)
    cache = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    ref = cache / "refs" / "main"
    if ref.exists():
        path = cache / "snapshots" / ref.read_text().strip()
        if any(w.stat().st_size > 0 for w in [*path.glob("*.safetensors"), *path.glob("*.bin")]):
            return str(path)
    for attempt in range(30):
        try:
            files = list_repo_files(repo)
            skip = SKIP_FILES + (["*.bin"] if any(f.endswith(".safetensors") for f in files) else [])
            print(f"Downloading model {repo} (one time)...", file=sys.stderr, flush=True)
            return snapshot_download(repo, ignore_patterns=skip)
        except Exception as e:  # dropped connection: partial files are kept and resumed
            print(f"  download interrupted ({e.__class__.__name__}), resuming [{attempt + 1}/30]",
                  file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"could not download {repo}; check the internet connection and run again")


@lru_cache(None)
def chunk_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(local_model(config.CHUNK_MODEL), device=device())


@lru_cache(None)
def paper_embedder():
    from sentence_transformers import SentenceTransformer, models
    word = models.Transformer(local_model(config.PAPER_MODEL), max_seq_length=512)
    dim = getattr(word, "get_embedding_dimension", None) or word.get_word_embedding_dimension  # renamed in v5
    pool = models.Pooling(dim(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool], device=device())


@lru_cache(None)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(local_model(config.RERANK_MODEL), device=device(), max_length=512)


def chroma():
    """Vector store of the active workspace."""
    return _chroma(str(config.ws().chroma_dir))


@lru_cache(None)
def _chroma(path: str):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def chunk_collection():
    return chroma().get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})


def paper_collection():
    return chroma().get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})


def approach_collection():
    """One vector per paper describing only its method (built from the paper card)."""
    return chroma().get_or_create_collection("approaches", metadata={"hnsw:space": "cosine"})


def embed_chunks(texts: list[str]):
    return chunk_embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str):
    return chunk_embedder().encode([config.CHUNK_QUERY_PREFIX + text], normalize_embeddings=True)[0]


def embed_papers(texts: list[str]):
    return paper_embedder().encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
