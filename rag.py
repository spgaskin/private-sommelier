"""
rag.py - a tiny, fully local RAG: facts live in retrieval, not in weights.

Mirrors the report's architecture decision: tune the model for BEHAVIOR, keep
FACTS in a retrieval layer that stays governable and updatable. Here that layer
is a local Ollama embedding model + cosine search in numpy. No vector DB, no
external service - the whole index is a single .npz file on disk.

Usage:
    uv run rag.py ingest --dir ./kb        # build the index from a folder
    uv run rag.py query "what port is billing on?"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

BASE_URL = os.environ.get("PRIVATE_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
EMBED_MODEL = os.environ.get("PRIVATE_LLM_EMBED_MODEL", "nomic-embed-text")
INDEX_PATH = Path(os.environ.get("PRIVATE_LLM_INDEX", "kb_index.npz"))

client = OpenAI(base_url=BASE_URL, api_key="ollama")

TEXT_EXTS = {".md", ".txt", ".csv", ".json", ".log"}


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings with the local embedding model. L2-normalized."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


def _chunk(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Split into overlapping word windows so retrieval lands on focused passages."""
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks


def ingest(directory: str) -> int:
    """Read a folder, chunk + embed every text file, write the index to disk."""
    root = Path(directory).resolve()
    texts: list[str] = []
    sources: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTS:
            body = path.read_text(encoding="utf-8", errors="replace")
            for chunk in _chunk(body):
                texts.append(chunk)
                sources.append(str(path.relative_to(root)))
    if not texts:
        raise SystemExit(f"no text files found under {root}")
    vecs = embed(texts)
    np.savez(
        INDEX_PATH,
        vectors=vecs,
        texts=np.array(texts, dtype=object),
        sources=np.array(sources, dtype=object),
    )
    print(f"ingested {len(texts)} chunks from {len(set(sources))} files -> {INDEX_PATH}")
    return len(texts)


def search(query: str, k: int = 3) -> list[tuple[float, str, str]]:
    """Return the top-k (score, source, passage) for a query. Cosine over the index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"no index at {INDEX_PATH} - run `rag.py ingest --dir ...` first")
    data = np.load(INDEX_PATH, allow_pickle=True)
    vecs, texts, sources = data["vectors"], data["texts"], data["sources"]
    q = embed([query])[0]
    scores = vecs @ q  # both L2-normalized -> cosine similarity
    top = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), str(sources[i]), str(texts[i])) for i in top]


def retrieve(query: str, k: int = 3) -> str:
    """Agent-facing: format the top passages as a single grounded context string."""
    hits = search(query, k)
    return "\n\n".join(f"[source: {src} | score {score:.2f}]\n{text}" for score, src, text in hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny local RAG over a folder.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="Build the index from a folder.")
    p_ing.add_argument("--dir", default="./kb")
    p_q = sub.add_parser("query", help="Search the index.")
    p_q.add_argument("text")
    p_q.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.dir)
    else:
        for score, src, text in search(args.text, args.k):
            print(f"\n[{score:.2f}] {src}\n{text[:300]}")


if __name__ == "__main__":
    main()
