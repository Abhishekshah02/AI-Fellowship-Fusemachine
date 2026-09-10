"""RAG pipeline: ingest -> chunk -> embed -> Chroma -> retrieve.

Chroma's default embedding function is all-MiniLM-L6-v2 running on
onnxruntime, so embeddings need no API key, no GPU and no torch. The model is
downloaded once on first use and baked into the Docker image.
"""
from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from pathlib import Path

import chromadb

from .config import settings

_HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # imported lazily: only PDFs pay for it

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split on markdown headings first so a chunk keeps one topic, then window
    anything still too long. Cheap, and it beats blind fixed-width slicing on
    structured docs like these."""
    size = size or settings().chunk_size
    overlap = overlap or settings().chunk_overlap

    bounds = [m.start() for m in _HEADING.finditer(text)]
    sections = (
        [text[a:b] for a, b in zip([0] + bounds, bounds + [len(text)])] if bounds else [text]
    )

    out: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= size:
            out.append(section)
            continue
        step = max(1, size - overlap)
        for start in range(0, len(section), step):
            piece = section[start : start + size].strip()
            if piece:
                out.append(piece)
            if start + size >= len(section):
                break
    return out


@lru_cache
def collection():
    s = settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(s.chroma_dir))
    return client.get_or_create_collection(
        name=s.collection, metadata={"hnsw:space": "cosine"}
    )


def ingest(kb_dir: Path | None = None) -> dict:
    """(Re)build the index from disk. Idempotent: ids are file#position, so
    re-running replaces chunks instead of duplicating them."""
    s = settings()
    kb_dir = kb_dir or s.kb_dir
    col = collection()

    ids, docs, metas = [], [], []
    files = sorted(
        p for p in kb_dir.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}
    )
    for path in files:
        for i, piece in enumerate(chunk(read_document(path))):
            ids.append(f"{path.name}#{i}")
            docs.append(piece)
            metas.append({"source": path.name, "chunk": i})

    if ids:
        # upsert in batches; Chroma embeds them with the bundled ONNX MiniLM
        for i in range(0, len(ids), 128):
            sl = slice(i, i + 128)
            col.upsert(ids=ids[sl], documents=docs[sl], metadatas=metas[sl])

    return {"files": len(files), "chunks": len(ids), "collection_size": col.count()}


def search(query: str, k: int | None = None) -> list[dict]:
    k = k or settings().top_k
    col = collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "id": f"{meta['source']}#{meta['chunk']}",
                "source": meta["source"],
                "text": doc,
                "score": round(1 - float(dist), 4),  # cosine distance -> similarity
            }
        )
    return hits


async def asearch(query: str, k: int | None = None) -> list[dict]:
    """Chroma is sync; keep the event loop free under concurrent requests."""
    return await asyncio.to_thread(search, query, k)


def as_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['id']}]\n{h['text']}" for h in hits)
