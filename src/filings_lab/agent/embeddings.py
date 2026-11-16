"""Paragraph embeddings for dense retrieval, cached in the lake.

Filings repeat themselves: boilerplate recurs across quarters and companies, so the cache is keyed by a
hash of the paragraph text and each distinct text is embedded once. A rebuild after new filings arrive
embeds only the texts it has not seen. Vectors are unit length, so a dot product is the cosine similarity.

``FastEmbedder`` runs a small ONNX sentence encoder on CPU (no PyTorch in the serving image).
``HashEmbedder`` needs no model at all: hashed character trigrams. It exists so tests and CI exercise the
whole dense path offline; it checks the plumbing and says nothing about retrieval quality.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.name, self.batch_size = model, batch_size
        self.model = TextEmbedding(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _unit(np.asarray(list(self.model.embed(texts, batch_size=self.batch_size)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        # the encoder was trained with an instruction in front of queries; query_embed adds it
        return _unit(np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32))


class HashEmbedder:
    """Character trigram counts hashed into ``dim`` buckets. Deterministic, dependency free, for tests."""

    def __init__(self, dim: int = 512):
        self.name, self.dim = f"hash-{dim}", dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            v[int.from_bytes(hashlib.blake2b(t[i:i + 3].encode(), digest_size=4).digest(), "little") % self.dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _unit(np.stack([self._one(t) for t in texts])) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _unit(self._one(text))


def get_embedder(name: str | None = None) -> Embedder:
    name = name or os.environ.get("FLAB_EMBED_MODEL", DEFAULT_MODEL)
    if name.startswith("hash"):
        return HashEmbedder(int(name.split("-")[1]) if "-" in name else 512)
    return FastEmbedder(name)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:20]


def cache_path(lake: str, model: str) -> Path:
    return Path(lake) / "gold" / "embeddings" / f"{model.replace('/', '__')}.npz"


def load_cache(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        return [], np.zeros((0, 0), dtype=np.float32)
    z = np.load(path, allow_pickle=False)
    return z["keys"].tolist(), z["vectors"].astype(np.float32)


def embed_texts(texts: list[str], embedder: Embedder, path: Path | None = None, chunk: int = 2048,
                log=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Embed ``texts`` through the cache at ``path``.

    Returns ``(unique_vectors, inverse, stats)`` with ``unique_vectors[inverse[i]]`` the vector of ``texts[i]``,
    so a corpus that repeats itself costs memory and time only for its distinct texts.
    """
    keys = [text_key(t) for t in texts]
    order: dict[str, int] = {}
    first_text: list[str] = []
    inverse = np.empty(len(texts), dtype=np.int64)
    for i, k in enumerate(keys):
        j = order.get(k)
        if j is None:
            j = order[k] = len(first_text)
            first_text.append(texts[i])
        inverse[i] = j
    cached_keys, cached_vecs = load_cache(path) if path else ([], np.zeros((0, 0), dtype=np.float32))
    have = {k: i for i, k in enumerate(cached_keys)}
    missing = [k for k in order if k not in have]
    t0 = time.perf_counter()

    def save() -> None:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            # half precision on disk: retrieval scores move in the fourth decimal, the file halves
            np.savez(tmp, keys=np.asarray(cached_keys), vectors=cached_vecs.astype(np.float16))
            tmp.replace(path)

    for n, a in enumerate(range(0, len(missing), chunk), start=1):
        part = missing[a:a + chunk]
        vecs = embedder.embed_documents([first_text[order[k]] for k in part])
        cached_vecs = vecs if cached_vecs.size == 0 else np.concatenate([cached_vecs, vecs])
        cached_keys = cached_keys + part
        if n % 4 == 0:
            save()  # an interrupted build resumes from here instead of starting over
        if log:
            log(f"embedded {min(a + chunk, len(missing))} of {len(missing)} new texts")
    seconds = time.perf_counter() - t0
    if missing:
        save()
        have = {k: i for i, k in enumerate(cached_keys)}
    unique = cached_vecs[[have[k] for k in order]] if order else np.zeros((0, 0), dtype=np.float32)
    stats = {"texts": len(texts), "distinct": len(order), "embedded_now": len(missing), "seconds": round(seconds, 1),
             "model": embedder.name, "dim": int(unique.shape[1]) if unique.size else 0}
    return np.ascontiguousarray(unique, dtype=np.float32), inverse, stats
