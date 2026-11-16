"""Paragraph retrieval: BM25, dense embeddings, and the two fused.

``BM25Index`` scores every paragraph with one sparse matrix product. ``DenseIndex`` scores them with a dot
product against cached sentence embeddings, which finds a paragraph that says the same thing in other
words ("cut its forecast" against "lowering its guidance"). ``HybridIndex`` merges the two rankings by
reciprocal rank fusion, which needs no score calibration between them. All three share one interface, so
the agent's tools do not know which they hold.

Filters (ticker, form, date range) are applied as a boolean mask before ranking, so a company scoped
question never sees another company's text. An optional diversity step (maximal marginal relevance on
paragraph token sets) stops the top results from all coming from one boilerplate block.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b|\b\d+(?:\.\d+)?%?"
# conversational words of a question that no filing would contain; not worth reporting as missing
QUESTION_WORDS = frozenset("""did does doing say says said tell told talk talked mention mentioned explain explained
describe described discuss discussed company companies filing filings management""".split())


def filter_mask(meta: pd.DataFrame, ticker=None, form=None, date_from=None, date_to=None, accession=None) -> np.ndarray:
    m = np.ones(len(meta), dtype=bool)
    if ticker:
        m &= meta["ticker"].str.upper().to_numpy() == ticker.upper()
    if form:
        m &= meta["form"].str.upper().to_numpy() == form.upper()
    if accession:
        m &= meta["accession"].to_numpy() == accession
    if date_from or date_to:
        dates = pd.to_datetime(meta["date_filed"]).to_numpy()
        if date_from:
            m &= dates >= np.datetime64(pd.Timestamp(date_from))
        if date_to:
            m &= dates <= np.datetime64(pd.Timestamp(date_to))
    return m


def text_groups(meta: pd.DataFrame) -> np.ndarray:
    """One id per distinct paragraph text. Filings repeat boilerplate every quarter; a search result list
    that shows the same risk paragraph ten times has one result, not ten."""
    return pd.factorize(meta["text"].fillna(""))[0]


def recency_order(meta: pd.DataFrame) -> np.ndarray:
    """Paragraph positions, most recently filed first: the tie break between copies of one text."""
    dates = pd.to_datetime(meta["date_filed"]).to_numpy()
    return np.argsort(-dates.astype("int64"), kind="stable")


def ranked(scores: np.ndarray, floor: float, recency: np.ndarray, groups: np.ndarray | None, limit: int) -> np.ndarray:
    """Positions scoring above ``floor``, best first, equal scores most recent first. With ``groups`` only the
    first paragraph of each distinct text is kept."""
    order = recency[np.argsort(-scores[recency], kind="stable")]
    order = order[scores[order] > floor]
    if groups is not None and len(order):
        _, first = np.unique(groups[order], return_index=True)
        order = order[np.sort(first)]
    return order[:limit]


class BM25Index:
    lexical_only = True  # ranks on shared words alone, so a query word no filing uses is worth reporting

    def __init__(self, paragraphs: pd.DataFrame, k1: float = 1.4, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.meta = paragraphs.reset_index(drop=True)
        self.vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, stop_words="english", min_df=1)
        tf = self.vec.fit_transform(self.meta["text"].fillna("")).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.diff(tf.tocsc().indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1))
        tf = tf.tocoo()
        weights = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = sparse.csr_matrix((weights * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self.binary = (self.matrix > 0).astype(np.float32).tocsr()
        self.groups, self.recency = text_groups(self.meta), recency_order(self.meta)

    def _mask(self, **filters) -> np.ndarray:
        return filter_mask(self.meta, **filters)

    def unknown_terms(self, query: str) -> list[str]:
        """Content words of the query that no indexed paragraph contains. BM25 silently ranks on the remaining
        words, so callers use this to say so instead of presenting weak matches as an answer."""
        analyzer = self.vec.build_analyzer()
        vocab = self.vec.vocabulary_
        return list(dict.fromkeys(t for t in analyzer(query)
                                  if t not in vocab and t not in QUESTION_WORDS and not t[0].isdigit()))

    def coverage(self, query: str, positions: np.ndarray) -> np.ndarray:
        """Share of the query's content words that each paragraph contains. A paragraph that shares one word
        of five with the query matched by accident; one that holds four of five is a keyword hit."""
        vocab = self.vec.vocabulary_
        terms = [t for t in dict.fromkeys(self.vec.build_analyzer()(query)) if t not in QUESTION_WORDS]
        known = [vocab[t] for t in terms if t in vocab]
        if not terms or not known or not len(positions):
            return np.zeros(len(positions))
        return np.asarray(self.binary[positions][:, known].sum(axis=1)).ravel() / len(terms)

    def scores(self, query: str, **filters) -> np.ndarray:
        """BM25 score of every paragraph; minus one where a filter excludes it, zero where no word matches."""
        q = self.vec.transform([query])
        q.data[:] = 1.0  # each query term counts once
        scores = np.asarray(self.matrix @ q.T.toarray()).ravel()
        scores[~self._mask(**filters)] = -1.0
        return scores

    def search(self, query: str, k: int = 8, diversify: bool = True, collapse: bool = True, **filters) -> pd.DataFrame:
        scores = self.scores(query, **filters)
        cand = ranked(scores, 0.0, self.recency, self.groups if collapse else None, max(k * 5, 20))
        if diversify and len(cand) > k:
            cand = self._mmr(cand, scores, k)
        out = self.meta.iloc[cand[:k]].copy()
        out["score"] = scores[cand[:k]]
        return out

    def _mmr(self, cand: np.ndarray, scores: np.ndarray, k: int, lam: float = 0.75) -> np.ndarray:
        vecs = self.binary[cand]
        sizes = np.asarray(vecs.sum(axis=1)).ravel() + 1e-9
        overlap = (vecs @ vecs.T).toarray()
        sim = overlap / np.sqrt(np.outer(sizes, sizes))
        rel = scores[cand] / (scores[cand].max() + 1e-9)
        chosen = [0]
        while len(chosen) < k:
            rest = [i for i in range(len(cand)) if i not in chosen]
            mmr = [lam * rel[i] - (1 - lam) * sim[i, chosen].max() for i in rest]
            chosen.append(rest[int(np.argmax(mmr))])
        return cand[chosen]


class DenseIndex:
    """Cosine similarity against cached paragraph embeddings (see ``embeddings.py``)."""

    lexical_only = False

    def __init__(self, paragraphs: pd.DataFrame, embedder=None, lake: str | None = None, log=None):
        from .embeddings import cache_path, embed_texts, get_embedder

        self.meta = paragraphs.reset_index(drop=True)
        self.embedder = embedder or get_embedder()
        path = cache_path(lake, self.embedder.name) if lake else None
        self.unique, self.inverse, self.stats = embed_texts(self.meta["text"].fillna("").tolist(), self.embedder, path, log=log)
        self.groups, self.recency = self.inverse, recency_order(self.meta)
        self._queries: dict[str, np.ndarray] = {}

    def scores(self, query: str, **filters) -> np.ndarray:
        q = self._queries.get(query)
        if q is None:
            q = self._queries[query] = self.embedder.embed_query(query)
        scores = (self.unique @ q)[self.inverse]
        scores[~filter_mask(self.meta, **filters)] = -1.0
        return scores

    def unknown_terms(self, query: str) -> list[str]:
        return []

    def search(self, query: str, k: int = 8, diversify: bool = True, collapse: bool = True, **filters) -> pd.DataFrame:
        scores = self.scores(query, **filters)
        keep = ranked(scores, -1.0, self.recency, self.groups if collapse else None, k)
        out = self.meta.iloc[keep].copy()
        out["score"] = scores[keep]
        return out


class HybridIndex:
    """BM25 and dense rankings merged by reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009).

    Each paragraph scores ``sum over rankers of 1 / (rrf_k + rank)`` over the top ``depth`` of each ranking.
    Only ranks are used, so a BM25 score in the tens and a cosine below one need no common scale.

    Rank fusion has a blind spot: it cannot tell a keyword ranking that found the answer from one that found
    nothing and ranked the noise. When a question is worded unlike the filing, BM25's best paragraphs share
    one incidental word with it, and fusing them in halves the precision of the dense ranking. So a BM25
    result enters the fusion only if it holds at least ``min_coverage`` of the query's content words. Below
    that, keyword overlap is not evidence and the dense ranking stands alone.
    """

    lexical_only = False

    def __init__(self, bm25: BM25Index, dense: DenseIndex, rrf_k: int = 60, depth: int = 50, min_coverage: float = 0.5):
        if len(bm25.meta) != len(dense.meta):
            raise ValueError("both indexes must cover the same paragraphs in the same order")
        self.bm25, self.dense, self.rrf_k, self.depth, self.min_coverage = bm25, dense, rrf_k, depth, min_coverage
        self.meta = bm25.meta

    def unknown_terms(self, query: str) -> list[str]:
        return self.bm25.unknown_terms(query)

    def fused(self, query: str, collapse: bool = True, **filters) -> tuple[np.ndarray, np.ndarray]:
        """(candidate paragraph positions best first, their fused scores)."""
        fused: dict[int, float] = {}
        for index, floor in ((self.bm25, 0.0), (self.dense, -1.0)):
            top = ranked(index.scores(query, **filters), floor, index.recency, index.groups if collapse else None, self.depth)
            if index is self.bm25 and self.min_coverage > 0:
                top = top[self.bm25.coverage(query, top) >= self.min_coverage]
            for rank, i in enumerate(top, start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + rank)
        order = sorted(fused, key=lambda i: (-fused[i], i))
        return np.asarray(order, dtype=np.int64), np.asarray([fused[i] for i in order], dtype=np.float64)

    def search(self, query: str, k: int = 8, diversify: bool = True, collapse: bool = True, **filters) -> pd.DataFrame:
        cand, score = self.fused(query, collapse=collapse, **filters)
        if diversify and len(cand) > k:
            full = np.zeros(len(self.meta))
            full[cand] = score
            cand = self.bm25._mmr(cand[: max(k * 5, 20)], full, k)
            score = full[cand]
        out = self.meta.iloc[cand[:k]].copy()
        out["score"] = score[:k]
        return out


RETRIEVAL_MODES = ("bm25", "dense", "hybrid")


def build_index(paragraphs: pd.DataFrame, mode: str | None = None, lake: str | None = None, embedder=None,
                bm25: BM25Index | None = None, log=None):
    """The index for a retrieval mode: ``bm25`` (default), ``dense`` or ``hybrid``. ``FLAB_RETRIEVAL`` sets
    the default. Dense modes read the embedding cache in the lake and embed whatever it lacks."""
    mode = (mode or os.environ.get("FLAB_RETRIEVAL", "bm25")).lower()
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
    if mode == "bm25":
        return bm25 or BM25Index(paragraphs)
    dense = DenseIndex(paragraphs, embedder=embedder, lake=lake, log=log)
    if mode == "dense":
        return dense
    return HybridIndex(bm25 or BM25Index(paragraphs), dense, min_coverage=float(os.environ.get("FLAB_HYBRID_MIN_COVERAGE", 0.5)))
