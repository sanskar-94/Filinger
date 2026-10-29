"""BM25 over paragraphs as a sparse matrix product: one CSR multiply scores every paragraph for a query.

Filters (ticker, form, date range) are applied as a boolean mask before ranking, so a company scoped
question never sees another company's text. An optional diversity step (maximal marginal relevance on
paragraph token sets) stops the top results from all coming from one boilerplate block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b|\b\d+(?:\.\d+)?%?"


class BM25Index:
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

    def _mask(self, ticker=None, form=None, date_from=None, date_to=None, accession=None) -> np.ndarray:
        m = np.ones(len(self.meta), dtype=bool)
        if ticker:
            m &= self.meta["ticker"].str.upper().to_numpy() == ticker.upper()
        if form:
            m &= self.meta["form"].str.upper().to_numpy() == form.upper()
        if accession:
            m &= self.meta["accession"].to_numpy() == accession
        dates = pd.to_datetime(self.meta["date_filed"]).to_numpy()
        if date_from:
            m &= dates >= np.datetime64(pd.Timestamp(date_from))
        if date_to:
            m &= dates <= np.datetime64(pd.Timestamp(date_to))
        return m

    def search(self, query: str, k: int = 8, diversify: bool = True, **filters) -> pd.DataFrame:
        q = self.vec.transform([query])
        q.data[:] = 1.0  # each query term counts once
        scores = np.asarray(self.matrix @ q.T.toarray()).ravel()
        scores[~self._mask(**filters)] = -1.0
        cand = np.argsort(-scores)[: max(k * 5, 20)]
        cand = cand[scores[cand] > 0]
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
