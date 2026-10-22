"""Leakage safe cross validation for event data whose labels span several days.

Each observation i has an information interval [t0_i, t1_i]: t0 is the event date (features known before
it), t1 is the last day used by its label (event date + 2 trading days for CAR[0,+2]).

``PurgedWalkForward`` (the main scheme): time ordered test blocks; the model for a block is trained only on
observations whose *label* ended before the block starts (purging) and which started at least ``embargo``
before it. Optionally a rolling rather than expanding training window.

``PurgedKFold`` (Lopez de Prado, AFML ch. 7): k contiguous folds; training rows whose interval overlaps the
test interval are purged and rows starting within ``embargo`` after the test block are embargoed. Used for
hyperparameter search inside a training window, never for the headline result.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _as_dt(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x)).to_numpy().astype("datetime64[ns]")


@dataclass
class PurgedWalkForward:
    n_splits: int = 6
    min_train_frac: float = 0.35
    embargo: pd.Timedelta = pd.Timedelta(days=5)
    rolling: pd.Timedelta | None = None

    def boundaries(self, t0) -> list[tuple[np.datetime64, np.datetime64]]:
        t = np.sort(_as_dt(t0))
        start = t[int(len(t) * self.min_train_frac)]
        edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(t[-1]), periods=self.n_splits + 1).to_numpy().copy()
        edges[-1] = edges[-1] + np.timedelta64(1, "D")  # include the last observation
        return list(zip(edges[:-1], edges[1:]))

    def split(self, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0, t1 = _as_dt(t0), _as_dt(t1)
        emb = np.timedelta64(self.embargo)
        for lo, hi in self.boundaries(t0):
            test = np.where((t0 >= lo) & (t0 < hi))[0]
            train_mask = (t1 < lo) & (t0 < lo - emb)
            if self.rolling is not None:
                train_mask &= t0 >= lo - np.timedelta64(self.rolling)
            train = np.where(train_mask)[0]
            if len(test) and len(train):
                yield train, test


@dataclass
class PurgedKFold:
    n_splits: int = 5
    embargo: pd.Timedelta = pd.Timedelta(days=5)

    def split(self, t0, t1) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        t0, t1 = _as_dt(t0), _as_dt(t1)
        order = np.argsort(t0, kind="stable")
        emb = np.timedelta64(self.embargo)
        for block in np.array_split(order, self.n_splits):
            lo, hi = t0[block].min(), t1[block].max()
            overlaps = (t0 <= hi) & (t1 >= lo)
            embargoed = (t0 > hi) & (t0 <= hi + emb)
            train = np.where(~overlaps & ~embargoed)[0]
            yield train, np.sort(block)


def assert_no_leakage(t0, t1, train: np.ndarray, test: np.ndarray) -> None:
    """Raise if any training label interval overlaps the test period."""
    t0, t1 = _as_dt(t0), _as_dt(t1)
    lo, hi = t0[test].min(), t1[test].max()
    bad = (t1[train] >= lo) & (t0[train] <= hi)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} training rows overlap the test window")
