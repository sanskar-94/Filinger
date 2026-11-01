"""Hyperparameter search inside each training window with purged k fold.

Tuning on the walk forward test blocks would leak the test into the model choice. Instead every outer fold
draws the same random configurations and scores them with PurgedKFold on its own training window only; the
best one is refit on the whole window and evaluated once on the untouched test block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import PurgedKFold

SPACE = {
    "num_leaves": [7, 15, 31],
    "min_data_in_leaf": [30, 60, 120],
    "feature_fraction": [0.5, 0.7, 0.9],
    "lambda_l2": [1.0, 5.0, 20.0],
    "learning_rate": [0.02, 0.05],
}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(n)]


def select(X: pd.DataFrame, y: np.ndarray, t0, t1, base: dict, configs: list[dict], k: int = 3, rounds: int = 250,
           embargo_days: int = 5) -> tuple[dict, list[float]]:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    cv = PurgedKFold(n_splits=k, embargo=pd.Timedelta(days=embargo_days))
    splits = [(tr, te) for tr, te in cv.split(t0, t1) if len(np.unique(y[te])) == 2 and len(tr) > 50]
    scores = []
    for cfg in configs:
        params = {**base, **cfg}
        aucs = []
        for tr, te in splits:
            m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), rounds)
            aucs.append(roc_auc_score(y[te], m.predict(X.iloc[te])))
        scores.append(float(np.mean(aucs)) if aucs else 0.5)
    return {**base, **configs[int(np.argmax(scores))]}, scores
