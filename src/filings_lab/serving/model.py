"""The production reaction model artefact: a LightGBM booster, its feature list, the training reference
distributions for drift detection and the walk forward metrics it earned."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import reference_stats

DRIFT_FEATURES = 12  # monitor the most important features plus the prediction itself


class ReactionModel:
    def __init__(self, booster, features: list[str], meta: dict):
        self.booster, self.features, self.meta = booster, features, meta

    @classmethod
    def load(cls, path: str | Path) -> ReactionModel:
        import lightgbm as lgb

        p = Path(path)
        booster = lgb.Booster(model_file=str(p / "model.txt"))
        meta = json.loads((p / "meta.json").read_text())
        return cls(booster, meta["features"], meta)

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "unregistered"))

    def predict(self, rows: list[dict] | pd.DataFrame) -> np.ndarray:
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        X = df.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        return self.booster.predict(X)

    @property
    def reference(self) -> dict:
        return self.meta.get("reference", {})

    def explain(self, features: dict, top: int = 5) -> dict:
        """Probability plus the features that moved it most (LightGBM TreeSHAP contributions)."""
        X = pd.DataFrame([features]).reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        p = float(self.booster.predict(X)[0])
        contrib = self.booster.predict(X, pred_contrib=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:top]
        drivers = [{"feature": self.features[i], "value": _num(features.get(self.features[i])),
                    "contribution": float(contrib[i])} for i in order]
        return {"probability_up": p, "model_version": self.version, "drivers": drivers,
                "missing_features": int(X.isna().sum(axis=1).iloc[0])}


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def build_artifact(features_df: pd.DataFrame, out_dir: str | Path, feature_set: str = "all", label: str = "label_up",
                   num_rounds: int | None = None, metrics: dict | None = None) -> Path:
    """Fit the final model on all labelled history and write a self describing artefact directory."""
    import lightgbm as lgb

    from ..modeling.train import LGB_PARAMS, feature_sets

    df = features_df.dropna(subset=[label]).sort_values("event_date")
    cols = feature_sets(df.columns)[feature_set]
    rounds = num_rounds or int(np.median((metrics or {}).get("rounds", [300])))
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(df[cols], df[label].astype(int)), rounds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    gain = pd.Series(booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
    drift_cols = list(gain.index[:DRIFT_FEATURES])
    ref_df = df[drift_cols].copy()
    ref_df["prediction"] = booster.predict(df[cols])
    meta = {
        "features": cols, "feature_set": feature_set, "label": label, "num_rounds": rounds,
        "trained_through": str(pd.to_datetime(df["event_date"]).max().date()), "n_train": int(len(df)),
        "metrics": metrics or {}, "reference": reference_stats(ref_df, drift_cols + ["prediction"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return out
