"""Batch drift report: has the population of recent filings moved away from what the model was trained on?

The live service watches drift on its own traffic (``serving.drift.DriftMonitor``). This job does the same
for the data itself: it scores the most recent months of filings with the production model and compares
each monitored feature and the prediction against the reference distributions stored in the model artefact.
It recommends retraining when any monitored feature has PSI above 0.25 or the prediction above 0.2, and
can push the numbers to a Prometheus Pushgateway so the same Grafana alerts cover it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import ks_pvalue, numeric_psi

FEATURE_LIMIT = 0.25
PREDICTION_LIMIT = 0.20


def drift_report(model, features: pd.DataFrame, since: pd.Timestamp) -> dict:
    recent = features[pd.to_datetime(features["event_date"]) >= since]
    if recent.empty:
        raise ValueError(f"no filings since {since.date()}")
    ref = model.reference.get("numeric", {})
    rows = []
    preds = model.predict(recent)
    for name, r in ref.items():
        values = preds if name == "prediction" else recent[name].to_numpy() if name in recent else np.array([])
        psi = numeric_psi(r, values)
        limit = PREDICTION_LIMIT if name == "prediction" else FEATURE_LIMIT
        rows.append({"feature": name, "psi": psi, "ks_p": ks_pvalue(r, values), "limit": limit,
                     "breached": bool(np.isfinite(psi) and psi > limit), "n": int(np.isfinite(pd.to_numeric(pd.Series(values), errors="coerce")).sum())})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False)
    breached = table[table["breached"]]["feature"].tolist()
    return {"model_version": model.version, "since": str(since.date()), "filings": int(len(recent)),
            "retrain_recommended": bool(breached), "breached": breached, "table": table}


def render(rep: dict) -> str:
    t = rep["table"].copy()
    t["psi"] = t["psi"].round(4)
    t["ks_p"] = t["ks_p"].round(4)
    verdict = ("Retraining recommended: " + ", ".join(rep["breached"])) if rep["retrain_recommended"] else "No drift beyond the limits."
    return (f"# Drift report\n\nModel version {rep['model_version']}, {rep['filings']} filings since {rep['since']}.\n\n"
            f"**{verdict}**\n\n" + t.to_markdown(index=False) + "\n")


def push(rep: dict, gateway: str, job: str = "filings_drift_report") -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    g = Gauge("flab_batch_feature_psi", "PSI of recent filings against training", ["feature"], registry=reg)
    for r in rep["table"].itertuples(index=False):
        if np.isfinite(r.psi):
            g.labels(r.feature).set(r.psi)
    Gauge("flab_retrain_recommended", "1 when the drift report recommends retraining", registry=reg).set(float(rep["retrain_recommended"]))
    push_to_gateway(gateway, job=job, registry=reg)


def save(rep: dict, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift.md").write_text(render(rep))
    (out / "drift.json").write_text(json.dumps({**{k: v for k, v in rep.items() if k != "table"},
                                                "table": rep["table"].to_dict("records")}, indent=2, default=float))
