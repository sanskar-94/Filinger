"""Star schema extracts for Power BI or Tableau (CSV for portability, Parquet for size).

Facts: experiment exposures and filing events (with out of fold model predictions).
Dimensions: company, date, variant. Pre aggregated tables keep the dashboard fast and make the numbers in
the BI tool identical to the Python readout (the summary table is written from the same readout object).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assign import Experiment
from .log import EventLog


def _write(df: pd.DataFrame, out: Path, name: str) -> Path:
    df.to_csv(out / f"{name}.csv", index=False)
    return out / f"{name}.csv"


def export_all(lake: str, db: str, experiment: str, phase2_dir: str, out_dir: str, readout_path: str | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    lake = lake.rstrip("/")
    idx = pd.read_parquet(f"{lake}/bronze_index.parquet")
    synth_companies = Path(lake).parent / "synthetic" / "companies.parquet"
    comp = idx.groupby(["cik", "ticker", "company"], as_index=False)["sic"].first()
    if synth_companies.exists():
        comp = comp.merge(pd.read_parquet(synth_companies)[["cik", "sector"]], on="cik", how="left")
    paths.append(_write(comp, out, "dim_company"))

    # filing events with out of fold predictions (Phase 2)
    ev = pd.read_parquet(f"{lake}/gold/events")
    oof_path = Path(phase2_dir) / "oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)[["accession", "pred", "fold"]]
        ev = ev.merge(oof, on="accession", how="left")
        month = pd.to_datetime(ev["event_date"]).dt.to_period("M")
        ev["pred_quintile"] = ev.groupby(month)["pred"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if s.notna().sum() >= 5 else np.nan)
    ev = ev.merge(comp[["cik", "company"] + (["sector"] if "sector" in comp else [])], on="cik", how="left")
    keep = [c for c in ["accession", "cik", "ticker", "company", "sector", "form", "event_date", "ret_0", "car", "car_mm",
                        "label_up", "pred", "pred_quintile", "fold"] if c in ev.columns]
    paths.append(_write(ev[keep], out, "fact_filing_events"))

    if oof_path.exists():
        from ..modeling.stats import long_short, monthly_ic

        o = ev.dropna(subset=["pred"])
        ic = monthly_ic(o, "pred", "car").rename("ic")
        ls = long_short(o, "pred", "car").rename("long_short")
        cnt = o.groupby(pd.to_datetime(o["event_date"]).dt.to_period("M")).size().rename("events")
        m = pd.concat([cnt, ic, ls], axis=1).reset_index().rename(columns={"index": "month", "event_date": "month"})
        m["month"] = m["month"].astype(str)
        paths.append(_write(m, out, "agg_model_monthly"))
        q = o.dropna(subset=["pred_quintile"]).groupby("pred_quintile")["car"].agg(["count", "mean", "std"]).reset_index()
        q = pd.DataFrame({"quintile": q["pred_quintile"].astype(int), "events": q["count"],
                          "mean_car_bps": q["mean"] * 1e4, "se_bps": q["std"] / np.sqrt(q["count"]) * 1e4})
        paths.append(_write(q, out, "model_quintiles"))
    for src, name in (("backtest_monthly.csv", "model_backtest"), ("backtest_cost_curve.csv", "model_cost_curve"),
                      ("calibration.csv", "model_calibration")):
        if (Path(phase2_dir) / src).exists():
            t = pd.read_csv(Path(phase2_dir) / src)
            if name == "model_backtest":
                t = t.assign(gross_bps=t["gross"] * 1e4, net_bps=t["net"] * 1e4, cum_net_bps=t["cum_net"] * 1e4)
            paths.append(_write(t, out, name))
    metrics_path = Path(phase2_dir) / "metrics.json"
    if metrics_path.exists():
        res = json.loads(metrics_path.read_text())["results"]
        labels = {"market_meta": "Market and filing type", "market_fundamentals": "Market and fundamentals",
                  "text_only": "Filing text only", "all": "All features (LightGBM)", "all_tuned": "All features, tuned", "logistic_all": "All features (logistic)",
                  "llm_only": "LLM signals only", "all_plus_llm": "All plus LLM signals"}
        rows = [{"feature_set": k, "label": labels.get(k, k), "events": v["n"], "auc": v["auc"]["auc"],
                 "auc_low": v["auc"]["ci_low"], "auc_high": v["auc"]["ci_high"], "ic": v["ic"]["mean"], "ic_t": v["ic"]["t"],
                 "long_short_bps": v["long_short_bps"]["mean"], "long_short_t": v["long_short_bps"]["t"]} for k, v in res.items()]
        paths.append(_write(pd.DataFrame(rows), out, "model_feature_sets"))

    # experiment
    exp = Experiment.load(experiment)
    df = EventLog(db).frame(exp.name)
    readout = json.loads(Path(readout_path).read_text()) if readout_path and Path(readout_path).exists() else None
    cut = (readout or {}).get("primary", {}).get("truncated_at_ts")
    if cut is not None:  # the test stopped at its maximum sample: report exactly the data the decision used
        df = df[df["ts"] <= cut]
    if not df.empty:
        df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date.astype(str)
        seed = df["user_id"].str.extract(r"^sim-user-(\d+)-")[0]
        df["period"] = (pd.to_numeric(seed, errors="coerce").fillna(0).astype(int) + 1)
        df["user_key"] = df["user_id"].map(lambda u: hashlib.sha256(u.encode()).hexdigest()[:12])  # no raw ids in BI
        df["rated"] = df["rating"].notna().astype(int)
        df["thumbs_up"] = (df["rating"] == 1).astype(int)
        fact = df[["request_id", "date", "variant", "user_key", "ticker", "mode", "latency_s", "n_citations",
                   "citation_valid_rate", "cost_usd", "rated", "thumbs_up"]]
        paths.append(_write(fact, out, "fact_ab_exposures"))
        daily = df.groupby(["date", "variant"]).agg(period=("period", "max"), exposures=("request_id", "count"), rated=("rated", "sum"),
                                                    thumbs_up=("thumbs_up", "sum"),
                                                    latency_p95_s=("latency_s", lambda s: s.quantile(0.95))).reset_index()
        daily["thumbs_up_rate"] = daily["thumbs_up"] / daily["rated"].replace(0, np.nan)
        daily = daily.sort_values(["variant", "date"])
        daily["cum_rated"] = daily.groupby("variant")["rated"].cumsum()
        daily["cum_thumbs_up_rate"] = daily.groupby("variant")["thumbs_up"].cumsum() / daily["cum_rated"]
        paths.append(_write(daily, out, "agg_ab_daily"))
    if readout is not None:
        r = readout
        paths += _readout_tables(r, df, exp, out, Path(db))
        p = r["primary"]
        rows = [{"metric": "thumbs_up_rate_diff", "estimate": p["diff"], "ci_low": p["ci_clustered"][0],
                 "ci_high": p["ci_clustered"][1], "p_value": p["p_value_clustered"], "method": "user clustered delta"},
                {"metric": "thumbs_up_rate_diff", "estimate": p["diff"], "ci_low": p["ci_newcombe"][0],
                 "ci_high": p["ci_newcombe"][1], "p_value": p["p_value_z"], "method": "Newcombe (independent answers)"},
                {"metric": "relative_lift", "estimate": p["relative_lift"], "ci_low": p["relative_lift_ci"][0],
                 "ci_high": p["relative_lift_ci"][1], "p_value": p["p_value_clustered"], "method": "delta"}]
        for k, v in r["guardrails"].items():
            ci = v.get("ci", [None, None])
            rows.append({"metric": k, "estimate": v["estimate"], "ci_low": ci[0], "ci_high": ci[1], "p_value": None,
                         "method": f"limit {v['limit']}, breached={v['breached']}"})
        summary = pd.DataFrame(rows)
        summary["decision"] = r["decision"]
        paths.append(_write(summary, out, "ab_summary"))
    return paths


def _readout_tables(r: dict, df: pd.DataFrame, exp: Experiment, out: Path, db: Path) -> list[Path]:
    """Dashboard tables taken straight from the readout, so the page and the readout agree by construction."""
    paths = []
    a, b = r["arms"][r["control"]], r["arms"][r["treatment"]]
    arms = pd.DataFrame([{
        "variant": v, "description": exp.variants.get(v, {}).get("description", ""), "users": x["users"],
        "exposures": x["exposures"], "rated": x["rated"], "thumbs_up": x["thumbs_up"], "rate": x["thumbs_up_rate"],
        "wilson_low": x["wilson_low"], "wilson_high": x["wilson_high"], "rating_rate": x["rating_rate"],
        "latency_p95_ms": x["latency_p95_s"] * 1000, "citation_valid_rate": x["citation_valid_rate"],
        "cost_per_answer_usd": x["cost_per_answer_usd"]} for v, x in ((r["control"], a), (r["treatment"], b))])
    paths.append(_write(arms, out, "ab_arms"))

    p, g = r["primary"], r["guardrails"]
    ra = a["thumbs_up_rate"]
    lat = g["latency_p95_relative_change"]
    rows = [
        {"key": "thumbs_up_lift", "metric": "Thumbs up rate", "method": "user clustered delta, 95%",
         "estimate_pct": p["relative_lift"] * 100, "low_pct": p["relative_lift_ci"][0] * 100, "high_pct": p["relative_lift_ci"][1] * 100},
        {"key": "thumbs_up_lift_always_valid", "metric": "Thumbs up rate, always valid", "method": "mSPRT confidence sequence, 95%",
         "estimate_pct": p["relative_lift"] * 100, "low_pct": p["msprt_confidence_sequence"][0] / ra * 100,
         "high_pct": p["msprt_confidence_sequence"][1] / ra * 100},
        {"key": "latency_p95_faster", "metric": "p95 latency, faster is right", "method": "user cluster bootstrap, 95%",
         "estimate_pct": -lat["estimate"] * 100, "low_pct": -lat["ci"][1] * 100, "high_pct": -lat["ci"][0] * 100},
        {"key": "citation_validity", "metric": "Citation validity", "method": "difference in mean",
         "estimate_pct": (b["citation_valid_rate"] / a["citation_valid_rate"] - 1) * 100,
         "low_pct": (b["citation_valid_rate"] / a["citation_valid_rate"] - 1) * 100,
         "high_pct": (b["citation_valid_rate"] / a["citation_valid_rate"] - 1) * 100},
    ]
    paths.append(_write(pd.DataFrame(rows), out, "ab_intervals"))

    truth_path = db.with_suffix(".truth.json")
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else []
    truth = truth if isinstance(truth, list) else [truth]
    planted = float(np.mean([t["true_diff"] for t in truth])) if truth else float("nan")
    dates = pd.to_datetime(df["ts"], unit="s")
    overview = pd.DataFrame([{
        "experiment": r["experiment"], "decision": r["decision"], "mode": p.get("mode", ""),
        "diff_pts": p["diff"] * 100, "diff_low_pts": p["ci_clustered"][0] * 100, "diff_high_pts": p["ci_clustered"][1] * 100,
        "cs_low_pts": p["msprt_confidence_sequence"][0] * 100, "cs_high_pts": p["msprt_confidence_sequence"][1] * 100,
        "p_always_valid": p["msprt_p_final"], "p_clustered": p["p_value_clustered"], "design_effect": p["design_effect"],
        "users": a["users"] + b["users"], "rated": a["rated"] + b["rated"], "srm_p": r["srm"]["p_value"],
        "start": dates.min().date().isoformat(), "end": dates.max().date().isoformat(),
        "simulated": bool(truth), "planted_diff_pts": planted * 100}])
    paths.append(_write(overview, out, "ab_overview"))

    from .analysis import msprt_path

    rated = df[df["rating"].notna()].assign(thumbs_up=lambda d: (d["rating"] == 1).astype(int))
    path = msprt_path(rated[["ts", "variant", "thumbs_up"]], r["control"], r["treatment"], deff=p["design_effect"])
    step = max(1, len(path) // 150)
    pts = pd.concat([path.iloc[::step], path.iloc[[-1]]]).drop_duplicates("n")
    pts = pts[pts["diff"].notna() & (pts["cs_low"] > -1.0)].assign(  # only once both arms have enough answers
        date=pd.to_datetime(pts["ts"], unit="s").dt.date.astype(str), diff_pts=pts["diff"] * 100,
        cs_low_pts=pts["cs_low"] * 100, cs_high_pts=pts["cs_high"] * 100)
    paths.append(_write(pts[["n", "date", "diff_pts", "p_always_valid", "cs_low_pts", "cs_high_pts"]], out, "ab_sequential"))
    return paths
