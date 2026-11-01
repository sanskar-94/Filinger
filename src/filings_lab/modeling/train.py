"""Phase 2: predict the post filing reaction with LightGBM under purged walk forward validation, then ask
whether the result is statistically real.

Experiment design:
1. Feature sets: market only (the null hypothesis that text adds nothing), market plus fundamentals,
   text only, and everything.
2. Each set gets identical folds, identical model capacity and early stopping on a purged, time ordered
   validation slice carved from the end of each training window.
3. Out of fold predictions feed: DeLong AUC CI, monthly IC with Newey West t, quintile long short spread
   with Newey West t, log loss and Brier, a DeLong and a Diebold Mariano comparison against market only,
   a drop one group ablation with Benjamini Hochberg correction, and a within month permutation test.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.build import feature_groups
from . import stats
from .cv import PurgedWalkForward, assert_no_leakage

LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 60,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "seed": 0, "num_threads": 4,
}


@dataclass
class Phase2Config:
    label: str = "label_up"
    target: str = "car"
    n_splits: int = 6
    min_train_frac: float = 0.35
    embargo_days: int = 5
    val_frac: float = 0.15
    max_rounds: int = 1000
    early_stopping: int = 50
    n_permutations: int = 20
    seed: int = 0
    tune: bool = False
    n_configs: int = 8
    cost_bps: float = 10.0
    lgb_params: dict = field(default_factory=lambda: dict(LGB_PARAMS))


def feature_sets(columns) -> dict[str, list[str]]:
    g = feature_groups(columns)
    text = g["text_tone"] + g["text_change"] + g["text_rules"]
    sets = {
        "market_meta": g["market"] + g["meta"],
        "market_fundamentals": g["market"] + g["financial"] + g["meta"],
        "text_only": text + g["meta"],
        "all": g["market"] + g["financial"] + text + g["meta"],
    }
    if g["llm"]:  # Phase 3 signals: does the LLM add information on top of everything else?
        sets["llm_only"] = g["llm"] + g["meta"]
        sets["all_plus_llm"] = sets["all"] + g["llm"]
    return sets


def _fit_lgb(X: pd.DataFrame, y: np.ndarray, t0, t1, cfg: Phase2Config, params: dict | None = None):
    """Fit with early stopping on the most recent ``val_frac`` of the window, purged from the fit part."""
    import lightgbm as lgb

    t0 = pd.to_datetime(pd.Series(t0)).to_numpy().astype("datetime64[ns]")
    t1 = pd.to_datetime(pd.Series(t1)).to_numpy().astype("datetime64[ns]")
    cut = np.datetime64(pd.Series(t0).quantile(1 - cfg.val_frac), "ns")
    fit = (t1 < cut) & (t0 < cut - np.timedelta64(cfg.embargo_days, "D"))
    val = t0 >= cut
    params = {**(params or cfg.lgb_params), "seed": cfg.seed}
    dfit = lgb.Dataset(X[fit], y[fit])
    dval = lgb.Dataset(X[val], y[val], reference=dfit)
    booster = lgb.train(params, dfit, cfg.max_rounds, valid_sets=[dval],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    best = max(booster.best_iteration, 20)
    # refit on the whole window with the chosen number of rounds
    return lgb.train(params, lgb.Dataset(X, y), best), best


def walk_forward(df: pd.DataFrame, cols: list[str], cfg: Phase2Config, y: np.ndarray | None = None,
                 tune: bool = False) -> pd.DataFrame:
    from .tuning import sample_configs, select

    y = df[cfg.label].astype(int).to_numpy() if y is None else y
    cv = PurgedWalkForward(cfg.n_splits, cfg.min_train_frac, pd.Timedelta(days=cfg.embargo_days))
    preds = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    rounds, chosen = [], []
    configs = sample_configs(cfg.n_configs, cfg.seed) if tune else []
    for k, (tr, te) in enumerate(cv.split(df["event_date"], df["t1"])):
        assert_no_leakage(df["event_date"], df["t1"], tr, te)
        params = None
        if tune:  # chosen inside the training window only, with purged k fold
            params, _ = select(df.iloc[tr][cols], y[tr], df["event_date"].iloc[tr], df["t1"].iloc[tr], cfg.lgb_params,
                               configs, embargo_days=cfg.embargo_days)
            chosen.append({key: params[key] for key in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "learning_rate")})
        model, best = _fit_lgb(df.iloc[tr][cols], y[tr], df["event_date"].iloc[tr], df["t1"].iloc[tr], cfg, params)
        preds[te] = model.predict(df.iloc[te][cols])
        fold[te] = k
        rounds.append(best)
    extra = [c for c in ("car_tradable",) if c in df.columns]
    out = df[["accession", "cik", "form", "event_date", cfg.target, *extra]].copy()
    out["y"] = y
    out["pred"] = preds
    out["fold"] = fold
    out.attrs["rounds"] = rounds
    out.attrs["params"] = chosen
    res = out[out["fold"] >= 0]
    res.attrs.update(out.attrs)
    return res


def logistic_baseline(df: pd.DataFrame, cols: list[str], cfg: Phase2Config) -> pd.DataFrame:
    """Regularised logistic regression on the same folds: is the tree model's extra capacity earning its keep?"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = df[cfg.label].astype(int).to_numpy()
    cv = PurgedWalkForward(cfg.n_splits, cfg.min_train_frac, pd.Timedelta(days=cfg.embargo_days))
    preds = np.full(len(df), np.nan)
    for tr, te in cv.split(df["event_date"], df["t1"]):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(df.iloc[tr][cols], y[tr])
        preds[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    mask = ~np.isnan(preds)
    return pd.DataFrame({"y": y[mask], "pred": preds[mask], "event_date": df["event_date"].to_numpy()[mask],
                         cfg.target: df[cfg.target].to_numpy()[mask]})


def _log_loss_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(oof: pd.DataFrame, target: str = "car") -> dict:
    ic = stats.monthly_ic(oof, "pred", target)
    ls = stats.long_short(oof, "pred", target)
    return {
        "n": int(len(oof)),
        "auc": stats.auc_ci(oof["y"], oof["pred"]),
        "log_loss": float(_log_loss_vec(oof["y"], oof["pred"]).mean()),
        "base_rate_log_loss": float(_log_loss_vec(oof["y"], np.full(len(oof), oof["y"].mean())).mean()),
        "brier": float(((oof["pred"] - oof["y"]) ** 2).mean()),
        "ic": {**stats.newey_west(ic), "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 1 else None,
               "months": int(len(ic))},
        "long_short_bps": {k: (v * 1e4 if k in ("mean", "se") else v) for k, v in stats.newey_west(ls).items()},
        "long_short_ci_bps": {k: (v * 1e4 if isinstance(v, float) and k != "block" else v)
                              for k, v in stats.block_bootstrap_ci(ls).items()},
    }


def importance(df: pd.DataFrame, cols: list[str], cfg: Phase2Config) -> pd.DataFrame:
    """Gain importance and mean |SHAP| (LightGBM pred_contrib) from a model on the full sample."""
    import lightgbm as lgb

    y = df[cfg.label].astype(int).to_numpy()
    model = lgb.train({**cfg.lgb_params}, lgb.Dataset(df[cols], y), 300)
    contrib = model.predict(df[cols], pred_contrib=True)[:, :-1]
    return (
        pd.DataFrame({"feature": cols, "gain": model.feature_importance("gain"),
                      "mean_abs_shap": np.abs(contrib).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def run(features: pd.DataFrame, out_dir: str | Path, cfg: Phase2Config | None = None) -> dict:
    from .. import tracking

    cfg = cfg or Phase2Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = features.dropna(subset=[cfg.label, cfg.target]).sort_values("event_date").reset_index(drop=True)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["t1"] = pd.to_datetime(df["t1"])
    sets = feature_sets(df.columns)
    results, oofs = {}, {}
    with tracking.run("phase2-baseline", "lgbm-walk-forward") as tr:
        tr.params({**{k: v for k, v in asdict(cfg).items() if k != "lgb_params"}, **cfg.lgb_params})
        for name, cols in sets.items():
            oof = walk_forward(df, cols, cfg)
            oofs[name] = oof
            results[name] = {"n_features": len(cols), "rounds": oof.attrs["rounds"], **summarize(oof, cfg.target)}
            tr.metrics({f"{name}.auc": results[name]["auc"]["auc"], f"{name}.ic": results[name]["ic"]["mean"],
                        f"{name}.log_loss": results[name]["log_loss"]})

        base, full = oofs["market_meta"], oofs["all"]
        m = full.merge(base[["accession", "pred"]], on="accession", suffixes=("", "_base"))
        comparison = {
            "delong_all_vs_market": stats.delong_test(m["y"], m["pred"], m["pred_base"]),
            "dm_log_loss_all_vs_market": stats.diebold_mariano(
                _log_loss_vec(m["y"], m["pred"]), _log_loss_vec(m["y"], m["pred_base"])),
        }
        if "all_plus_llm" in oofs:
            ml = oofs["all_plus_llm"].merge(full[["accession", "pred"]], on="accession", suffixes=("", "_all"))
            comparison["delong_all_plus_llm_vs_all"] = stats.delong_test(ml["y"], ml["pred"], ml["pred_all"])

        # drop one group ablation on the full model, BH corrected
        groups = feature_groups(df.columns)
        abl = []
        for gname, gcols in groups.items():
            if not gcols or gname in ("meta", "llm"):
                continue
            cols = [c for c in sets["all"] if c not in gcols]
            o = walk_forward(df, cols, cfg).merge(full[["accession", "pred"]], on="accession", suffixes=("", "_full"))
            t = stats.delong_test(o["y"], o["pred_full"], o["pred"])
            abl.append({"dropped": gname, "auc_full": t["auc_a"], "auc_without": t["auc_b"], "delta_auc": t["diff"], "p_value": t["p_value"]})
        abl = pd.DataFrame(abl)
        abl["q_value"] = stats.benjamini_hochberg(abl["p_value"])["q_value"].to_numpy()

        # permutation test: refit the full pipeline on labels shuffled within month
        month = df["event_date"].dt.to_period("M").astype(str).to_numpy()
        y = df[cfg.label].astype(int).to_numpy()

        def perm_score(yp: np.ndarray) -> float:
            o = walk_forward(df, sets["all"], cfg, y=yp)
            return stats.auc_ci(o["y"], o["pred"])["auc"]

        perm = stats.permutation_test(y, month, perm_score, n_perm=cfg.n_permutations, seed=cfg.seed) if cfg.n_permutations else None

        if cfg.tune:
            tuned = walk_forward(df, sets["all"], cfg, tune=True)
            results["all_tuned"] = {"n_features": len(sets["all"]), "rounds": tuned.attrs["rounds"],
                                    "params": tuned.attrs["params"], **summarize(tuned, cfg.target)}
            mt = tuned.merge(full[["accession", "pred"]], on="accession", suffixes=("", "_default"))
            comparison["delong_tuned_vs_default"] = stats.delong_test(mt["y"], mt["pred"], mt["pred_default"])

        from .backtest import backtest
        from .calibration import calibration_report

        bt = backtest(full, cfg.cost_bps)
        cal = calibration_report(full)
        bt["monthly"].to_csv(out / "backtest_monthly.csv", index=False)
        bt["cost_curve"].to_csv(out / "backtest_cost_curve.csv", index=False)
        cal["table"].to_csv(out / "calibration.csv", index=False)

        logit = logistic_baseline(df, sets["all"], cfg)
        results["logistic_all"] = summarize(logit, cfg.target)
        imp = importance(df, sets["all"], cfg)

        full.to_parquet(out / "oof_predictions.parquet", index=False)
        abl.to_csv(out / "ablation.csv", index=False)
        imp.to_csv(out / "feature_importance.csv", index=False)
        metrics = {"config": {k: v for k, v in asdict(cfg).items()}, "results": results,
                   "comparison": comparison, "permutation": perm, "backtest": bt["summary"],
                   "calibration": {"raw": cal["raw"], "isotonic": cal["isotonic"]}}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        (out / "report.md").write_text(render_report(metrics, abl, imp, bt["cost_curve"]))
        for f in ("metrics.json", "report.md", "ablation.csv", "feature_importance.csv", "backtest_monthly.csv"):
            tr.artifact(str(out / f))
        tr.metrics({"comparison.delong_p": comparison["delong_all_vs_market"]["p_value"]})
    return metrics


def _fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def render_report(metrics: dict, abl: pd.DataFrame, imp: pd.DataFrame, cost_curve: pd.DataFrame | None = None) -> str:
    r = metrics["results"]
    lines = ["# Phase 2 report: classical baseline", "",
             "Out of fold results from purged walk forward validation. Each feature set uses identical folds.", "",
             "| Feature set | n | AUC [95% DeLong CI] | p (AUC=0.5) | log loss (base) | IC mean | IC NW t | LS spread bps | LS NW t |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, v in r.items():
        a, ic, ls = v["auc"], v["ic"], v["long_short_bps"]
        lines.append(
            f"| {name} | {v['n']} | {_fmt(a['auc'])} [{_fmt(a['ci_low'])}, {_fmt(a['ci_high'])}] | {_fmt(a['p_vs_0.5'], 4)} | "
            f"{_fmt(v['log_loss'])} ({_fmt(v['base_rate_log_loss'])}) | {_fmt(ic['mean'])} | {_fmt(ic['t'], 2)} | "
            f"{_fmt(ls['mean'], 1)} | {_fmt(ls['t'], 2)} |")
    c = metrics["comparison"]
    d, dm = c["delong_all_vs_market"], c["dm_log_loss_all_vs_market"]
    lines += ["", "market_meta holds price based features plus form and 8-K item indicators; it is the null model that",
              "says filing language adds nothing beyond what kind of filing it is and how the stock was trading.", "",
              "## Does text add information beyond market features?", "",
              f"DeLong paired test, all vs market and meta: AUC difference {_fmt(d['diff'])} (se {_fmt(d['se'])}), p = {_fmt(d['p_value'])}.",
              f"Diebold Mariano on per event log loss: DM = {_fmt(dm['dm'], 2)}, p = {_fmt(dm['p_value'])} (negative favours the full model).", ""]
    if "delong_all_plus_llm_vs_all" in c:
        x = c["delong_all_plus_llm_vs_all"]
        lines.append(f"LLM signals on top of all classical features: AUC difference {_fmt(x['diff'])} "
                     f"(se {_fmt(x['se'])}), DeLong p = {_fmt(x['p_value'])}.")
        lines.append("")
    if metrics.get("permutation"):
        p = metrics["permutation"]
        lines.append(f"Permutation test ({p['n_perm']} refits with labels shuffled within month): observed AUC {_fmt(p['observed'])}, "
                     f"null mean {_fmt(p['null_mean'])} (sd {_fmt(p['null_sd'])}), p = {_fmt(p['p_value'], 3)} "
                     f"(smallest attainable p with {p['n_perm']} permutations is {1 / (p['n_perm'] + 1):.3f}).")
    if "delong_tuned_vs_default" in c:
        x = c["delong_tuned_vs_default"]
        lines.append(f"Purged k fold tuning inside each window against default parameters: AUC difference {_fmt(x['diff'])}, "
                     f"DeLong p = {_fmt(x['p_value'])}.")
        lines.append("")
    b = metrics.get("backtest") or {}
    if b:
        lines += ["## Cost aware backtest (all features, LightGBM)", "",
                  f"Monthly top minus bottom quintile, entering at the day 0 close ({b['target']}), {b['months']} months, "
                  f"{b['cost_bps_per_side']:.0f} bps per side: mean net {_fmt(b['mean_net_bps'], 1)} bps per month "
                  f"(gross {_fmt(b['mean_gross_bps'], 1)}), annualised Sharpe {_fmt(b['sharpe_annual'], 2)}, Newey West t "
                  f"{_fmt(b['nw_t'], 2)}, hit rate {b['hit_rate']:.0%}, max drawdown {_fmt(b['max_drawdown_bps'], 0)} bps, "
                  f"break even cost {_fmt(b['break_even_cost_bps'], 1)} bps per side.", ""]
        if cost_curve is not None and len(cost_curve):
            lines += [cost_curve[["cost_bps_per_side", "mean_net_bps", "sharpe_annual", "nw_t", "hit_rate"]].round(3).to_markdown(index=False), ""]
    cal = metrics.get("calibration")
    if cal:
        lines += ["## Calibration (folds after the first)", "",
                  f"Expected calibration error {_fmt(cal['raw']['ece'])} raw, {_fmt(cal['isotonic']['ece'])} after out of sample isotonic "
                  f"recalibration. Brier {_fmt(cal['raw']['brier'])} raw, {_fmt(cal['isotonic']['brier'])} recalibrated "
                  f"(reliability {_fmt(cal['raw']['reliability'])}, resolution {_fmt(cal['raw']['resolution'])}, "
                  f"uncertainty {_fmt(cal['raw']['uncertainty'])}).", ""]
    lines += ["", "## Drop one group ablation (Benjamini Hochberg q values)", "", abl.round(4).to_markdown(index=False), "",
              "## Top features by mean |SHAP|", "", imp.head(20).round(5).to_markdown(index=False), ""]
    return "\n".join(lines)
