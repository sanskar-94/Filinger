"""``flab train``: the Phase 2 experiment."""
from __future__ import annotations


def _run(a):
    import pandas as pd

    from .train import Phase2Config, run

    feats = pd.read_parquet(f"{a.lake.rstrip('/')}/gold/features")
    for extra in a.extra or []:  # e.g. llm_signals.parquet from flab llm-features
        feats = feats.merge(pd.read_parquet(extra), on="accession", how="left")
    cfg = Phase2Config(label=a.label, n_splits=a.splits, n_permutations=a.permutations, tune=a.tune, cost_bps=a.cost_bps)
    m = run(feats, a.out, cfg)
    for name, v in m["results"].items():
        print(f"{name:22s} AUC {v['auc']['auc']:.4f} [{v['auc']['ci_low']:.4f}, {v['auc']['ci_high']:.4f}]  "
              f"IC {v['ic']['mean']:.4f} (t {v['ic']['t']:.2f})")
    b = m.get("backtest") or {}
    if b:
        print(f"backtest: {b['mean_net_bps']:.1f} bps net per month at {b['cost_bps_per_side']:.0f} bps per side, "
              f"Sharpe {b['sharpe_annual']:.2f}, break even {b['break_even_cost_bps']:.1f} bps")
    print(f"report -> {a.out}/report.md")


def _register(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--out", default="artifacts/phase2")
    p.add_argument("--label", default="label_up")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--extra", action="append", help="extra per accession feature parquet (repeatable)")
    p.add_argument("--tune", action="store_true", help="purged k fold hyperparameter search inside each window")
    p.add_argument("--cost-bps", type=float, default=10.0, help="trading cost per side for the backtest")
    p.set_defaults(fn=_run)


COMMANDS = {"train": (_register, "Phase 2: LightGBM with purged walk forward validation and significance tests")}
