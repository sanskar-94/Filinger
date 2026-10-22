"""``flab train``: the Phase 2 experiment."""
from __future__ import annotations


def _run(a):
    import pandas as pd

    from .train import Phase2Config, run

    feats = pd.read_parquet(f"{a.lake.rstrip('/')}/gold/features")
    cfg = Phase2Config(label=a.label, n_splits=a.splits, n_permutations=a.permutations)
    m = run(feats, a.out, cfg)
    for name, v in m["results"].items():
        print(f"{name:22s} AUC {v['auc']['auc']:.4f} [{v['auc']['ci_low']:.4f}, {v['auc']['ci_high']:.4f}]  "
              f"IC {v['ic']['mean']:.4f} (t {v['ic']['t']:.2f})")
    print(f"report -> {a.out}/report.md")


def _register(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--out", default="artifacts/phase2")
    p.add_argument("--label", default="label_up")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--permutations", type=int, default=20)
    p.set_defaults(fn=_run)


COMMANDS = {"train": (_register, "Phase 2: LightGBM with purged walk forward validation and significance tests")}
