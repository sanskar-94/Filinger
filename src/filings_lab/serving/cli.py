"""Phase 4 commands: register, promote, serve, gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def cmd_register(a):
    import tempfile

    import pandas as pd

    from .model import build_artifact
    from .registry import get_registry

    feats = pd.read_parquet(f"{a.lake.rstrip('/')}/gold/features")
    metrics = json.loads(Path(a.metrics).read_text()) if a.metrics and Path(a.metrics).exists() else {}
    res = metrics.get("results", {}).get(a.feature_set, {})
    summary = {"auc": res.get("auc", {}).get("auc"), "ic": res.get("ic", {}).get("mean"),
               "ic_t": res.get("ic", {}).get("t"), "log_loss": res.get("log_loss")}
    with tempfile.TemporaryDirectory() as tmp:
        build_artifact(feats, tmp, a.feature_set, metrics={"summary": summary, "rounds": res.get("rounds", [300])})
        reg = get_registry()
        version = reg.register(a.name, tmp, metrics=summary, tags={"feature_set": a.feature_set})
    meta_path = reg.resolve(f"models:/{a.name}/{version}") / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["version"] = version
        meta_path.write_text(json.dumps(meta, indent=2))
    if a.alias:
        reg.set_alias(a.name, a.alias, version)
    print(f"registered {a.name} v{version}" + (f" as @{a.alias}" if a.alias else ""), json.dumps(summary))


def cmd_promote(a):
    from .registry import get_registry

    get_registry().set_alias(a.name, a.alias, a.version)
    print(f"{a.name}@{a.alias} -> v{a.version}")


def cmd_serve(a):
    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(), host=a.host, port=a.port, workers=1, log_level="info")


def cmd_gate(a):
    """Exit non zero when the candidate is worse than production. Used by the CI model gate."""
    from ..llm.eval_harness import gate

    prod = json.loads(Path(a.production).read_text())
    failures = []
    if a.extractor_eval:
        ev = json.loads(Path(a.extractor_eval).read_text())["summary"]
        cand = ev[a.system] if a.system else next(iter(ev.values()))
        ok, why = gate(cand, prod["extractor"]["metrics"])
        failures += [f"extractor {w}" for w in why]
        print(f"extractor gate: {'PASS' if ok else 'FAIL'}")
    if a.reaction_metrics:
        res = json.loads(Path(a.reaction_metrics).read_text())["results"][a.feature_set]
        cand = {"auc": res["auc"]["auc"], "ic": res["ic"]["mean"], "auc_p": res["auc"]["p_vs_0.5"]}
        base = prod["reaction_model"]["metrics"]
        ok, why = gate(cand, base, {"auc": ("max", 0.005), "ic": ("max", 0.01)})
        if cand["auc_p"] > 0.05:
            ok, why = False, why + [f"AUC not significantly above 0.5 (p = {cand['auc_p']:.3f})"]
        failures += [f"reaction {w}" for w in why]
        print(f"reaction model gate: {'PASS' if ok else 'FAIL'}")
    for f in failures:
        print("  " + f)
    if a.summary:
        with open(a.summary, "a") as fh:
            fh.write("## Model gate\n\n" + ("All gated metrics pass.\n" if not failures else
                     "\n".join(f"1. {f}" for f in failures) + "\n"))
    sys.exit(1 if failures else 0)


def _reg_register(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--metrics", default="artifacts/phase2/metrics.json")
    p.add_argument("--feature-set", default="all")
    p.add_argument("--name", default="filings-reaction")
    p.add_argument("--alias", default="challenger")
    p.set_defaults(fn=cmd_register)


def _reg_promote(p):
    p.add_argument("--name", default="filings-reaction")
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--alias", default="champion")
    p.set_defaults(fn=cmd_promote)


def _reg_serve(p):
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=cmd_serve)


def _reg_gate(p):
    p.add_argument("--production", default="models/production.json")
    p.add_argument("--extractor-eval", help="eval.json from flab llm-eval")
    p.add_argument("--system", help="system name inside eval.json")
    p.add_argument("--reaction-metrics", help="metrics.json from flab train")
    p.add_argument("--feature-set", default="all")
    p.add_argument("--summary", help="append a markdown summary (e.g. $GITHUB_STEP_SUMMARY)")
    p.set_defaults(fn=cmd_gate)


COMMANDS = {
    "register": (_reg_register, "fit the final reaction model and register it"),
    "promote": (_reg_promote, "point a registry alias (champion) at a version"),
    "serve": (_reg_serve, "run the FastAPI service"),
    "gate": (_reg_gate, "CI gate: fail if a candidate model is worse than production"),
}
