"""Phase 6 commands: ab-plan, ab-simulate, ab-analyze, bi-export."""
from __future__ import annotations

import json
from pathlib import Path


def cmd_plan(a):
    from .power import plan

    p = plan(a.baseline, a.mde, a.alpha, a.power, a.rated_per_user, a.icc, a.rating_rate, a.daily_users)
    print(json.dumps(p.to_dict(), indent=2))


def cmd_simulate(a):
    import time

    from ..agent.service import AgentService
    from ..agent.store import DataStore
    from .assign import Experiment
    from .log import EventLog
    from .simulate import simulate

    if not a.append:
        Path(a.db).unlink(missing_ok=True)
    exp = Experiment.load(a.experiment)
    svc = AgentService(DataStore(a.lake), exp, EventLog(a.db), mode="offline")
    start = time.time() - (a.start_days_ago or a.days) * 86400
    truth = simulate(svc, a.users, a.mean_requests, a.rating_rate, {"A": a.rate_a, "B": a.rate_b}, a.user_sd,
                     a.days, start_ts=start, seed=a.seed, progress=True)
    truth_path = Path(a.db).with_suffix(".truth.json")
    periods = json.loads(truth_path.read_text()) if a.append and truth_path.exists() else []
    periods = periods if isinstance(periods, list) else [periods]
    truth_path.write_text(json.dumps(periods + [truth], indent=2))
    print(json.dumps(truth, indent=2))


def cmd_analyze(a):
    from .analysis import analyze, render_markdown, save
    from .assign import Experiment
    from .log import EventLog

    exp = Experiment.load(a.experiment)
    df = EventLog(a.db).frame(exp.name)
    g = exp.guardrails if isinstance(exp.guardrails, dict) else None
    r = analyze(df, exp.name, {k: v.get("weight", 1.0) for k, v in exp.variants.items()}, planned_mde=a.mde, guardrails=g,
                sequential=exp.analysis == "sequential", max_rated_per_arm=exp.max_rated_per_arm)
    save(r, a.out)
    print(render_markdown(r))


def cmd_export(a):
    from .export import export_all

    paths = export_all(a.lake, a.db, a.experiment, a.phase2, a.out, a.readout)
    for p in paths:
        print(p)


def _reg_plan(p):
    p.add_argument("--baseline", type=float, default=0.60)
    p.add_argument("--mde", type=float, default=0.04, help="absolute lift to detect")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--rated-per-user", type=float, default=1.8)
    p.add_argument("--icc", type=float, default=0.15)
    p.add_argument("--rating-rate", type=float, default=0.45)
    p.add_argument("--daily-users", type=float, default=None)
    p.set_defaults(fn=cmd_plan)


def _common(p):
    p.add_argument("--experiment", default="configs/experiment.json")
    p.add_argument("--db", default="data/experiments.db")


def _reg_sim(p):
    _common(p)
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--mean-requests", type=float, default=4.0)
    p.add_argument("--rating-rate", type=float, default=0.45)
    p.add_argument("--rate-a", type=float, default=0.60)
    p.add_argument("--rate-b", type=float, default=0.64)
    p.add_argument("--user-sd", type=float, default=0.6)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true", help="extend an existing test instead of starting over")
    p.add_argument("--start-days-ago", type=float, default=None)
    p.set_defaults(fn=cmd_simulate)


def _reg_analyze(p):
    _common(p)
    p.add_argument("--mde", type=float, default=0.04)
    p.add_argument("--out", default="artifacts/phase6")
    p.set_defaults(fn=cmd_analyze)


def _reg_export(p):
    _common(p)
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--phase2", default="artifacts/phase2")
    p.add_argument("--readout", default="artifacts/phase6/readout.json")
    p.add_argument("--out", default="dashboards/data")
    p.set_defaults(fn=cmd_export)


COMMANDS = {
    "ab-plan": (_reg_plan, "power analysis and sample size for the A/B test"),
    "ab-simulate": (_reg_sim, "drive simulated analyst traffic through the real agent service"),
    "ab-analyze": (_reg_analyze, "read out the A/B test with confidence intervals and guardrails"),
    "bi-export": (_reg_export, "export star schema tables for Power BI or Tableau"),
}
