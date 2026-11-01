"""Read out an A/B test from the event log.

Unit of randomisation: the user. Unit of analysis for the primary metric: rated answers. Because one user
can rate many answers, every interval for the primary metric is computed two ways: the per answer
(independence) intervals analysts usually quote, and a user clustered delta method interval that is
actually valid. The decision uses the clustered one.

Checks, in the order an analyst should look at them:
1. Sample ratio mismatch (chi square on users per variant against the configured weights).
2. Primary metric: thumbs up rate, Wilson interval per arm, Newcombe interval for the difference,
   clustered delta method interval for difference and relative lift, z test, always valid mSPRT p value.
3. Guardrails: p95 latency (user cluster bootstrap), citation validity, cost per answer.
4. Achieved power at the planned minimum detectable effect, so a null result is read correctly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .power import achieved_power


# --------------------------------------------------------------------------------------------- intervals
def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Hybrid score interval for p2 - p1 (Newcombe 1998, method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, alpha)
    l2, u2 = wilson(k2, n2, alpha)
    d = p2 - p1
    return (d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k2 / n2 - k1 / n1) / se if se > 0 else 0.0
    return z, float(2 * (1 - stats.norm.cdf(abs(z))))


def ratio_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ratio of sums R = sum(x) / sum(y) over clusters and its delta method variance."""
    n = len(x)
    mx, my = x.mean(), y.mean()
    r = mx / my
    cov = np.cov(x, y, ddof=1) if n > 1 else np.zeros((2, 2))
    var = (cov[0, 0] - 2 * r * cov[0, 1] + r * r * cov[1, 1]) / (n * my * my)
    return float(r), float(max(var, 0.0))


# ----------------------------------------------------------------------------------------------- mSPRT
def msprt_path(events: pd.DataFrame, a: str, b: str, tau: float = 0.05, alpha: float = 0.05,
               deff: float = 1.0) -> pd.DataFrame:
    """Always valid p values for p_b - p_a over time (Johari, Pekelis and Walsh, normal mixture mSPRT).

    Peeking at this p value after every observation keeps the false positive rate at ``alpha``, unlike
    repeatedly checking a fixed horizon z test. ``deff`` inflates the variance for answers clustered within
    users. The ``cs_low`` and ``cs_high`` columns are the matching always valid confidence sequence.
    """
    ev = events.sort_values("ts")
    is_b = (ev["variant"] == b).to_numpy()
    up = ev["thumbs_up"].to_numpy().astype(float)
    na = np.cumsum(~is_b)
    nb = np.cumsum(is_b)
    ka = np.cumsum(up * ~is_b)
    kb = np.cumsum(up * is_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa, pb = ka / na, kb / nb
        v = (pa * (1 - pa) / na + pb * (1 - pb) / nb) * deff
        theta = pb - pa
        t2 = tau * tau
        lam = np.sqrt(v / (v + t2)) * np.exp(theta**2 * t2 / (2 * v * (v + t2)))
        half = np.sqrt(2 * v * (v + t2) / t2 * (np.log(1 / alpha) + 0.5 * np.log((v + t2) / v)))
    ok = (na > 20) & (nb > 20) & (v > 0)
    p = np.minimum.accumulate(np.where(ok, np.minimum(1.0, 1.0 / lam), 1.0))
    lo = np.maximum.accumulate(np.where(ok, theta - half, -1.0))  # running intersection stays valid
    hi = np.minimum.accumulate(np.where(ok, theta + half, 1.0))
    return pd.DataFrame({"ts": ev["ts"].to_numpy(), "n": np.arange(1, len(ev) + 1), "diff": theta,
                         "p_always_valid": p, "cs_low": lo, "cs_high": hi})


# --------------------------------------------------------------------------------------------- readout
@dataclass
class Readout:
    experiment: str
    control: str
    treatment: str
    srm: dict
    arms: dict
    primary: dict
    guardrails: dict
    power: dict
    decision: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def _cluster_boot_p95(df: pd.DataFrame, a: str, b: str, n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    out = []
    groups = {v: [g["latency_s"].to_numpy() for _, g in df[df["variant"] == v].groupby("user_id")] for v in (a, b)}
    for _ in range(n_boot):
        q = {}
        for v, gs in groups.items():
            pick = rng.integers(0, len(gs), len(gs))
            q[v] = np.quantile(np.concatenate([gs[i] for i in pick]), 0.95)
        out.append(q[b] / q[a] - 1)
    point = np.quantile(df[df.variant == b]["latency_s"], 0.95) / np.quantile(df[df.variant == a]["latency_s"], 0.95) - 1
    return float(point), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def analyze(df: pd.DataFrame, experiment: str, weights: dict[str, float], control: str = "A", treatment: str = "B",
            alpha: float = 0.05, planned_mde: float = 0.04, guardrails: dict | None = None,
            sequential: bool = False, max_rated_per_arm: int | None = None) -> Readout:
    """``sequential=True`` when the result is watched continuously and may be extended: the decision then
    rests on the always valid mSPRT p value and confidence sequence, and a not yet significant test
    returns "continue" until ``max_rated_per_arm`` is reached."""
    guardrails = guardrails or {"latency_p95_s": {"max_relative_increase": 0.2},
                                "citation_valid_rate": {"max_absolute_decrease": 0.01}}
    df = df.copy()
    df["rated"] = df["rating"].notna()
    df["thumbs_up"] = (df["rating"] == 1).astype(int)
    reasons: list[str] = []
    truncated_at = None
    if sequential and max_rated_per_arm:
        # a sequential test stops at its maximum sample: data after both arms reach it is not used
        r = df[df["rated"]].sort_values("ts")
        ca, cb = (r["variant"] == control).cumsum(), (r["variant"] == treatment).cumsum()
        reached = r[(ca >= max_rated_per_arm) & (cb >= max_rated_per_arm)]
        if len(reached):
            truncated_at = float(reached["ts"].iloc[0])
            df = df[df["ts"] <= truncated_at]

    # 1. sample ratio mismatch on users
    users = df.groupby("variant")["user_id"].nunique()
    obs = np.array([users.get(v, 0) for v in (control, treatment)])
    w = np.array([weights.get(control, 0.5), weights.get(treatment, 0.5)])
    chi = stats.chisquare(obs, obs.sum() * w / w.sum())
    srm = {"users": dict(zip((control, treatment), obs.tolist(), strict=True)), "p_value": float(chi.pvalue),
           "mismatch": bool(chi.pvalue < 0.001)}
    if srm["mismatch"]:
        reasons.append(f"sample ratio mismatch (p = {chi.pvalue:.2e}): fix assignment before reading results")

    # 2. primary metric
    arms = {}
    rated = df[df["rated"]]
    for v in (control, treatment):
        d, r = df[df.variant == v], rated[rated.variant == v]
        k, n = int(r["thumbs_up"].sum()), int(len(r))
        per_user = r.groupby("user_id").agg(x=("thumbs_up", "sum"), y=("thumbs_up", "size"))
        rate, var = ratio_delta(per_user["x"].to_numpy(float), per_user["y"].to_numpy(float))
        lo, hi = wilson(k, n, alpha)
        arms[v] = {"users": int(d["user_id"].nunique()), "exposures": int(len(d)), "rated": n, "thumbs_up": k,
                   "thumbs_up_rate": k / n if n else float("nan"), "wilson_low": lo, "wilson_high": hi,
                   "clustered_se": math.sqrt(var), "rating_rate": n / len(d) if len(d) else float("nan"),
                   "latency_p95_s": float(d["latency_s"].quantile(0.95)),
                   "citation_valid_rate": float(d["citation_valid_rate"].mean()),
                   "cost_per_answer_usd": float(d["cost_usd"].mean())}
    a, b = arms[control], arms[treatment]
    z = stats.norm.ppf(1 - alpha / 2)
    diff = b["thumbs_up_rate"] - a["thumbs_up_rate"]
    se_c = math.sqrt(a["clustered_se"] ** 2 + b["clustered_se"] ** 2)
    zstat, p_naive = two_proportion_z(a["thumbs_up"], a["rated"], b["thumbs_up"], b["rated"])
    p_clustered = float(2 * (1 - stats.norm.cdf(abs(diff / se_c)))) if se_c > 0 else float("nan")
    lift = b["thumbs_up_rate"] / a["thumbs_up_rate"] - 1
    se_lift = (1 + lift) * math.sqrt((a["clustered_se"] / a["thumbs_up_rate"]) ** 2 + (b["clustered_se"] / b["thumbs_up_rate"]) ** 2)
    se_iid = math.sqrt(a["thumbs_up_rate"] * (1 - a["thumbs_up_rate"]) / a["rated"]
                       + b["thumbs_up_rate"] * (1 - b["thumbs_up_rate"]) / b["rated"])
    deff = max((se_c / se_iid) ** 2, 1.0) if se_iid > 0 else 1.0
    path = msprt_path(rated[["ts", "variant", "thumbs_up"]], control, treatment, alpha=alpha, deff=deff)
    crossed = path[path["p_always_valid"] < alpha]
    primary = {
        "metric": "thumbs_up_rate", "diff": diff,
        "ci_newcombe": newcombe_diff(a["thumbs_up"], a["rated"], b["thumbs_up"], b["rated"], alpha),
        "ci_clustered": (diff - z * se_c, diff + z * se_c), "p_value_z": p_naive, "p_value_clustered": p_clustered,
        "relative_lift": lift, "relative_lift_ci": (lift - z * se_lift, lift + z * se_lift),
        "design_effect": deff,
        "msprt_p_final": float(path["p_always_valid"].iloc[-1]),
        "msprt_confidence_sequence": (float(path["cs_low"].iloc[-1]), float(path["cs_high"].iloc[-1])),
        "msprt_first_significant_n": int(crossed["n"].iloc[0]) if len(crossed) else None,
        "mode": "sequential" if sequential else "fixed horizon",
        "max_rated_per_arm": max_rated_per_arm,
        "truncated_at_ts": truncated_at,
    }

    # 3. guardrails
    lat, lat_lo, lat_hi = _cluster_boot_p95(df, control, treatment)
    g = {"latency_p95_relative_change": {"estimate": lat, "ci": (lat_lo, lat_hi),
                                         "limit": guardrails["latency_p95_s"]["max_relative_increase"]},
         "citation_valid_rate_change": {"estimate": b["citation_valid_rate"] - a["citation_valid_rate"],
                                        "limit": -guardrails["citation_valid_rate"]["max_absolute_decrease"]}}
    g["latency_p95_relative_change"]["breached"] = lat_lo > g["latency_p95_relative_change"]["limit"]
    g["citation_valid_rate_change"]["breached"] = g["citation_valid_rate_change"]["estimate"] < g["citation_valid_rate_change"]["limit"]
    for name, val in g.items():
        if val["breached"]:
            reasons.append(f"guardrail breached: {name}")

    # 4. power and decision
    pw = {"planned_mde": planned_mde, "achieved_power_at_mde_iid": achieved_power(a["rated"], a["thumbs_up_rate"],
                                                                                  a["thumbs_up_rate"] + planned_mde, alpha)}
    pw["achieved_power_at_mde_clustered"] = achieved_power(int(a["rated"] / max(primary["design_effect"], 1)),
                                                           a["thumbs_up_rate"], a["thumbs_up_rate"] + planned_mde, alpha)
    lo, hi = primary["ci_clustered"]
    if srm["mismatch"] or any(v["breached"] for v in g.values()):
        decision = "do not ship"
    elif sequential:
        cs_lo, cs_hi = primary["msprt_confidence_sequence"]
        enough = max_rated_per_arm is not None and min(a["rated"], b["rated"]) >= max_rated_per_arm
        if primary["msprt_p_final"] < alpha and diff > 0:
            decision = "ship treatment"
            reasons.append(f"always valid p = {primary['msprt_p_final']:.4f}; confidence sequence "
                           f"[{cs_lo * 100:+.2f}, {cs_hi * 100:+.2f}] points excludes zero")
        elif primary["msprt_p_final"] < alpha:
            decision = "keep control"
            reasons.append("treatment is significantly worse (always valid test)")
        elif enough:
            decision = "inconclusive"
            reasons.append("maximum sample reached without a significant difference")
        else:
            decision = "continue"
            reasons.append(f"always valid p = {primary['msprt_p_final']:.3f}: keep collecting; peeking is safe in sequential mode")
    elif lo > 0:
        decision = "ship treatment"
        reasons.append(f"thumbs up rate improved by {diff * 100:+.2f} points, clustered 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]")
    elif hi < 0:
        decision = "keep control"
        reasons.append("treatment is significantly worse on the primary metric")
    else:
        decision = "inconclusive"
        reasons.append(f"CI includes zero; power at the planned MDE was {pw['achieved_power_at_mde_clustered']:.0%}")
    return Readout(experiment, control, treatment, srm, arms, primary, g, pw, decision, reasons)


def render_markdown(r: Readout) -> str:
    a, b = r.arms[r.control], r.arms[r.treatment]
    p = r.primary

    def pct(x):
        return f"{x * 100:.2f}%"

    lines = [f"# A/B readout: {r.experiment}", "", f"**Decision: {r.decision}.** " + " ".join(r.reasons), "",
             "## Sample ratio", "", f"Users per variant {r.srm['users']}, chi square p = {r.srm['p_value']:.3f}"
             + (" (MISMATCH)" if r.srm["mismatch"] else " (ok)"), "",
             "## Arms", "", "| | users | exposures | rated | thumbs up rate [Wilson 95%] | p95 latency | citation valid | cost per answer |",
             "|---|---|---|---|---|---|---|---|"]
    for name, x in ((r.control, a), (r.treatment, b)):
        lines.append(f"| {name} | {x['users']} | {x['exposures']} | {x['rated']} | {pct(x['thumbs_up_rate'])} "
                     f"[{pct(x['wilson_low'])}, {pct(x['wilson_high'])}] | {x['latency_p95_s'] * 1000:.0f} ms | "
                     f"{x['citation_valid_rate']:.4f} | ${x['cost_per_answer_usd']:.4f} |")
    lines += ["", "## Primary metric: thumbs up rate", "",
              f"1. Difference {r.treatment} minus {r.control}: {p['diff'] * 100:+.2f} points.",
              f"2. Newcombe 95% CI (treats answers as independent): [{p['ci_newcombe'][0] * 100:+.2f}, {p['ci_newcombe'][1] * 100:+.2f}], z test p = {p['p_value_z']:.4f}.",
              f"3. User clustered delta method 95% CI: [{p['ci_clustered'][0] * 100:+.2f}, {p['ci_clustered'][1] * 100:+.2f}], p = {p['p_value_clustered']:.4f}; design effect {p['design_effect']:.2f}.",
              f"4. Relative lift {p['relative_lift'] * 100:+.1f}% (95% CI {p['relative_lift_ci'][0] * 100:+.1f}% to {p['relative_lift_ci'][1] * 100:+.1f}%).",
              f"5. Always valid mSPRT (design effect applied): p = {p['msprt_p_final']:.4f}, confidence sequence "
              f"[{p['msprt_confidence_sequence'][0] * 100:+.2f}, {p['msprt_confidence_sequence'][1] * 100:+.2f}]"
              + (f"; first below 0.05 after {p['msprt_first_significant_n']} rated answers." if p["msprt_first_significant_n"] else "; never below 0.05."),
              f"6. Analysis mode: {p['mode']}" + (f", stopped at the maximum of {p['max_rated_per_arm']:,} rated answers per arm."
                                                  if p.get("truncated_at_ts") else "."),
              "", "## Guardrails", ""]
    for k, v in r.guardrails.items():
        ci = f" (95% CI {v['ci'][0] * 100:+.1f}% to {v['ci'][1] * 100:+.1f}%)" if "ci" in v else ""
        lines.append(f"1. {k}: {v['estimate'] * 100:+.2f}%{ci}, limit {v['limit'] * 100:+.1f}%, {'BREACHED' if v['breached'] else 'ok'}")
    lines += ["", "## Power", "", f"At the planned MDE of {r.power['planned_mde'] * 100:.1f} points: power "
              f"{r.power['achieved_power_at_mde_iid']:.0%} if answers were independent, "
              f"{r.power['achieved_power_at_mde_clustered']:.0%} after the design effect.", ""]
    return "\n".join(lines)


def save(r: Readout, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readout.json").write_text(json.dumps(r.to_dict(), indent=2, default=float))
    (out / "readout.md").write_text(render_markdown(r))
