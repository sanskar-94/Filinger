"""Agent evaluation on questions whose answers the synthetic world knows.

Four question types, built from the generator's ground truth:

* ``guidance``: "Did X raise, lower or keep its guidance in the release filed in May 2019?" The answer is
  right when it cites the guidance paragraph of that filing and states the direction.
* ``reaction``: "How did the stock react to the release X filed in May 2019?" Right when the answer gives
  that filing's 3 day abnormal return.
* ``disclosure_yes``: "Has a securities class action been filed against X?" for a company that did disclose
  one. Right when the answer cites a paragraph that states it.
* ``disclosure_no``: the same questions for companies that never disclosed the event. Right when the answer
  says so instead of presenting unrelated passages as evidence.

The same questions score any agent that returns an ``AgentAnswer``: the offline extractive agent (no model,
runs anywhere) and the hosted model through the tool loop (``--agent claude``, needs an API key). The checks
are mechanical on purpose: citations are compared with known paragraphs and numbers with known returns, so
no model grades another model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval_eval import label_paragraphs

GUIDANCE_NEEDS = {"raised": "guidance_raised", "lowered": "guidance_lowered", "maintained": "guidance_maintained"}
DIRECTION = {
    "raised": re.compile(r"\b(rais\w+|increased its outlook|above the high end|lift\w+|boost\w+)\b", re.I),
    "lowered": re.compile(r"\b(lower\w+|reduced its\b|below the low end|cut\b|trimm\w+)", re.I),
    "maintained": re.compile(r"\b(reaffirm\w*|unchanged|maintain\w*|kept\b)", re.I),
}
DISCLOSURE_QUESTIONS = {
    "going_concern": "Has {company} said there is doubt about its ability to continue as a going concern?",
    "covenant_breach": "Has {company} breached its debt covenants or asked its lenders for a waiver?",
    "capital_need": "Does {company} say it needs to raise more capital to fund its operations?",
    "class_action": "Has a securities class action been filed against {company}?",
    "sec_subpoena": "Has {company} received a subpoena from the Securities and Exchange Commission?",
    "patent_verdict": "Did {company} lose a patent infringement case?",
    "ceo_departure": "Did the chief executive of {company} step down?",
    "cfo_appointed": "Did {company} appoint a new chief financial officer?",
    "credit_facility": "Did {company} enter into a new revolving credit facility?",
    "buyback": "Did {company} announce a new share repurchase authorization?",
}
ABSENCE = re.compile(
    r"\bno filing paragraphs? matched|\bnot (been )?disclos|\bno (such )?(disclosure|evidence|mention|record|indication)|"
    r"\b(do|does|did) not (find|appear|mention|disclose|report|show|indicate|contain)|\bnot find\b|\bnothing in\b|"
    r"\bno paragraph\b|\bnone of the\b|\bfound no\b|\bcould not find\b|\bno passage\b", re.I)
PERCENT = re.compile(r"([+\-−]?\d+(?:\.\d+)?)\s*%")
HEADLINE = {"guidance": "answer_correct", "reaction": "car_correct", "disclosure_yes": "support_cited", "disclosure_no": "abstained"}


def build_questions(store, truth: pd.DataFrame, per_type: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    paras = store.paragraphs.reset_index(drop=True)
    labels = label_paragraphs(paras["text"])
    idx = store.index.set_index("accession")
    events = store.events
    ids, accs, tickers = paras["paragraph_id"].to_numpy(), paras["accession"].to_numpy(), paras["ticker"].to_numpy()

    def sample(items: list, n: int) -> list:
        return items if len(items) <= n else [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]

    questions: list[dict] = []
    guided = truth[(truth["form"] == "8-K") & truth["guidance_direction"].isin(GUIDANCE_NEEDS)]
    guided = guided[guided["accession"].isin(idx.index)]
    for r in sample(list(guided.itertuples(index=False)), per_type):
        support = ids[(accs == r.accession) & labels[GUIDANCE_NEEDS[r.guidance_direction]]]
        if not len(support):
            continue
        f = idx.loc[r.accession]
        questions.append({"type": "guidance", "ticker": f["ticker"], "accession": r.accession, "expected": r.guidance_direction,
                          "support": sorted(support),
                          "question": f"Did {f['company']} raise, lower or keep its full year guidance in the earnings "
                                      f"release it filed in {f['date_filed']:%B %Y}?"})
    earnings = idx[(idx["form"] == "8-K") & idx["items"].astype(str).str.contains("2.02", regex=False)]
    earnings = earnings[earnings.index.isin(events.index)] if not events.empty else earnings.iloc[:0]
    for acc in sample(list(earnings.index), per_type):
        f = idx.loc[acc]
        questions.append({"type": "reaction", "ticker": f["ticker"], "accession": acc, "expected": float(events.loc[acc]["car"]),
                          "support": [],
                          "question": f"How did the stock react to the earnings release {f['company']} filed in "
                                      f"{f['date_filed']:%B %Y}?"})
    companies = store.companies()[["ticker", "company"]].to_dict("records")
    yes: list[dict] = []
    no: list[dict] = []
    for need, template in DISCLOSURE_QUESTIONS.items():
        for c in companies:
            support = ids[labels[need] & (tickers == c["ticker"])]
            q = {"ticker": c["ticker"], "accession": None, "need": need, "question": template.format(company=c["company"])}
            (yes if len(support) else no).append({**q, "type": "disclosure_yes" if len(support) else "disclosure_no",
                                                  "expected": bool(len(support)), "support": sorted(support)})
    questions += sample(yes, per_type * 2) + sample(no, per_type * 2)
    for i, q in enumerate(questions):
        q["id"] = i
    return questions


def score_answer(q: dict, ans: dict) -> dict:
    """Mechanical checks of one answer against what the generator knows."""
    text = ans.get("answer") or ""
    cited = [c["paragraph_id"] for c in ans.get("citations", [])]
    support = set(q["support"])
    out = {"citation_valid_rate": ans.get("citation_valid_rate", 1.0), "n_citations": len(cited),
           "latency_s": ans.get("latency_s", 0.0), "cost_usd": ans.get("cost_usd", 0.0), "error": bool(ans.get("error"))}
    if q["type"] == "guidance":
        out["support_cited"] = float(bool(support & set(cited)))
        out["first_citation_correct"] = float(bool(cited) and cited[0] in support)
        out["direction_stated"] = float(bool(DIRECTION[q["expected"]].search(text)))
        out["answer_correct"] = out["support_cited"] * out["direction_stated"]
    elif q["type"] == "reaction":
        values = [float(v.replace("−", "-")) for v in PERCENT.findall(text)]
        out["car_correct"] = float(any(abs(v - q["expected"] * 100) <= 0.011 for v in values))
    elif q["type"] == "disclosure_yes":
        out["support_cited"] = float(bool(support & set(cited)))
        out["citation_precision"] = float(len(support & set(cited)) / len(set(cited))) if cited else 0.0
    else:
        out["abstained"] = float(bool(ABSENCE.search(text)))
    out["headline"] = out[HEADLINE[q["type"]]]
    return out


def run_agent(agent, questions: list[dict], system: str, log=None) -> pd.DataFrame:
    rows = []
    for n, q in enumerate(questions, start=1):
        ans = agent.ask(q["question"], q["ticker"]).to_dict()
        rows.append({"system": system, "id": q["id"], "type": q["type"], **score_answer(q, ans),
                     "answer": (ans.get("answer") or "")[:600]})
        if log and n % 50 == 0:
            log(f"{system}: {n} of {len(questions)} questions")
    return pd.DataFrame(rows)


def _boot(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(rows: pd.DataFrame) -> dict:
    systems = list(dict.fromkeys(rows["system"]))
    out: dict = {"systems": {}, "paired": {}}
    for s in systems:
        d = rows[rows["system"] == s]
        per_type = {}
        for t, metric in HEADLINE.items():
            part = d[d["type"] == t]
            mean, lo, hi = _boot(part[metric].to_numpy(dtype=float))
            per_type[t] = {"n": int(len(part)), "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi}
        extra = {"guidance_support_cited": float(d[d["type"] == "guidance"]["support_cited"].mean()),
                 "guidance_first_citation_correct": float(d[d["type"] == "guidance"]["first_citation_correct"].mean()),
                 "disclosure_citation_precision": float(d[d["type"] == "disclosure_yes"]["citation_precision"].mean())}
        out["systems"][s] = {"types": per_type, **extra,
                             "macro_score": float(np.mean([v["mean"] for v in per_type.values()])),
                             "citation_valid_rate": float(d["citation_valid_rate"].mean()),
                             "errors": int(d["error"].sum()), "latency_p50_s": float(d["latency_s"].median()),
                             "latency_p95_s": float(d["latency_s"].quantile(0.95)), "cost_usd": float(d["cost_usd"].sum())}
    base = systems[0]
    for s in systems[1:]:
        a = rows[rows["system"] == s].set_index("id")["headline"]
        b = rows[rows["system"] == base].set_index("id")["headline"].reindex(a.index)
        types = rows[rows["system"] == s].set_index("id")["type"]
        out["paired"][f"{s} minus {base}"] = {}
        for t in HEADLINE:
            diff = (a - b)[types == t].to_numpy(dtype=float)
            mean, lo, hi = _boot(diff)
            out["paired"][f"{s} minus {base}"][t] = {"diff": mean, "ci_low": lo, "ci_high": hi}
    return out


def report_md(summary: dict, info: dict) -> str:
    names = {"guidance": "Guidance, dated (right paragraph cited and direction stated)",
             "reaction": "Stock reaction, dated (right return given)",
             "disclosure_yes": "Disclosure made (a stating paragraph cited)",
             "disclosure_no": "No such disclosure (says so)"}
    lines = ["# Agent evaluation (synthetic world)", "",
             f"{info['questions']} questions built from the generator's ground truth: "
             + ", ".join(f"{n} {t}" for t, n in info["by_type"].items()) + ". "
             f"Agent: {info['agent']}. Intervals are 95% bootstrap intervals over questions.", "",
             "| System | " + " | ".join(names[t] for t in HEADLINE) + " | Mean of the four | Citations valid |",
             "|---|" + "---|" * (len(HEADLINE) + 2)]
    for s, d in summary["systems"].items():
        cells = [f"{d['types'][t]['mean']:.3f} [{d['types'][t]['ci_low']:.3f}, {d['types'][t]['ci_high']:.3f}]" for t in HEADLINE]
        lines.append(f"| {s} | " + " | ".join(cells) + f" | {d['macro_score']:.3f} | {d['citation_valid_rate']:.3f} |")
    lines += ["", "| System | Guidance: right paragraph cited | Guidance: cited first | Disclosure: share of citations that state it | "
              "Latency p50 / p95 (s) | Cost (USD) | Errors |", "|---|---|---|---|---|---|---|"]
    for s, d in summary["systems"].items():
        lines.append(f"| {s} | {d['guidance_support_cited']:.3f} | {d['guidance_first_citation_correct']:.3f} | "
                     f"{d['disclosure_citation_precision']:.3f} | {d['latency_p50_s']:.3f} / {d['latency_p95_s']:.3f} | "
                     f"{d['cost_usd']:.2f} | {d['errors']} |")
    if summary["paired"]:
        lines += ["", "| Paired difference on the same questions | " + " | ".join(t for t in HEADLINE) + " |",
                  "|---|" + "---|" * len(HEADLINE)]
        for name, d in summary["paired"].items():
            lines.append(f"| {name} | " + " | ".join(
                f"{d[t]['diff']:+.3f} [{d[t]['ci_low']:+.3f}, {d[t]['ci_high']:+.3f}]" for t in HEADLINE) + " |")
    lines += ["", "Every check is mechanical: citations against the paragraphs the generator wrote, returns against the event "
              "table, the direction and a statement of absence by pattern. The patterns are strict for the offline agent, which "
              "quotes filing sentences; a model that words an answer freely can be marked wrong for phrasing, so read its "
              "direction and absence columns with answers.jsonl next to them."]
    return "\n".join(lines) + "\n"


def evaluate(lake: str, truth_path: str, out_dir: str | Path, modes=("bm25", "hybrid"), agent: str = "offline",
             per_type: int = 40, seed: int = 0, embed_model: str | None = None, client=None, log=print) -> dict:
    from .agent import AGENT_MODEL, ClaudeAgent, ExtractiveAgent
    from .embeddings import get_embedder
    from .retrieval import BM25Index, build_index
    from .store import DataStore
    from .tools import ToolBox

    store = DataStore(lake)
    questions = build_questions(store, pd.read_parquet(truth_path), per_type, seed)
    bm25 = BM25Index(store.paragraphs)
    embedder = get_embedder(embed_model) if any(m != "bm25" for m in modes) else None
    frames = []
    for mode in modes:
        index = build_index(store.paragraphs, mode, lake=lake, embedder=embedder, bm25=bm25, log=log)
        if agent == "claude":
            frames.append(run_agent(ClaudeAgent(ToolBox(store, index), client=client), questions, f"{AGENT_MODEL}, {mode}", log))
        else:
            for scoped in (False, True):
                name = f"offline, {mode}" + (", dated questions scoped" if scoped else "")
                frames.append(run_agent(ExtractiveAgent(ToolBox(store, index), scope_period=scoped), questions, name, log))
            if mode == "hybrid":  # ablation: the same agent with the company's name left in the search
                kept = ToolBox(store, index, strip_names=False)
                frames.append(run_agent(ExtractiveAgent(kept), questions, "offline, hybrid, dated questions scoped, company name kept", log))
    rows = pd.concat(frames, ignore_index=True)
    summary = summarise(rows)
    info = {"questions": len(questions), "by_type": {t: sum(q["type"] == t for q in questions) for t in HEADLINE},
            "agent": "offline extractive" if agent == "offline" else f"hosted model ({AGENT_MODEL})", "retrieval": list(modes),
            "per_type": per_type, "seed": seed}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"info": info, **summary}, indent=2))
    (out / "report.md").write_text(report_md(summary, info))
    (out / "questions.jsonl").write_text("".join(json.dumps(q, default=str) + "\n" for q in questions))
    rows.drop(columns=["answer"]).round(4).to_csv(out / "scores.csv", index=False)
    if agent == "claude":  # a model's wording is worth reading next to its marks; the offline agent only quotes
        (out / "answers.jsonl").write_text("".join(
            json.dumps({"system": r.system, "id": int(r.id), "answer": r.answer}) + "\n" for r in rows.itertuples(index=False)))
    return {"info": info, **summary}
