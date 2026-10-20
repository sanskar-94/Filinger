"""Finance tone lexicon.

The repo ships a compact seed lexicon written for this project (finance specific in the spirit of
Loughran and McDonald 2011: words like "liability" or "tax" are *not* negative here). If you have the
official Loughran McDonald master dictionary CSV, ``Lexicon.from_lm_csv`` loads it instead; its licence
terms are the user's responsibility, which is why it is not redistributed here.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field

SEED: dict[str, set[str]] = {
    "negative": set("""
        abandon abandoned adverse adversely against breach breached closure closed closing concern concerns
        decline declined declines declining decrease decreased decreases default defaults deficiency deficit
        delay delayed delays delinquent deteriorate deteriorated deterioration difficult difficulties disappoint
        disappointing discontinue discontinued dispute disputes disrupt disruption disruptions doubt downgrade
        downturn drag fail failed failure failures fraud harm harmed headwind headwinds impair impaired impairment
        impairments inability inadequate investigation lawsuit layoff layoffs lose loss losses lower lowered
        misconduct negative negatively penalty penalties pressure problem problems recall restate restated
        restatement restructuring shortfall slowdown slowing softer softness suspend suspended terminate
        terminated termination unable uncollectible underperform underperforming unfavorable unfavorably volatile
        weak weaken weakened weaker weakness worse worsen worsened writedown writeoff
    """.split()),
    "positive": set("""
        achieve achieved achievement advance advances advantage beneficial benefit benefited best better boost
        breakthrough confident delivered efficiency efficient enhance enhanced enhancement excellent exceed
        exceeded exceeding expand expanded expansion favorable favorably gain gained gains great greater growth
        improve improved improvement improvements improving increase increased innovation innovative leadership
        momentum opportunities opportunity optimistic outperform outperformed positive profitable profitability
        progress raised raising rebound record robust solid stable strength strengthen strengthened strong
        stronger success successful successfully surpass upside
    """.split()),
    "uncertainty": set("""
        almost ambiguity anticipate apparent appear appears approximate approximately assume assumption
        assumptions believe cannot contingency contingent could depend dependent depends doubt estimate
        estimated estimates exposure fluctuate fluctuation fluctuations may maybe might nearly pending possible
        possibly predict preliminary presume probable probably random risk risks roughly seldom sometimes
        speculative suggest uncertain uncertainties uncertainty unclear unknown unpredictable variability vary
    """.split()),
    "litigious": set("""
        allegation allegations alleged alleging appeal arbitration attorney claimant claims class complaint
        counsel court defendant defendants deposition indictment infringement injunction judgment jury lawsuit
        lawsuits legal litigation plaintiff plaintiffs proceeding proceedings prosecution putative regulator
        settlement settlements subpoena sued testimony tribunal verdict violation violations
    """.split()),
    "constraining": set("""
        commit commitment commitments compliance comply constrain constrained constraint constraints covenant
        covenants obligate obligated obligation obligations prohibit prohibited require required requirement
        requirements restrict restricted restriction restrictions waiver
    """.split()),
    "strong_modal": set("always best clearly definitely highest must never strongly undoubtedly will".split()),
    "weak_modal": set("almost appear could depend may maybe might nearly possibly sometimes suggest".split()),
    "forward_looking": set("""
        anticipate anticipates believe believes continue estimate estimates expect expects forecast future
        goal guidance intend intends outlook plan plans project projected target will
    """.split()),
}
NEGATIONS = {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "nobody", "isn't", "don't", "doesn't"}
_TOKEN = re.compile(r"[a-z][a-z'\-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Lexicon:
    categories: dict[str, set[str]] = field(default_factory=lambda: {k: set(v) for k, v in SEED.items()})

    @classmethod
    def from_lm_csv(cls, path: str) -> Lexicon:
        """Load the Loughran McDonald master dictionary (columns are category names, nonzero = member)."""
        mapping = {"Negative": "negative", "Positive": "positive", "Uncertainty": "uncertainty",
                   "Litigious": "litigious", "Constraining": "constraining", "Strong_Modal": "strong_modal",
                   "Weak_Modal": "weak_modal"}
        cats: dict[str, set[str]] = {v: set() for v in mapping.values()}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                word = row["Word"].lower()
                for col, cat in mapping.items():
                    if row.get(col) and row[col] not in ("0", ""):
                        cats[cat].add(word)
        cats["forward_looking"] = set(SEED["forward_looking"])
        return cls(cats)

    def score(self, tokens: list[str]) -> dict[str, float]:
        """Counts per category. Positive words within three tokens after a negation count as negative."""
        counts = {k: 0 for k in self.categories}
        pos, neg = self.categories["positive"], self.categories["negative"]
        for i, tok in enumerate(tokens):
            for cat, words in self.categories.items():
                if tok not in words:
                    continue
                if cat == "positive" and any(t in NEGATIONS for t in tokens[max(0, i - 3) : i]):
                    counts["negative"] += 1
                    continue
                counts[cat] += 1
        n = max(len(tokens), 1)
        out = {f"lex_{k}": v / n for k, v in counts.items()}
        p, q = counts["positive"], counts["negative"]
        out["lex_net_tone"] = (p - q) / (p + q + 1)
        _ = pos, neg
        return out
