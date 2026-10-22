"""A synthetic, EDGAR shaped world for tests, CI and the offline demo.

It produces raw HTML filings (with tables of contents, page numbers, hidden inline XBRL headers and numeric
tables, like the real thing), a filing index with acceptance timestamps, daily prices for every ticker and
a market proxy, and the ground truth structured signals for every filing.

The data generating process plants a *weak* link between language and returns:
1. Each company has a latent quality state that follows an AR(1) process across quarters.
2. Filing language is sampled conditionally on that state (tone, risk factor count, liquidity language).
3. Earnings 8-Ks carry explicit guidance changes, phrased in many ways, some of them hard for a lexicon.
4. The abnormal return around the filing is a small function of those signals plus a lot of noise.

Nothing measured on this data says anything about real markets. It exists so every stage of the pipeline
can be executed and tested, and so the evaluation code can be checked against a known answer.
"""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .prices.events import event_day

SECTORS = [
    ("Software", 7372, ["cloud platform", "analytics suite", "security software", "subscription services"]),
    ("Semiconductors", 3674, ["data center chips", "automotive semiconductors", "memory products", "RF modules"]),
    ("Retail", 5331, ["store network", "e-commerce channel", "private label brands", "membership program"]),
    ("Biotech", 2836, ["oncology pipeline", "clinical programs", "biologics portfolio", "gene therapy platform"]),
    ("Industrials", 3560, ["aftermarket services", "industrial equipment", "automation systems", "aerospace parts"]),
    ("Banking", 6022, ["commercial lending", "deposit franchise", "wealth management", "mortgage banking"]),
    ("Energy", 1311, ["upstream production", "midstream assets", "refining operations", "renewable projects"]),
    ("Healthcare", 8062, ["hospital network", "outpatient services", "specialty pharmacy", "care management"]),
]
_ADJ = ["Northern", "Atlas", "Summit", "Pioneer", "Harbor", "Crescent", "Granite", "Vertex", "Beacon", "Sterling",
        "Meridian", "Cobalt", "Evergreen", "Liberty", "Keystone", "Orion", "Redwood", "Silverline", "Trident", "Zenith"]
_NOUN = ["Dynamics", "Systems", "Holdings", "Technologies", "Industries", "Networks", "Labs", "Partners", "Group",
         "Therapeutics", "Resources", "Brands", "Solutions", "Financial", "Energy", "Health"]

# ---- sentence banks. {p} product, {n} percent, {y} year, {a}/{b} money --------------------------------------
POS = [
    "Revenue increased {n}% compared to the prior year period, driven by strong demand for our {p}.",
    "Gross margin expanded as a result of favorable mix and improved pricing across our {p}.",
    "We delivered record operating income and continued to gain market share in our {p}.",
    "Customer retention improved and bookings growth accelerated during the period.",
    "Operating cash flow increased {n}%, which strengthened our balance sheet and gives us flexibility to invest.",
    "Demand trends remained robust and our backlog reached an all-time high.",
    "We achieved meaningful efficiency gains that more than offset inflationary pressure on costs.",
]
NEG = [
    "Revenue decreased {n}% compared to the prior year period due to weaker demand for our {p}.",
    "Gross margin declined as a result of unfavorable mix, pricing pressure and higher input costs.",
    "We recorded an impairment charge related to goodwill associated with our {p}.",
    "Operating results were adversely affected by lower volumes and elevated costs.",
    "We experienced delays in customer orders and a deterioration in our backlog.",
    "Restructuring charges increased as we reduced headcount and closed underperforming facilities.",
    "Losses widened during the period and we do not expect conditions to improve in the near term.",
]
NEUTRAL = [
    "Results for the period reflect the factors described below.",
    "We operate in a single reportable segment focused on our {p}.",
    "The following discussion should be read in conjunction with the consolidated financial statements.",
    "Selling, general and administrative expenses were {n}% of revenue for the period.",
    "We continue to evaluate opportunities to allocate capital consistent with our long term strategy.",
    "Interest expense was consistent with the prior year reflecting our fixed rate debt structure.",
]
TRICKY_POS = [  # positive meaning, negative looking words: lexicons misread these
    "We do not expect the resolution of these matters to have a material adverse effect on our results.",
    "Losses from the discontinued unit declined substantially, and the unit is no longer a drag on margins.",
    "The previously disclosed weakness in our {p} has been fully remediated.",
]
RISKS = [
    ("Competition", "We face intense competition in the markets for our {p}, and if we fail to compete effectively our results could suffer."),
    ("Macro", "Adverse economic conditions could reduce demand for our {p} and harm our results of operations."),
    ("Supply", "Disruptions in our supply chain could delay shipments and increase our costs."),
    ("Cyber", "A cybersecurity incident could disrupt our operations and expose us to liability."),
    ("Regulation", "Changes in laws and regulations could increase our compliance costs and restrict our business."),
    ("Talent", "Our success depends on our ability to attract and retain key personnel."),
    ("FX", "Fluctuations in foreign currency exchange rates could adversely affect our reported results."),
    ("Debt", "Our indebtedness could limit our flexibility and we may be unable to refinance on acceptable terms."),
    ("Customer", "A significant portion of our revenue comes from a limited number of customers."),
    ("IP", "We may be unable to protect our intellectual property, which could harm our competitive position."),
    ("Climate", "Climate related events and regulation could disrupt our facilities and increase costs."),
    ("Pandemic", "Public health crises could adversely affect our workforce, suppliers and customers."),
]
LIQUIDITY = [
    "These conditions raise substantial doubt about our ability to continue as a going concern.",
    "We were not in compliance with certain financial covenants and have requested a waiver from our lenders.",
    "We will need to raise additional capital within the next twelve months to fund our operations, which may not be available on acceptable terms.",
]
LITIGATION = [
    "On {d}, a putative securities class action was filed against the Company and certain officers alleging false and misleading statements.",
    "The Company received a subpoena from the Securities and Exchange Commission requesting documents related to revenue recognition.",
    "A jury returned a verdict against the Company in a patent infringement case and awarded damages of ${a} million, which we intend to appeal.",
]
GUIDANCE = {
    "raised": [
        "The Company is raising its full year {y} revenue guidance to ${a} billion to ${b} billion.",
        "Based on the strength of the quarter, we now expect full year {y} results above the high end of our prior outlook.",
        "Management increased its outlook for fiscal {y} earnings per share.",
    ],
    "lowered": [
        "The Company is lowering its full year {y} revenue guidance to ${a} billion to ${b} billion.",
        "We now expect full year {y} results to come in below the low end of our previously issued outlook.",
        "Given the softer environment, management reduced its fiscal {y} earnings outlook.",
    ],
    "maintained": [
        "The Company reaffirms its full year {y} revenue guidance of ${a} billion to ${b} billion.",
        "Our outlook for fiscal {y} is unchanged from the guidance provided last quarter.",
    ],
}
EIGHTK_OTHER = {
    "5.02": ["On {d}, the Company announced that its Chief Executive Officer will step down, effective immediately.",
             "The Board of Directors appointed a new Chief Financial Officer, effective {d}."],
    "1.01": ["On {d}, the Company entered into a definitive agreement to supply its {p} to a large customer under a multi year contract.",
             "The Company entered into a new credit agreement providing a revolving credit facility of ${a} million."],
    "8.01": ["The Company issued a press release announcing a new share repurchase authorization of ${a} million.",
             "The Company announced the opening of a new facility to support its {p}."],
}


@dataclass
class SyntheticConfig:
    n_companies: int = 60
    start: str = "2012-01-01"
    end: str = "2024-12-31"
    seed: int = 7
    tone_effect: float = 0.006      # abnormal return per 1 sd of latent tone (10-K / 10-Q)
    guidance_effect: float = 0.02   # abnormal return for raised (+) / lowered (-) guidance (8-K 2.02)
    liquidity_effect: float = -0.03
    litigation_effect: float = -0.012
    periodic_noise: float = 0.035
    eightk_noise: float = 0.05
    other_8k_rate: float = 1.5      # extra 8-Ks per company per year


def _ticker(rng, used: set[str]) -> str:
    while True:
        t = "".join(rng.choice(list(string.ascii_uppercase), size=rng.integers(3, 5)))
        if t not in used and t != "SPY":
            used.add(t)
            return t


def _fill(tpl: str, rng, product: str, year: int) -> str:
    a = int(rng.integers(2, 40))
    return tpl.format(
        p=product, n=int(rng.integers(2, 25)), y=year, a=a, b=a + int(rng.integers(1, 5)),
        d=f"{pd.Timestamp(year=year, month=int(rng.integers(1, 13)), day=int(rng.integers(1, 28))):%B %d, %Y}",
    )


def _tone_paragraphs(rng, tone: float, products: list[str], year: int, n_paras: int) -> list[str]:
    p_pos = 1 / (1 + np.exp(-1.6 * tone))
    paras = []
    for _ in range(n_paras):
        sents = []
        for _ in range(int(rng.integers(3, 6))):
            u = rng.random()
            prod = products[int(rng.integers(len(products)))]
            if u < 0.35:
                bank = POS if rng.random() < p_pos else NEG
            elif u < 0.42 and tone > 0:
                bank = TRICKY_POS
            else:
                bank = NEUTRAL
            sents.append(_fill(bank[int(rng.integers(len(bank)))], rng, prod, year))
        paras.append(" ".join(sents))
    return paras


def _risk_section(rng, risk_keys: list[str], products: list[str], year: int) -> list[str]:
    lookup = dict(RISKS)
    paras = []
    for k in risk_keys:
        body = _fill(lookup[k], rng, products[int(rng.integers(len(products)))], year)
        extra = " There can be no assurance that we will be able to mitigate these risks, and the occurrence of any of them could materially and adversely affect our business, financial condition and results of operations."
        paras.append(f"{k} risk. {body}{extra}")
    return paras


def _html(company: str, form: str, period: str, sections: list[tuple[str, list[str]]], rng) -> str:
    toc = "".join(f"<tr><td>{h}</td><td>{i + 3}</td></tr>" for i, (h, _) in enumerate(sections))
    body = []
    for i, (header, paras) in enumerate(sections):
        body.append(f'<p style="font-weight:bold">{header}</p>')
        body.extend(f"<p>{p}</p>" for p in paras)
        body.append(f'<p style="text-align:center">{i + 3}</p><hr/>')
        if header.lower().startswith(("item 8", "item 1. financial")):
            rev = int(rng.integers(500, 9000))
            body.append(
                "<table><tr><td></td><td>Current</td><td>Prior</td></tr>"
                f"<tr><td>Revenue</td><td>$ {rev:,}</td><td>$ {int(rev * 0.93):,}</td></tr>"
                f"<tr><td>Net income</td><td>{int(rev * 0.1):,}</td><td>({int(rev * 0.02):,})</td></tr></table>"
            )
    return (
        "<html><head><title>{c} {f}</title><style>p{{margin:0}}</style></head><body>"
        '<div style="display:none"><ix:header><ix:hidden>dei:EntityRegistrantName</ix:hidden></ix:header></div>'
        "<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM {f}</p><p>{c}</p><p>For the period ended {pe}</p>"
        "<p>TABLE OF CONTENTS</p><table>{toc}</table>{body}</body></html>"
    ).format(c=company, f=form, pe=period, toc=toc, body="".join(body))


def _accepted(rng, day: pd.Timestamp, kind: str) -> pd.Timestamp:
    if kind == "earnings":  # before the open or just after the close
        hour, minute = (7, int(rng.integers(0, 59))) if rng.random() < 0.5 else (16, int(rng.integers(1, 59)))
    else:
        hour, minute = int(rng.integers(6, 22)), int(rng.integers(0, 59))
    ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour, minute=minute, tz="America/New_York")
    return ts.tz_convert("UTC")


def generate(cfg: SyntheticConfig | None = None, out_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Generate the synthetic world. Writes files to ``out_dir`` when given and returns the frames."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    days = pd.bdate_range(cfg.start, cfg.end)
    n_days = len(days)
    mkt = rng.normal(0.0003, 0.011, n_days)
    out = Path(out_dir) if out_dir else None
    used: set[str] = set()
    companies, filings, truth, facts = [], [], [], []
    price_frames = []
    years = range(days[0].year, days[-1].year)  # fiscal years whose reports land inside the window

    for i in range(cfg.n_companies):
        sector, sic, products = SECTORS[i % len(SECTORS)]
        cik = 1_000_000 + i * 37
        ticker = _ticker(rng, used)
        name = f"{_ADJ[int(rng.integers(len(_ADJ)))]} {_NOUN[int(rng.integers(len(_NOUN)))]} Inc"
        beta, idio = rng.uniform(0.6, 1.5), rng.uniform(0.012, 0.028)
        companies.append({"cik": cik, "ticker": ticker, "company": name, "sector": sector, "sic": sic})
        ret = beta * mkt + rng.normal(0, idio, n_days)
        volume = rng.lognormal(13, 0.3, n_days)
        q_prev, q = 0.0, rng.normal()
        revenue = rng.uniform(200, 5000)
        assets = revenue * rng.uniform(2, 4)
        risk_keys = list(rng.choice([k for k, _ in RISKS], size=6, replace=False))
        seq = 0
        first = len(filings)
        for year in years:
            for qtr in (1, 2, 3, 4):
                q_prev, q = q, 0.7 * q + rng.normal(0, 0.7)
                is_annual = qtr == 4
                rep_year = year + 1 if is_annual else year
                month = {1: 5, 2: 8, 3: 11, 4: 2}[qtr]
                base = pd.Timestamp(year=rep_year, month=month, day=1)
                earn_day = base + pd.Timedelta(days=int(rng.integers(0, 12)))
                rep_day = earn_day + pd.Timedelta(days=int(rng.integers(1, 25)))
                if rep_day > days[-1]:
                    continue
                delta = q - q_prev
                guidance = (
                    "raised" if delta > 0.6 else "lowered" if delta < -0.6 else "maintained" if rng.random() < 0.6 else "none"
                )
                liquidity = bool(q < -1.6 and rng.random() < 0.7)
                litigation = bool(rng.random() < 0.06)
                # risk factor list evolves: deteriorating companies add risks
                risk_change = "stable"
                if delta < -0.5 and len(risk_keys) < len(RISKS):
                    risk_keys.append(next(k for k, _ in RISKS if k not in risk_keys))
                    risk_change = "increasing"
                elif delta > 0.8 and len(risk_keys) > 4:
                    risk_keys.pop(int(rng.integers(len(risk_keys))))
                    risk_change = "decreasing"
                tone_obs = q + rng.normal(0, 0.5)
                tone_label = "positive" if tone_obs > 0.5 else "negative" if tone_obs < -0.5 else "neutral"
                period = f"{year}-{qtr * 3:02d}-{[31, 30, 30, 31][qtr - 1]}"

                # ---- earnings 8-K (item 2.02) -----------------------------------------------------------
                seq += 1
                paras = _tone_paragraphs(rng, q, products, year, 2)
                guid_sentence = None
                if guidance != "none":
                    guid_sentence = _fill(GUIDANCE[guidance][int(rng.integers(len(GUIDANCE[guidance])))], rng, products[0], year + (1 if is_annual else 0))
                    paras.insert(int(rng.integers(0, len(paras) + 1)), guid_sentence)
                docs = [("Item 2.02 Results of Operations and Financial Condition", [
                    f"On {earn_day:%B %d, %Y}, {name} issued a press release announcing its financial results for the quarter ended {period}.", *paras]),
                    ("Item 9.01 Financial Statements and Exhibits", ["Exhibit 99.1 Press release dated " + f"{earn_day:%B %d, %Y}."])]
                filings.append(_mk_filing(cik, name, ticker, sic, "8-K", earn_day, _accepted(rng, earn_day, "earnings"), seq, "2.02,9.01", period))
                truth.append(_truth(filings[-1], guidance, tone_label, False, False, "stable", guid_sentence,
                                    cfg.guidance_effect * {"raised": 1, "lowered": -1}.get(guidance, 0) + 0.004 * q, cfg.eightk_noise))
                filings[-1]["_html"] = _html(name, "8-K", period, docs, rng)

                # ---- periodic report (10-K or 10-Q) ------------------------------------------------------
                seq += 1
                form = "10-K" if is_annual else "10-Q"
                mdna = _tone_paragraphs(rng, q, products, year, int(rng.integers(4, 8)))
                if liquidity:
                    mdna.insert(int(rng.integers(0, len(mdna) + 1)), _fill(LIQUIDITY[int(rng.integers(len(LIQUIDITY)))], rng, products[0], year))
                legal = ["From time to time we are involved in legal proceedings arising in the ordinary course of business. " * 2]
                lit_sentence = None
                if litigation:
                    lit_sentence = _fill(LITIGATION[int(rng.integers(len(LITIGATION)))], rng, products[0], year)
                    legal.append(lit_sentence)
                risks = _risk_section(rng, risk_keys, products, year)
                if form == "10-K":
                    sections = [
                        ("Item 1. Business", [f"{name} is a leading provider of {', '.join(products)}. " * 3]),
                        ("Item 1A. Risk Factors", risks),
                        ("Item 2. Properties", ["Our headquarters are leased. We believe our facilities are adequate for our current needs. " * 4]),
                        ("Item 3. Legal Proceedings", legal),
                        ("Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations", mdna),
                        ("Item 8. Financial Statements and Supplementary Data", ["The consolidated financial statements are included below. " * 8]),
                    ]
                else:
                    sections = [
                        ("PART I. FINANCIAL INFORMATION", ["The following unaudited interim financial statements are presented. " * 7]),
                        ("Item 1. Financial Statements", ["Condensed consolidated financial statements are presented below. " * 7]),
                        ("Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations", mdna),
                        ("PART II. OTHER INFORMATION", ["Other information required by Form 10-Q follows. " * 9]),
                        ("Item 1. Legal Proceedings", legal),
                        ("Item 1A. Risk Factors", risks),
                    ]
                effect = (
                    cfg.tone_effect * q + cfg.liquidity_effect * liquidity + cfg.litigation_effect * litigation
                    - 0.004 * (risk_change == "increasing")
                )
                filings.append(_mk_filing(cik, name, ticker, sic, form, rep_day, _accepted(rng, rep_day, "periodic"), seq, "", period))
                revenue *= 1 + 0.015 + 0.03 * q + rng.normal(0, 0.02)
                assets *= 1 + 0.01 + 0.01 * q + rng.normal(0, 0.01)
                margin = 0.08 + 0.05 * q + rng.normal(0, 0.02)
                values = {
                    "revenue": revenue, "net_income": revenue * margin, "assets": assets,
                    "liabilities": assets * float(np.clip(0.5 - 0.08 * q + rng.normal(0, 0.05), 0.1, 0.95)),
                    "cash": assets * float(np.clip(0.12 + 0.03 * q + rng.normal(0, 0.02), 0.01, 0.6)),
                    "op_cash_flow": revenue * (margin + rng.normal(0.03, 0.02)),
                }
                for metric, val in values.items():
                    facts.append({"cik": cik, "metric": metric, "value": float(val), "period_end": pd.Timestamp(period),
                                  "fp": "FY" if is_annual else f"Q{qtr}", "form": form, "filed": rep_day.normalize(),
                                  "accession": filings[-1]["accession"]})
                truth.append(_truth(filings[-1], "none", tone_label, liquidity, litigation, risk_change,
                                    lit_sentence, effect, cfg.periodic_noise))
                filings[-1]["_html"] = _html(name, form, period, sections, rng)

                # ---- other 8-Ks ---------------------------------------------------------------------------
                for _ in range(rng.poisson(cfg.other_8k_rate / 4)):
                    item = ["5.02", "1.01", "8.01"][int(rng.integers(3))]
                    d = rep_day + pd.Timedelta(days=int(rng.integers(5, 60)))
                    if d > days[-1]:
                        continue
                    seq += 1
                    sent = _fill(EIGHTK_OTHER[item][int(rng.integers(2))], rng, products[0], d.year)
                    eff = -0.01 if "step down" in sent else 0.006 if item == "1.01" else 0.0
                    docs = [(f"Item {item} {'Departure of Directors or Certain Officers' if item == '5.02' else 'Entry into a Material Definitive Agreement' if item == '1.01' else 'Other Events'}",
                             [sent + " " + _fill(NEUTRAL[int(rng.integers(len(NEUTRAL)))], rng, products[0], d.year) * 2])]
                    filings.append(_mk_filing(cik, name, ticker, sic, "8-K", d, _accepted(rng, d, "periodic"), seq, item, ""))
                    truth.append(_truth(filings[-1], "none", "neutral", False, False, "stable", sent, eff, cfg.eightk_noise * 0.7))
                    filings[-1]["_html"] = _html(name, "8-K", "", docs, rng)

        # inject the event reactions into this company's return path
        for f, t in zip(filings[first:], truth[first:]):
            d0 = event_day(pd.Timestamp(f["accepted_at"]), days)
            if d0 is None:
                continue
            i0 = days.get_loc(d0)
            shock = t["true_effect"] + rng.normal(0, t["noise_sd"])
            for k, w in enumerate((0.6, 0.25, 0.15)):
                if i0 + k < n_days:
                    ret[i0 + k] += shock * w
                    volume[i0 + k] *= 1 + 2.5 * abs(shock) / 0.03 * w
        close = 40 * np.cumprod(1 + np.clip(ret, -0.6, 2))
        price_frames.append(pd.DataFrame({"ticker": ticker, "date": days, "close": close, "volume": volume.round()}))

    price_frames.append(pd.DataFrame({"ticker": "SPY", "date": days, "close": 200 * np.cumprod(1 + mkt), "volume": 8e7}))
    prices = pd.concat(price_frames, ignore_index=True)
    comp = pd.DataFrame(companies)
    idx = pd.DataFrame([{k: v for k, v in f.items() if k != "_html"} for f in filings])
    tr = pd.DataFrame(truth)
    fx = pd.DataFrame(facts)
    result = {"companies": comp, "index": idx, "prices": prices, "truth": tr, "facts": fx}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        for f in filings:
            p = out / "raw" / str(f["cik"]) / f"{f['accession']}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f["_html"])
        idx = idx.assign(raw_uri=[str(out / "raw" / str(c) / f"{a}.txt") for c, a in zip(idx["cik"], idx["accession"])])
        result["index"] = idx
        idx.to_parquet(out / "index.parquet", index=False)
        comp.to_parquet(out / "companies.parquet", index=False)
        prices.to_parquet(out / "prices.parquet", index=False)
        tr.to_parquet(out / "truth.parquet", index=False)
        fx.to_parquet(out / "facts.parquet", index=False)
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return result


def _mk_filing(cik, name, ticker, sic, form, day, accepted, seq, items, period) -> dict:
    acc = f"{cik:010d}-{day.year % 100:02d}-{seq:06d}"
    return {
        "cik": cik, "company": name, "ticker": ticker, "sic": sic, "form": form, "accession": acc,
        "date_filed": day.normalize(), "accepted_at": accepted.isoformat(), "items": items,
        "primary_document": f"{acc.replace('-', '')}.htm", "period_of_report": period,
    }


def _truth(filing, guidance, tone, liquidity, litigation, risk_change, quote, effect, noise) -> dict:
    return {
        "accession": filing["accession"], "cik": filing["cik"], "form": filing["form"],
        "guidance_direction": guidance, "management_tone": tone, "liquidity_concern": bool(liquidity),
        "litigation_risk": bool(litigation), "risk_trend": risk_change, "evidence": quote,
        "true_effect": float(effect), "noise_sd": float(noise),
    }
