"""Assemble the modelling table: one row per filing event with text, change, financial and market features.

``company_text_features`` is a pure function over one company's sections, used directly by the local
builder and via ``groupBy("cik").applyInPandas`` in ``features/spark_features.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .text_features import section_features, similarity

# section name -> (column prefix, which features to keep; None keeps all)
SECTION_SPEC: dict[str, tuple[str, list[str] | None]] = {
    "mdna": ("mdna", None),
    "8k_results_of_operations": ("ear", None),
    "risk_factors": ("rf", ["log_words", "lex_negative", "lex_uncertainty", "lex_litigious", "lex_constraining", "fog"]),
    "legal": ("legal", ["log_words", "lex_litigious", "lex_negative"]),
}
CHANGE_SECTIONS = {"mdna": "mdna", "risk_factors": "rf"}
COMPARE_WINDOW_DAYS = (300, 430)  # same form, roughly one year earlier (10-Q compares like quarters)
MARKET_FEATURES = ["mom_250_30", "ret_5_1", "vol_60_1", "beta", "mkt_ret_20_1", "abn_volume_5_1"]
EIGHTK_ITEMS = ["1.01", "2.02", "5.02", "8.01"]


def company_text_features(sections: pd.DataFrame) -> pd.DataFrame:
    """``sections`` for one company with columns accession, form, section, text, date_filed."""
    rows: dict[str, dict] = {}
    for r in sections.itertuples(index=False):
        spec = SECTION_SPEC.get(r.section)
        if spec is None:
            continue
        prefix, keep = spec
        feats = section_features(r.text)
        if keep is not None:
            feats = {k: feats[k] for k in keep}
        rows.setdefault(r.accession, {}).update({f"{prefix}_{k}": v for k, v in feats.items()})

    # year over year change ("Lazy Prices"): compare with the same form and section about a year earlier
    sec = sections[sections["section"].isin(CHANGE_SECTIONS)].sort_values("date_filed")
    for (form, section), g in sec.groupby(["form", "section"]):
        dates = pd.to_datetime(g["date_filed"]).to_numpy()
        texts, accs = g["text"].tolist(), g["accession"].tolist()
        for i in range(len(g)):
            lag = (dates[i] - dates[:i]) / np.timedelta64(1, "D")
            cand = np.where((lag >= COMPARE_WINDOW_DAYS[0]) & (lag <= COMPARE_WINDOW_DAYS[1]))[0]
            if len(cand) == 0:
                continue
            j = cand[-1]
            sim = similarity(texts[i], texts[j])
            p = CHANGE_SECTIONS[section]
            rows.setdefault(accs[i], {}).update({f"{p}_{k}": v for k, v in sim.items()})
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "accession"
    return out.reset_index()


def build_features(
    sections: pd.DataFrame, events: pd.DataFrame, index: pd.DataFrame, facts: pd.DataFrame | None = None
) -> pd.DataFrame:
    meta = index[["accession", "date_filed", "items"]].copy()
    meta["date_filed"] = pd.to_datetime(meta["date_filed"])
    sec = sections.merge(meta[["accession", "date_filed"]], on="accession")
    parts = [company_text_features(g) for _, g in sec.groupby("cik")]
    text = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["accession"])
    return assemble(events, meta, text, facts)


def assemble(events: pd.DataFrame, meta: pd.DataFrame, text: pd.DataFrame, facts: pd.DataFrame | None) -> pd.DataFrame:
    from .financial import fundamentals_asof

    df = events.merge(meta, on="accession", how="left").merge(text, on="accession", how="left")
    items = df["items"].fillna("").astype(str)
    for it in EIGHTK_ITEMS:
        df[f"item_{it.replace('.', '_')}"] = items.str.contains(it, regex=False).astype(np.int8)
    for f in ("10-K", "10-Q", "8-K"):
        df[f"form_{f.replace('-', '').lower()}"] = (df["form"].str.replace("/A", "") == f).astype(np.int8)
    df["month"] = pd.to_datetime(df["event_date"]).dt.month.astype(np.int8)
    if facts is not None and not facts.empty:
        df = df.merge(fundamentals_asof(facts, df[["accession", "cik", "date_filed"]]), on="accession", how="left")
    df["t1"] = pd.to_datetime(df["event_date"]) + pd.offsets.BDay(2)  # label end, used for purging
    df["year"] = pd.to_datetime(df["event_date"]).dt.year
    return df.drop(columns=["items"]).sort_values("event_date").reset_index(drop=True)


def feature_groups(columns) -> dict[str, list[str]]:
    cols = list(columns)
    text_tone = [c for c in cols if c.startswith(("mdna_", "ear_", "rf_", "legal_")) and not c.endswith(
        ("sim_cosine", "sim_jaccard", "len_change")) and "guidance" not in c and "going_concern" not in c]
    return {
        "market": [c for c in MARKET_FEATURES if c in cols],
        "text_tone": text_tone,
        "text_change": [c for c in cols if c.endswith(("sim_cosine", "sim_jaccard", "len_change"))],
        "text_rules": [c for c in cols if "guidance" in c or "going_concern" in c],
        "financial": [c for c in cols if c.startswith("fin_")],
        "meta": [c for c in cols if c.startswith(("form_", "item_"))] + (["month"] if "month" in cols else []),
    }
