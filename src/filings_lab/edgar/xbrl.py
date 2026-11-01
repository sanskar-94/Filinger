"""Point in time fundamentals from XBRL companyfacts."""
from __future__ import annotations

import pandas as pd

from . import urls

CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


def parse_company_facts(payload: dict) -> pd.DataFrame:
    """Long table: cik, metric, value, period_end, fy, fp, form, filed, accession."""
    rows = []
    gaap = payload.get("facts", {}).get("us-gaap", {})
    for metric, candidates in CONCEPTS.items():
        for concept in candidates:
            if concept not in gaap:
                continue
            for unit, facts in gaap[concept].get("units", {}).items():
                for f in facts:
                    rows.append(
                        {
                            "cik": int(payload["cik"]),
                            "metric": metric,
                            "concept": concept,
                            "unit": unit,
                            "value": f.get("val"),
                            "period_start": f.get("start"),
                            "period_end": f.get("end"),
                            "fy": f.get("fy"),
                            "fp": f.get("fp"),
                            "form": f.get("form"),
                            "filed": f.get("filed"),
                            "accession": f.get("accn"),
                        }
                    )
            break  # first concept that exists wins
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("period_start", "period_end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_company_facts(client, cik: int) -> pd.DataFrame:
    return parse_company_facts(client.get_json(urls.company_facts(cik)))


def point_in_time(facts: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """Latest value of each metric that was *filed* strictly before ``as_of``."""
    if facts.empty:
        return {}
    known = facts[facts["filed"] < as_of]
    if known.empty:
        return {}
    known = known.sort_values(["filed", "period_end"])
    return known.groupby("metric")["value"].last().to_dict()


def facts_from_api(client, ciks) -> pd.DataFrame:
    frames = []
    for cik in ciks:
        try:
            frames.append(fetch_company_facts(client, int(cik)))
        except Exception as exc:  # companies without XBRL (older or foreign filers) return 404
            print(f"cik {cik}: {exc}")
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def facts_from_bulk(zip_path: str, ciks: set[int] | None = None) -> pd.DataFrame:
    """Parse the nightly companyfacts.zip (one CIK##########.json per filer) without any API calls."""
    import json
    import zipfile

    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not (name.startswith("CIK") and name.endswith(".json")):
                continue
            if ciks is not None and int(name[3:13]) not in ciks:
                continue
            df = parse_company_facts(json.loads(zf.read(name)))
            if not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
