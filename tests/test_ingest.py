import json
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from filings_lab.edgar.xbrl import facts_from_bulk
from filings_lab.prices.build import build_prices
from filings_lab.prices.sources import CsvPriceSource, StooqPriceSource

CSV = "Date,Open,High,Low,Close,Volume\n2021-01-04,10,11,9,10.5,1000\n2021-01-05,10.5,11,10,10.8,1200\n"


class FakeSession:
    def __init__(self, known):
        self.known, self.urls = known, []

    def get(self, url, timeout):
        self.urls.append(url)
        sym = url.split("s=")[1].split("&")[0]
        ok = sym.split(".")[0] in self.known
        return SimpleNamespace(status_code=200, text=CSV if ok else "No data")


def test_stooq_prices_and_coverage_report():
    idx = pd.DataFrame({"ticker": ["AAA", "BBB", None], "cik": [1, 2, 3]})
    src = StooqPriceSource(session=FakeSession({"aaa", "spy"}))
    prices, rep = build_prices(idx, src, "2021-01-01", "2021-12-31")
    assert set(prices["ticker"]) == {"AAA", "SPY"} and rep["missing"] == ["BBB"]
    assert rep["market_found"] and rep["filings_with_prices"] == pytest.approx(1 / 3)
    assert prices["close"].tolist()[:2] == [10.5, 10.8]


def test_missing_market_proxy_is_an_error():
    idx = pd.DataFrame({"ticker": ["AAA"], "cik": [1]})
    with pytest.raises(ValueError):
        build_prices(idx, StooqPriceSource(session=FakeSession({"aaa"})), "2021-01-01", "2021-12-31")


def test_csv_directory_source(tmp_path):
    (tmp_path / "AAA.csv").write_text(CSV.replace("Close", "Adj Close"))
    df = CsvPriceSource(tmp_path).history(["AAA", "ZZZ"], "2021-01-01", "2021-01-31")
    assert len(df) == 2 and df["ticker"].unique().tolist() == ["AAA"]


def test_companyfacts_bulk(tmp_path):
    payload = {"cik": 5, "facts": {"us-gaap": {"Assets": {"units": {"USD": [
        {"val": 10, "end": "2020-12-31", "filed": "2021-02-01", "form": "10-K", "accn": "x", "fy": 2020, "fp": "FY"}]}}}}}
    z = tmp_path / "companyfacts.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("CIK0000000005.json", json.dumps(payload))
        zf.writestr("CIK0000000006.json", json.dumps({"cik": 6, "facts": {}}))
    facts = facts_from_bulk(str(z), {5, 6})
    assert len(facts) == 1 and facts["metric"].iloc[0] == "assets"
    assert facts_from_bulk(str(z), {6}).empty
