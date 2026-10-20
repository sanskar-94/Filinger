import time

import pandas as pd
import pytest

from filings_lab.edgar import urls
from filings_lab.edgar.client import EdgarClient, RateLimiter
from filings_lab.edgar.index import parse_master_index
from filings_lab.edgar.submissions import parse_submissions
from filings_lab.edgar.xbrl import parse_company_facts, point_in_time

MASTER = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2021

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|Apple Inc.|10-Q|2021-01-28|edgar/data/320193/0000320193-21-000010.txt
320193|Apple Inc.|SC 13G/A|2021-02-05|edgar/data/320193/0000320193-21-000011.txt
789019|MICROSOFT CORP|8-K|2021-01-26|edgar/data/789019/0001193125-21-018233.txt
789019|MICROSOFT CORP|10-K/A|2021-02-01|edgar/data/789019/0001193125-21-020000.txt
"""


def test_master_index_filters_forms_and_parses_accession():
    df = parse_master_index(MASTER)
    assert list(df["form"]) == ["10-Q", "8-K", "10-K/A"]
    assert df.loc[0, "accession"] == "0000320193-21-000010"
    assert df.loc[2, "base_form"] == "10-K" and bool(df.loc[2, "is_amendment"])
    assert df["date_filed"].dtype.kind == "M"


def test_client_requires_declared_user_agent():
    with pytest.raises(ValueError):
        EdgarClient(user_agent="python-requests/2.0")
    with pytest.raises(ValueError):
        EdgarClient(user_agent="Lab ops@lab.org", max_rps=20)
    assert EdgarClient(user_agent="Lab Research ops@lab.org").session.headers["User-Agent"].startswith("Lab")


def test_rate_limiter_caps_throughput():
    lim = RateLimiter(rate=50)
    t0 = time.monotonic()
    for _ in range(26):
        lim.acquire()
    assert time.monotonic() - t0 >= 0.45  # 25 waits at 50/s


def test_client_retries_then_caches(tmp_path):
    class Resp:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

        def raise_for_status(self):
            raise RuntimeError(self.status_code)

    class Sess:
        headers: dict = {}
        calls = 0

        def get(self, url, timeout):
            Sess.calls += 1
            return Resp(429) if Sess.calls == 1 else Resp(200, b'{"ok": 1}')

    c = EdgarClient(user_agent="Lab ops@lab.org", max_rps=10, cache_dir=tmp_path, session=Sess())
    c.max_retries = 3
    import filings_lab.edgar.client as mod

    orig = mod.time.sleep
    mod.time.sleep = lambda s: None
    try:
        assert c.get_json("https://x/y.json") == {"ok": 1}
        assert c.get_json("https://x/y.json") == {"ok": 1}
    finally:
        mod.time.sleep = orig
    assert Sess.calls == 2  # one retry, then served from cache


def test_urls():
    assert urls.submissions(320193).endswith("CIK0000320193.json")
    assert urls.filing_document(320193, "0000320193-21-000010", "a.htm").endswith("/320193/000032019321000010/a.htm")


def test_submissions_converts_acceptance_to_eastern():
    payload = {
        "cik": "320193", "name": "Apple Inc.", "sic": "3571", "tickers": ["AAPL"],
        "filings": {"recent": {
            "accessionNumber": ["0000320193-21-000010"], "form": ["10-Q"], "filingDate": ["2021-01-28"],
            "acceptanceDateTime": ["2021-01-28T21:05:12.000Z"], "primaryDocument": ["aapl.htm"],
            "items": [""], "reportDate": ["2020-12-26"],
        }},
    }
    df = parse_submissions(payload)
    ts = df.loc[0, "accepted_at"]
    assert ts.hour == 16 and ts.minute == 5  # 21:05 UTC is 16:05 ET in winter: after the close


def test_xbrl_point_in_time_never_uses_future_filings():
    payload = {"cik": 1, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"val": 100, "end": "2019-12-31", "filed": "2020-02-10", "form": "10-K", "accn": "a"},
        {"val": 120, "end": "2020-12-31", "filed": "2021-02-10", "form": "10-K", "accn": "b"},
    ]}}}}}
    facts = parse_company_facts(payload)
    assert point_in_time(facts, pd.Timestamp("2021-02-10"))["revenue"] == 100  # same day is not yet known
    assert point_in_time(facts, pd.Timestamp("2021-02-11"))["revenue"] == 120
    assert point_in_time(facts, pd.Timestamp("2019-01-01")) == {}


def test_index_from_bulk_reads_main_and_overflow_pages(tmp_path):
    import json
    import zipfile

    from filings_lab.edgar.universe import index_from_bulk

    def block(acc, form, date):
        return {"accessionNumber": [acc], "form": [form], "filingDate": [date],
                "acceptanceDateTime": [f"{date}T20:00:00.000Z"], "primaryDocument": ["d.htm"],
                "items": [""], "reportDate": [date]}

    main = {"cik": "320193", "name": "Apple Inc.", "sic": "3571", "tickers": ["AAPL"],
            "filings": {"recent": block("0000320193-21-000010", "10-Q", "2021-01-28"),
                        "files": [{"name": "CIK0000320193-submissions-001.json"}]}}
    older = block("0000320193-15-000001", "10-K", "2015-10-28")
    path = tmp_path / "submissions.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("CIK0000320193.json", json.dumps(main))
        zf.writestr("CIK0000320193-submissions-001.json", json.dumps(older))
        zf.writestr("CIK0000000001.json", json.dumps({"cik": "1", "filings": {}}))
    idx = index_from_bulk(str(path), "2014-01-01", "2024-12-31")
    assert sorted(idx["form"]) == ["10-K", "10-Q"]
    assert set(idx["ticker"]) == {"AAPL"}
