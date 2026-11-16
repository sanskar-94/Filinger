"""An earnings 8-K as EDGAR really ships it: the 8-K is a cover note and the substance is Exhibit 99.1."""
import gzip
from types import SimpleNamespace

import pandas as pd

from filings_lab import storage
from filings_lab.edgar import urls
from filings_lab.edgar.download import download_filings
from filings_lab.pipeline.core import process_document, process_partition
from filings_lab.text.clean import press_release_text, prune_submission, split_documents

ACC, CIK = "0001234567-24-000042", 1234567
COVER = """<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 8-K</p>
<p>Item 2.02 Results of Operations and Financial Condition.</p>
<p>On February 1, 2024, Example Corp issued a press release announcing its financial results for the quarter
ended December 31, 2023. A copy of the press release is furnished as Exhibit 99.1 to this report.</p>
<p>Item 9.01 Financial Statements and Exhibits.</p><p>99.1 Press release dated February 1, 2024.</p>
</body></html>"""
RELEASE = """<html><body><p>Example Corp Reports Fourth Quarter Results</p>
<p>Revenue declined 8% compared to the prior year period as customers delayed orders, and gross margin
contracted on higher input costs across the product portfolio.</p>
<p>The Company is lowering its full year 2024 revenue guidance to $4.1 billion to $4.3 billion, from a prior
range of $4.6 billion to $4.8 billion, reflecting the softer demand environment described above.</p>
<table><tr><td>Revenue</td><td>1,020</td><td>1,109</td></tr><tr><td>Net income</td><td>88</td><td>131</td></tr></table>
</body></html>"""


def _doc(dtype: str, name: str, body: str) -> str:
    return f"<DOCUMENT>\n<TYPE>{dtype}\n<SEQUENCE>1\n<FILENAME>{name}\n<TEXT>\n{body}\n</TEXT>\n</DOCUMENT>\n"


SUBMISSION = ("<SEC-DOCUMENT>0001234567-24-000042.txt : 20240201\n<SEC-HEADER>ACCESSION NUMBER: 0001234567-24-000042\n</SEC-HEADER>\n"
              + _doc("8-K", "ex-8k.htm", COVER) + _doc("EX-99.1", "ex991.htm", RELEASE)
              + _doc("GRAPHIC", "logo.jpg", "begin 644 logo.jpg\nM_]C_X  02D9)1@ ! 0$ 2 !(  #_VP!#  $! 0$! 0$! 0$! 0$! 0$!\nend")
              + _doc("EX-101.SCH", "ex.xsd", "<xs:schema></xs:schema>") + "</SEC-DOCUMENT>\n")


def test_press_release_is_read_from_the_exhibit():
    assert [d["type"] for d in split_documents(SUBMISSION)] == ["8-K", "EX-99.1", "GRAPHIC", "EX-101.SCH"]
    text = press_release_text(SUBMISSION)
    assert "lowering its full year 2024 revenue guidance" in text
    assert "1,020" not in text and "begin 644" not in text  # the numeric table and the image are not prose
    assert press_release_text(COVER) == ""  # a lone document has no exhibits


def test_earnings_item_carries_the_release():
    sections, paragraphs = process_document(SUBMISSION, ACC, CIK, "8-K")
    by_name = {s["section"]: s["text"] for s in sections}
    earnings = by_name["8k_results_of_operations"]
    assert earnings.startswith("On February 1, 2024")  # the item's own sentence first, so its paragraph ids hold
    assert "lowering its full year 2024 revenue guidance" in earnings
    assert any("lowering" in p["text"] and p["section"] == "8k_results_of_operations" for p in paragraphs)
    # the cover alone, which is all the primary document holds, says nothing about guidance
    cover_only = {s["section"]: s["text"] for s in process_document(COVER, ACC, CIK, "8-K")[0]}
    assert "guidance" not in cover_only["8k_results_of_operations"]


def test_release_without_an_earnings_item_is_its_own_section():
    other = SUBMISSION.replace("Item 2.02 Results of Operations and Financial Condition.", "Item 7.01 Regulation FD Disclosure.")
    names = {s["section"] for s in process_document(other, ACC, CIK, "8-K")[0]}
    assert "8k_press_release" in names and "8k_results_of_operations" not in names


def test_prune_keeps_only_what_is_read():
    pruned = prune_submission(SUBMISSION, "8-K")
    assert [d["type"] for d in split_documents(pruned)] == ["8-K", "EX-99.1"]
    assert pruned.startswith("<SEC-DOCUMENT>") and len(pruned) < len(SUBMISSION)
    assert process_document(pruned, ACC, CIK, "8-K") == process_document(SUBMISSION, ACC, CIK, "8-K")
    assert prune_submission(COVER, "8-K") == COVER


class FakeClient:
    def __init__(self):
        self.urls = []

    def get_bytes(self, url):
        self.urls.append(url)
        return (SUBMISSION if url.endswith(".txt") else COVER).encode()


def test_download_fetches_the_submission_for_8k_and_stores_gzip(tmp_path):
    idx = pd.DataFrame({"cik": [CIK, CIK], "accession": [ACC, "0001234567-24-000050"], "form": ["8-K", "10-Q"],
                        "primary_document": ["ex-8k.htm", "q.htm"]})
    client = FakeClient()
    out = download_filings(client, idx, str(tmp_path), workers=1, compress=True)
    assert out["download_status"].tolist() == ["ok", "ok"]
    assert urls.submission_txt(CIK, ACC) in client.urls and urls.filing_document(CIK, "0001234567-24-000050", "q.htm") in client.urls
    stored = storage.read_bytes(out["raw_uri"].iloc[0])
    assert stored[:2] == b"\x1f\x8b" and b"GRAPHIC" not in gzip.decompress(stored)
    # both runners read raw bytes through process_partition, which has to see through the compression
    batch = out.assign(content=[storage.read_bytes(u) for u in out["raw_uri"]])[["accession", "cik", "form", "content"]]
    sections, _ = process_partition(batch.iloc[:1], reader=None)
    assert sections["text"].str.contains("lowering its full year 2024").any()
    # a rerun skips what is there, and --no-exhibits goes back to the primary document
    assert download_filings(client, idx, str(tmp_path), workers=1)["download_status"].tolist() == ["cached", "cached"]
    plain = download_filings(FakeClient(), idx.iloc[:1], str(tmp_path / "plain"), workers=1, exhibits=False)
    assert storage.read_bytes(plain["raw_uri"].iloc[0]) == COVER.encode()


def test_yahoo_prices_use_the_adjusted_close():
    from filings_lab.prices.build import build_prices
    from filings_lab.prices.sources import YahooPriceSource

    def payload(closes, adjusted):
        stamps = [1598275800 + 86400 * i for i in range(len(closes))]  # 2020-08-24 09:30 New York, daily
        return {"chart": {"result": [{"meta": {"gmtoffset": -14400}, "timestamp": stamps,
                                      "indicators": {"quote": [{"close": closes, "volume": [10] * len(closes)}],
                                                     "adjclose": [{"adjclose": adjusted}]}}]}}

    class Session:
        def get(self, url, params, timeout, headers):
            sym = url.rsplit("/", 1)[1]
            if sym == "ZZZ":
                return SimpleNamespace(status_code=404, json=lambda: {})
            # a 4 for 1 split on the third day: raw close falls 75%, the adjusted series does not
            return SimpleNamespace(status_code=200, json=lambda: payload([400.0, 404.0, 101.0, None], [100.0, 101.0, 101.0, None]))

    idx = pd.DataFrame({"ticker": ["AAA", "ZZZ"], "cik": [1, 2]})
    prices, rep = build_prices(idx, YahooPriceSource(session=Session(), pause=0), "2020-08-24", "2020-08-31")
    assert rep["missing"] == ["ZZZ"] and rep["market_found"]
    a = prices[prices["ticker"] == "AAA"]
    assert a["close"].tolist() == [100.0, 101.0, 101.0] and str(a["date"].iloc[0].date()) == "2020-08-24"
