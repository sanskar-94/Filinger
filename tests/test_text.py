from filings_lab.text.chunk import chunk_section, paragraph_id, split_paragraphs
from filings_lab.text.clean import clean_filing, select_main_document
from filings_lab.text.sections import extract_sections, key_sections

BODY_RF = "Our business faces many risks that could adversely affect results. " * 12
BODY_MDNA = "Revenue increased due to strong demand across all regions and segments. " * 12

TENK = f"""<html><body>
<p>TABLE OF CONTENTS</p>
<table><tr><td>Item 1A. Risk Factors</td><td>5</td></tr><tr><td>Item 7. Management's Discussion and Analysis</td><td>20</td></tr></table>
<p>Item 1A. Risk Factors</p><p>{BODY_RF}</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 7. Management&#8217;s Discussion and Analysis of Financial Condition</p><p>{BODY_MDNA}</p>
<p>Item 8. Financial Statements</p>
<table><tr><td>Revenue</td><td>1,000</td><td>900</td></tr></table>
</body></html>"""


def test_sections_skip_table_of_contents():
    text = clean_filing(TENK, "10-K")
    secs = key_sections(text, "10-K")
    assert set(secs) >= {"risk_factors", "mdna"}
    assert secs["risk_factors"].startswith("Our business faces")
    assert "Revenue increased" in secs["mdna"] and "Unresolved" not in secs["mdna"]


def test_numeric_tables_removed():
    assert "1,000" not in clean_filing(TENK, "10-K")


def test_full_submission_selects_main_document():
    raw = (
        "<SEC-DOCUMENT><DOCUMENT><TYPE>EX-99.1<TEXT>press release</TEXT></DOCUMENT>"
        "<DOCUMENT><TYPE>10-Q<TEXT><html><p>main body</p></html></TEXT></DOCUMENT></SEC-DOCUMENT>"
    )
    assert "main body" in select_main_document(raw, "10-Q")


def test_10q_parts_are_disambiguated():
    body = "Text body that is long enough to count as a real section of the filing. " * 3
    text = "\n".join([
        "PART I", "Item 1. Financial Statements", body, "Item 2. Management's Discussion and Analysis", body + " MDNA",
        "PART II", "Item 1. Legal Proceedings", body + " LEGAL", "Item 1A. Risk Factors", body + " RISK",
    ])
    secs = key_sections(text, "10-Q")
    assert secs["mdna"].endswith("MDNA") and secs["legal"].endswith("LEGAL") and secs["risk_factors"].endswith("RISK")


def test_8k_items():
    text = "Item 2.02 Results of Operations and Financial Condition\nRevenue grew.\nItem 9.01 Financial Statements and Exhibits\nExhibit 99.1"
    secs = key_sections(text, "8-K")
    assert secs["8k_results_of_operations"] == "Revenue grew."
    assert "8k_2.02" in extract_sections(text, "8-K")


def test_paragraph_ids_are_stable_and_unique():
    text = "\n\n".join(f"Paragraph number {i} has enough words in it to stand alone as a unit of text, so it is not merged." for i in range(5))
    a = chunk_section("acc-1", "mdna", text)
    b = chunk_section("acc-1", "mdna", text)
    assert [x["paragraph_id"] for x in a] == [x["paragraph_id"] for x in b]
    assert len({x["paragraph_id"] for x in a}) == 5
    assert paragraph_id("acc-1", "mdna", 0) != paragraph_id("acc-2", "mdna", 0)


def test_long_paragraphs_split_and_short_headings_merged():
    long = " ".join(["This is a sentence about results."] * 200)
    parts = split_paragraphs("Overview\n\n" + long)
    assert all(len(p) <= 2100 for p in parts)
    assert parts[0].startswith("Overview")
