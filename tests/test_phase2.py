import numpy as np
import pandas as pd
import pytest

from filings_lab.features.build import build_features, feature_groups
from filings_lab.features.financial import fundamentals_asof
from filings_lab.features.lexicon import Lexicon, tokenize
from filings_lab.features.text_features import section_features, similarity


def test_lexicon_negation_flips_positive_words():
    lex = Lexicon()
    plain = lex.score(tokenize("results improved and growth was strong"))
    negated = lex.score(tokenize("results have not improved and growth was not strong"))
    assert plain["lex_net_tone"] > 0 > negated["lex_net_tone"]


def test_finance_words_are_not_negative():
    s = Lexicon().score(tokenize("tax liability cost expense depreciation"))
    assert s["lex_negative"] == 0


def test_lm_csv_loader(tmp_path):
    p = tmp_path / "lm.csv"
    p.write_text("Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining\n"
                 "LOSS,2009,0,0,0,0,0,0\nGAIN,0,2009,0,0,0,0,0\n")
    lex = Lexicon.from_lm_csv(str(p))
    assert lex.categories["negative"] == {"loss"} and lex.categories["positive"] == {"gain"}


def test_guidance_rules():
    up = section_features("The Company is raising its full year 2024 revenue guidance to $5 billion.")
    down = section_features("We now expect results below the low end of our previously issued outlook.")
    flat = section_features("The Company reaffirms its full year guidance.")
    assert up["guidance_up"] == 1 and up["guidance_down"] == 0
    assert down["guidance_down"] == 1
    assert flat["guidance_flat"] == 1 and flat["guidance_up"] == 0


def test_similarity_bounds():
    a = "revenue increased due to strong demand in cloud services"
    assert similarity(a, a)["sim_cosine"] == pytest.approx(1.0)
    assert similarity(a, "litigation risk factors changed materially")["sim_jaccard"] == 0


def test_fundamentals_are_point_in_time():
    facts = pd.DataFrame({
        "cik": [1, 1], "metric": ["assets", "assets"], "value": [100.0, 200.0],
        "period_end": pd.to_datetime(["2020-12-31", "2021-03-31"]), "fp": ["FY", "Q1"],
        "filed": pd.to_datetime(["2021-02-01", "2021-05-01"]),
    })
    ev = pd.DataFrame({"accession": ["a", "b", "c"], "cik": [1, 1, 1],
                       "date_filed": pd.to_datetime(["2021-01-15", "2021-05-01", "2021-05-02"])})
    out = fundamentals_asof(facts, ev).set_index("accession")
    assert np.isnan(out.loc["a", "fin_log_assets"])
    assert out.loc["b", "fin_log_assets"] == pytest.approx(np.log(100))  # same day filing is not yet known
    assert out.loc["c", "fin_log_assets"] == pytest.approx(np.log(200))


@pytest.fixture(scope="module")
def features(small_world, small_lake):
    sections = pd.read_parquet(small_lake / "silver" / "sections")
    events = pd.read_parquet(small_lake / "gold" / "events").drop(columns=["year"])
    return build_features(sections, events, small_world["index"], small_world["facts"])


def test_feature_table(features):
    g = feature_groups(features.columns)
    assert features["accession"].is_unique
    assert len(g["market"]) == 6 and g["text_tone"] and g["text_change"] and g["financial"]
    assert (features["t1"] > features["event_date"]).all()
    periodic = features["form"].isin(["10-K", "10-Q"])
    assert features.loc[periodic, "mdna_lex_net_tone"].notna().all()


def test_phase2_runs_and_text_beats_market(features, tmp_path):
    from filings_lab.modeling.train import Phase2Config, run

    cfg = Phase2Config(n_splits=3, n_permutations=0, max_rounds=200)
    m = run(features, tmp_path, cfg)
    assert (tmp_path / "report.md").exists() and (tmp_path / "oof_predictions.parquet").exists()
    r = m["results"]
    assert set(r) >= {"market_meta", "text_only", "all", "logistic_all"}
    assert 0.3 < r["all"]["auc"]["auc"] < 0.8
