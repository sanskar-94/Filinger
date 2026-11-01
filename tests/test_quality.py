import shutil

import pandas as pd

from filings_lab.pipeline.quality import check_events, check_features, check_paragraphs, check_prices, run_checks


def test_clean_lake_passes(small_lake, small_world):
    from filings_lab.features.build import build_features

    lake = small_lake
    sections = pd.read_parquet(lake / "silver" / "sections")
    events = pd.read_parquet(lake / "gold" / "events").drop(columns=["year"])
    feats = build_features(sections, events, small_world["index"], small_world["facts"])
    checks = run_checks(str(lake)) + check_features(feats)
    failed = [c for c in checks if c.status == "fail"]
    assert not failed, failed
    assert {c.name for c in checks} >= {"periodic_mdna_extracted", "leakage_canary", "event_not_before_acceptance"}


def test_corruptions_are_caught(small_lake, small_world, tmp_path):
    events = pd.read_parquet(small_lake / "gold" / "events")
    bad = events.copy()
    bad.loc[bad.index[:3], "event_date"] = pd.Timestamp("2000-01-03")  # reaction before the filing existed
    bad.loc[bad.index[3], "car"] = 4.2
    names = {c.name: c.status for c in check_events(small_world["index"], bad)}
    assert names["event_not_before_acceptance"] == "fail" and names["car_plausible"] == "fail"

    prices = pd.read_parquet(small_lake / "silver" / "prices.parquet")
    dup = pd.concat([prices, prices.head(5)])
    assert {c.name: c.status for c in check_prices(dup)}["one_price_per_day"] == "fail"

    paras = pd.read_parquet(small_lake / "silver" / "paragraphs")
    assert {c.name: c.status for c in check_paragraphs(pd.concat([paras, paras.head(2)]))}["paragraph_id_unique"] == "fail"


def test_leakage_canary_fires_on_future_information(small_lake, small_world):
    from filings_lab.features.build import build_features

    sections = pd.read_parquet(small_lake / "silver" / "sections")
    events = pd.read_parquet(small_lake / "gold" / "events").drop(columns=["year"])
    feats = build_features(sections, events, small_world["index"], small_world["facts"])
    feats["ret_5_1"] = feats["car"] * 0.9  # simulate a feature computed with post event prices
    canary = {c.name: c for c in check_features(feats)}["leakage_canary"]
    assert canary.status == "fail" and "ret_5_1" in canary.detail


def test_missing_tables_are_tolerated(small_lake, tmp_path):
    lake = tmp_path / "lake"
    shutil.copytree(small_lake, lake)
    shutil.rmtree(lake / "gold")
    assert all(c.table not in ("gold_events", "gold_features") for c in run_checks(str(lake)))
