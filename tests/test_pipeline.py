import pandas as pd


def test_local_pipeline_outputs(small_world, small_lake):
    sections = pd.read_parquet(small_lake / "silver" / "sections")
    paragraphs = pd.read_parquet(small_lake / "silver" / "paragraphs")
    events = pd.read_parquet(small_lake / "gold" / "events")
    idx = small_world["index"]
    periodic = idx[idx["form"].isin(["10-K", "10-Q"])]["accession"]
    got = sections[sections["section"] == "mdna"]["accession"]
    assert set(periodic) <= set(got), "every periodic report should yield an MD&A section"
    assert paragraphs["paragraph_id"].is_unique
    assert events["car"].notna().all() and len(events) > 100
    # every event is on or after its acceptance date
    acc = pd.to_datetime(events["accepted_at"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    assert (pd.to_datetime(events["event_date"]) >= acc).all()


def test_planted_signal_is_recoverable(small_world, small_lake):
    events = pd.read_parquet(small_lake / "gold" / "events")
    truth = small_world["truth"].set_index("accession")
    merged = events.join(truth[["true_effect"]], on="accession")
    assert merged["car"].corr(merged["true_effect"]) > 0.05
