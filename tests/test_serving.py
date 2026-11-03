import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from filings_lab.llm.providers import ExtractionResult, Extractor
from filings_lab.serving.app import CircuitBreaker, OfflineFeatureStore, ServiceConfig, create_app
from filings_lab.serving.drift import DriftMonitor, categorical_psi, numeric_psi, reference_stats
from filings_lab.serving.model import ReactionModel, build_artifact
from filings_lab.serving.registry import FileRegistry


@pytest.fixture(scope="module")
def features(small_world, small_lake):
    from filings_lab.features.build import build_features

    sections = pd.read_parquet(small_lake / "silver" / "sections")
    events = pd.read_parquet(small_lake / "gold" / "events").drop(columns=["year"])
    return build_features(sections, events, small_world["index"], small_world["facts"])


ARTIFACT_DIR: dict = {}


@pytest.fixture(scope="module")
def model(features, tmp_path_factory):
    d = build_artifact(features, tmp_path_factory.mktemp("model"), num_rounds=50)
    ARTIFACT_DIR["dir"] = d
    return ReactionModel.load(d)


class FailingExtractor(Extractor):
    name = "llm"

    def __init__(self):
        self.calls = 0

    def extract(self, inp):
        self.calls += 1
        return ExtractionResult("llm", inp.accession, None, None, False, False, "http: timeout")


def _client(model, features, extractor=None, keys=frozenset()):
    store = OfflineFeatureStore(None)
    store.df = features.set_index("accession")
    cfg = ServiceConfig(api_keys=keys, breaker_failures=2, breaker_cooldown_s=60)
    return TestClient(create_app(cfg, model=model, extractor=extractor, feature_store=store))


def test_auth_and_health(model, features):
    c = _client(model, features, keys=frozenset({"secret"}))
    assert c.get("/healthz").status_code == 200 and c.get("/readyz").json()["status"] == "ready"
    assert c.get("/v1/model").status_code == 401
    assert c.get("/v1/model", headers={"X-API-Key": "secret"}).status_code == 200


def test_score_by_accession_matches_model(model, features):
    c = _client(model, features)
    acc = features["accession"].iloc[-1]
    r = c.post("/v1/score", json={"accession": acc}).json()
    expected = model.predict(features[features["accession"] == acc])[0]
    assert r["probability_up"] == pytest.approx(expected)
    assert len(r["drivers"]) == 5
    assert c.post("/v1/score", json={"accession": "nope"}).status_code == 404


def test_extract_falls_back_and_breaker_opens(model, features):
    bad = FailingExtractor()
    c = _client(model, features, extractor=bad)
    body = {"text": "The Company is lowering its full year guidance.\n\nDemand weakened.", "form": "8-K"}
    for _ in range(4):
        r = c.post("/v1/extract", json=body).json()
        assert r["fallback"] and r["signals"]["guidance_direction"] == "lowered"
    assert bad.calls == 2  # breaker opened after two failures; later requests skip the LLM
    metrics = c.get("/metrics").text
    assert 'flab_extractions_total{outcome="fallback",system="rules"} 4.0' in metrics
    assert "flab_llm_breaker_open 1.0" in metrics


class QuotingExtractor(Extractor):
    """An LLM that returns one real quote and one it made up."""
    name = "llm"

    def extract(self, inp):
        sig = {"guidance_direction": "raised", "management_tone": "positive", "liquidity_concern": False,
               "litigation_risk": False, "summary": "Guidance raised.",
               "key_quotes": [{"paragraph": 2, "text": "we are RAISING our full-year guidance"},
                              {"paragraph": 1, "text": "Zenith Brands issued a press release"}]}
        return ExtractionResult("llm", inp.accession, json.dumps(sig), sig, True, True, None)


def test_extract_flags_quotes_the_cited_paragraph_does_not_contain(model, features):
    c = _client(model, features, extractor=QuotingExtractor())
    body = {"paragraphs": [{"paragraph_id": "p-1", "section": "s", "text": "Revenue grew 12%."},
                           {"paragraph_id": "p-2", "section": "s", "text": "We are raising our full year guidance."}]}
    r = c.post("/v1/extract", json=body).json()
    assert not r["fallback"] and r["quotes_grounded"] == 1
    assert [(q["grounded"], q["paragraph_id"]) for q in r["quotes"]] == [(True, "p-2"), (False, None)]
    metrics = c.get("/metrics").text
    assert 'flab_extraction_quotes_total{grounded="false",system="llm"} 1.0' in metrics


def test_circuit_breaker_half_opens():
    b = CircuitBreaker(failures=1, cooldown_s=0.0)
    b.record(False)
    assert not b.open  # cooldown elapsed immediately: half open
    b.record(True)
    assert b.count == 0


def test_drift_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = reference_stats(pd.DataFrame({"x": rng.normal(size=5000), "g": rng.choice(["a", "b"], 5000)}), ["x"], ["g"])
    assert numeric_psi(ref["numeric"]["x"], rng.normal(size=2000)) < 0.05
    assert numeric_psi(ref["numeric"]["x"], rng.normal(1.0, 1, size=2000)) > 0.25
    assert categorical_psi(ref["categorical"]["g"], ["a"] * 900 + ["b"] * 100) > 0.25
    mon = DriftMonitor(ref, window=300, min_obs=100)
    for v in rng.normal(2.0, 1, size=300):
        mon.observe({"x": float(v)})
    assert mon.report()["x"]["psi"] > 0.5


def test_file_registry_aliases(tmp_path):
    reg = FileRegistry(tmp_path / "reg")
    art = tmp_path / "art"
    art.mkdir()
    (art / "model.txt").write_text("m")
    v1 = reg.register("m", art, metrics={"auc": 0.55})
    v2 = reg.register("m", art)
    reg.set_alias("m", "champion", v1)
    assert reg.resolve("models:/m@champion").name == str(v1)
    assert reg.resolve(f"models:/m/{v2}").name == str(v2)
    with pytest.raises(KeyError):
        reg.resolve("models:/m@challenger")


def test_file_registry_publishes_readable_versions(tmp_path):
    """A version staged from a 700 temp dir must still be readable by the API's non root user, carry its own
    version number in meta.json, and leave no staging directory behind."""
    import os
    import stat
    import tempfile

    reg = FileRegistry(tmp_path / "reg")
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmp:
        assert stat.S_IMODE(os.stat(tmp).st_mode) == 0o700
        (Path(tmp) / "model.txt").write_text("m")
        (Path(tmp) / "meta.json").write_text(json.dumps({"feature_set": "all"}))
        v = reg.register("m", tmp)
    reg.set_alias("m", "champion", v)
    vdir = reg.resolve("models:/m@champion")
    assert stat.S_IMODE(os.stat(vdir).st_mode) == 0o755
    assert stat.S_IMODE(os.stat(vdir / "model.txt").st_mode) == 0o644
    assert json.loads((vdir / "meta.json").read_text())["version"] == v
    assert stat.S_IMODE(os.stat(tmp_path / "reg" / "m" / "aliases.json").st_mode) == 0o644
    assert not [p for p in (tmp_path / "reg" / "m").iterdir() if p.name.startswith(".")]


def test_gate_cli_exit_codes(tmp_path):
    prod = {"extractor": {"metrics": {"extraction_score": 0.80, "schema_valid_rate": 1.0}},
            "reaction_model": {"metrics": {"auc": 0.55, "ic": 0.05}}}
    (tmp_path / "prod.json").write_text(json.dumps(prod))
    good = {"summary": {"cand": {"extraction_score": 0.85, "schema_valid_rate": 1.0}}}
    bad = {"summary": {"cand": {"extraction_score": 0.70, "schema_valid_rate": 0.97}}}
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text(json.dumps(bad))

    def run(f):
        return subprocess.run([sys.executable, "-m", "filings_lab.cli", "gate", "--production", str(tmp_path / "prod.json"),
                               "--extractor-eval", str(tmp_path / f), "--system", "cand"], capture_output=True, text=True)

    assert run("good.json").returncode == 0
    out = run("bad.json")
    assert out.returncode == 1 and "extraction_score" in out.stdout

    # a candidate scored on a different number of gold items than production is refused, however good it looks
    prod["extractor"]["gold_limit"] = 400
    (tmp_path / "prod.json").write_text(json.dumps(prod))
    (tmp_path / "other.json").write_text(json.dumps({"summary": {"cand": {**good["summary"]["cand"], "n": 300}}}))
    (tmp_path / "same.json").write_text(json.dumps({"summary": {"cand": {**good["summary"]["cand"], "n": 400}}}))
    out = run("other.json")
    assert out.returncode == 1 and "--limit 400" in out.stdout
    assert run("same.json").returncode == 0

    # with a fingerprint on record, the same size is not enough: the inputs and labels must be the same
    prod["extractor"].update(gold_sha256="a" * 64, system="rules")
    (tmp_path / "prod.json").write_text(json.dumps(prod))
    (tmp_path / "fp.json").write_text(json.dumps({"summary": {"cand": {**good["summary"]["cand"], "n": 400}},
                                                  "gold": {"n": 400, "sha256": "b" * 64}}))
    assert run("fp.json").returncode == 1
    # production scored in the same run is the baseline, whatever production.json recorded elsewhere
    same_run = {"summary": {"cand": {"extraction_score": 0.62, "schema_valid_rate": 1.0, "n": 400},
                            "rules": {"extraction_score": 0.60, "schema_valid_rate": 1.0, "n": 400}},
                "gold": {"n": 400, "sha256": "b" * 64}}
    (tmp_path / "same_run.json").write_text(json.dumps(same_run))
    out = run("same_run.json")
    assert out.returncode == 0 and "same evaluation run" in out.stdout
    same_run["summary"]["cand"]["extraction_score"] = 0.55
    (tmp_path / "same_run.json").write_text(json.dumps(same_run))
    assert run("same_run.json").returncode == 1


def test_batch_drift_report(model, features):
    from filings_lab.serving.drift_report import drift_report, render

    since = pd.to_datetime(features["event_date"]).max() - pd.DateOffset(months=24)
    rep = drift_report(model, features, since)
    assert rep["filings"] > 0 and "prediction" in set(rep["table"]["feature"])
    stationary = rep["table"][rep["table"]["feature"].isin(["ret_5_1", "vol_60_1", "mom_250_30", "fin_size_rank"])]
    assert (stationary["psi"] < 0.25).all(), stationary
    shifted = features.copy()
    recent = pd.to_datetime(shifted["event_date"]) >= since
    top = rep["table"][rep["table"]["feature"] != "prediction"]["feature"].iloc[0]
    shifted.loc[recent, top] = shifted.loc[recent, top] + 5 * shifted[top].std()
    rep2 = drift_report(model, shifted, since)
    assert rep2["retrain_recommended"] and top in rep2["breached"]
    assert "Retraining recommended" in render(rep2)


def test_mlflow_registry_round_trip(model, features, tmp_path, monkeypatch):
    pytest.importorskip("mlflow.pyfunc")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("alembic")
    import mlflow

    from filings_lab.serving.registry import MlflowRegistry

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    reg = MlflowRegistry()
    v = reg.register("filings-reaction-test", ARTIFACT_DIR["dir"], metrics={"auc": 0.56})
    reg.set_alias("filings-reaction-test", "champion", v)
    assert reg.describe("models:/filings-reaction-test@champion")["version"] == v
    loaded = ReactionModel.load(reg.resolve("models:/filings-reaction-test@champion"))
    acc = features["accession"].iloc[-1]
    assert loaded.predict(features[features["accession"] == acc])[0] == pytest.approx(model.predict(features[features["accession"] == acc])[0])
