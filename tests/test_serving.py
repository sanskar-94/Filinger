import json
import subprocess
import sys

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


@pytest.fixture(scope="module")
def model(features, tmp_path_factory):
    d = build_artifact(features, tmp_path_factory.mktemp("model"), num_rounds=50)
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
