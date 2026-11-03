import json
from types import SimpleNamespace

import numpy as np
import pytest

from filings_lab.agent.agent import ClaudeAgent, ExtractiveAgent, build_citations
from filings_lab.agent.retrieval import BM25Index
from filings_lab.agent.store import DataStore
from filings_lab.agent.tools import TOOL_DEFS, ToolBox
from filings_lab.experiment.assign import Experiment
from filings_lab.experiment.log import EventLog


@pytest.fixture(scope="module")
def store(small_lake):
    return DataStore(str(small_lake))


@pytest.fixture(scope="module")
def index(store):
    return BM25Index(store.paragraphs)


def test_search_respects_ticker_filter_and_ranks_relevant(store, index):
    t = store.index["ticker"].iloc[0]
    hits = index.search("raising full year revenue guidance", k=5, ticker=t)
    assert len(hits) and (hits["ticker"] == t).all()
    assert "guidance" in hits.iloc[0]["text"].lower()
    assert hits["paragraph_id"].is_unique


def test_tool_schemas_are_strict_compatible():
    for t in TOOL_DEFS:
        s = t["input_schema"]
        assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])


def test_toolbox_tracks_returned_ids(store, index):
    tb = ToolBox(store, index)
    res = json.loads(tb.execute("search_filings", {"query": "guidance", "ticker": None, "form": "8-K", "date_from": None,
                                                    "date_to": None, "k": 3}))["results"]
    assert {r["paragraph_id"] for r in res} <= tb.seen_ids
    para = json.loads(tb.execute("get_paragraph", {"paragraph_id": res[0]["paragraph_id"]}))
    assert para["url"].startswith("https://www.sec.gov/Archives/edgar/data/")
    valid, invalid = build_citations(f"x [cite:{res[0]['paragraph_id']}] y [cite:{'0' * 16}]", tb)
    assert [c.paragraph_id for c in valid] == [res[0]["paragraph_id"]] and invalid == ["0" * 16]


class ScriptedClaude:
    """Turn 1: call search. Turn 2: answer citing one real and one fabricated id. Turn 3: corrected answer."""

    def __init__(self, refuse=False):
        self.calls, self.refuse = [], refuse
        outer = self

        class _M:
            def create(self, **kw):
                outer.calls.append(kw)
                return outer.respond(kw)

        self.beta = SimpleNamespace(messages=_M())
        self.messages = _M()

    @staticmethod
    def _msg(content, stop):
        return SimpleNamespace(content=content, stop_reason=stop, usage=SimpleNamespace(
            input_tokens=2000, output_tokens=300, cache_read_input_tokens=0, cache_creation_input_tokens=0))

    def respond(self, kw):
        if self.refuse:
            return self._msg([], "refusal")
        n = len(self.calls)
        if n == 1:
            return self._msg([SimpleNamespace(type="thinking", thinking=""),
                              SimpleNamespace(type="tool_use", id="tu_1", name="search_filings",
                                              input={"query": "guidance", "ticker": None, "form": "8-K",
                                                     "date_from": None, "date_to": None, "k": 3})], "tool_use")
        tool_msg = kw["messages"][2]["content"][0]
        real = json.loads(tool_msg["content"])["results"][0]["paragraph_id"]
        if n == 2:
            text = f"Guidance was raised [cite:{real}] and margins rose [cite:{'f' * 16}]."
        else:
            text = f"Guidance was raised [cite:{real}]."
        return self._msg([SimpleNamespace(type="text", text=text)], "end_turn")


def test_claude_agent_loop_repairs_fabricated_citation(store, index):
    client = ScriptedClaude()
    ans = ClaudeAgent(ToolBox(store, index), client=client).ask("What happened to guidance?")
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert all(t["strict"] for t in first["tools"]) and "tool_choice" not in first
    assert first["output_config"] == {"effort": "high"}
    # turn 2 history: user, assistant (with its thinking block intact), one user message holding the tool result
    second = client.calls[1]["messages"]
    assert second[1]["content"][0].type == "thinking" and second[2]["content"][0]["tool_use_id"] == "tu_1"
    assert ans.meta["repaired"] and ans.invalid_citations == [] and len(ans.citations) == 1
    assert ans.turns == 3 and ans.tool_calls[0]["tool"] == "search_filings"
    assert ans.cost_usd == pytest.approx(3 * (2000 * 4 + 300 * 20) / 1e6)


def test_claude_agent_handles_refusal(store, index):
    ans = ClaudeAgent(ToolBox(store, index), client=ScriptedClaude(refuse=True)).ask("anything")
    assert ans.stop_reason == "refusal" and "declined" in ans.answer and ans.citations == []


def test_extractive_agent_answers_with_valid_citations(store, index):
    t = store.index["ticker"].iloc[0]
    ans = ExtractiveAgent(ToolBox(store, index)).ask(f"How did {t} stock react to its guidance changes?")
    assert ans.citations and ans.citation_valid_rate == 1.0
    assert all(c.ticker == t for c in ans.citations)


def test_extractive_agent_says_when_a_term_is_not_in_any_filing(store, index):
    assert index.unknown_terms("What about zyxwvut and guidance?") == ["zyxwvut"]
    assert index.unknown_terms("guidance outlook") == []
    ans = ExtractiveAgent(ToolBox(store, index)).ask("What did management say about zyxwvut guidance?")
    assert ans.answer.startswith("No indexed filing uses the term 'zyxwvut'")
    assert ans.citations and ans.citation_valid_rate == 1.0


def test_assignment_is_sticky_balanced_and_independent():
    a = Experiment("x", "salt-a", {"A": {"weight": 0.5}, "B": {"weight": 0.5}})
    b = Experiment("y", "salt-b", {"A": {"weight": 0.5}, "B": {"weight": 0.5}})
    users = [f"user-{i}" for i in range(20000)]
    va = np.array([a.assign(u) for u in users])
    assert all(a.assign(u) == v for u, v in zip(users[:500], va[:500], strict=True))
    assert abs((va == "A").mean() - 0.5) < 0.015
    vb = np.array([b.assign(u) for u in users])
    agree = (va == vb).mean()
    assert abs(agree - 0.5) < 0.02  # no correlation between experiments
    w = Experiment("z", "s", {"A": {"weight": 0.9}, "B": {"weight": 0.1}})
    assert abs(np.mean([w.assign(u) == "B" for u in users]) - 0.1) < 0.01


def test_event_log(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.exposure(request_id="r1", experiment="x", variant="A", user_id="u", latency_s=0.2, n_citations=2,
                 citation_valid_rate=1.0, cost_usd=0.01)
    assert log.feedback("r1", 1) and not log.feedback("missing", 1)
    df = log.frame("x")
    assert len(df) == 1 and df["rating"].iloc[0] == 1


def test_feedback_reaching_another_replica_is_kept_and_joined(store, tmp_path, monkeypatch):
    """Two API replicas, each with its own log: a rating that lands on the other replica is accepted because the
    request id is signed, a forged id is not, and the analysis joins the two logs back together."""
    from filings_lab.agent.service import AgentService
    from filings_lab.experiment.log import load_frame

    monkeypatch.setenv("FLAB_REQUEST_SIGNING_KEY", "shared-secret")
    exp = Experiment("x", "s", {"A": {"weight": 0.5}, "B": {"weight": 0.5}})
    pod_a = AgentService(store, exp, EventLog(tmp_path / "pod-a.db"), mode="offline")
    pod_b = AgentService(store, exp, EventLog(tmp_path / "pod-b.db"), mode="offline")
    out = pod_a.ask("revenue guidance outlook", user_id="u1")
    rid, _, sig = out["request_id"].partition(".")
    assert pod_b.verify_request_id(out["request_id"]) and len(sig) == 24
    assert pod_b.feedback(out["request_id"], 1)
    assert not pod_b.feedback(f"{rid}.{'0' * 24}", 1) and not pod_b.feedback("nope", 1)
    assert EventLog(tmp_path / "pod-a.db").frame("x")["rating"].isna().all()  # the rating is not in pod a's file
    merged = load_frame(f"{tmp_path}/pod-*.db", "x")
    assert len(merged) == 1 and merged["rating"].iloc[0] == 1
    assert list(merged.columns) == list(EventLog(tmp_path / "pod-a.db").frame("x").columns)
    monkeypatch.setenv("FLAB_REQUEST_SIGNING_KEY", "another-secret")
    other_key = AgentService(store, exp, EventLog(tmp_path / "pod-c.db"), mode="offline")
    assert not other_key.feedback(out["request_id"], 1)


def test_event_log_snapshot_is_a_consistent_copy(tmp_path):
    log = EventLog(tmp_path / "live.db")
    for i in range(50):
        log.exposure(request_id=f"r{i}", experiment="x", variant="AB"[i % 2], user_id=f"u{i % 7}")
    log.feedback("r3", -1)
    log.snapshot(str(tmp_path / "exports" / "pod-0.db"))
    copy = EventLog(tmp_path / "exports" / "pod-0.db").frame("x")
    assert len(copy) == 50 and copy.set_index("request_id").loc["r3", "rating"] == -1


def test_api_ships_its_event_log_at_shutdown(store, tmp_path):
    import socket

    from fastapi.testclient import TestClient

    from filings_lab.agent.service import AgentService
    from filings_lab.serving.app import ServiceConfig, create_app

    exp = Experiment("x", "s", {"A": {"weight": 0.5}, "B": {"weight": 0.5}})
    svc = AgentService(store, exp, EventLog(tmp_path / "live.db"), mode="offline")
    cfg = ServiceConfig(lake=str(tmp_path / "nolake"), event_export=str(tmp_path / "exports"), event_export_seconds=3600)
    with TestClient(create_app(cfg, model=None, agent_service=svc)) as c:
        rid = c.post("/v1/agent/ask", json={"question": "revenue guidance outlook", "user_id": "u9"}).json()["request_id"]
        assert c.post("/v1/feedback", json={"request_id": rid, "rating": -1}).json() == {"ok": True}
    shipped = EventLog(tmp_path / "exports" / f"{socket.gethostname()}.db").frame("x")
    assert shipped["request_id"].tolist() == [rid] and shipped["rating"].iloc[0] == -1


def test_agent_routes(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from filings_lab.agent.service import AgentService
    from filings_lab.serving.app import ServiceConfig, create_app

    exp = Experiment("x", "s", {"A": {"weight": 0.5}, "B": {"weight": 0.5, "diversify": False}})
    svc = AgentService(store, exp, EventLog(tmp_path / "e.db"), mode="offline")
    c = TestClient(create_app(ServiceConfig(lake=str(tmp_path / "nolake")), model=None, agent_service=svc))
    t = store.index["ticker"].iloc[0]
    r = c.post("/v1/agent/ask", json={"question": "revenue guidance outlook", "ticker": t, "user_id": "u1"}).json()
    assert r["variant"] == exp.assign("u1") and r["citations"]
    assert c.post("/v1/feedback", json={"request_id": r["request_id"], "rating": 1}).json() == {"ok": True}
    assert c.post("/v1/feedback", json={"request_id": "nope", "rating": 1}).status_code == 404
    assert c.get(f"/v1/companies/{t}/prices").json()["series"]
    assert 'flab_ab_requests_total{experiment="x",variant=' in c.get("/metrics").text


def test_each_variant_serves_its_own_model_version(store, tmp_path, monkeypatch):
    from filings_lab.agent.service import AgentService

    sent = []

    class FakeVllm:
        def post(self, url, json, timeout):
            sent.append(json["model"])
            body = {"guidance_direction": "raised", "management_tone": "positive", "liquidity_concern": False,
                    "litigation_risk": False, "key_quotes": [{"paragraph": 1, "text": "x"}], "summary": "s"}
            import json as _j
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"choices": [{"message": {"content": _j.dumps(body)}}], "usage": {}})

    monkeypatch.setenv("FLAB_LLM_BASE_URL", "http://vllm:8000/v1")
    exp = Experiment("x", "s", {"A": {"weight": 0.5, "extractor_model": "filings-extractor"},
                                 "B": {"weight": 0.5, "extractor_model": "filings-extractor-b"}})
    svc = AgentService(store, exp, EventLog(tmp_path / "e.db"), mode="offline")
    for v in ("A", "B"):
        svc.extractor_for(v, exp.config(v)).session = FakeVllm()
    users = {exp.assign(f"user-{i}"): f"user-{i}" for i in range(50)}
    t = store.index["ticker"].iloc[0]
    for v, user in sorted(users.items()):
        out = svc.ask("Did the company change its guidance?", t, user)
        assert out["variant"] == v and any(c["tool"] == "extract_signals" for c in out["tool_calls"])
    assert sent == ["filings-extractor", "filings-extractor-b"]
    logged = svc.log.frame("x").sort_values("variant")["model_version"].tolist()
    assert logged == ["vllm:filings-extractor", "vllm:filings-extractor-b"]


def test_a_variant_whose_model_falls_back_is_logged_as_such(store, tmp_path, monkeypatch):
    """Variant B's model returns invalid JSON, so the rules answer instead. The exposure must say so: otherwise the
    experiment silently compares A's model with the rules while claiming to test B's model."""
    from filings_lab.agent.service import AgentService

    class FakeVllm:
        def post(self, url, json, timeout):
            if json["model"] == "filings-extractor-b":
                content = "Revenue guidance for the year is"  # never reaches a JSON object
            else:
                import json as _j
                content = _j.dumps({"guidance_direction": "raised", "management_tone": "positive",
                                    "liquidity_concern": False, "litigation_risk": False, "summary": "s",
                                    "key_quotes": [{"paragraph": 1, "text": "a sentence that is not in the filing"}]})
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"choices": [{"message": {"content": content}}], "usage": {}})

    monkeypatch.setenv("FLAB_LLM_BASE_URL", "http://vllm:8000/v1")
    exp = Experiment("x", "s", {"A": {"weight": 0.5, "extractor_model": "filings-extractor"},
                                 "B": {"weight": 0.5, "extractor_model": "filings-extractor-b"}})
    svc = AgentService(store, exp, EventLog(tmp_path / "e.db"), mode="offline")
    for v in ("A", "B"):
        svc.extractor_for(v, exp.config(v)).session = FakeVllm()
    users = {exp.assign(f"user-{i}"): f"user-{i}" for i in range(50)}
    t = store.index["ticker"].iloc[0]
    outs = {v: svc.ask("Did the company change its guidance?", t, user) for v, user in sorted(users.items())}
    assert outs["A"]["extractions"] == {"n": 1, "fallbacks": 0}
    assert outs["B"]["extractions"] == {"n": 1, "fallbacks": 1}
    df = svc.log.frame("x").set_index("variant")
    assert df.loc["B", "n_extraction_fallbacks"] == 1 and df.loc["A", "n_extraction_fallbacks"] == 0
    # A's quote is not in the paragraph it cites, so it never becomes a citable id
    sig = json.loads(svc.toolbox_for("A", exp.config("A")).execute("extract_signals", {"accession": outs["A"]["tool_calls"][1]["input"]["accession"]}))
    assert sig["quotes"] == [] and not sig["fallback"]


def test_event_log_migrates_old_schema(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE exposures (request_id TEXT PRIMARY KEY, ts REAL, experiment TEXT, variant TEXT, user_id TEXT)")
    EventLog(db)
    with sqlite3.connect(db) as c:
        assert "model_version" in {r[1] for r in c.execute("PRAGMA table_info(exposures)")}


def test_filing_viewer_readout_and_review_routes(store, tmp_path, monkeypatch):
    import json as _json

    from fastapi.testclient import TestClient

    from filings_lab.agent.service import AgentService
    from filings_lab.serving.app import ServiceConfig, create_app

    acc = store.paragraphs["accession"].iloc[0]
    queue = tmp_path / "gold.jsonl"
    label = {"guidance_direction": "raised", "management_tone": "positive", "liquidity_concern": False,
             "litigation_risk": False, "key_quotes": [], "summary": "s"}
    queue.write_text("\n".join(_json.dumps({"accession": a, "form": "8-K", "company": "C", "paragraphs": [{"section": "x", "text": "t"}],
                                            "label": label}) for a in ("a1", "a2", "a3")) + "\n")
    monkeypatch.setenv("FLAB_REVIEW_QUEUE", str(queue))
    monkeypatch.setenv("FLAB_REVIEWED", str(tmp_path / "reviewed.jsonl"))
    exp = Experiment("x", "s", {"A": {"weight": 0.5}, "B": {"weight": 0.5}})
    svc = AgentService(store, exp, EventLog(tmp_path / "e.db"), mode="offline")
    c = TestClient(create_app(ServiceConfig(lake=str(tmp_path / "nolake")), model=None, agent_service=svc))

    doc = c.get(f"/v1/filings/{acc}/paragraphs").json()
    ids = [p["paragraph_id"] for s in doc["sections"] for p in s["paragraphs"]]
    assert doc["url"].startswith("https://www.sec.gov/") and len(ids) == len(set(ids)) > 0
    assert c.get("/v1/filings/nope/paragraphs").status_code == 404

    assert c.get("/v1/experiments/current/readout").status_code == 409  # nothing rated yet
    t = store.index["ticker"].iloc[0]
    for i in range(160):
        out = c.post("/v1/agent/ask", json={"question": "guidance outlook", "ticker": t, "user_id": f"u{i}"}).json()
        c.post("/v1/feedback", json={"request_id": out["request_id"], "rating": 1 if i % 3 else -1})
    ro = c.get("/v1/experiments/current/readout").json()
    assert ro["decision"] and set(ro["arms"]) == {"A", "B"}

    item = c.get("/v1/review/next").json()
    assert item["accession"] == "a1" and item["proposed"]["guidance_direction"] == "raised"
    c.post("/v1/review", json={"accession": "a1", "action": "accept"})
    c.post("/v1/review", json={"accession": "a2", "action": "correct", "label": {"guidance_direction": "lowered"}})
    assert c.post("/v1/review", json={"accession": "a3", "action": "correct", "label": {"management_tone": "upbeat"}}).status_code == 422
    stats = c.get("/v1/review/stats").json()
    assert stats["reviewed"] == 2 and stats["corrected"] == 1
    assert stats["fields"]["guidance_direction"]["agreement"] == 0.5
    assert c.get("/v1/review/next").json()["accession"] == "a3"


def test_cohen_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    from filings_lab.llm.review import cohen_kappa

    a = ["raised", "none", "none", "lowered", "none", "raised", "maintained", "none"]
    b = ["raised", "none", "maintained", "lowered", "none", "none", "maintained", "none"]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
