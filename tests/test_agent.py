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
