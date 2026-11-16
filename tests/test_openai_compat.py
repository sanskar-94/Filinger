"""The free hosted model path: an OpenAI compatible client, the agent on top of it, labelling through it."""
import json
from types import SimpleNamespace

import pytest

from filings_lab.agent import agent_eval
from filings_lab.agent.openai_agent import OpenAIChatAgent, clean_arguments, openai_tools
from filings_lab.agent.retrieval import BM25Index
from filings_lab.agent.service import AgentService
from filings_lab.agent.store import DataStore
from filings_lab.agent.tools import ToolBox
from filings_lab.config import load_dotenv
from filings_lab.openai_compat import NIM_BASE_URL, ChatClient, ChatError


def _resp(status, payload=None, headers=None):
    return SimpleNamespace(status_code=status, json=lambda: payload or {}, headers=headers or {}, text=json.dumps(payload or {}))


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "body": json, "headers": headers})
        return self.responses.pop(0)


def test_dotenv_sets_only_what_is_missing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# contact\nSEC_USER_AGENT="Lab Research ops@lab.org"\nexport NVIDIA_API_KEY=nvapi-test\nEMPTY=\nFLAB_LAKE_ROOT=./other\n')
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("FLAB_LAKE_ROOT", "./mine")
    added = load_dotenv(env)
    import os

    assert added == ["SEC_USER_AGENT", "NVIDIA_API_KEY"]
    assert os.environ["SEC_USER_AGENT"] == "Lab Research ops@lab.org" and os.environ["FLAB_LAKE_ROOT"] == "./mine"
    assert load_dotenv(tmp_path / "missing.env") == []
    for k in added:
        monkeypatch.delenv(k, raising=False)


def test_client_sends_the_key_and_backs_off_on_a_rate_limit(monkeypatch):
    monkeypatch.delenv("FLAB_OPENAI_BASE_URL", raising=False)
    slept = []
    ok = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    session = Session([_resp(429, headers={"Retry-After": "7"}), _resp(503), _resp(200, ok)])
    client = ChatClient(key="nvapi-test", rpm=0, session=session, sleep=slept.append)
    assert client.complete(model="m", messages=[]) == ok
    assert slept == [7.0, 4.0] and client.requests == 3
    assert session.calls[0]["url"] == f"{NIM_BASE_URL}/chat/completions"
    assert session.calls[0]["headers"] == {"Authorization": "Bearer nvapi-test"}
    with pytest.raises(ChatError, match="401"):  # a bad key is not retried
        ChatClient(key="x", rpm=0, session=Session([_resp(401, {"detail": "unauthorized"})]), sleep=slept.append).complete(model="m")
    always_limited = ChatClient(key="x", rpm=0, session=Session([_resp(429)] * 3), sleep=slept.append, max_retries=3)
    with pytest.raises(ChatError, match="429"):
        always_limited.complete(model="m")


@pytest.fixture(scope="module")
def store(small_lake):
    return DataStore(str(small_lake))


class ScriptedChat:
    """Stands in for ChatClient: answers each request with the next scripted assistant message."""

    def __init__(self, script):
        self.script, self.bodies = list(script), []

    def complete(self, **body):
        self.bodies.append(json.loads(json.dumps(body)))  # a snapshot: the agent keeps appending to its message list
        msg = self.script.pop(0)
        if callable(msg):
            msg = msg(body)
        finish = "tool_calls" if msg.get("tool_calls") else "stop"
        return {"choices": [{"message": {"role": "assistant", **msg}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def _call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def test_tools_translate_and_arguments_are_forgiven():
    tools = openai_tools()
    assert tools[0]["type"] == "function" and tools[0]["function"]["name"] == "search_filings"
    assert tools[0]["function"]["parameters"]["required"][0] == "query"
    assert clean_arguments("search_filings", '{"query": "guidance", "top_k": 3}') == {"query": "guidance"}  # an invented key is dropped
    assert clean_arguments("get_paragraph", {"paragraph_id": "abc"}) == {"paragraph_id": "abc"}
    # seen on the first live calls: the text "None" for an empty filter, and a number sent as text
    live = clean_arguments("search_filings", '{"query": "guidance", "ticker": "AIZ", "form": "None", "date_from": "null", "k": "5"}')
    assert live == {"query": "guidance", "ticker": "AIZ", "form": None, "date_from": None, "k": 5}
    assert clean_arguments("get_paragraph", {"paragraph_id": "None"}) == {"paragraph_id": "None"}  # only nullable arguments
    with pytest.raises(ValueError):
        clean_arguments("search_filings", "{not json")


def test_openai_agent_runs_tools_and_repairs_a_fabricated_citation(store):
    ticker = store.index["ticker"].iloc[0]
    tb = ToolBox(store, BM25Index(store.paragraphs))

    def cite_one_real_one_fake(body):
        results = json.loads(next(m for m in reversed(body["messages"]) if m["role"] == "tool")["content"])["results"]
        cite_one_real_one_fake.pid = results[0]["paragraph_id"]
        return {"content": f"<think>the search worked</think>Guidance changed [cite:{results[0]['paragraph_id']}] [cite:{'0' * 16}]"}

    chat = ScriptedChat([
        {"content": None, "tool_calls": [_call("search_filings", "{broken json", "bad"),
                                         _call("search_filings", {"query": "revenue guidance", "ticker": ticker, "limit": 5}, "good")]},
        cite_one_real_one_fake,
        lambda body: {"content": f"Guidance changed [cite:{cite_one_real_one_fake.pid}]"},
    ])
    ans = OpenAIChatAgent(tb, model="meta/test-model", client=chat).ask("Did the company change its guidance?", ticker)
    assert ans.error is None and ans.mode == "openai:meta/test-model" and ans.turns == 3
    assert [c.paragraph_id for c in ans.citations] == [cite_one_real_one_fake.pid] and ans.invalid_citations == []
    assert ans.meta["repaired"] and "<think>" not in ans.answer and ans.input_tokens == 300
    assert [c["error"] is None for c in ans.tool_calls] == [False, True]  # the broken call was reported, the good one ran
    second = chat.bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert second[3]["tool_call_id"] == "bad" and second[3]["content"].startswith("Error:")
    assert "not returned by any tool" in chat.bodies[2]["messages"][-1]["content"]
    assert chat.bodies[0]["tools"][0]["function"]["name"] == "search_filings" and chat.bodies[0]["temperature"] == 0.0


def test_api_failure_becomes_an_error_and_the_service_answers_offline(store, monkeypatch):
    class Down:
        def complete(self, **body):
            raise ChatError("503 from the API")

    tb = ToolBox(store, BM25Index(store.paragraphs))
    ans = OpenAIChatAgent(tb, client=Down()).ask("Did the company change its guidance?")
    assert ans.error and ans.error.startswith("api:") and ans.answer == ""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    svc = AgentService(store)  # auto: no paid key, a free one is set
    assert svc.mode == "openai"
    svc.client = Down()
    out = svc.ask("Did the company change its guidance?", store.index["ticker"].iloc[0])
    assert out["mode"] == "offline-extractive" and out["meta"]["fallback_from_error"] and out["citations"]
    monkeypatch.delenv("NVIDIA_API_KEY")
    assert AgentService(store).mode == "offline"


def test_agent_eval_scores_an_openai_compatible_agent(small_lake, small_world, tmp_path, monkeypatch):
    monkeypatch.setenv("FLAB_OPENAI_MODEL", "meta/test-model")

    class Client:
        def complete(self, **body):
            return {"choices": [{"message": {"content": "The filings do not disclose any such matter."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 9}}

    out = tmp_path / "hosted"
    summary = agent_eval.evaluate(str(small_lake), str(small_world["dir"] / "truth.parquet"), out, modes=["bm25"], agent="openai",
                                  per_type=4, client=Client(), log=None)
    (name, system), = summary["systems"].items()
    assert name == "meta/test-model, bm25" and system["errors"] == 0 and system["cost_usd"] == 0.0
    assert system["types"]["disclosure_no"]["mean"] == 1.0 and system["types"]["guidance"]["mean"] == 0.0
    assert "OpenAI compatible API (meta/test-model)" in summary["info"]["agent"]
    assert len((out / "answers.jsonl").read_text().splitlines()) == summary["info"]["questions"]


def test_labelling_through_the_api_spec_resumes(small_lake, tmp_path, monkeypatch):
    from filings_lab.llm import cli as llm_cli
    from filings_lab.llm.providers import OpenAICompatExtractor, RuleExtractor

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    ex = llm_cli.make_system("api:nvidia/nemotron-3-ultra-550b-a55b")
    assert isinstance(ex, OpenAICompatExtractor) and ex.name == "api:nvidia/nemotron-3-ultra-550b-a55b" and ex.client.key == "nvapi-test"

    seen = []

    class Chat:
        def complete(self, **body):
            seen.append(body["model"])
            return {"choices": [{"message": {"content": "not json at all"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    ex.client = Chat()
    inputs, _ = llm_cli._inputs(str(small_lake), 4000)
    bad = ex.extract(next(iter(inputs.values())))
    assert seen == ["nvidia/nemotron-3-ultra-550b-a55b"] and not bad.json_valid and bad.input_tokens == 5

    monkeypatch.setattr(llm_cli, "make_system", lambda spec, style="full": RuleExtractor())
    out = tmp_path / "labels.jsonl"
    args = SimpleNamespace(lake=str(small_lake), budget=4000, limit=6, out=str(out), system="rules", mode="realtime", model=None)
    llm_cli.cmd_label(args)
    first = out.read_text().splitlines()
    assert len(first) == 6 and all(json.loads(x)["schema_valid"] for x in first)
    args.limit = 9
    llm_cli.cmd_label(args)  # a rerun labels only what is missing
    assert len(out.read_text().splitlines()) == 9 and out.read_text().splitlines()[:6] == first


def test_console_entry_point_keeps_exit_codes_and_output(tmp_path):
    import subprocess
    import sys

    def flab(*args):
        code = "import sys; from filings_lab.cli import run; sys.argv = ['flab', *sys.argv[1:]]; run()"
        return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, cwd=tmp_path, timeout=120)

    ok = flab("synth", "--out", "world", "--companies", "2", "--start", "2019-01-01", "--end", "2020-12-31")
    assert ok.returncode == 0 and '"index"' in ok.stdout and (tmp_path / "world" / "index.parquet").exists()
    assert flab("--help").returncode == 0
    unknown = flab("no-such-command")
    assert unknown.returncode == 2 and "invalid choice" in unknown.stderr
    crash = flab("local", "--index", "missing.parquet", "--prices", "missing.parquet")
    assert crash.returncode == 1 and "Traceback" in crash.stderr  # an error still shows its traceback


def test_agent_concludes_when_the_turn_budget_runs_out(store):
    """Seen live: a model that kept searching for a disclosure that does not exist returned nothing at all."""
    from filings_lab.agent.openai_agent import tidy_answer

    ticker = store.index["ticker"].iloc[0]
    search = {"content": None, "tool_calls": [_call("search_filings", {"query": "going concern", "ticker": ticker})]}
    still_searching = {"content": "The filings do not disclose any going concern doubt.",
                       "tool_calls": [_call("search_filings", {"query": "insolvency"})]}
    chat = ScriptedChat([search, search, still_searching])
    tb = ToolBox(store, BM25Index(store.paragraphs))
    ans = OpenAIChatAgent(tb, model="m", client=chat, max_turns=3).ask("Has it flagged a going concern?", ticker)
    assert ans.turns == 3 and ans.answer == "The filings do not disclose any going concern doubt." and ans.error is None
    assert len(ans.tool_calls) == 2  # the tool call on the last turn is not run
    last = chat.bodies[-1]
    assert last["tool_choice"] == "none" and [b["tool_choice"] for b in chat.bodies[:2]] == ["auto", "auto"]
    assert last["messages"][-1]["role"] == "user" and "search budget" in last["messages"][-1]["content"]
    # seen live: a smaller model copied the brackets of the instructions around the id
    assert tidy_answer("<think>plan</think>Lowered [cite:<E9E5CB809B1CF6D6>] and [cite: e9e5cb809b1cf6d6 ].") == \
        "Lowered [cite:e9e5cb809b1cf6d6] and [cite:e9e5cb809b1cf6d6]."
    assert "[cite:3fa2b1c4d5e6f708]" in chat.bodies[0]["messages"][0]["content"]  # the format is shown by example


def test_agent_eval_with_workers_gives_each_thread_its_own_tool_box(small_lake, small_world, tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv("FLAB_OPENAI_MODEL", "meta/test-model")
    threads = set()

    class Client:
        def complete(self, **body):
            threads.add(threading.get_ident())
            if body["messages"][-1]["role"] == "tool":
                first = json.loads(body["messages"][-1]["content"])["results"][0]["paragraph_id"]
                return {"choices": [{"message": {"content": f"It lowered guidance [cite:{first}]"}, "finish_reason": "stop"}], "usage": {}}
            call = _call("search_filings", {"query": "full year guidance", "ticker": body["messages"][1]["content"].rsplit(" ", 1)[1][:-1]})
            return {"choices": [{"message": {"content": None, "tool_calls": [call]}, "finish_reason": "tool_calls"}], "usage": {}}

    truth = str(small_world["dir"] / "truth.parquet")
    one = agent_eval.evaluate(str(small_lake), truth, tmp_path / "w1", modes=["bm25"], agent="openai", per_type=6, client=Client(), log=None)
    many = agent_eval.evaluate(str(small_lake), truth, tmp_path / "w4", modes=["bm25"], agent="openai", per_type=6, client=Client(),
                               log=None, workers=4)
    a, b = one["systems"]["meta/test-model, bm25"], many["systems"]["meta/test-model, bm25"]
    assert a["citation_valid_rate"] == b["citation_valid_rate"] == 1.0  # no thread saw another's paragraphs as its own
    assert {t: v["mean"] for t, v in a["types"].items()} == {t: v["mean"] for t, v in b["types"].items()}
    assert a["mean_turns"] == 2.0 and a["empty_answers"] == 0 and len(threads) > 1
