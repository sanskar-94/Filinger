"""Dense and hybrid retrieval, the embedding cache, and the two evaluations built on the synthetic truth."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from filings_lab.agent import agent_eval, retrieval_eval
from filings_lab.agent.agent import ExtractiveAgent, question_period
from filings_lab.agent.embeddings import HashEmbedder, cache_path, embed_texts, get_embedder
from filings_lab.agent.retrieval import BM25Index, DenseIndex, HybridIndex, build_index, ranked
from filings_lab.agent.service import AgentService
from filings_lab.agent.store import DataStore
from filings_lab.agent.tools import ToolBox, strip_entity

RISK = "Debt risk. Our indebtedness could limit our flexibility and we may be unable to refinance on acceptable terms."
CORPUS = pd.DataFrame({
    "paragraph_id": list("abcdefg"),
    "ticker": ["AAA"] * 6 + ["BBB"],
    "form": ["8-K", "10-Q", "10-K", "10-Q", "10-K", "8-K", "8-K"],
    "accession": ["a1", "a2", "a3", "a4", "a5", "a6", "b1"],
    "date_filed": pd.to_datetime(["2020-02-01", "2020-05-01", "2021-02-01", "2021-05-01", "2022-02-01", "2022-03-01", "2020-02-01"]),
    "text": ["The Company is lowering its full year 2020 revenue guidance to $4 billion to $5 billion.",
             RISK, RISK, RISK, RISK,
             "On March 1, 2022, the Company announced that its Chief Executive Officer will step down.",
             "The Company is lowering its full year 2020 revenue guidance to $9 billion to $10 billion."],
})


class TopicEmbedder:
    """Two dimensions, forecast and leadership: a paragraph or query points along the topic it is about."""

    name = "topic-test"
    TOPICS = (("guidance", "forecast", "projection"), ("chief executive", "ceo", "boss", "resign"))

    def _one(self, text):
        t = text.lower()
        v = np.array([float(any(w in t for w in words)) for words in self.TOPICS] + [0.05], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._one(t) for t in texts])

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture(scope="module")
def indexes():
    bm25 = BM25Index(CORPUS)
    dense = DenseIndex(CORPUS, embedder=TopicEmbedder())
    return bm25, dense, HybridIndex(bm25, dense)


def test_embedding_cache_embeds_each_distinct_text_once(tmp_path):
    path = cache_path(str(tmp_path), "hash-64")
    emb = HashEmbedder(64)
    texts = CORPUS["text"].tolist()
    unique, inverse, stats = embed_texts(texts, emb, path)
    assert stats["texts"] == 7 and stats["distinct"] == 4 and stats["embedded_now"] == 4 and path.exists()
    assert np.allclose(np.linalg.norm(unique, axis=1), 1.0, atol=1e-5)
    assert np.allclose(unique[inverse[1]], unique[inverse[4]])  # the repeated risk paragraph shares one vector
    again, inv2, stats2 = embed_texts(texts + ["A new paragraph about something else entirely."], emb, path)
    assert stats2["embedded_now"] == 1 and again.shape[0] == 5 and (inv2[:7] == inverse).all()
    # stored at half precision, so a reload agrees to about three decimals
    assert np.allclose(again[:4], unique, atol=2e-3)
    assert get_embedder("hash-64").name == "hash-64"


def test_repeated_paragraphs_collapse_to_the_most_recent_copy(indexes):
    bm25, dense, _ = indexes
    hits = bm25.search("indebtedness refinance", k=5, diversify=False, ticker="AAA")
    assert hits["paragraph_id"].tolist() == ["e"]  # four copies, one result, the latest filing's
    raw = bm25.search("indebtedness refinance", k=5, diversify=False, collapse=False, ticker="AAA")
    assert sorted(raw["paragraph_id"]) == ["b", "c", "d", "e"]
    scores = np.array([1.0, 3.0, 3.0, 0.0])
    assert ranked(scores, 0.0, np.array([3, 2, 1, 0]), None, 10).tolist() == [2, 1, 0]  # ties go to the more recent


def test_dense_finds_a_paraphrase_that_shares_no_words(indexes):
    bm25, dense, hybrid = indexes
    q = "the boss resigned"
    assert bm25.search(q, k=3, ticker="AAA").empty
    assert dense.search(q, k=1, ticker="AAA")["paragraph_id"].tolist() == ["f"]
    assert hybrid.search(q, k=1, diversify=False, ticker="AAA")["paragraph_id"].tolist() == ["f"]
    assert (dense.search(q, k=5, ticker="BBB")["ticker"] == "BBB").all()  # the filter holds for dense scores too


def test_keyword_hits_enter_the_fusion_only_when_they_cover_the_query(indexes):
    bm25, dense, hybrid = indexes
    # "step" is the only word shared with paragraph f; one word of four is an accident, not a keyword hit
    q = "forecast projections trimmed step"
    assert bm25.search(q, k=1, ticker="AAA")["paragraph_id"].tolist() == ["f"]
    assert bm25.coverage(q, np.array([5]))[0] == pytest.approx(0.25)
    assert hybrid.search(q, k=1, diversify=False, ticker="AAA")["paragraph_id"].tolist() == ["a"]
    ungated = HybridIndex(bm25, dense, min_coverage=0.0)
    cand, score = ungated.fused(q, ticker="AAA")
    assert 5 in cand.tolist()  # without the gate the accidental hit is fused in
    # a real keyword hit passes the gate and both rankers agree, so it scores twice
    cand, score = hybrid.fused("lowering revenue guidance", ticker="AAA")
    assert cand[0] == 0 and score[0] == pytest.approx(2 / 61)


def test_build_index_modes(tmp_path, indexes):
    assert isinstance(build_index(CORPUS, "bm25"), BM25Index)
    hybrid = build_index(CORPUS, "hybrid", lake=str(tmp_path), embedder=HashEmbedder(64))
    assert isinstance(hybrid, HybridIndex) and not hybrid.lexical_only and cache_path(str(tmp_path), "hash-64").exists()
    with pytest.raises(ValueError):
        build_index(CORPUS, "semantic")


def test_strip_entity_and_question_period():
    q = "Has Cobalt Therapeutics Inc said there is doubt about its ability to continue?"
    assert strip_entity(q, "Cobalt Therapeutics Inc", "CBT") == "Has said there is doubt about its ability to continue?"
    assert strip_entity("Did Apple's chief executive step down?", "Apple Inc.", "AAPL") == "Did chief executive step down?"
    assert strip_entity("Apple", "Apple Inc.", "AAPL") == "Apple"  # never strip a query down to nothing
    assert question_period("the release it filed in May 2019?") == ("2019-05-01", "2019-05-31")
    assert question_period("What happened in 2020?") == ("2020-01-01", "2020-12-31")
    assert question_period("February 2024 results") == ("2024-02-01", "2024-02-29")
    assert question_period("Did the company change its guidance?") == (None, None)


# ------------------------------------------------------------------ evaluations on the synthetic world
@pytest.fixture(scope="module")
def store(small_lake):
    return DataStore(str(small_lake))


def test_generator_sentences_label_the_paragraphs_that_state_a_fact(store, small_world):
    paras = store.paragraphs.reset_index(drop=True)
    labels = retrieval_eval.label_paragraphs(paras["text"])
    truth = small_world["truth"]
    lowered = set(truth[(truth["form"] == "8-K") & (truth["guidance_direction"] == "lowered")]["accession"])
    assert lowered and lowered == set(paras["accession"][labels["guidance_lowered"]])
    assert retrieval_eval.template_regex("Revenue fell {n}% in {y}.").search("Revenue fell 12% in 2019.")


def test_rank_metrics_count_distinct_texts():
    m = retrieval_eval.rank_metrics(np.array([7, 7, 3, 9]), np.array([7, 9]))
    assert m["hit@1"] == 1.0 and m["recall@10"] == 1.0 and m["mrr@10"] == 1.0
    assert m["ndcg@10"] == pytest.approx((1 + 1 / np.log2(5)) / (1 + 1 / np.log2(3)))  # the repeat of 7 earns nothing
    miss = retrieval_eval.rank_metrics(np.array([1, 2]), np.array([5]))
    assert miss == {"hit@1": 0.0, "hit@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}


def test_retrieval_eval_writes_a_report(store, small_lake, tmp_path):
    out = tmp_path / "retrieval"
    summary = retrieval_eval.evaluate(store.paragraphs, str(small_lake), out, embedder=HashEmbedder(256), companies_per_need=3,
                                      log=None)
    assert summary["systems"] == ["bm25", "dense", "hybrid"] and summary["info"]["cases"] > 20
    lexical = summary["by_style"]["lexical"]
    assert lexical["bm25"]["hit@5"]["mean"] > 0.9  # a keyword query finds the sentence it was cut from
    assert set(summary["paired"]["all"]) == {"hybrid minus bm25", "hybrid minus dense"}
    assert "exact" in summary["by_style"] and (out / "report.md").read_text().startswith("# Retrieval evaluation")
    assert json.loads((out / "metrics.json").read_text())["info"]["embed_model"] == "hash-256"
    cases = pd.read_csv(out / "cases.csv")
    assert set(cases["style"]) == {"lexical", "paraphrase", "exact"}


def test_bootstrap_resamples_whole_queries():
    rows = pd.DataFrame({"query": ["q1"] * 3 + ["q2"] * 3, "m": [1, 1, 1, 0, 0, 0]})
    b = retrieval_eval.bootstrap_mean(rows, "m", n_boot=500)
    assert b["mean"] == 0.5 and b["ci_low"] == 0.0 and b["ci_high"] == 1.0  # two clusters: the interval is honest about it
    d = retrieval_eval.bootstrap_diff(rows, rows.assign(m=0), "m", n_boot=200)
    assert d["diff"] == 0.5 and d["ci_low"] >= 0.0


def test_agent_questions_and_scoring(store, small_world):
    qs = agent_eval.build_questions(store, small_world["truth"], per_type=6, seed=1)
    types = {q["type"] for q in qs}
    assert {"guidance", "reaction", "disclosure_yes", "disclosure_no"} <= types
    g = next(q for q in qs if q["type"] == "guidance")
    by_id = store.by_id
    assert g["support"] and all(by_id.loc[p]["accession"] == g["accession"] for p in g["support"])
    cite = [{"paragraph_id": g["support"][0]}]
    word = {"raised": "raising", "lowered": "lowering", "maintained": "reaffirms"}[g["expected"]]
    good = agent_eval.score_answer(g, {"answer": f"The Company is {word} its guidance.", "citations": cite})
    assert good["answer_correct"] == 1.0 and good["first_citation_correct"] == 1.0
    wrong_para = agent_eval.score_answer(g, {"answer": f"It is {word} guidance.", "citations": [{"paragraph_id": "0" * 16}]})
    assert wrong_para["answer_correct"] == 0.0 and wrong_para["direction_stated"] == 1.0
    r = next(q for q in qs if q["type"] == "reaction")
    right = agent_eval.score_answer(r, {"answer": f"Reaction: {r['expected'] * 100:+.2f}% over 3 days.", "citations": []})
    off = agent_eval.score_answer(r, {"answer": f"Reaction: {r['expected'] * 100 + 0.5:+.2f}%.", "citations": []})
    assert right["car_correct"] == 1.0 and off["car_correct"] == 0.0
    no = next(q for q in qs if q["type"] == "disclosure_no")
    assert agent_eval.score_answer(no, {"answer": "The filings do not disclose any such matter.", "citations": []})["abstained"] == 1.0
    notice = "No indexed filing uses the term 'breached'; these passages only match the other words of the question."
    assert agent_eval.score_answer(no, {"answer": notice, "citations": cite})["abstained"] == 0.0  # not a statement of absence
    # what a hosted model writes: a typographic minus before the number, and emphasis in the middle of the denial
    for sign in ("\u2013", "\u2011", "\u2212", "-"):
        typed = agent_eval.score_answer({**r, "expected": -0.064}, {"answer": f"3 day abnormal return: {sign}6.40 %", "citations": []})
        assert typed["car_correct"] == 1.0
    assert agent_eval.score_answer({**r, "expected": 0.064}, {"answer": "Return: \u20136.40%", "citations": []})["car_correct"] == 0.0
    # a return rounded to one decimal is the right return; rounded to a whole number it is not an answer
    assert agent_eval.states_percent("3 day abnormal return of about +5.6 %", 5.63) and agent_eval.states_percent("\u20112.7%", -2.68)
    assert not agent_eval.states_percent("roughly 6%", 5.63) and not agent_eval.states_percent("+5.7%", 5.63)
    assert not agent_eval.states_percent("+5.60%", 5.63)  # two decimals stated, and they are wrong
    bold = "Orion Health does **not** state that it needs to raise more capital."
    assert agent_eval.score_answer(no, {"answer": bold, "citations": cite})["abstained"] == 1.0


def test_agent_eval_asks_again_after_an_api_error(store, small_world):
    from filings_lab.agent.agent import AgentAnswer

    qs = agent_eval.build_questions(store, small_world["truth"], per_type=2, seed=1)
    down = {qs[0]["id"]: 1, qs[1]["id"]: 5}  # the first recovers on the second try, the second never does

    class Flaky:
        def __init__(self):
            self.asked = []

        def ask(self, question, ticker=None):
            q = next(x for x in qs if x["question"] == question)
            self.asked.append(q["id"])
            if down.get(q["id"], 0) > 0:
                down[q["id"]] -= 1
                return AgentAnswer("", [], [], [], "test", 1, 0.0, error="api: 503")
            return AgentAnswer("The filings do not disclose it.", [], [], [{"tool": "search_filings", "input": {"query": "x"}}],
                               "test", 2, 0.0)

    agent, slept, said = Flaky(), [], []
    rows = agent_eval.run_agent(agent, qs, "flaky", log=said.append, retries=2, pause=30.0, sleep=slept.append)
    assert rows["id"].tolist() == [q["id"] for q in qs] and slept == [30.0, 30.0]
    by_id = rows.set_index("id")
    assert by_id.loc[qs[0]["id"], "attempts"] == 2 and not by_id.loc[qs[0]["id"], "error"]
    assert by_id.loc[qs[1]["id"], "attempts"] == 3 and by_id.loc[qs[1]["id"], "error"] and by_id.loc[qs[1]["id"], "error_text"] == "api: 503"
    assert agent.asked.count(qs[2]["id"]) == 1 and by_id.loc[qs[2]["id"], "calls"] == [{"tool": "search_filings", "input": {"query": "x"}}]
    assert any("2 questions ended in an API error" in m for m in said)
    summary = agent_eval.summarise(rows)["systems"]["flaky"]
    assert summary["asked_again"] == 2 and summary["errors"] == 1


def test_agent_eval_resumes_an_interrupted_hosted_run(small_lake, small_world, tmp_path, monkeypatch):
    from filings_lab.agent import openai_agent
    from filings_lab.agent.agent import AgentAnswer

    truth = str(small_world["dir"] / "truth.parquet")
    asked = []

    class Chat:  # the agent never reaches it: ask is replaced below
        requests = limited = 0

    def answer(self, question, ticker=None):
        asked.append(question)
        if stop["after"] is not None and len(asked) > stop["after"]:
            raise KeyboardInterrupt
        return AgentAnswer("The filings do not disclose it.", [], [], [], "test", 1, 0.0)

    stop: dict = {"after": 5}
    monkeypatch.setattr(openai_agent.OpenAIChatAgent, "ask", answer)
    out = tmp_path / "hosted"
    run = dict(modes=["bm25"], agent="openai", per_type=3, client=Chat())
    with pytest.raises(KeyboardInterrupt):
        agent_eval.evaluate(str(small_lake), truth, out, log=None, **run)
    saved = (out / "progress.jsonl").read_text().splitlines()
    assert len(saved) == 5 and not (out / "report.md").exists()
    stop["after"], said = None, []
    summary = agent_eval.evaluate(str(small_lake), truth, out, log=said.append, **run)
    total = summary["info"]["questions"]
    assert len(asked) == 6 + (total - 5)  # the five saved answers were not asked again
    assert any("kept 5 answers from an interrupted run" in m for m in said)
    assert not (out / "progress.jsonl").exists() and len((out / "answers.jsonl").read_text().splitlines()) == total
    asked.clear()
    (out / "progress.jsonl").write_text(saved[0] + "\n")
    agent_eval.evaluate(str(small_lake), truth, out, log=None, resume=False, **run)
    assert len(asked) == total  # --fresh discards what was saved


def test_agent_eval_runs_offline_and_scoping_helps(small_lake, small_world, tmp_path):
    out = tmp_path / "agent"
    summary = agent_eval.evaluate(str(small_lake), str(small_world["dir"] / "truth.parquet"), out, modes=["bm25"], per_type=8,
                                  log=None)
    plain, scoped = summary["systems"]["offline, bm25"], summary["systems"]["offline, bm25, dated questions scoped"]
    assert scoped["types"]["guidance"]["mean"] >= plain["types"]["guidance"]["mean"]
    assert scoped["types"]["reaction"]["mean"] == 1.0  # the month names one earnings release, whose return is quoted
    assert scoped["citation_valid_rate"] == 1.0 and scoped["errors"] == 0
    assert (out / "report.md").exists() and len((out / "questions.jsonl").read_text().splitlines()) == summary["info"]["questions"]


def test_agent_eval_scores_the_hosted_agent_through_the_same_questions(small_lake, small_world, tmp_path):
    class Messages:
        def create(self, **kw):
            usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            text = "The filings do not disclose any such matter."
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", usage=usage)

    client = SimpleNamespace(messages=Messages(), beta=SimpleNamespace(messages=Messages()))
    summary = agent_eval.evaluate(str(small_lake), str(small_world["dir"] / "truth.parquet"), tmp_path / "hosted", modes=["bm25"],
                                  agent="claude", per_type=4, client=client, log=None)
    (name, system), = summary["systems"].items()
    assert name.endswith(", bm25") and system["errors"] == 0
    assert system["types"]["disclosure_no"]["mean"] == 1.0 and system["types"]["disclosure_yes"]["mean"] == 0.0


def test_offline_agent_scopes_a_dated_question_and_drops_the_keyword_notice_for_hybrid(store):
    bm25 = BM25Index(store.paragraphs)
    filing = store.index[store.index["items"].astype(str).str.contains("2.02", regex=False)].iloc[3]
    q = f"What did the company say in the earnings release it filed in {filing['date_filed']:%B %Y}? zzzunknownword"
    ans = ExtractiveAgent(ToolBox(store, bm25)).ask(q, filing["ticker"])
    assert ans.tool_calls[0]["input"]["date_from"] == f"{filing['date_filed']:%Y-%m}-01"
    assert ans.citations and all(c.date_filed[:7] == f"{filing['date_filed']:%Y-%m}" for c in ans.citations)
    assert "zzzunknownword" in ans.answer  # a keyword index says which word no filing uses
    hybrid = HybridIndex(bm25, DenseIndex(store.paragraphs, embedder=HashEmbedder(128)))
    assert "zzzunknownword" not in ExtractiveAgent(ToolBox(store, hybrid)).ask(q, filing["ticker"]).answer
    unscoped = ExtractiveAgent(ToolBox(store, bm25), scope_period=False).ask(q, filing["ticker"])
    assert unscoped.tool_calls[0]["input"]["date_from"] is None


def test_service_falls_back_to_bm25_when_the_dense_index_cannot_load(store, monkeypatch):
    import filings_lab.agent.service as service_module

    def broken(*a, **k):
        raise RuntimeError("no embedding model in this image")

    monkeypatch.setattr(service_module, "build_index", broken)
    monkeypatch.setenv("FLAB_RETRIEVAL", "hybrid")
    svc = AgentService(store, mode="offline")
    assert svc.index is svc._bm25
    assert svc.toolbox_for("B", {"retrieval": "dense"}).index is svc._bm25
    out = svc.ask("Did the company change its guidance?", store.index["ticker"].iloc[0])
    assert out["citations"] and out["citation_valid_rate"] == 1.0


def test_cli_commands_run_on_a_small_lake(small_lake, small_world, tmp_path, capsys):
    from filings_lab.cli import main

    main(["embed", "--lake", str(small_lake), "--model", "hash-64"])
    stats = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert stats["model"] == "hash-64" and stats["distinct"] <= stats["texts"]
    main(["retrieval-eval", "--lake", str(small_lake), "--model", "hash-64", "--companies-per-need", "2", "--out", str(tmp_path / "r")])
    main(["agent-eval", "--lake", str(small_lake), "--truth", str(small_world["dir"] / "truth.parquet"), "--retrieval", "bm25",
          "--per-type", "4", "--out", str(tmp_path / "a")])
    out = capsys.readouterr().out
    assert "# Retrieval evaluation" in out and "# Agent evaluation" in out
    assert (tmp_path / "r" / "metrics.json").exists() and (tmp_path / "a" / "scores.csv").exists()
