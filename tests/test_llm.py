import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from filings_lab.llm.dataset import assign_splits, build_inputs, labels_from_synthetic_truth, load_gold, write_dataset
from filings_lab.llm.eval_harness import evaluate, gate, item_table, macro_f1, summarize
from filings_lab.llm.prompts import ExtractionInput, chat_messages, select_paragraphs
from filings_lab.llm.providers import (AnthropicBatchLabeler, AnthropicExtractor, LabelExtractor,
                                       OpenAICompatExtractor, RuleExtractor, token_cost)
from filings_lab.llm.schema import FilingSignals, json_schema, parse

LABEL = {"guidance_direction": "raised", "management_tone": "positive", "liquidity_concern": False,
         "litigation_risk": False, "key_quotes": [{"paragraph": 1, "text": "We are raising our full year guidance."}],
         "summary": "Guidance raised."}
INP = ExtractionInput("0000000001-24-000001", "8-K", "Acme", [
    {"paragraph_id": "a", "section": "8k_results_of_operations", "text": "Revenue grew. We are raising our full year guidance."},
    {"paragraph_id": "b", "section": "8k_results_of_operations", "text": "Margins expanded strongly."},
])


def test_json_schema_is_strict_and_inlined():
    s = json_schema()
    assert s["additionalProperties"] is False and set(s["required"]) == set(FilingSignals.model_fields)
    assert "$defs" not in json.dumps(s) and "$ref" not in json.dumps(s)
    item = s["properties"]["key_quotes"]["items"]
    assert item["additionalProperties"] is False and item["required"] == ["paragraph", "text"]


def test_parse_handles_fences_and_reports_errors():
    sig, ok, err = parse("```json\n" + json.dumps(LABEL) + "\n```")
    assert sig and ok and err is None
    sig, ok, err = parse('{"guidance_direction": "sideways"}')
    assert sig is None and ok and err.startswith("schema")
    sig, ok, err = parse("not json at all")
    assert sig is None and not ok


def test_select_paragraphs_keeps_signal_and_order():
    paras = [{"section": "mdna", "text": f"filler paragraph {i} " * 20} for i in range(20)]
    paras[13]["text"] = "We now expect full year results above our prior outlook and guidance."
    kept = select_paragraphs(paras, budget_chars=800)
    assert paras[13] in kept
    idx = [paras.index(p) for p in kept]
    assert idx == sorted(idx) and sum(len(p["text"]) for p in kept) <= 800


def test_chat_messages_include_schema_for_small_models():
    msgs = chat_messages(INP)
    assert msgs[0]["role"] == "system" and "JSON schema" in msgs[0]["content"]
    assert "[P2]" in msgs[1]["content"]


def test_rule_extractor_finds_guidance_with_grounded_quote():
    r = RuleExtractor().extract(INP)
    assert r.schema_valid and r.signals["guidance_direction"] == "raised"
    t = item_table([r], [(INP, LABEL)])
    assert t["n_grounded"].iloc[0] == t["n_quotes"].iloc[0] >= 1


def test_macro_f1_known_values():
    g = np.array(["a", "a", "b", "b"], dtype=object)
    assert macro_f1(g, g.copy(), ["a", "b"]) == 1.0
    assert macro_f1(g, np.array(["a", "b", "a", "b"], dtype=object), ["a", "b"]) == pytest.approx(0.5)


def test_invalid_outputs_score_zero_and_gate_blocks_regressions(tmp_path):
    gold = [(INP, LABEL)] * 20
    perfect = LabelExtractor({INP.accession: LABEL}, name="perfect")
    broken = LabelExtractor({}, name="broken")
    res = evaluate([perfect, broken], gold, tmp_path, reference="perfect")
    s = res["summary"]
    assert s["perfect"]["extraction_score"] == 1.0 and s["broken"]["schema_valid_rate"] == 0
    ok, why = gate(s["broken"], s["perfect"])
    assert not ok and any("extraction_score" in w for w in why)
    ok, why = gate(s["perfect"], s["perfect"])
    assert ok and not why
    assert (tmp_path / "report.md").exists() and (tmp_path / "items.parquet").exists()


class _Usage(SimpleNamespace):
    pass


def _fake_message(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop, stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=_Usage(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class FakeAnthropic:
    def __init__(self, reply):
        self.calls = []
        outer = self

        class _M:
            def create(self, **kw):
                outer.calls.append(kw)
                return reply

        self.messages = _M()
        self.beta = SimpleNamespace(messages=_M())


def test_frontier_request_shape_and_cost():
    client = FakeAnthropic(_fake_message(json.dumps(LABEL)))
    ex = AnthropicExtractor(model="claude-opus-5-5", client=client)
    r = ex.extract(INP)
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "medium"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert r.schema_valid and r.signals["guidance_direction"] == "raised"
    assert r.cost_usd == pytest.approx((1000 * 4.0 + 200 * 20.0) / 1e6)


def test_frontier_refusal_is_reported_not_parsed():
    client = FakeAnthropic(_fake_message("", stop="refusal"))
    r = AnthropicExtractor(client=client).extract(INP)
    assert not r.schema_valid and r.error == "refusal: cyber"


def test_batch_labeler_uses_custom_ids_and_half_price():
    msg = _fake_message(json.dumps(LABEL))

    class Batches:
        def __init__(self):
            self.reqs = None

        def create(self, requests):
            self.reqs = requests
            return SimpleNamespace(id="batch_1")

        def results(self, bid):
            return [SimpleNamespace(custom_id=INP.accession, result=SimpleNamespace(type="succeeded", message=msg))]

    client = FakeAnthropic(msg)
    client.messages.batches = Batches()
    lab = AnthropicBatchLabeler(AnthropicExtractor(client=client, use_fallbacks=False))
    assert lab.submit([INP]) == ["batch_1"]
    assert "fallbacks" not in client.messages.batches.reqs[0]["params"]  # not accepted on the Batches API
    out = lab.collect("batch_1", {INP.accession: INP})
    assert out[0].schema_valid and out[0].cost_usd == pytest.approx(token_cost("claude-opus-5-5", 1000, 200) / 2)


def test_vllm_extractor_guided_request():
    class Sess:
        def post(self, url, json, timeout):
            self.url, self.body = url, json
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "choices": [{"message": {"content": __import__("json").dumps(LABEL)}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 120}})

    s = Sess()
    r = OpenAICompatExtractor("http://vllm:8000/v1", "filings-extractor", guided=True, session=s).extract(INP)
    assert s.url.endswith("/chat/completions") and s.body["response_format"]["type"] == "json_schema"
    assert r.schema_valid and r.input_tokens == 900


def test_dataset_splits_and_gold(small_world, small_lake, tmp_path):
    paragraphs = pd.read_parquet(small_lake / "silver" / "paragraphs")
    inputs = build_inputs(paragraphs, small_world["index"], budget_chars=4000)
    labels = labels_from_synthetic_truth(small_world["truth"], inputs)
    splits = assign_splits(small_world["index"], test_years=1, holdout_frac=0.2)
    by_cik = small_world["index"].assign(split=small_world["index"]["accession"].map(splits)).groupby("cik")["split"]
    held = {c for c, s in by_cik if set(s) == {"test"}}  # companies whose every filing is in test
    train_ciks = set(small_world["index"]["cik"][small_world["index"]["accession"].map(splits) == "train"])
    assert held and not (held & train_ciks)
    stats = write_dataset(inputs, labels, splits, tmp_path)
    assert stats["train"] > 0 and stats["test"] > 0
    rec = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert rec["prompt"][0]["role"] == "system" and rec["completion"][0]["role"] == "assistant"
    FilingSignals.model_validate_json(rec["completion"][0]["content"])
    gold = load_gold(tmp_path / "gold.jsonl")
    assert len(gold) == stats["test"]
    s = summarize(item_table([RuleExtractor().extract(i) for i, _ in gold], gold))
    assert s["schema_valid_rate"] == 1.0 and 0 <= s["extraction_score"] <= 1
