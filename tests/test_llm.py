import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from filings_lab.llm.dataset import assign_splits, build_inputs, labels_from_synthetic_truth, load_gold, write_dataset
from filings_lab.llm.eval_harness import evaluate, gate, item_table, macro_f1, summarize
from filings_lab.llm.prompts import ExtractionInput, chat_messages, select_paragraphs
from filings_lab.llm.providers import (
    AnthropicBatchLabeler,
    AnthropicExtractor,
    LabelExtractor,
    OpenAICompatExtractor,
    RuleExtractor,
    token_cost,
)
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


def test_select_paragraphs_keeps_short_evidence_over_long_keyword_rich_text():
    long = ("Revenue increased while margins declined and costs increased; we expect results to improve after the "
            "impairment, and orders decreased before demand increased again. ") * 3
    paras = [{"section": "mdna", "text": long} for _ in range(6)]
    paras.insert(4, {"section": "legal", "text": "A putative securities class action was filed against the Company."})
    kept = select_paragraphs(paras, budget_chars=900)
    assert paras[4] in kept  # one keyword, but it is the only evidence for litigation_risk
    idx = [paras.index(p) for p in kept]
    assert idx == sorted(idx) and sum(len(p["text"]) for p in kept) <= 900


def test_unsupported_labels_counts_positives_the_prompt_cannot_support():
    from filings_lab.llm.dataset import unsupported_labels

    inp = ExtractionInput("a", "10-K", "Acme", [{"paragraph_id": "p", "section": "mdna", "text": "Revenue grew."}])
    lab = {"guidance_direction": "raised", "management_tone": "positive", "liquidity_concern": False, "litigation_risk": True}
    assert unsupported_labels({"a": inp}, {"a": lab}) == {"guidance_direction": 1, "liquidity_concern": 0, "litigation_risk": 1}


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


def test_run_system_batches_local_models_and_keeps_order():
    from filings_lab.llm.eval_harness import run_system

    class Batched(RuleExtractor):
        def __init__(self):
            super().__init__()
            self.batches = []

        def extract_many(self, inputs, batch_size=16):
            self.batches.append(batch_size)
            return [self.extract(x) for x in inputs]

    gold = [(INP, LABEL), (ExtractionInput("0000000002-24-000002", "8-K", "Beta", INP.paragraphs), LABEL)]
    ex = Batched()
    assert [r.accession for r in run_system(ex, gold, batch_size=8)] == [g[0].accession for g in gold]
    assert ex.batches == [8]
    run_system(ex, gold)  # default stays one request at a time, which is what serving latency means
    assert ex.batches == [8]


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
    # every positive ground truth label has its evidence in the prompt, even at a tight text budget
    for split in ("train", "val", "test"):
        assert set(stats[f"{split}_unsupported_labels"].values()) == {0}, (split, stats)
    tight = build_inputs(paragraphs, small_world["index"], budget_chars=1500)
    tight_stats = write_dataset(tight, labels_from_synthetic_truth(small_world["truth"], tight), splits, tmp_path / "tight")
    assert all(set(tight_stats[f"{s}_unsupported_labels"].values()) == {0} for s in ("train", "val", "test")), tight_stats
    rec = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert rec["prompt"][0]["role"] == "system" and rec["completion"][0]["role"] == "assistant"
    FilingSignals.model_validate_json(rec["completion"][0]["content"])
    gold = load_gold(tmp_path / "gold.jsonl")
    assert len(gold) == stats["test"]
    s = summarize(item_table([RuleExtractor().extract(i) for i, _ in gold], gold))
    assert s["schema_valid_rate"] == 1.0 and 0 <= s["extraction_score"] <= 1


def test_register_adapter_with_eval_scores(tmp_path, monkeypatch):
    from filings_lab.llm.cli import cmd_register
    from filings_lab.serving.registry import FileRegistry

    train = tmp_path / "lora" / "tiny-champion"
    (train / "adapter").mkdir(parents=True)
    (train / "adapter" / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "artifacts/tiny-base"}))
    (train / "train_metrics.json").write_text(json.dumps({"train_loss": 2.0, "final_eval_loss": 1.43, "run_id": "abc"}))
    (train / "train_config.json").write_text(json.dumps({"base_model": "artifacts/tiny-base", "lora_r": 16}))
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"summary": {"tiny-lora-champion": {"extraction_score": 0.71, "schema_valid_rate": 0.98}}}))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("FLAB_REGISTRY", str(tmp_path / "registry"))
    cmd_register(SimpleNamespace(adapter=str(train / "adapter"), name="filings-extractor", alias="champion", eval=str(ev),
                                 system="tiny-lora-champion", run_id=None, prompt_style="compact",
                                 tag=["gate=failed"]))
    reg = FileRegistry(tmp_path / "registry")
    path = reg.resolve("models:/filings-extractor@champion")
    assert (path / "adapter_config.json").exists()
    meta = reg.metadata("filings-extractor", 1)
    assert meta["kind"] == "adapter" and meta["metrics"]["extraction_score"] == 0.71
    assert meta["metrics"]["val_loss"] == 1.43 and meta["tags"]["lora_r"] == "16"
    assert meta["tags"]["gate"] == "failed"
    with pytest.raises(SystemExit):
        cmd_register(SimpleNamespace(adapter=str(train / "adapter"), name="filings-extractor", alias=None, eval=str(ev),
                                     system="missing", run_id=None, prompt_style="compact", tag=None))


def test_bi_extractor_table_applies_the_gate_against_production_in_the_same_run(tmp_path):
    from filings_lab.experiment.export import llm_extractor_table

    ev = {"summary": {"rules": {"n": 400, "extraction_score": 0.85, "schema_valid_rate": 1.0},
                      "good": {"n": 400, "extraction_score": 0.86, "schema_valid_rate": 1.0},
                      "bad": {"n": 400, "extraction_score": 0.40, "schema_valid_rate": 1.0}}}
    (tmp_path / "eval.json").write_text(json.dumps(ev))
    t = llm_extractor_table(tmp_path / "eval.json").set_index("system")
    assert t.loc["rules", "gate"] == "production" and t.loc["good", "gate"] == "pass"
    assert t.loc["bad", "gate"] == "fail" and "extraction_score" in t.loc["bad", "gate_reasons"]


def test_replay_system_scores_labels_made_elsewhere(tmp_path):
    from filings_lab.llm.cli import make_system

    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"accession": INP.accession, "signals": LABEL}) + "\n")
    ex = make_system(f"replay:{labels}@annotator")
    r = ex.extract(INP)
    assert ex.name == "annotator" and r.schema_valid and r.signals["guidance_direction"] == "raised"
    missing = ex.extract(ExtractionInput("0000000009-24-000009", "8-K", "Other", INP.paragraphs))
    assert not missing.schema_valid  # a filing the annotator never labelled scores as a miss, not a pass


def test_mlflow_reporting_only_with_a_tracking_server(monkeypatch):
    from filings_lab.llm.train_lora import _reporters

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert _reporters(["mlflow"]) == ["none"] and _reporters(["mlflow", "tensorboard"]) == ["tensorboard"]
    assert _reporters(["none"]) == ["none"]


def test_mlx_dataset_conversion_keeps_messages_and_names_splits_for_mlx(tmp_path):
    from filings_lab.llm.mlx_backend import to_mlx_dataset

    src = tmp_path / "ds"
    src.mkdir()
    rec = {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
           "completion": [{"role": "assistant", "content": json.dumps(LABEL)}]}
    (src / "train.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n")
    (src / "val.jsonl").write_text(json.dumps(rec) + "\n")
    counts = to_mlx_dataset(src, tmp_path / "mlx")
    assert counts == {"train": 2, "valid": 1}
    row = json.loads((tmp_path / "mlx" / "train.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert not (tmp_path / "mlx" / "val.jsonl").exists()  # mlx-lm reads valid.jsonl


def test_mlx_extractor_runs_a_local_qwen2_model(tmp_path):
    """Builds a two layer Qwen2 model and tokenizer like flab llm-pretrain does, then generates through MLX."""
    pytest.importorskip("mlx_lm")
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from filings_lab.llm.mlx_backend import MLXExtractor
    from filings_lab.llm.tiny import build_tokenizer, init_model

    tok = build_tokenizer(["We are raising our full year guidance."] * 50, vocab_size=300)
    model = init_model(tok, hidden=32, layers=2, heads=2, kv_heads=1, intermediate=64)
    model.save_pretrained(tmp_path / "m")
    tok.save_pretrained(tmp_path / "m")
    ex = MLXExtractor(str(tmp_path / "m"), name="mlx-test", max_new_tokens=8)
    r = ex.extract(INP)
    assert r.system == "mlx-test" and r.input_tokens > 0 and r.latency_s >= 0
