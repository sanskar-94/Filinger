"""The OpenAI compatible server with a base model and two adapters. Runs where torch, transformers and peft are
installed (CI job llm-smoke)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from filings_lab.llm.tiny import build_tokenizer

    d = tmp_path_factory.mktemp("llm")
    tok = build_tokenizer(["revenue guidance raised lowered maintained tone positive negative"] * 50, vocab_size=300)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, eos_token_id=tok.eos_token_id,
                                         pad_token_id=tok.pad_token_id))
    model.save_pretrained(d / "base")
    tok.save_pretrained(d / "base")
    for name in ("a", "b"):
        peft = get_peft_model(Qwen2ForCausalLM.from_pretrained(d / "base"), LoraConfig(r=2, target_modules="all-linear"))
        peft.save_pretrained(d / name)
    return d


def test_chat_completions_route_to_adapters_and_report_metrics(served):
    from fastapi.testclient import TestClient

    from filings_lab.serving.llm_server import create_llm_app

    app = create_llm_app(str(served / "base"), {"filings-extractor": str(served / "a"), "filings-extractor-b": str(served / "b")})
    c = TestClient(app)
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} == {"base", "filings-extractor", "filings-extractor-b"}
    for model in ("filings-extractor", "filings-extractor-b", "base"):
        r = c.post("/v1/chat/completions", json={"model": model, "max_tokens": 8,
                                                 "messages": [{"role": "user", "content": "revenue guidance"}]}).json()
        assert r["model"] == model and r["usage"]["completion_tokens"] <= 8 and r["choices"][0]["message"]["role"] == "assistant"
    assert c.post("/v1/chat/completions", json={"model": "nope", "messages": []}).status_code == 404
    text = c.get("/metrics").text
    assert 'vllm:generation_tokens_total{model_name="filings-extractor-b"}' in text


def test_openai_compat_extractor_against_the_server(served):
    from fastapi.testclient import TestClient

    from filings_lab.llm.prompts import ExtractionInput
    from filings_lab.llm.providers import OpenAICompatExtractor
    from filings_lab.serving.llm_server import create_llm_app

    c = TestClient(create_llm_app(str(served / "base"), {"filings-extractor": str(served / "a")}))

    class Session:  # route the extractor's requests into the test client
        def post(self, url, json, timeout):
            return c.post(url.replace("http://llm/v1", "/v1"), json=json)

    ex = OpenAICompatExtractor("http://llm/v1", "filings-extractor", session=Session(), style="compact", max_tokens=16)
    res = ex.extract(ExtractionInput("acc", "8-K", "C", [{"section": "mdna", "text": "Revenue guidance raised."}]))
    assert res.error is None or res.error.startswith(("no json", "json", "schema"))  # random weights: not valid JSON
    assert res.output_tokens > 0
