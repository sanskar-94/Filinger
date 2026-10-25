"""Real TRL + PEFT run on a tiny random model. Runs where torch, transformers, peft and trl are installed
(the CI ``llm`` job); skipped elsewhere."""
import pytest

pytest.importorskip("trl")
pytest.importorskip("peft")


def test_lora_smoke(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("smoke", Path(__file__).parents[1] / "scripts" / "smoke_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--work", str(tmp_path), "--steps", "3"])
    assert (tmp_path / "run" / "adapter" / "adapter_model.safetensors").exists()
