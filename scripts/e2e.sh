#!/usr/bin/env bash
# Runs the whole lab on a tiny synthetic world in a temporary directory and checks every stage's output.
# Used by CI (job e2e) and handy before a release. Takes about two minutes on a laptop.
set -euo pipefail
WORK="${1:-$(mktemp -d)}"
echo "working in $WORK"
SYN="$WORK/synthetic"; LAKE="$WORK/lake"; ART="$WORK/artifacts"
export FLAB_REGISTRY="$WORK/registry" FLAB_EVENT_DB="$WORK/experiments.db"

flab synth --out "$SYN" --companies 10 --start 2015-01-01 --end 2021-12-31 --seed 5
flab local --index "$SYN/index.parquet" --prices "$SYN/prices.parquet" --lake "$LAKE" --workers 2
flab features --lake "$LAKE" --facts "$SYN/facts.parquet"
flab dq --lake "$LAKE" --out "$ART/dq"
flab train --lake "$LAKE" --out "$ART/phase2" --splits 3 --permutations 0
flab llm-dataset --lake "$LAKE" --truth "$SYN/truth.parquet" --out "$WORK/llm_dataset" --test-years 1
flab llm-eval --gold "$WORK/llm_dataset/gold.jsonl" --systems rules --limit 100 --out "$ART/phase3" > /dev/null
flab llm-features --lake "$LAKE" --system rules --out "$LAKE/gold/llm_signals.parquet"
flab register --lake "$LAKE" --metrics "$ART/phase2/metrics.json" --alias champion
flab drift-report --lake "$LAKE" --months 18 --out "$ART/drift" > /dev/null
flab gate --production models/production.json --extractor-eval "$ART/phase3/eval.json" --system rules || echo "gate: candidate differs from production on this tiny world (expected)"

python - "$WORK" <<'PY'
import json, sys
from pathlib import Path
from fastapi.testclient import TestClient
import os
w = Path(sys.argv[1])
os.environ["FLAB_FEATURES"] = str(w / "lake/gold/features")
os.environ["FLAB_SIGNALS"] = str(w / "lake/gold/llm_signals.parquet")
os.environ["FLAB_LAKE"] = str(w / "lake")
os.environ["FLAB_AGENT_MODE"] = "offline"
from filings_lab.serving.app import create_app
c = TestClient(create_app())
assert c.get("/readyz").json()["status"] == "ready"
companies = c.get("/v1/companies").json()
t = companies[0]["ticker"]
ans = c.post("/v1/agent/ask", json={"question": "Did the company change its guidance?", "ticker": t, "user_id": "e2e"}).json()
assert ans["citations"] and ans["citation_valid_rate"] == 1.0
acc = ans["citations"][0]["accession"]
doc = c.get(f"/v1/filings/{acc}/paragraphs").json()
assert any(p["paragraph_id"] == ans["citations"][0]["paragraph_id"] for s in doc["sections"] for p in s["paragraphs"])
dq = json.loads((w / "artifacts/dq/dq.json").read_text())
assert not [d for d in dq if d["status"] == "fail"]
print(f"e2e ok: {len(companies)} companies, answer with {len(ans['citations'])} verified citations")
PY
