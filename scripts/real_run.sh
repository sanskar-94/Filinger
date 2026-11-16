#!/usr/bin/env bash
# The lab on real EDGAR filings: index, download, prices, fundamentals, lake, features, quality gates and the
# walk forward model, for the companies in configs/universe_large_cap.txt. Resumable: every stage skips what
# is already on disk. Raw filings are stored compressed (about 1 GB for the default run).
#
#   echo 'SEC_USER_AGENT="Your Name you@example.com"' >> .env   # the SEC refuses undeclared tools
#   bash scripts/real_run.sh
#
# The universe is today's large companies, so it is survivorship biased: a company that failed or was bought
# between START and END is not in it. Read the results as "does this work on real text", not as a strategy.
set -euo pipefail
if [ -z "${SEC_USER_AGENT:-}" ] && [ -f .env ]; then set -a; . ./.env; set +a; fi  # the contact line can live in .env
: "${SEC_USER_AGENT:?set SEC_USER_AGENT to a name and contact email (in .env or the shell), as the SEC fair access policy asks}"
UNIVERSE="${UNIVERSE:-configs/universe_large_cap.txt}"
START="${START:-2015-01-01}"; END="${END:-2024-12-31}"
OUT="${OUT:-data/real}"; ART="${ART:-artifacts/real}"; WORKERS="${WORKERS:-4}"
PRICE_START="$(python -c "import sys, pandas as pd; print((pd.Timestamp(sys.argv[1]) - pd.DateOffset(years=2)).date())" "$START")"
mkdir -p "$OUT" "$ART"

flab index --tickers "$UNIVERSE" --start "$START" --end "$END" --out "$OUT/index.parquet" --cache "$OUT/http_cache"
flab download --index "$OUT/index.parquet" --raw "$OUT/raw" --workers "$WORKERS" --compress
[ -f "$OUT/prices.parquet" ] || flab prices --index "$OUT/index.parquet" --source yahoo --start "$PRICE_START" --end "$END" --out "$OUT/prices.parquet"
[ -f "$OUT/facts.parquet" ] || flab facts --index "$OUT/index.parquet" --cache "$OUT/http_cache" --out "$OUT/facts.parquet"
flab local --index "$OUT/index.parquet" --raw "$OUT/raw" --prices "$OUT/prices.parquet" --lake "$OUT/lake" --workers "$WORKERS"
flab features --lake "$OUT/lake" --facts "$OUT/facts.parquet"
# the thresholds were set on the synthetic world; on real filings a failed gate is a finding to read, not a stop
flab dq --lake "$OUT/lake" --out "$ART/dq" || echo "data quality gates failed: see $ART/dq"
flab train --lake "$OUT/lake" --out "$ART/phase2" --permutations "${PERMUTATIONS:-20}"
echo "report: $ART/phase2/report.md"
