# Common entry points. Every target is a thin wrapper over the flab CLI.
LAKE ?= data/lake
SYN ?= data/synthetic

.PHONY: install synth pipeline features dq train llm register serve web test lint e2e ab dashboard-data drift loadtest spark-test

install:
	pip install -e ".[ml,serve,agent,dev]"
	cd web && npm ci

synth:
	flab synth --out $(SYN) --companies 60

pipeline: synth
	flab local --index $(SYN)/index.parquet --prices $(SYN)/prices.parquet --lake $(LAKE)

features:
	flab features --lake $(LAKE) --facts $(SYN)/facts.parquet

dq:
	flab dq --lake $(LAKE)

train: features dq
	flab train --lake $(LAKE) --permutations 20

llm:
	flab llm-dataset --lake $(LAKE) --truth $(SYN)/truth.parquet
	flab llm-eval --systems rules --limit 400
	flab llm-features --lake $(LAKE) --system rules

register:
	flab register --lake $(LAKE) --alias champion

serve:
	flab serve --port 8080

web:
	cd web && FLAB_API_URL=http://localhost:8080 npm run dev

ab:
	flab ab-plan
	flab ab-simulate --users 3000 --days 14 --start-days-ago 14
	flab ab-analyze

dashboard-data:
	flab bi-export

drift:
	flab drift-report --lake $(LAKE)

loadtest:
	locust -f loadtest/locustfile.py --headless -u 40 -r 10 -t 60s --host http://localhost:8080 --csv artifacts/loadtest/run

test:
	pytest -q

spark-test:
	pytest -q tests/test_spark_parity.py

lint:
	ruff check src tests scripts loadtest
	cd web && npm run lint && npm run typecheck

e2e:
	bash scripts/e2e.sh
