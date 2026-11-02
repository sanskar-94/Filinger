"""Load test for the API: the mix an analyst session produces.

    locust -f loadtest/locustfile.py --headless -u 40 -r 10 -t 60s --host http://localhost:8080 --csv artifacts/loadtest/run

Accessions and tickers are read from the lake (FLAB_LAKE, default data/lake) so requests hit real rows.
Set FLAB_API_KEY when the API requires a key.
"""
from __future__ import annotations

import os
import random

import pandas as pd
from locust import HttpUser, between, task

LAKE = os.environ.get("FLAB_LAKE", "data/lake")
FEATURES = pd.read_parquet(f"{LAKE}/gold/features", columns=["accession", "ticker"])
ACCESSIONS = FEATURES["accession"].tolist()
TICKERS = sorted(FEATURES["ticker"].dropna().unique().tolist())
QUESTIONS = ["Did the company change its revenue guidance?", "Any litigation or regulatory investigation?",
             "Has management flagged liquidity concerns?", "How did demand and margins develop?"]
TEXT = ("Revenue increased 12% compared to the prior year driven by strong demand.\n\n"
        "The Company is raising its full year revenue guidance to $5 billion to $6 billion.")


class Analyst(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        key = os.environ.get("FLAB_API_KEY")
        if key:
            self.client.headers["X-API-Key"] = key

    @task(5)
    def score(self):
        self.client.post("/v1/score", json={"accession": random.choice(ACCESSIONS)}, name="/v1/score")

    @task(3)
    def filing(self):
        self.client.get(f"/v1/filings/{random.choice(ACCESSIONS)}", name="/v1/filings/[accession]")

    @task(2)
    def extract(self):
        self.client.post("/v1/extract", json={"text": TEXT, "form": "8-K"}, name="/v1/extract")

    @task(1)
    def ask(self):
        self.client.post("/v1/agent/ask", json={"question": random.choice(QUESTIONS), "ticker": random.choice(TICKERS),
                                                "user_id": f"load-{random.randint(0, 10_000)}"}, name="/v1/agent/ask")

    @task(1)
    def health(self):
        self.client.get("/healthz")
