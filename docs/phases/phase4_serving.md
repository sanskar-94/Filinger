# Phase 4: Serve it in production

## Topology

```
                      ALB ingress (TLS)
                            |
                 filings-api  (FastAPI, CPU nodes, HPA 2 to 12)
                   |                    |                     |
     reaction model (LightGBM,   OfflineFeatureStore     vLLM service (GPU nodes, KEDA 1 to 4)
     from registry @champion)    (gold Parquet; Redis    Qwen2.5 3B + LoRA adapters:
                                  or DynamoDB at scale)   filings-extractor (champion)
                                                          filings-extractor-b (challenger, Phase 6)
     Prometheus  <- /metrics (API) and vLLM metrics, DCGM GPU metrics
     Grafana dashboard + alert rules            MLflow tracking and model registry
```

## API (`filings_lab/serving/app.py`)

| Route | Purpose |
|---|---|
| `GET /healthz`, `GET /readyz` | liveness, readiness (model loaded) |
| `GET /metrics` | Prometheus; drift gauges recomputed on scrape |
| `GET /v1/model` | active version, feature set, training window |
| `POST /v1/extract` | paragraphs or text to structured signals |
| `POST /v1/score` | accession or feature vector to P(positive reaction) with the top five SHAP drivers |
| `GET /v1/filings/{accession}` | features, stored LLM signals and a live score in one call |

Resilience: the vLLM call sits behind a circuit breaker (opens after 5 consecutive failures, half opens after
30 s). Errors, invalid JSON and an open breaker all fall back to the rule extractor, and the response carries
`fallback: true`. Authentication is an `X-API-Key` header (keys from a Kubernetes secret). Metrics use route
templates as labels, so per accession URLs do not explode cardinality.

## Models and tracking

1. Every training entry point logs through `filings_lab/tracking.py`: MLflow when `MLFLOW_TRACKING_URI` is
   set, a JSON run log otherwise.
2. `flab register` fits the final reaction model on all labelled history and writes a self describing
   artefact: booster, feature list, training window, walk forward metrics and the reference distributions of
   the top features and of the prediction (for drift).
3. The registry (`serving/registry.py`) resolves `models:/name@alias`. With an MLflow server it is the MLflow
   Model Registry; otherwise a directory with the same alias semantics (synced to S3 in Kubernetes).

## Drift (`serving/drift.py`)

PSI against training quantile bins for the twelve most important features and for the predicted probability,
a KS test for numeric features, and PSI on category shares for extraction outputs (for example the share of
"guidance lowered"). Alerts fire at PSI above 0.25 for an hour.

## Batch drift and retraining (`serving/drift_report.py`, `infra/k8s/base/cronjobs.yaml`)

`flab drift-report` scores the most recent months of filings with the production model and compares each
monitored feature and the prediction with the reference distributions stored in the model artefact. It
recommends retraining when a feature's PSI exceeds 0.25 or the prediction's exceeds 0.2, and can push the
numbers to a Prometheus Pushgateway. A nightly CronJob runs it; a weekly CronJob runs the quality gates,
retrains, and registers a challenger (promotion to champion stays a reviewed pull request).

It has already earned its keep twice on the synthetic data. It flagged log assets (PSI 1.5) because company
size trends upward, which led to a stationary within month size percentile. On the latest run it flags the
year over year change in risk factor length (PSI 0.30): risk sections stop growing once companies reach the
generator's cap, the same kind of drift real 10-Ks show as risk sections lengthen over a decade.

## Load test (`loadtest/locustfile.py`)

The analyst request mix (scores, filing lookups, extraction, agent questions, health checks) against two
uvicorn workers on the sandbox CPU, 40 concurrent users for 60 seconds:

| Endpoint | Requests | Failures | Median | p95 | p99 |
|---|---|---|---|---|---|
| `POST /v1/score` | 1,543 | 0 | 21 ms | 61 ms | 95 ms |
| `GET /v1/filings/{accession}` | 940 | 0 | 21 ms | 54 ms | 92 ms |
| `POST /v1/extract` (rules fallback) | 617 | 0 | 5 ms | 30 ms | 47 ms |
| `POST /v1/agent/ask` (offline agent) | 304 | 0 | 42 ms | 140 ms | 180 ms |
| All | 3,705 | 0 | 19 ms | 61 ms | 110 ms |

About 63 requests per second without errors. With vLLM in the loop the extraction path is GPU bound and
scales on queue depth through KEDA, which is why its autoscaling does not use CPU.

## Kubernetes (`infra/k8s`, `infra/eks`)

1. API: non root, read only root filesystem, init container syncs the registry and gold tables from S3 via
   IRSA, rolling updates with zero unavailable, PDB, zone spread, HPA on CPU and on requests per second (via
   prometheus-adapter).
2. vLLM: GPU node group (g6.xlarge, one L4) with a taint, init container syncs the champion and challenger
   adapters, startup probe for model load, KEDA scaling on `vllm:num_requests_waiting` because CPU says
   nothing about a GPU server's load, NetworkPolicy so only the API and Prometheus can reach it.
3. Terraform creates the VPC, EKS with both node groups, ECR repositories, the IRSA role and the GitHub OIDC
   deploy role (no long lived keys).

## CI/CD (`.github/workflows`)

| Workflow | What it does |
|---|---|
| `ci.yml` | ruff, tests on Python 3.11 to 3.13, Spark parity tests with Java 17, LoRA smoke test on CPU torch, web lint, typecheck, tests and build, the end to end script, kubeconform on both rendered overlays, terraform fmt and validate |
| `model-gate.yml` | on changes to models, configs, features or training code: rebuilds the frozen evaluation world, runs the data quality gates, retrains the reaction model, evaluates the candidate extractor on the gold set (or a staging vLLM endpoint when `CANDIDATE_LLM_URL` is set) and runs `flab gate` against `models/production.json`; fails the PR if anything gated is worse |
| `deploy.yml` | on main: builds and pushes both images tagged with the commit SHA, then a protected `production` environment applies the prod overlay, waits for rollouts and smoke tests `/readyz` |

Promotion is GitOps: a pull request edits `models/production.json`, and the gate must pass before merge.
Gated metrics: extraction score (may drop at most 0.01), schema validity, quote grounding, guidance F1,
p95 latency SLO of 8 s, and for the reaction model AUC, IC, and AUC significantly above 0.5.

## Local stack

```bash
flab register --alias champion
docker compose up --build              # API on :8080, Prometheus :9090, Grafana :3000, MLflow :5000
docker compose --profile gpu up        # adds vLLM on :8000 when an NVIDIA GPU is present
```

## Runbook

| Alert | First action |
|---|---|
| LlmCircuitBreakerOpen | check vLLM pod events (OOM, GPU node scale up), traffic is already on rules |
| ExtractionFallbackRateHigh | inspect `flab_extractions_total{outcome="invalid"}`: a new adapter emitting bad JSON means roll the alias back |
| FeatureDriftHigh / PredictionDrift | compare the drifting feature's recent distribution with training, check upstream parsing changes, schedule retraining |
| VllmQueueBacklog | GPU capacity: check KEDA and node group limits |
