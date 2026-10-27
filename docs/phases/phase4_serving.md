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
| `ci.yml` | ruff, tests on Python 3.11 to 3.13, Spark parity test with Java 17, LoRA smoke test on CPU torch, web lint, typecheck, tests and build |
| `model-gate.yml` | on changes to models, configs, features or training code: rebuilds the frozen evaluation world, retrains the reaction model, evaluates the candidate extractor on the gold set (or a staging vLLM endpoint when `CANDIDATE_LLM_URL` is set) and runs `flab gate` against `models/production.json`; fails the PR if anything gated is worse |
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
