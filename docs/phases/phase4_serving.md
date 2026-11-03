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
`fallback: true`. Every quote the model returns is checked against the paragraph it cites; the response lists
each quote with `grounded` and the resolved `paragraph_id`, so a fabricated quote is never presented as
evidence (`flab_extraction_quotes_total{grounded}` counts them). Authentication is an `X-API-Key` header (keys
from a Kubernetes secret). Metrics use route templates as labels, so per accession URLs do not explode
cardinality. The image runs one uvicorn process per container: counters, the breaker and the experiment log
are per process, so pods scale through the HPA rather than workers inside a pod.

## Models and tracking

1. Every training entry point logs through `filings_lab/tracking.py`: MLflow when `MLFLOW_TRACKING_URI` is
   set, a JSON run log otherwise.
2. `flab register` fits the final reaction model on all labelled history and writes a self describing
   artefact: booster, feature list, training window, walk forward metrics and the reference distributions of
   the top features and of the prediction (for drift).
3. The registry (`serving/registry.py`) resolves `models:/name@alias`. With an MLflow server it is the MLflow
   Model Registry; otherwise a directory with the same alias semantics (synced to S3 in Kubernetes). A file
   registry version is staged and published with one rename, with permissions the non root API user can read.
4. Fine tuned adapters are registered the same way (`flab llm-register`, model `filings-extractor`), logged
   into their training run with the gold set scores and the gate outcome. Run against a real MLflow 3 server
   here: `filings-reaction` v1 at `@champion`, and the two tiny adapters as `filings-extractor` v2 and v3, each
   linked to its `phase3-lora` training run.

## Serving the fine tuned model (measured here, `flab llm-serve`)

vLLM needs a GPU, so `serving/llm_server.py` stands in for it on a CPU: the same OpenAI compatible routes
(`/v1/models`, `/v1/chat/completions`), one base model with several named LoRA adapters switched per request,
and vLLM's metric names (`vllm:num_requests_waiting`, `vllm:e2e_request_latency_seconds`, token counters), so
the API, the KEDA trigger and the dashboards work unchanged against either.

```bash
flab llm-serve --base artifacts/tiny-base \
  --adapter filings-extractor=artifacts/lora/tiny-champion/adapter \
  --adapter filings-extractor-b=artifacts/lora/tiny-challenger/adapter --port 8000
FLAB_LLM_BASE_URL=http://127.0.0.1:8000/v1 FLAB_LLM_PROMPT_STYLE=compact flab serve
```

What the run showed (`artifacts/phase4_llm_serving.json`):

1. Parity: the champion served over HTTP scores exactly what the in process model scores on the 400 gold
   filings (extraction score 0.3773, every field identical), at p50 0.39 s and p95 0.48 s per request on CPU.
2. A/B routing: each experiment variant calls its own adapter, and the experiment log records the version per
   request (variant A `vllm:filings-extractor`, variant B `vllm:filings-extractor-b`).
3. It also showed why the configured version is not enough: the challenger never produces valid JSON, so every
   B extraction was answered by the rules fallback while the log said B. Each exposure now records how many
   extractions it made and how many fell back, and `flab_agent_extractions_total{variant,outcome}` shows it
   live; a variant whose model keeps falling back is not testing that model.
4. Breaker: with the LLM server stopped, seven requests in a row were all answered by the rules (no errors), the
   breaker opened after five failures, and after the cooldown a probe on a real filing closed it again.

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

The analyst request mix (scores, filing lookups, extraction, agent questions, health checks) against the image
that ships, run the way the cluster runs it: one uvicorn process, a 2 CPU limit, read only root filesystem,
all capabilities dropped, non root; 40 concurrent users for 60 seconds:

| Endpoint | Requests | Failures | Median | p95 | p99 |
|---|---|---|---|---|---|
| `POST /v1/score` | 1,502 | 0 | 39 ms | 160 ms | 230 ms |
| `GET /v1/filings/{accession}` | 885 | 0 | 41 ms | 170 ms | 230 ms |
| `POST /v1/extract` (rules fallback) | 579 | 0 | 18 ms | 110 ms | 150 ms |
| `POST /v1/agent/ask` (offline agent) | 306 | 0 | 73 ms | 350 ms | 490 ms |
| All | 3,545 | 0 | 35 ms | 170 ms | 280 ms |

About 60 requests per second per pod without errors, so the HPA target of 40 requests per second per pod
leaves headroom. An earlier run with two workers in one container was faster at the tail (p95 61 ms) but its
metrics depended on which worker answered the scrape; one process per pod and more pods is the trade. With
vLLM in the loop the extraction path is GPU bound and scales on queue depth through KEDA, which is why its
autoscaling does not use CPU.

## Kubernetes (`infra/k8s`, `infra/eks`)

1. API: non root (numeric UID 10001 in both images, so `runAsNonRoot` can be verified), read only root
   filesystem, all capabilities dropped. An init container syncs the registry and the lake tables the scorer
   and the agent read from S3 via IRSA. Rolling updates with zero unavailable, PDB, zone spread, HPA on CPU and
   on requests per second (via prometheus-adapter). Both images were run locally with exactly these settings.
2. Experiment log across replicas: each pod appends to its own SQLite file on a pod volume and ships a
   consistent snapshot (SQLite online backup) to `s3://.../experiments/events/<pod>.db` every five minutes and
   at shutdown. Request ids carry an HMAC from a shared secret, so a rating that reaches a different pod than
   its answer is still accepted, and `flab ab-analyze --db s3://.../experiments/events/` joins every pod's
   file. A daily CronJob writes the readout back to S3.
3. The experiment definition is a ConfigMap generated from `configs/experiment.json`, and the cluster alert
   rules are a PrometheusRule generated from `infra/monitoring/alerts.yml` (`scripts/render_k8s.py`); CI fails
   if either copy is stale.
4. Jobs that write results back to S3 pair the work container with an uploader that waits for the work
   container's exit code and uploads only on success, so the Job reports the real outcome. The retraining Job
   runs the data quality gate directly against the S3 lake.
5. vLLM: GPU node group (g6.xlarge, one L4) with a taint, init container syncs the champion and challenger
   adapters, startup probe for model load, KEDA scaling on `vllm:num_requests_waiting` because CPU says
   nothing about a GPU server's load, NetworkPolicy so only the API and Prometheus can reach it.
6. Terraform creates the VPC, EKS with both node groups, ECR repositories, two IRSA roles (serving: read the
   registry and lake, append experiment logs; trainer: read the lake and logs, write registry versions and
   readouts) and the GitHub OIDC deploy role (no long lived keys).

## CI/CD (`.github/workflows`)

| Workflow | What it does |
|---|---|
| `ci.yml` | ruff, tests on Python 3.11 to 3.13 (including the S3 lake tests against a moto server), Spark parity tests with Java 17, LoRA smoke and local LLM server tests on CPU torch, web lint, typecheck, tests and build, the end to end script, kubeconform on both rendered overlays, generated Kubernetes objects up to date, promtool config check and alert rule unit tests, terraform fmt and validate |
| `model-gate.yml` | on changes to models, configs, features or training code: rebuilds the frozen evaluation world, runs the data quality gates, retrains the reaction model, evaluates the candidate extractor on the gold set (or a staging vLLM endpoint when `CANDIDATE_LLM_URL` is set) and runs `flab gate` against `models/production.json`; fails the PR if anything gated is worse |
| `deploy.yml` | on main: builds and pushes both images tagged with the commit SHA, then a protected `production` environment applies the prod overlay, waits for rollouts and smoke tests `/readyz` |

Promotion is GitOps: a pull request edits `models/production.json`, and the gate must pass before merge.
Gated metrics: extraction score (may drop at most 0.01), schema validity, quote grounding, guidance F1,
p95 latency SLO of 8 s, and for the reaction model AUC, IC, and AUC significantly above 0.5.

## Local stack

```bash
flab register --alias champion
docker compose up --build              # API :8080, web :3000, Prometheus :9090, Grafana :3001, MLflow :5000
docker compose --profile gpu up        # adds vLLM on :8000 when an NVIDIA GPU is present
```

Alert rules have unit tests (`infra/monitoring/alerts_test.yml`, run with `promtool test rules`): every alert
must fire on a series that breaches it for longer than its hold time and stay silent on a healthy one.

## Runbook

| Alert | First action |
|---|---|
| LlmCircuitBreakerOpen | check vLLM pod events (OOM, GPU node scale up), traffic is already on rules |
| ExtractionFallbackRateHigh | inspect `flab_extractions_total{outcome="invalid"}`: a new adapter emitting bad JSON means roll the alias back |
| FeatureDriftHigh / PredictionDrift | compare the drifting feature's recent distribution with training, check upstream parsing changes, schedule retraining |
| VllmQueueBacklog | GPU capacity: check KEDA and node group limits |
