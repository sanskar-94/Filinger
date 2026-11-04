# Phase 3: Fine tune an LLM for structured signal extraction

The LLM does not predict returns. It reads a filing and extracts a small set of well defined signals,
which then become features for the Phase 2 model and citations for the Phase 5 agent. That framing keeps
the task checkable: every label can be verified against the text by a human.

## The task (`filings_lab/llm/schema.py`)

| Field | Values | Definition (abridged; full text in `prompts.py`) |
|---|---|---|
| guidance_direction | raised, lowered, maintained, none | what management did to its own forward outlook in this document |
| management_tone | positive, neutral, negative | characterisation of recent results; negations read carefully |
| liquidity_concern | bool | going concern doubt, covenant breach or waiver, explicit need to raise capital |
| litigation_risk | bool | a specific non routine proceeding, not boilerplate |
| key_quotes | 1 to 3 | verbatim excerpts with the [P#] paragraph they come from |
| summary | text | at most 60 words |

Long sections are cut to a character budget by `select_paragraphs`, which keeps the paragraphs with the most
signal keywords and restores document order. Paragraphs are numbered so quotes resolve to stable paragraph
ids from Phase 1.

## Systems compared (`filings_lab/llm/providers.py`)

| System | How it runs | Notes |
|---|---|---|
| rules | regex and lexicon | free floor; strong on templated language, weak on tone |
| base | vLLM or transformers, Qwen2.5 3B Instruct or Llama 3.2 3B Instruct | prompt with schema, no fine tuning |
| fine tuned | same base plus the LoRA adapter (or merged weights) | the distilled student |
| frontier | a hosted frontier model (`FLAB_FRONTIER_MODEL` sets which) | teacher and upper reference |

Frontier calls use structured outputs (`output_config.format` with the strict JSON schema), an explicit
`effort` (default medium), a cacheable system prompt, and the server side refusal fallback
(`fallbacks="default"` with beta `server-side-fallback-2026-07-01`) so a classifier decline is retried on
the provider's recommended fallback model instead of losing the label. Refusals that still occur are recorded
as errors rather than parsed. Bulk labelling goes through the Message Batches API at half price; fallbacks
are not accepted there, so failed batch items are retried in realtime afterwards (`flab llm-label`).

## Data (`filings_lab/llm/dataset.py`)

1. Teacher labels: `flab llm-label --mode batch --limit 3000` writes labels with token usage and cost.
2. Gold set: a stratified sample of a few hundred test filings reviewed by a human; store it in the same
   JSONL format as `gold.jsonl`.
3. Splits: test is the last two years plus all filings of 10% held out companies; validation is the year
   before; train is everything earlier. A test asserts held out companies never reach training.
4. Training data is stratified so rare classes (guidance changes, liquidity, litigation) are all kept.
5. Format: TRL conversational prompt and completion, so the loss covers only the JSON answer.

Cost estimate for the teacher, stated as assumptions: about 3,500 input and 1,000 output tokens per filing
(output includes thinking) at $4 and $20 per million tokens gives about $0.034 per filing realtime and
$0.017 through batches, roughly $50 for 3,000 training labels. `flab llm-label` prints the exact figure
from the API's usage fields.

## Human review of teacher labels (`llm/review.py`, app page `/review`)

A reviewer sees the filing excerpt next to the teacher's proposed labels and accepts, corrects or skips.
Decisions are appended to `reviewed.jsonl` (resumable, never overwritten), and the page shows per field
agreement with the teacher and Cohen's kappa (checked against sklearn), which is the evidence for how far the
teacher can be trusted per field. `flab llm-gold` turns the reviewed items into the gold set, human labels
replacing the teacher's.

## Training (`filings_lab/llm/train_lora.py`)

TRL `SFTTrainer` with a PEFT `LoraConfig`: rank 16, alpha 32, dropout 0.05, all linear projections,
learning rate 2e-4 cosine, two epochs, completion only loss, early model selection on validation loss,
optional 4 bit NF4 QLoRA (`configs/qlora_llama3.1_8b.json`) and a merged checkpoint for vLLM. The script
adapts to API renames across TRL and transformers versions (for example `warmup_ratio` and `torch_dtype`).
Every run logs to MLflow (experiment `phase3-lora`) with its parameters, curves and the adapter folder, and
writes the run id next to the adapter so the registered version links back to the run that made it.

A real run of the intended models needs one 24 GB GPU for the 3B model:
`flab llm-train --config configs/lora_qwen2.5_3b.json`. The first runs had no GPU and could not reach the Hugging
Face hub, so the whole method was run end to end on a model small enough for four CPU cores:

1. `flab llm-pretrain` builds a 4,096 token byte level BPE tokenizer and pretrains a 5.2M parameter Qwen2 style
   model from scratch on the paragraphs of training period filings only (2.1M tokens, 250 steps; held out
   perplexity 1.25, which mostly says the synthetic filings are templated).
2. Two LoRA configurations fine tune it on 2,745 teacher labelled filings with the compact prompt (the schema
   instruction without the long field definitions, 1,500 character text budget, so a sample fits 1,024 tokens):

| Adapter | Config | Trainable | Steps | Validation loss | Token accuracy |
|---|---|---|---|---|---|
| champion | `configs/lora_tiny_champion.json`: r 16, all linear layers, 3 epochs, lr 2e-3, merged | 8.2% | 1,032 | 1.43 | 93.4% |
| challenger | `configs/lora_tiny_challenger.json`: r 4, q and v only, 1 epoch, lr 1e-3 | 0.4% | 344 | 5.66 | 20.4% |

`scripts/smoke_lora.py` (CI job `llm-smoke`) runs the same loop on a two layer model in seconds.

### Running the 3B to 8B fine tune

`scripts/gpu_finetune.sh` is the whole phase as one job: teacher labels when `ANTHROPIC_API_KEY` is set, the
dataset, the LoRA (Qwen2.5 3B) or QLoRA (`CONFIG=configs/qlora_llama3.1_8b.json`, Llama 3.1 8B) fine tune, an
evaluation of the production rules, the base model, the fine tuned model and the frontier model on the same
production gold filings, the gate, and registration of the adapter with its verdict (MLflow when
`MLFLOW_TRACKING_URI` is set). It runs in three places:

1. Any machine with one 24 GB GPU and access to the model hub: `bash scripts/gpu_finetune.sh`.
2. The training image: `docker build -f Dockerfile.train -t filings-train .`, then
   `docker run --gpus all -e HF_TOKEN -e ANTHROPIC_API_KEY filings-train` (the deploy workflow pushes it to ECR).
3. The cluster: `infra/k8s/jobs/finetune.yaml`, a Job on the GPU node group that syncs the lake from S3, runs the
   script and uploads the run and the candidate adapter; serving it stays a reviewed pull request.
4. An Apple Silicon Mac, including one with 8 GB: `scripts/mac_finetune.sh` (install with
   `pip install -e ".[ml,serve,mlx]"`), or `scripts/mac_full_run.sh`, which does everything from a fresh clone in
   one command: a Python 3.10+ environment, the synthetic lake, the fine tune, and the results copied to
   `eval/mac_3b/` for committing. Half precision 3B weights need about 6 GB, which an 8 GB Mac cannot spare for
   training, so this route trains LoRA adapters on Qwen2.5 3B Instruct quantized to 4 bit (about 2 GB) with Apple's
   MLX on the Mac's GPU: batch of one, compact prompts of about 600 tokens, gradient checkpointing, loss on the
   answer only. The evaluation harness scores MLX models through an `mlx:` system spec, loading one model at a
   time, and the same gate and registration follow. If memory still runs short, `MODEL` can name the 1.5B 4 bit
   model instead. Verified here on MLX's CPU build with the small model: the whole script runs end to end, and a
   600 step training run lowered validation loss from 10.0 to 6.3. That is too few examples for the small scratch
   model to emit valid JSON (the PyTorch run needed about 8,000), and MLX's CPU build trains at only about 12 tokens
   per second here, so the full learning check belongs on a Mac's GPU, where a 3B instruct model also starts from
   far stronger instruction following.

`SMOKE=1` runs every step of either script on the small CPU model. CI does that for both on every push, on a tiny
world it builds from scratch (the MLX one on MLX's CPU build), so both paths are exercised continuously even though
CI has neither a GPU nor a Mac. That run caught one
bug: with `report_to: mlflow` and no tracking server, the transformers callback fell back to a local store and
stopped the run; MLflow reporting now switches on only when a server is configured, as everywhere else.

## Evaluation (`filings_lab/llm/eval_harness.py`)

Per system: JSON validity, schema validity, macro F1 and accuracy per field, an `extraction_score` (mean
macro F1), quote grounding (is each quote really in the cited paragraph), latency p50 and p95, and cost per
1,000 filings from API usage or GPU hours. Systems are compared with a paired bootstrap over gold items.
Local models can generate in batches (`--batch-size`): quality is identical under greedy decoding, latency
becomes amortized, so serving latency is measured against the served endpoint one request at a time.

Every `eval.json` records a fingerprint of the exact inputs and labels it scored. Two evaluations are only
comparable when the fingerprints match: the compact gold set holds the same 400 filings as the production gold
set, but its shorter text budget changes 155 inputs and 51 labels.

```bash
flab llm-eval --gold data/llm_dataset_compact/gold.jsonl --limit 400 --prompt-style compact --batch-size 16 \
  --systems "rules,hf:artifacts/tiny-base@tiny-base,hf:artifacts/tiny-base|artifacts/lora/tiny-champion/adapter@tiny-lora-champion,hf:artifacts/tiny-base|artifacts/lora/tiny-challenger/adapter@tiny-lora-challenger" \
  --reference rules --out artifacts/phase3_tiny
```

## Results on the synthetic world (400 gold filings, compact inputs)

| System | JSON valid | Schema valid | Extraction score | Guidance F1 | Tone F1 | Quote grounding | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| rules (production) | 100% | 100% | 0.853 | 1.00 | 0.54 | 100% | under 1 ms |
| tiny base, no fine tuning | 0% | 0% | 0.000 | 0.00 | 0.00 | n/a | 0.27 / 0.36 s (batched) |
| tiny + LoRA champion | 100% | 100% | 0.377 | 0.30 | 0.22 | 29% | 0.39 / 0.48 s (served, one at a time) |
| tiny + LoRA challenger | 0% | 0% | 0.000 | 0.00 | 0.00 | n/a | 0.27 / 0.35 s (batched) |

Paired bootstrap, champion minus rules on extraction score: minus 0.48, 95% CI [minus 0.51, minus 0.38].
Reports: `artifacts/phase3_tiny/report.md` and, for the served model, `artifacts/phase3_served/report.md`.

What this shows, read plainly:

1. Fine tuning does what it is for. The base model writes filing style prose and never a JSON object; after
   LoRA the same model emits schema valid JSON for every gold filing, and the served model scores exactly what
   the in process model scores, field for field.
2. Capacity is the limit. A 5M parameter model learns the format but not the reading: it mostly predicts the
   majority class per field, and only 29% of its quotes are grounded. The rule baseline wins by 0.48 on the
   templated synthetic text, which is the expected result for a model this size.
3. The undertrained challenger never closes a JSON object. The gate exists for exactly this adapter.

## Frontier reference (blind labels)

Without an API key the teacher path could not run, so a frontier model labelled a fixed sample of 60 gold filings
directly, blind to the gold labels, from the same inputs and field definitions; the harness then scored those
labels like any other system (`replay:` system spec). How the sample and labels were made, and what is not
comparable (latency, cost), is in `eval/frontier_sample/README.md`. Every liquidity and litigation positive is in
the sample, so compare the rows with each other, not with the 400 filing table above.

| System, 60 filings | Schema valid | Extraction score | Guidance F1 | Tone F1 | Liquidity F1 | Litigation F1 | Quotes grounded |
|---|---|---|---|---|---|---|---|
| frontier reference | 100% | 0.909 | 1.00 | 0.67 | 1.00 | 0.97 | 100% |
| rules (production) | 100% | 0.859 | 1.00 | 0.57 | 0.90 | 0.97 | 100% |
| tiny + LoRA champion | 100% | 0.338 | 0.22 | 0.20 | 0.49 | 0.45 | 30% |
| tiny base, tiny + LoRA challenger | 0% | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 | n/a |

Paired bootstrap, frontier minus rules: plus 0.050, 95% CI [0.006, 0.162], P(no gain) = 0.011. The frontier model
gains where rules are weakest: tone, which needs weighing mixed statements, and liquidity, where the rules miss
a stated need to raise capital to fund operations. Its one litigation miss is a filing whose 1,500 character excerpt contains no legal paragraph
at all, which exposed a dataset fault (below). So the ordering the project expects holds on this data: frontier
above rules above the CPU sized student, and a 3B to 8B student is what has to close the gap.

### A dataset fault the frontier labels exposed

Evidence selection ranked paragraphs by how many keywords they contain, so at a tight budget a long results
paragraph full of generic words ("increased", "declined") could push out a one line class action or covenant
waiver, leaving a positive label the prompt could not support. In the compact build used for the tiny model, 4 of
128 litigation and 4 of 61 liquidity training positives, and 1 of 32 and 1 of 11 gold positives, had no supporting
text. Paragraphs with direct evidence for a labelled field now rank first, every dataset build reports
unsupported labels per field in `stats.json`, and a test requires zero at both budgets. Rebuilt with the fix, the
compact dataset has none; the production gold set (every paragraph fits its budget) keeps its fingerprint. The
tiny model results above predate the fix.

## The gate decides (`flab gate`)

`gate()` blocks a candidate that is worse than production on any gated metric beyond its tolerance. When the
evaluation run also scored the production system, that row is the baseline (same items, same labels);
otherwise the stored production metrics are used only if the gold fingerprint matches, and the gate refuses
anything else. CI's model gate scores production in the same run for exactly this reason, and a from scratch
rebuild of the evaluation world reproduces the recorded fingerprint.

```
tiny-lora-champion   FAIL (baseline: rules in the same evaluation run)
  extraction_score 0.3773 vs 0.8533, quote_grounding_rate 0.2900 vs 1.0000, guidance_direction_f1 0.2982 vs 1.0000
tiny-lora-challenger FAIL (baseline: rules in the same evaluation run)
  extraction_score 0.0000 vs 0.8533, schema_valid_rate 0.0000 vs 1.0000, guidance_direction_f1 0.0000 vs 1.0000
```

Production stays on the rules extractor (`models/production.json`).

## Registry and lineage (`flab llm-register`)

Both adapters are registered as versions of `filings-extractor` in the MLflow Model Registry, logged into their
own training runs (so each version links to its curves and config), with the gold set scores as metrics and
tags for the base model, LoRA rank, A/B variant and gate outcome (`gate=failed`). A registered version resolves
to the adapter folder that vLLM loads; as a pyfunc it can also run on its own (base model plus adapter, input
rows in, signals out). Without an MLflow server the same command writes the file registry.

## Feeding Phase 2

`flab llm-features --system <spec>` writes `llm_*` columns per filing and `flab train --extra <file>` adds
two feature sets (`llm_only`, `all_plus_llm`) plus a DeLong test of all plus LLM against all.

Run for real on all 6,830 filings with the fine tuned champion (schema valid on every filing; about 14
minutes on four CPU cores with batched generation), and with the rules for comparison, both against the
same base features and the same purged walk forward folds:

| Signals from | LLM signals alone, AUC [95% CI] | All features | All plus LLM signals | DeLong, plus LLM vs all |
|---|---|---|---|---|
| rules | 0.571 [0.553, 0.588] | 0.569 | 0.562 | minus 0.007, p = 0.041 |
| tiny LoRA champion | 0.534 [0.516, 0.551] | 0.569 | 0.558 | minus 0.011, p = 0.007 |

On the synthetic world the extracted signals carry information on their own, but nothing the text features
do not already hold: the templated language that drives the labels is in the lexicon features too, so the
extra columns only add variance, and the noisier fine tuned signals cost more. The gate kept this model out of
production on extraction quality; Phase 2 reaches the same verdict on predictive value, independently.
Reports: `artifacts/phase2_llm/report.md` (rules) and `artifacts/phase2_llm_tiny/report.md` (champion).

## What the synthetic world can and cannot show

The synthetic guidance sentences are templated, which is why rules reach a guidance F1 of 1.0 and why a small
model has nothing to add there; tone is where rules struggle (F1 0.54) and where the frontier reference gains.
Real filings phrase guidance in far more ways, which is where a 3B to 8B fine tuned model is expected to earn its
place. That run needs a GPU and access to the model hub, neither of which this environment has; it is packaged as
one GPU job (below) that trains, scores base, fine tuned and frontier systems on the gold set, gates the
candidate and registers it.
