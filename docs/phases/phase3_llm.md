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

Verified here: `scripts/smoke_lora.py` builds a tokenizer and a two layer Qwen2 style model from the
synthetic corpus, runs the real training loop, saves the adapter and merged weights, reloads the adapter
for inference and scores it with the harness (`tests/test_lora_smoke.py`, CI job `llm`). A real run needs
one 24 GB GPU for the 3B model: `flab llm-train --config configs/lora_qwen2.5_3b.json`.

## Evaluation (`filings_lab/llm/eval_harness.py`)

Per system: JSON validity, schema validity, macro F1 and accuracy per field, an `extraction_score` (mean
macro F1), quote grounding (is each quote really in the cited paragraph), latency p50 and p95, and cost per
1,000 filings from API usage or GPU hours. Systems are compared with a paired bootstrap over gold items.
`gate()` blocks a candidate that is worse than production on any gated metric beyond its tolerance; Phase 4
runs it in CI.

```bash
flab llm-eval --systems "rules,vllm:http://localhost:8000/v1|qwen-base,vllm:http://localhost:8000/v1|filings-extractor,frontier" \
              --reference "vllm:http://localhost:8000/v1|qwen-base"
```

## Feeding Phase 2

`flab llm-features --system <spec>` writes `llm_*` columns per filing and `flab train --extra <file>` adds
two feature sets (`llm_only`, `all_plus_llm`) plus a DeLong test of all plus LLM against all.

## What the synthetic world can and cannot show

On synthetic filings the rule baseline scores an extraction_score of 0.86 (guidance F1 1.0, tone F1 0.55)
on 400 gold items, because the synthetic guidance sentences are templated; tone is where it struggles.
Real filings phrase guidance in far more ways, which is exactly where the fine tuned model is expected to
earn its place. Results for the base, fine tuned and frontier systems require a GPU and an API key and are
deliberately not reported here; the commands above produce them, and the gate decides what ships.
