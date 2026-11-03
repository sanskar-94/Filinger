# Frontier reference labels (blind)

`frontier_labels.jsonl` holds extraction labels for 60 gold filings, written by a frontier model reading the filings directly,
because no API key was available for the `flab llm-label` teacher path. They are the
frontier row of the Phase 3 comparison; how they were made matters for reading that row.

## How the sample and the labels were produced

1. `python scripts/frontier_sample.py --gold data/llm_dataset_compact/gold.jsonl --size 60 --seed 2026` draws the
   sample from the first 400 compact gold filings: every liquidity and litigation positive (they are rare) plus a
   seeded random remainder, shuffled so position reveals nothing. It writes `sample_gold.jsonl` (with labels) and
   `sample_inputs.md` (company, form and numbered paragraphs only).
2. Labels were written blind from `sample_inputs.md`, following the field definitions of `SYSTEM_PROMPT` in
   `src/filings_lab/llm/prompts.py`. The gold labels were not opened until all 60 were written.
3. One rubric was fixed for tone after the first 30 filings and applied to all 60: count each substantive
   statement about results (repeats count as emphasis); if only one side appears, that is the tone; if both
   appear and differ by at most one, the tone is neutral. Three of the first 30 labels changed to follow it,
   before any gold label was seen.
4. Scored with the same harness as every other system:

```bash
flab llm-eval --gold data/frontier_sample/sample_gold.jsonl --prompt-style compact --batch-size 16 \
  --systems "rules,hf:artifacts/tiny-base@tiny-base,hf:artifacts/tiny-base|artifacts/lora/tiny-champion/adapter@tiny-lora-champion,replay:eval/frontier_sample/frontier_labels.jsonl@frontier-blind" \
  --reference rules --out artifacts/phase3_frontier_sample
```

## What differs from the API path

The labels went through no structured output decoding and no batch API, so latency and cost are not measured
and are reported as zero; only quality is comparable. The sample over represents liquidity and litigation
positives by design, so compare systems on this sample with each other, not with scores on the full gold set.
