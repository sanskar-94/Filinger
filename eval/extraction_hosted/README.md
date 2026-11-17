# Hosted extraction runs

Free hosted models, with no fine tuning, extracting the signal schema from 100 gold filings, scored beside the rules
by `flab llm-eval`. The reading is in [docs/phases/phase3_llm.md](../../docs/phases/phase3_llm.md).

| Folder | Evidence per filing | What it is |
|---|---|---|
| [nemotron-3-ultra](nemotron-3-ultra/report.md) | up to 12,000 characters (`data/llm_dataset`) | `api:nvidia/nemotron-3-ultra-550b-a55b` with 4,096 output tokens: 100% schema valid, extraction score 0.8995 against the rules' 0.8783 |
| [first_run](first_run/report.md) | up to 12,000 characters | the same model with the 768 token cap of the fine tuned extractor: 24% valid, because the model's reasoning used up the cap before it wrote the JSON. Kept as produced |
| [nemotron-3-ultra-compact](nemotron-3-ultra-compact/report.md) | 1,500 characters, what the 3B student sees (`data/llm_dataset_compact`) | the teacher of the distillation in the student's own setting: 100% valid, 0.8897 against the rules' 0.8662 |
| [nemotron-3.5-lightning-compact](nemotron-3.5-lightning-compact/report.md) | 1,500 characters | the smaller free model, considered as a faster teacher and rejected: 54 of 100 answers were cut off at 4,096 tokens while still reasoning, so 46% valid and 0.4883 |

`predictions/` holds every output with its error, if any, and `items.parquet` the per filing marks behind the
paired bootstrap.

To run them again (a free key from build.nvidia.com in `.env` as `NVIDIA_API_KEY`):

```bash
flab llm-eval --gold data/llm_dataset/gold.jsonl --limit 100 --systems "rules,api:nvidia/nemotron-3-ultra-550b-a55b" --reference rules --workers 4 --out eval/extraction_hosted/nemotron-3-ultra
```

```bash
flab llm-eval --gold data/llm_dataset_compact/gold.jsonl --limit 100 --systems "rules,api:nvidia/nemotron-3-ultra-550b-a55b" --reference rules --workers 4 --out eval/extraction_hosted/nemotron-3-ultra-compact
```
