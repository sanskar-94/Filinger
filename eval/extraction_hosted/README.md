# Hosted extraction runs

A free hosted model, with no fine tuning, extracting the signal schema from the 100 gold filings of
`data/llm_dataset/gold.jsonl`, scored beside the rules by `flab llm-eval`. The reading is in
[docs/phases/phase3_llm.md](../../docs/phases/phase3_llm.md).

| Folder | What it is |
|---|---|
| [nemotron-3-ultra](nemotron-3-ultra/report.md) | `api:nvidia/nemotron-3-ultra-550b-a55b` with 4,096 output tokens: 100% schema valid, extraction score 0.8995 against the rules' 0.8783 |
| [first_run](first_run/report.md) | the same model with the 768 token cap of the fine tuned extractor: 24% valid, because the model's reasoning used up the cap before it wrote the JSON. Kept as produced |

`predictions/` holds every output with its error, if any, and `items.parquet` the per filing marks behind the
paired bootstrap.

To run it again (a free key from build.nvidia.com in `.env` as `NVIDIA_API_KEY`):

```bash
flab llm-eval --gold data/llm_dataset/gold.jsonl --limit 100 --systems "rules,api:nvidia/nemotron-3-ultra-550b-a55b" --reference rules --workers 4 --out eval/extraction_hosted/nemotron-3-ultra
```
