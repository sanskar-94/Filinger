# Hosted agent runs

Each folder is one `flab agent-eval` run: the analyst agent answering questions with known answers from the
synthetic world, through a hosted model on NVIDIA's free endpoints, or the offline agent on the same questions for
comparison. What the numbers mean, and what changed between the runs, is in
[docs/phases/phase5_agent.md](../../docs/phases/phase5_agent.md#hosted-models-on-nvidias-free-endpoints).

| Folder | Agent | Questions | Role |
|---|---|---|---|
| [nemotron-3-ultra](nemotron-3-ultra/report.md) | `nvidia/nemotron-3-ultra-550b-a55b` | `--seed 2`, 60 per retrieval mode | final |
| [nemotron-3.5-lightning](nemotron-3.5-lightning/report.md) | `nvidia/nemotron-3.5-lightning-30b-a3b` | `--seed 2`, the same 60 | final |
| [offline](offline/report.md) | the offline extractive agent | `--seed 2`, the same 60 | baseline |
| [first_run](first_run/report.md) | `nvidia/nemotron-3-ultra-550b-a55b` | `--seed 0` | development, kept as produced |
| [second_run](second_run/report.md) | `nvidia/nemotron-3.5-lightning-30b-a3b` | `--seed 1` | development, kept as produced |

The development runs were scored by the checker and loop of their day. The first lost 24 of its 60 hybrid answers to
API errors and marks some right answers wrong (a minus written as an en dash, a denial in bold); the phase 5 doc
gives the scores that change under the fixed checker. The final runs used the code as committed with them.

In each folder:

| File | Holds |
|---|---|
| `report.md` | the tables: scores with 95% bootstrap intervals, citations, turns, latency, errors, paired differences |
| `scores.csv` | one row per question and system, every mark |
| `answers.jsonl` | each hosted answer in full, its error text if any, and every tool call with its arguments |
| `questions.jsonl` | the questions with what the generator knows to be the answer |
| `metrics.json` | the summary and the run's settings |

In the two final runs, `api_requests` and `rate_limited` in `metrics.json` count the hybrid half only: the client was
counted per retrieval mode until after these runs. The offline agent's answers are not saved; it is deterministic,
so the command below gives them again.

To run them again (a free key from build.nvidia.com in `.env` as `NVIDIA_API_KEY`):

```bash
flab agent-eval --agent openai --retrieval bm25,hybrid --per-type 10 --seed 2 --workers 3 --out eval/agent_hosted/nemotron-3-ultra
```

```bash
FLAB_OPENAI_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b flab agent-eval --agent openai --retrieval bm25,hybrid --per-type 10 --seed 2 --workers 4 --out eval/agent_hosted/nemotron-3.5-lightning
```

```bash
flab agent-eval --per-type 10 --seed 2 --retrieval bm25,hybrid --out eval/agent_hosted/offline
```

A hosted run saves each answer to `progress.jsonl` as it arrives; run the same command again after a stop and it
resumes (`--fresh` starts over).
