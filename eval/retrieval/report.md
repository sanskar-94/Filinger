# Retrieval evaluation (synthetic world)

57,486 paragraphs (25,751 distinct texts) from 60 companies; 182 distinct queries, 1,012 query and company pairs. Dense model: `BAAI/bge-small-en-v1.5` (384 dimensions). Every search is scoped to one company, as the agent's searches are. Scores count distinct texts. Intervals are 95%, from a bootstrap over whole queries.

## Keyword style queries (words taken from the filing sentence)

| System | hit@1 | hit@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25 | 0.979 | 0.992 [0.975, 1.000] | 0.858 | 0.982 | 0.887 [0.785, 0.969] |
| dense | 0.988 | 0.988 [0.967, 1.000] | 0.852 | 0.988 | 0.884 [0.786, 0.967] |
| hybrid | 0.988 | 0.992 [0.975, 1.000] | 0.856 | 0.989 | 0.888 [0.790, 0.969] |

| Paired difference | hit@5 | nDCG@10 | P(nDCG difference <= 0) |
|---|---|---|---|
| hybrid minus bm25 | +0.000 [+0.000, +0.000] | +0.002 [+0.000, +0.005] | 0.371 |
| hybrid minus dense | +0.004 [+0.000, +0.013] | +0.005 [+0.000, +0.014] | 0.124 |

## Paraphrased queries (the same need in other words)

| System | hit@1 | hit@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25 | 0.088 | 0.165 [0.078, 0.263] | 0.156 | 0.118 | 0.118 [0.053, 0.192] |
| dense | 0.626 | 0.745 [0.627, 0.848] | 0.633 | 0.676 | 0.615 [0.502, 0.720] |
| hybrid | 0.624 | 0.745 [0.627, 0.848] | 0.632 | 0.674 | 0.613 [0.500, 0.718] |

| Paired difference | hit@5 | nDCG@10 | P(nDCG difference <= 0) |
|---|---|---|---|
| hybrid minus bm25 | +0.580 [+0.456, +0.694] | +0.495 [+0.382, +0.601] | 0.000 |
| hybrid minus dense | +0.000 [+0.000, +0.000] | -0.002 [-0.006, +0.000] | 1.000 |

## Exact figure lookups (generated from the paragraphs)

| System | hit@1 | hit@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] |
| dense | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] |
| hybrid | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 [1.000, 1.000] |

| Paired difference | hit@5 | nDCG@10 | P(nDCG difference <= 0) |
|---|---|---|---|
| hybrid minus bm25 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.000 |
| hybrid minus dense | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1.000 |

## All queries

| System | hit@1 | hit@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25 | 0.444 | 0.493 [0.404, 0.587] | 0.456 | 0.463 | 0.440 [0.360, 0.530] |
| dense | 0.771 | 0.843 [0.773, 0.907] | 0.743 | 0.801 | 0.739 [0.664, 0.808] |
| hybrid | 0.770 | 0.844 [0.774, 0.909] | 0.743 | 0.800 | 0.739 [0.664, 0.808] |

| Paired difference | hit@5 | nDCG@10 | P(nDCG difference <= 0) |
|---|---|---|---|
| hybrid minus bm25 | +0.351 [+0.258, +0.439] | +0.300 [+0.220, +0.377] | 0.000 |
| hybrid minus dense | +0.001 [+0.000, +0.003] | +0.000 [-0.003, +0.003] | 0.472 |

## Paraphrased queries by need (nDCG@10)

| Need | bm25 | dense | hybrid |
|---|---|---|---|
| buyback | 0.000 | 0.883 | 0.883 |
| capital_need | 0.000 | 0.333 | 0.333 |
| ceo_departure | 0.000 | 0.983 | 0.983 |
| cfo_appointed | 0.602 | 0.988 | 0.988 |
| class_action | 0.000 | 0.935 | 0.935 |
| covenant_breach | 0.000 | 0.870 | 0.870 |
| credit_facility | 0.340 | 0.736 | 0.736 |
| going_concern | 0.000 | 0.000 | 0.000 |
| guidance_lowered | 0.169 | 0.351 | 0.351 |
| guidance_maintained | 0.000 | 0.236 | 0.236 |
| guidance_raised | 0.094 | 0.291 | 0.291 |
| impairment | 0.008 | 0.343 | 0.343 |
| new_facility | 0.445 | 0.884 | 0.884 |
| patent_verdict | 0.333 | 0.969 | 0.969 |
| restructuring | 0.000 | 0.890 | 0.858 |
| sec_subpoena | 0.000 | 0.333 | 0.333 |
| supply_contract | 0.013 | 0.426 | 0.426 |

## Effect of collapsing repeated paragraphs (nDCG@10, in brackets without it)

| System | keyword style | paraphrased | exact figure |
|---|---|---|---|
| bm25 | 0.887 (was 0.883) | 0.118 (was 0.116) | 1.000 (was 1.000) |
| dense | 0.884 (was 0.878) | 0.615 (was 0.600) | 1.000 (was 0.999) |
| hybrid | 0.888 (was 0.886) | 0.613 (was 0.595) | 1.000 (was 1.000) |

## Effect of the coverage gate on fusion (nDCG@10)

| System | keyword style | paraphrased | exact figure |
|---|---|---|---|
| hybrid, no coverage gate | 0.897 | 0.551 | 1.000 |
| dense | 0.884 | 0.615 | 1.000 |
| hybrid | 0.888 | 0.613 | 1.000 |

## As the agent serves it (top 6 with the diversity step, hit@5)

| System | keyword style | paraphrased | exact figure |
|---|---|---|---|
| bm25 | 0.992 | 0.190 | 1.000 |
| dense | 0.988 | 0.745 | 1.000 |
| hybrid | 0.992 | 0.794 | 1.000 |

## Search time per query on this machine (milliseconds)

| System | p50 | p95 |
|---|---|---|
| bm25 | 4.7 | 16.3 |
| dense | 6.6 | 36.2 |
| hybrid | 10.7 | 39.9 |

Dense and hybrid times exclude encoding the query, which took a median of 7.7 ms (p95 14.3 ms) and is cached per query text.

Relevance is exact (the generator's own sentences). The paraphrases were written with those sentences in view, so read the gap between the query sets as the mechanism at work, not as an estimate for real filings.
