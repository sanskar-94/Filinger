# Phase 3 evaluation

|                                       |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:--------------------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                                 | 100 |                 1 |                   1 |             0.8783 |                       1 |               0.5131 |                      1 |                    1 |                 1      |          0.0002 |          0.0005 |                 0 |
| api:nvidia/nemotron-3-ultra-550b-a55b | 100 |                 1 |                   1 |             0.8995 |                       1 |               0.5978 |                      1 |                    1 |                 0.9421 |         13.7386 |         27.1833 |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| api:nvidia/nemotron-3-ultra-550b-a55b vs rules | 0.0212 | [-0.0050, 0.0476] | 0.048 |
