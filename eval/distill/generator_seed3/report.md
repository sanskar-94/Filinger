# Phase 3 evaluation

|                              |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:-----------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                        | 400 |                 1 |                   1 |             0.86   |                       1 |               0.5408 |                 0.8994 |                1     |                 1      |          0.0002 |          0.0003 |                 0 |
| student-generator-1000-seed3 | 400 |                 1 |                   1 |             0.7642 |                       1 |               0.5857 |                 0.4981 |                0.973 |                 0.9975 |          3.1452 |          4.0119 |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| student-generator-1000-seed3 vs rules | -0.0958 | [-0.1353, 0.0131] | 0.902 |
