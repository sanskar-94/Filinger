# Phase 3 evaluation

|                              |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:-----------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                        | 400 |                 1 |                   1 |             0.86   |                       1 |               0.5408 |                 0.8994 |               1      |                 1      |          0.0002 |          0.0003 |                 0 |
| student-generator-1000-seed2 | 400 |                 1 |                   1 |             0.6478 |                       1 |               0.5992 |                 0.4981 |               0.4937 |                 0.9825 |          3.0138 |          3.3953 |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| student-generator-1000-seed2 vs rules | -0.2123 | [-0.2478, -0.1062] | 1.000 |
