# Phase 3 evaluation

|                              |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:-----------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                        | 400 |             1     |               1     |             0.86   |                  1      |               0.5408 |                 0.8994 |               1      |                 1      |          0.0002 |          0.0003 |                 0 |
| student-generator-1000-seed1 | 400 |             0.985 |               0.985 |             0.8069 |                  0.9973 |               0.5775 |                 0.9241 |               0.7288 |                 0.9949 |          3.0155 |          3.3629 |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| student-generator-1000-seed1 vs rules | -0.0531 | [-0.1757, 0.0646] | 0.868 |
