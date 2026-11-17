# Phase 3 evaluation

|                                           |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:------------------------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                                     | 100 |              1    |                1    |             0.8662 |                  1      |               0.4647 |                 1      |               1      |                 1      |          0.0002 |          0.0003 |                 0 |
| api:nvidia/nemotron-3.5-lightning-30b-a3b | 100 |              0.46 |                0.46 |             0.4883 |                  0.6029 |               0.4029 |                 0.6301 |               0.3172 |                 0.9898 |         71.4449 |        227.951  |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| api:nvidia/nemotron-3.5-lightning-30b-a3b vs rules | -0.3779 | [-0.4556, -0.2336] | 1.000 |
