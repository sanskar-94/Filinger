# Phase 3 evaluation

|                                       |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:--------------------------------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules                                 | 100 |                 1 |                   1 |             0.8662 |                  1      |               0.4647 |                      1 |                    1 |                 1      |          0.0002 |          0.0003 |                 0 |
| api:nvidia/nemotron-3-ultra-550b-a55b | 100 |                 1 |                   1 |             0.8897 |                  0.9882 |               0.5704 |                      1 |                    1 |                 0.9325 |         52.6511 |        179.538  |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| api:nvidia/nemotron-3-ultra-550b-a55b vs rules | 0.0235 | [-0.0043, 0.0505] | 0.054 |
