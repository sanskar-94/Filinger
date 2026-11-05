# Phase 3 evaluation

|                  |   n |   json_valid_rate |   schema_valid_rate |   extraction_score |   guidance_direction_f1 |   management_tone_f1 |   liquidity_concern_f1 |   litigation_risk_f1 |   quote_grounding_rate |   latency_p50_s |   latency_p95_s |   cost_per_1k_usd |
|:-----------------|----:|------------------:|--------------------:|-------------------:|------------------------:|---------------------:|-----------------------:|---------------------:|-----------------------:|----------------:|----------------:|------------------:|
| rules            | 400 |            1      |                   1 |             0.86   |                       1 |               0.5408 |                 0.8994 |                    1 |                      1 |          0.0002 |          0.0004 |                 0 |
| qwen3b-4bit-base | 400 |            0.3675 |                   0 |             0      |                       0 |               0      |                 0      |                    0 |                    nan |         10.6121 |         14.6111 |                 0 |
| qwen3b-4bit-lora | 400 |            1      |                   1 |             0.8541 |                       1 |               0.4164 |                 1      |                    1 |                      1 |          4.6678 |          7.1337 |                 0 |

## Paired bootstrap on extraction_score

| comparison | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| qwen3b-4bit-base vs rules | -0.8600 | [-0.8936, -0.7554] | 1.000 |
| qwen3b-4bit-lora vs rules | -0.0060 | [-0.0412, 0.0977] | 0.610 |
