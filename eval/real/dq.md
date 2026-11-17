# Data quality

20 passed, 2 warnings, 0 failed.

| status | table | check | observed | threshold | detail |
|---|---|---|---|---|---|
| WARN | silver_sections | no_empty_sections | 7 | 0 |  |
| WARN | gold_features | feature_null_rates | 53 | 0.6 | rf_log_words, rf_lex_negative, rf_lex_uncertainty, rf_lex_litigious, rf_lex_constraining, rf_fog, mdna_n_words, mdna_log_words |
| PASS | bronze_index | required_fields_present | 0.0 | 0.0 | share of rows missing cik, form, accession or date |
| PASS | bronze_index | accession_unique | 0 | 0 | duplicate accession numbers |
| PASS | bronze_index | accepted_at_parses | 1.0 | 0.98 |  |
| PASS | silver_sections | periodic_mdna_extracted | 0.980072 | 0.9 |  |
| PASS | silver_sections | tenk_risk_factors_extracted | 0.948875 | 0.8 |  |
| PASS | silver_sections | earnings_results_extracted | 0.989986 | 0.9 |  |
| PASS | silver_paragraphs | paragraph_id_unique | 0 | 0 | citations depend on it |
| PASS | silver_paragraphs | no_blank_paragraphs | 0 | 0 |  |
| PASS | silver_prices | one_price_per_day | 0 | 0 |  |
| PASS | silver_prices | positive_closes | 0 | 0 |  |
| PASS | silver_prices | no_extreme_daily_moves | 0 | 0.75 | unadjusted splits show up here |
| PASS | gold_events | event_coverage | 0.999787 | 0.6 | first year has no estimation window, so coverage below 100% is expected |
| PASS | gold_events | car_finite | 0 | 0 |  |
| PASS | gold_events | car_plausible | 0 | 1.0 |  |
| PASS | gold_events | event_not_before_acceptance | 0 | 0 | a reaction dated before the filing was public is look ahead |
| PASS | gold_events | label_balance | 0.5035 | 0.35 to 0.65 |  |
| PASS | gold_features | one_row_per_filing | 0 | 0 |  |
| PASS | gold_features | no_infinite_values | 0 | 0 |  |
| PASS | gold_features | label_window_after_event | True | True |  |
| PASS | gold_features | leakage_canary | 0.0406 | 0.3 | lower 95% bound of |IC|; strongest feature ear_lex_net_tone with |IC| 0.071 |
