# Phase 2 report: classical baseline

Out of fold results from purged walk forward validation. Each feature set uses identical folds.

| Feature set | n | AUC [95% DeLong CI] | p (AUC=0.5) | log loss (base) | IC mean | IC NW t | LS spread bps | LS NW t |
|---|---|---|---|---|---|---|---|---|
| market_meta | 1464 | 0.4986 [0.4690, 0.5283] | 0.9285 | 0.6983 (0.6930) | 0.0198 | 0.53 | 22.8 | 0.41 |
| market_fundamentals | 1464 | 0.4911 [0.4615, 0.5207] | 0.5546 | 0.6973 (0.6930) | 0.0193 | 0.52 | 3.6 | 0.08 |
| text_only | 1464 | 0.5061 [0.4764, 0.5357] | 0.6887 | 0.6943 (0.6930) | 0.0229 | 0.92 | -9.0 | -0.29 |
| all | 1464 | 0.4845 [0.4549, 0.5140] | 0.3031 | 0.7001 (0.6930) | -0.0088 | -0.23 | -7.6 | -0.18 |
| logistic_all | 1464 | 0.5292 [0.4996, 0.5587] | 0.0532 | 0.7129 (0.6930) | 0.0265 | 0.82 | 21.5 | 0.58 |

market_meta holds price based features plus form and 8-K item indicators; it is the null model that
says filing language adds nothing beyond what kind of filing it is and how the stock was trading.

## Does text add information beyond market features?

DeLong paired test, all vs market and meta: AUC difference -0.0142 (se 0.0157), p = 0.3651.
Diebold Mariano on per event log loss: DM = 0.53, p = 0.5978 (negative favours the full model).

Permutation test (20 refits with labels shuffled within month): observed AUC 0.4845, null mean 0.4946 (sd 0.0170), p = 0.714 (smallest attainable p with 20 permutations is 0.048).
## Cost aware backtest (all features, LightGBM)

Monthly top minus bottom quintile, entering at the day 0 close (car_tradable), 47 months, 10 bps per side: mean net -32.1 bps per month (gross 7.9), annualised Sharpe -0.91, Newey West t -1.60, hit rate 43%, max drawdown -2202 bps, break even cost 2.0 bps per side.

|   cost_bps_per_side |   mean_net_bps |   sharpe_annual |   nw_t |   hit_rate |
|--------------------:|---------------:|----------------:|-------:|-----------:|
|                   0 |          7.948 |           0.226 |  0.398 |      0.553 |
|                   5 |        -12.052 |          -0.342 | -0.603 |      0.489 |
|                  10 |        -32.052 |          -0.911 | -1.605 |      0.426 |
|                  20 |        -72.052 |          -2.047 | -3.607 |      0.234 |
|                  40 |       -152.052 |          -4.32  | -7.612 |      0.085 |

## Calibration (folds after the first)

Expected calibration error 0.0467 raw, 0.0380 after out of sample isotonic recalibration. Brier 0.2527 raw, 0.2514 recalibrated (reliability 0.0038, resolution 0.0015, uncertainty 0.2497).


## Drop one group ablation (Benjamini Hochberg q values)

| dropped     |   auc_full |   auc_without |   delta_auc |   p_value |   q_value |
|:------------|-----------:|--------------:|------------:|----------:|----------:|
| market      |     0.4845 |        0.5065 |     -0.022  |    0.1064 |    0.371  |
| text_tone   |     0.4845 |        0.4833 |      0.0012 |    0.8968 |    0.8968 |
| text_change |     0.4845 |        0.4942 |     -0.0097 |    0.1888 |    0.371  |
| text_rules  |     0.4845 |        0.4885 |     -0.004  |    0.5997 |    0.7496 |
| financial   |     0.4845 |        0.5002 |     -0.0158 |    0.2226 |    0.371  |

## Top features by mean |SHAP|

| feature              |     gain |   mean_abs_shap |
|:---------------------|---------:|----------------:|
| mkt_ret_20_1         | 1172.16  |         0.11306 |
| ret_5_1              | 1082.63  |         0.11185 |
| fin_size_rank        | 1293.99  |         0.10658 |
| beta                 | 1301.73  |         0.10622 |
| mom_250_30           | 1391.31  |         0.10377 |
| fin_leverage         | 1130.33  |         0.10023 |
| abn_volume_5_1       | 1259.09  |         0.08274 |
| fin_cash_ratio       |  729.356 |         0.08042 |
| fin_accruals         |  996.148 |         0.07659 |
| fin_rev_growth       |  854.449 |         0.07299 |
| fin_roa              |  759.911 |         0.07252 |
| month                |  483.843 |         0.04425 |
| vol_60_1             |  832.312 |         0.04108 |
| legal_lex_litigious  |  105.201 |         0.02637 |
| ear_lex_strong_modal |  191.378 |         0.02593 |
| ear_num_density      |  162.37  |         0.02573 |
| ear_lex_constraining |  172.571 |         0.02447 |
| mdna_lex_positive    |  150.659 |         0.02131 |
| ear_lex_litigious    |  111.396 |         0.01969 |
| ear_lex_uncertainty  |  125.164 |         0.01924 |
