# Phase 2 report: classical baseline

Out of fold results from purged walk forward validation. Each feature set uses identical folds.

| Feature set | n | AUC [95% DeLong CI] | p (AUC=0.5) | log loss (base) | IC mean | IC NW t | LS spread bps | LS NW t |
|---|---|---|---|---|---|---|---|---|
| market_meta | 12216 | 0.4940 [0.4838, 0.5043] | 0.2546 | 0.6944 (0.6931) | 0.0124 | 1.24 | 18.7 | 1.71 |
| market_fundamentals | 12216 | 0.4983 [0.4880, 0.5085] | 0.7383 | 0.6945 (0.6931) | 0.0299 | 2.63 | 21.9 | 1.22 |
| text_only | 12216 | 0.5041 [0.4939, 0.5143] | 0.4318 | 0.6940 (0.6931) | 0.0237 | 1.83 | 29.2 | 1.70 |
| all | 12216 | 0.5058 [0.4955, 0.5160] | 0.2691 | 0.6938 (0.6931) | 0.0346 | 2.89 | 31.1 | 2.10 |
| logistic_all | 12216 | 0.5137 [0.5035, 0.5240] | 0.0085 | 0.6983 (0.6931) | 0.0348 | 3.24 | 24.9 | 1.68 |

market_meta holds price based features plus form and 8-K item indicators; it is the null model that
says filing language adds nothing beyond what kind of filing it is and how the stock was trading.

## Does text add information beyond market features?

DeLong paired test, all vs market and meta: AUC difference 0.0117 (se 0.0053), p = 0.0283.
Diebold Mariano on per event log loss: DM = -1.15, p = 0.2516 (negative favours the full model).

Permutation test (20 refits with labels shuffled within month): observed AUC 0.5058, null mean 0.4941 (sd 0.0046), p = 0.048 (smallest attainable p with 20 permutations is 0.048).
## Cost aware backtest (all features, LightGBM)

Monthly top minus bottom quintile, entering at the day 0 close (car_tradable), 80 months, 10 bps per side: mean net -27.7 bps per month (gross 12.3), annualised Sharpe -1.21, Newey West t -3.27, hit rate 29%, max drawdown -2251 bps, break even cost 3.1 bps per side.

|   cost_bps_per_side |   mean_net_bps |   sharpe_annual |    nw_t |   hit_rate |
|--------------------:|---------------:|----------------:|--------:|-----------:|
|                   0 |         12.264 |           0.536 |   1.447 |      0.55  |
|                   5 |         -7.736 |          -0.338 |  -0.913 |      0.462 |
|                  10 |        -27.736 |          -1.211 |  -3.272 |      0.288 |
|                  20 |        -67.736 |          -2.958 |  -7.992 |      0.112 |
|                  40 |       -147.736 |          -6.451 | -17.43  |      0.038 |

## Calibration (folds after the first)

Expected calibration error 0.0206 raw, 0.0496 after out of sample isotonic recalibration. Brier 0.2505 raw, 0.2542 recalibrated (reliability 0.0007, resolution 0.0002, uncertainty 0.2500).


## Drop one group ablation (Benjamini Hochberg q values)

| dropped     |   auc_full |   auc_without |   delta_auc |   p_value |   q_value |
|:------------|-----------:|--------------:|------------:|----------:|----------:|
| market      |     0.5058 |        0.5187 |     -0.0129 |    0.0036 |    0.0178 |
| text_tone   |     0.5058 |        0.5105 |     -0.0047 |    0.2105 |    0.3509 |
| text_change |     0.5058 |        0.5108 |     -0.005  |    0.1165 |    0.2912 |
| text_rules  |     0.5058 |        0.5042 |      0.0016 |    0.6021 |    0.6021 |
| financial   |     0.5058 |        0.5016 |      0.0042 |    0.3586 |    0.4482 |

## Top features by mean |SHAP|

| feature           |     gain |   mean_abs_shap |
|:------------------|---------:|----------------:|
| fin_leverage      | 1770.79  |         0.05113 |
| fin_rev_growth    | 1648.24  |         0.03845 |
| abn_volume_5_1    | 1800.5   |         0.03682 |
| ret_5_1           | 1830.14  |         0.03349 |
| fin_accruals      | 1841.21  |         0.0332  |
| mkt_ret_20_1      | 1809.05  |         0.02924 |
| fin_size_rank     | 1314.77  |         0.02653 |
| fin_cash_ratio    | 1736.12  |         0.02638 |
| fin_roa           | 1681     |         0.02364 |
| month             |  778.64  |         0.02296 |
| mom_250_30        | 1457.64  |         0.02    |
| vol_60_1          | 1327.38  |         0.01962 |
| ear_guidance_up   |  331.194 |         0.01906 |
| beta              | 1328.39  |         0.01825 |
| ear_lex_net_tone  |  557.968 |         0.01562 |
| ear_lex_litigious |  589.495 |         0.01357 |
| mdna_fog          |  502.256 |         0.01181 |
| ear_lex_negative  |  330.429 |         0.00921 |
| mdna_num_density  |  324.931 |         0.00902 |
| ear_lex_positive  |  264.145 |         0.00826 |
