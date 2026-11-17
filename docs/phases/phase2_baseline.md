# Phase 2: Classical baseline and statistics

Question: does filing language carry information about the post filing reaction that price based features
and the type of filing do not already carry? And is the answer statistically defensible?

## Features (`filings_lab/features`)

| Group | Examples | Source |
|---|---|---|
| market | momentum (-250,-30), return (-5,-1), volatility (-60,-1), market model beta, market return (-20,-1), abnormal volume | prices, strictly before day 0 |
| text tone | lexicon shares (negative, positive with negation handling, uncertainty, litigious, constraining, modal), net tone, Fog index, numeric density, forward looking share | MD&A, earnings 8-K, risk factors, legal |
| text change | cosine and Jaccard similarity to the same section about a year earlier, log length change ("Lazy Prices") | MD&A, risk factors |
| text rules | regex guidance up, down, maintained; going concern language | earnings 8-K, MD&A |
| financial | log assets, leverage, ROA, cash ratio, accruals, revenue growth year over year | XBRL facts filed strictly before the filing date |
| meta | form type, 8-K item flags, month | index |

The same `company_text_features` function runs in pandas and inside Spark (`groupBy("cik").applyInPandas`).

## Validation (`filings_lab/modeling/cv.py`)

Purged walk forward: six time ordered test blocks after an initial 35% training period. A training row is
used only if its label window (event day to event day + 2 trading days) ended before the test block starts
and it started at least five days before it. Every fold calls `assert_no_leakage`. Early stopping uses the
latest 15% of each training window, purged the same way, and the model is then refit on the whole window.

## Statistics (`filings_lab/modeling/stats.py`)

| Question | Test |
|---|---|
| Is AUC above 0.5? | DeLong standard error and CI |
| Is the full model better than the null model on the same events? | Paired DeLong test, Diebold Mariano on per event log loss |
| Is the monthly IC positive given autocorrelation? | Newey West t statistic, IC information ratio |
| Is a tradable spread there? | Monthly top minus bottom quintile CAR, Newey West t and block bootstrap CI |
| Could the whole pipeline produce this by chance? | Permutation test: labels shuffled within month, full walk forward refit each time |
| Which feature groups matter after testing several? | Drop one group ablation, DeLong p values, Benjamini Hochberg q values |

All tests are checked against known answers in `tests/test_stats.py` (DeLong equals sklearn AUC and agrees
with the bootstrap; Newey West collapses to the ordinary t with zero lags; and so on).

## Result on the synthetic world (60 companies, 2012 to 2024, 4,175 out of fold events)

| Feature set | AUC [95% CI] | monthly IC (NW t) | quintile spread bps (NW t) |
|---|---|---|---|
| market and meta (null) | 0.500 [0.482, 0.517] | 0.054 (2.1) | 107 (2.9) |
| market and fundamentals | 0.512 [0.494, 0.529] | 0.020 (0.7) | -6 (-0.2) |
| text only | 0.569 [0.551, 0.586] | 0.142 (6.0) | 236 (5.9) |
| all, LightGBM | 0.569 [0.552, 0.587] | 0.080 (3.1) | 125 (3.0) |
| all, LightGBM tuned with purged k fold | 0.562 [0.545, 0.580] | 0.091 (4.6) | 157 (4.9) |
| all, logistic regression | 0.577 [0.559, 0.594] | 0.160 (7.0) | 257 (7.4) |

All versus null: DeLong AUC difference 0.070, p below 1e-9; Diebold Mariano DM = -4.7, p below 0.0001.
Permutation test, 20 full refits: p = 0.048, the smallest value 20 permutations can produce.
Ablation: only the guidance rule group is significant after BH correction (q below 1e-10).

## Tuning, calibration and a tradable backtest

1. Tuning: eight random configurations scored with purged k fold inside every training window, never on the
   test block. It did not beat the defaults (AUC difference -0.007, DeLong p = 0.052), so the defaults stay.
2. Calibration: the raw probabilities are already well calibrated (expected calibration error 0.012, Brier
   reliability 0.0002). Out of sample isotonic recalibration made them worse (ECE 0.042), so it is not used.
3. Backtest: monthly top minus bottom quintile, entering at the first close after the filing and exiting two
   trading days later, equal weight, 4x the per side cost per round:

| cost per side | net spread per month | Sharpe | Newey West t | months positive |
|---|---|---|---|---|
| 0 bps | +54.4 bps | 0.91 | 2.04 | 68% |
| 5 bps | +34.4 bps | 0.57 | 1.29 | 63% |
| 10 bps | +14.4 bps | 0.24 | 0.54 | 60% |
| 20 bps | -25.6 bps | -0.43 | -0.96 | 48% |

   The break even cost is 13.6 bps per side. The edge is thin because most of the planted reaction lands
   on day 0, before anyone can act on the filing; the informational spread on CAR[0,+2] is several times
   larger. That gap between "predictive" and "tradable" is the point of running the backtest at all.
4. A drift report (Phase 4) showed that log assets trends upward as companies grow; the feature is now the
   company's size percentile among filings in the same month, which is stationary by construction.

How to read this:
1. The synthetic world plants a linear signal in the text, so a regularised linear model beating LightGBM is
   the expected outcome; seeing it confirms the comparison is fair rather than tuned toward trees.
2. These numbers validate the machinery only. On real EDGAR data expect an AUC in the 0.52 to 0.55 range;
   anything above 0.65 should be treated as a leakage bug until proven otherwise.
3. Fundamentals add little here because the synthetic world does not price them; on real data they are a
   meaningful control.

Regenerate: `flab features && flab train --permutations 20 --tune` (writes `artifacts/phase2/report.md`,
`backtest_monthly.csv`, `backtest_cost_curve.csv`, `calibration.csv`). The SQL views in
`sql/02_tone_quintiles.sql` and `sql/03_monthly_ic.sql` reproduce the headline cuts in DuckDB.

## Result on real filings (100 companies, 2015 to 2024, 12,216 out of fold events)

`bash scripts/real_run.sh` on 100 of today's large companies: 18,784 real filings, the same features, folds and
tests as above, no setting changed for real data ([report](../../eval/real/phase2_report.md), the run itself in
[phase 1](phase1_data.md#run-on-real-edgar)).

| Feature set | AUC [95% CI] | monthly IC (NW t) | quintile spread bps (NW t) |
|---|---|---|---|
| market and meta (null) | 0.494 [0.484, 0.504] | 0.012 (1.2) | 19 (1.7) |
| market and fundamentals | 0.498 [0.488, 0.509] | 0.030 (2.6) | 22 (1.2) |
| text only | 0.504 [0.494, 0.514] | 0.024 (1.8) | 29 (1.7) |
| all, LightGBM | 0.506 [0.496, 0.516] | 0.035 (2.9) | 31 (2.1) |
| all, logistic regression | 0.514 [0.504, 0.524] | 0.035 (3.2) | 25 (1.7) |

All versus null: DeLong AUC difference +0.012 (se 0.005), p = 0.028; Diebold Mariano on log loss DM = -1.15,
p = 0.25. Permutation test, 20 full refits with labels shuffled within month: p = 0.048, again the smallest value
20 permutations can produce. Backtest at 10 bps per side: +12 bps a month gross, -28 bps net, Sharpe -1.21; break
even at 3.1 bps per side. Raw probabilities are close to calibrated (expected calibration error 0.021) and isotonic
recalibration again makes them worse.

How to read this:
1. Real filing text carries a weak signal about the next three days: a monthly rank correlation of 0.035 with a
   Newey West t near 3, and a paired AUC gain over the null model of about one point. That is below the 0.52 to
   0.55 that published work led us to expect, on a sample this size the intervals are about a point wide, and it
   is not tradable: costs above 3 bps per side erase it, and the reaction lands mostly on day 0.
2. The ablation says more than the headline. Dropping the market group raises the LightGBM AUC from 0.506 to
   0.519 (BH q = 0.018): price features add noise the trees fit, and no text group is individually significant.
   That is one exploratory cut among five, so it is a lead to test on new data, not a result.
3. The strongest single text feature is the net tone of the earnings release (|IC| 0.071), and the guidance
   raised indicator is among the twenty most used features. Fundamentals (leverage, revenue growth, accruals)
   lead the SHAP ranking, which is the control they were put in for.
4. The universe is today's large companies, so it is survivorship biased; and with 20 companies over 2019 to
   2024 (1,464 out of fold events, [pilot report](../../eval/real/pilot/phase2_report.md)) every interval spans
   0.5, which is the sample size argument for running all 100.
