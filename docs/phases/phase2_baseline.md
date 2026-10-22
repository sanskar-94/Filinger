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
| market and fundamentals | 0.511 [0.494, 0.529] | 0.010 (0.4) | -43 (-1.0) |
| text only | 0.569 [0.551, 0.586] | 0.142 (6.0) | 236 (5.9) |
| all, LightGBM | 0.564 [0.546, 0.581] | 0.082 (3.8) | 161 (4.9) |
| all, logistic regression | 0.577 [0.560, 0.594] | 0.160 (6.7) | 254 (7.1) |

All versus null: DeLong AUC difference 0.064, p < 0.0001; Diebold Mariano DM = -4.0, p = 0.0001.
Permutation test, 20 full refits: p = 0.048, the smallest value 20 permutations can produce.
Ablation: only the guidance rule group is significant after BH correction (q < 0.001).

How to read this:
1. The synthetic world plants a linear signal in the text, so a regularised linear model beating LightGBM is
   the expected outcome; seeing it confirms the comparison is fair rather than tuned toward trees.
2. Adding weak or irrelevant features to LightGBM costs a little AUC (text only beats all). On real data
   this argues for feature selection inside the training window, not on the full sample.
3. These numbers validate the machinery only. On real EDGAR data expect an AUC in the 0.52 to 0.55 range;
   anything above 0.65 should be treated as a leakage bug until proven otherwise.
4. Fundamentals add nothing here because the synthetic world does not price them; on real data they are a
   meaningful control.

Regenerate: `flab features && flab train --permutations 20` (writes `artifacts/phase2/report.md`).
The SQL views in `sql/02_tone_quintiles.sql` and `sql/03_monthly_ic.sql` reproduce the headline cuts in DuckDB.
