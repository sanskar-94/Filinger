# Phase 6: Experiment and dashboard

Two versions of the analyst experience run side by side inside the app, and the readout decides which one
ships, with a confidence interval, a power analysis and guardrails.

## Design (`configs/experiment.json`)

| Item | Choice |
|---|---|
| Variants | A (champion): retrieval with MMR diversification and the production adapter. B (challenger): plain BM25 ranking and the candidate adapter (`filings-extractor-b` on the same vLLM server) |
| Unit of randomisation | user (`flab_uid` cookie), hashed with the experiment salt: sticky, stateless, independent across experiments |
| Primary metric | thumbs up rate among rated answers |
| Guardrails | p95 latency may not rise more than 20%; citation validity may not fall more than 1 point |
| Planned MDE | 4 points on a 60% baseline, alpha 0.05, power 0.8 |

## Before the test: power (`experiment/power.py`, `flab ab-plan`)

The textbook two proportion formula gives the rated answers needed per arm if every answer were independent.
Answers from the same user are correlated, so the plan multiplies by the design effect 1 + (m - 1) * ICC,
then converts rated answers to exposures (rating rate) and users (answers per user). The sample size
function matches statsmodels to the unit (tests).

## After the test: the readout (`experiment/analysis.py`, `flab ab-analyze`)

1. Sample ratio mismatch: chi square on users per variant against the configured weights. A mismatch stops
   the analysis, because it means assignment or logging is broken.
2. Primary metric: Wilson interval per arm; Newcombe hybrid score interval and a z test for the difference
   (they assume independent answers); a user clustered delta method interval for the difference and the
   relative lift, which is the one the decision uses; the design effect actually observed.
3. Always valid inference: the mixture sequential probability ratio test (mSPRT) gives a p value that stays
   valid however often the dashboard is checked; a test confirms it holds the false positive rate under
   continuous peeking where a repeatedly checked z test does not.
4. Guardrails: p95 latency change with a user cluster bootstrap interval, citation validity, cost.
5. Achieved power at the planned MDE, independence and clustered, so an inconclusive result is read as
   "underpowered" or "no effect" correctly.
6. Decision rule: do not ship on SRM or a breached guardrail; ship when the clustered interval is above zero;
   keep control when it is below; otherwise inconclusive.

Every method is checked against a published value or an independent computation: Wilson against Newcombe's
(1998) worked example, Newcombe's interval against the same paper, the delta method against a cluster
bootstrap, the sample size against statsmodels.

## Traffic (`experiment/simulate.py`, `flab ab-simulate`)

No real analysts use this deployment yet, so traffic is simulated, but only the human part: every request
goes through the real `AgentService` (assignment, retrieval, citations, latency, logging), so guardrail
metrics are measured, not invented. Ratings follow logit(p) = logit(rate of the variant) + a per user
effect, which plants a known lift and realistic within user correlation. The readout is then checked against
the planted truth.

## Result

Three periods of 14 days, 9,000 simulated analysts; the test stopped when both arms reached the configured
maximum of 8,000 rated answers, and everything after that moment is excluded from the readout. Planted lift:
+4.2, +3.9 and +3.4 points in the three periods.

| | After period 1 (planned sample) | After period 2 | Final, at the maximum sample |
|---|---|---|---|
| Thumbs up rate A / B | 60.2% / 62.3% | 60.1% / 62.7% | 59.8% / 62.5% |
| Lift, user clustered 95% CI | +2.09 [-0.75, +4.92] | +2.60 [+0.64, +4.57] | +2.64 [+1.03, +4.25] |
| Fixed sample p (clustered) | 0.149 | 0.009, not valid after peeking | 0.001, not valid after peeking |
| Always valid p | 0.207 | 0.155 | 0.013 |
| Always valid confidence sequence | [-1.70, +6.42] | [-0.41, +4.53] | [+0.39, +4.52] |
| Decision | continue | continue | ship variant B |

How to read it:
1. At the planned sample the interval covered both zero and the true effect. With 80% power at the planned
   MDE, missing a real effect one time in five is expected.
2. After the first look, only the always valid test can decide. After period 2 the fixed sample test already
   said "significant" (p = 0.009), which is exactly the optional stopping error the sequential test exists to
   prevent; the confidence sequence still included zero, so the readout said keep collecting.
3. The always valid p first fell below 0.05 after 11,664 rated answers, during period 3, and at the stopping
   point the confidence sequence [+0.39, +4.52] excludes zero and contains the planted lift (about 3.9).
4. Guardrails are clean: citation validity unchanged, and B is reliably about 3.8% faster at p95 because
   plain ranking skips the diversification step. Sample ratio check p = 0.83.

Artifacts: `artifacts/phase6/period1/readout.md`, `artifacts/phase6/period2/readout.md`,
`artifacts/phase6/readout.md`, and the BI tables in `dashboards/data/`, which use exactly the data up to the
stopping point.

## Dashboard

The BI star schema (`flab bi-export`) feeds Power BI (`dashboards/powerbi/`, DAX measures included) and
Tableau (`dashboards/tableau/`, calculated fields included). Publishing to a Power BI or Tableau workspace
needs that account; the same dashboard is published as a private web dashboard built from the same
exported files, with an A/B test tab (lift, every metric's interval against zero, cumulative rate by arm,
the sequential evidence path, the arms table) and a reaction model tab (AUC by feature set, monthly IC,
abnormal return by prediction quintile, the tradable backtest and its cost sensitivity).
