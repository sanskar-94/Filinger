# Power BI build

1. Get data, Text/CSV (or Folder) and load every file in `dashboards/data/`. In the cloud, point the same
   queries at the S3 export through Athena, or at the Databricks SQL warehouse over the gold tables.
2. Model view: relate `fact_ab_exposures[ticker]` and `fact_filing_events[ticker]` to `dim_company[ticker]`.
   Set `variant`, `date`, `form`, `sector` as text; `latency_s`, `car`, `pred` as decimal.
3. New measures: paste `measures.dax` (one measure per definition).
4. Publish to a workspace and schedule refresh after the nightly `flab bi-export`.

## Page 1: Experiment readout

| Visual | Fields |
|---|---|
| Card row | Decision, Lift (points), Lift (relative), Users, Rated |
| Clustered bar with error bars | variant on axis, Thumbs Up Rate, error bars from Wilson Low and Wilson High |
| Line | `agg_ab_daily[date]` on axis, `cum_thumbs_up_rate` by variant |
| Table | `ab_summary` (metric, method, estimate, ci_low, ci_high, p_value) |
| Guardrail cards | Latency p95 (ms) and Citation Valid Rate by variant |

Note on the page: the decision uses the user clustered interval from the readout. DAX intervals treat every
answer as independent and are shown for orientation only.

## Page 2: Model performance

| Visual | Fields |
|---|---|
| Line | `agg_model_monthly[month]`, `ic` with a zero reference line; Mean IC card |
| Column | `fact_filing_events[pred_quintile]` on axis, Mean CAR (bps) (should rise from 1 to 5) |
| Line | month, `long_short` |
| Matrix | sector by form, Mean CAR (bps), count of filings |

## Page 3: Filings explorer

Slicers for ticker, form and date; a table of filings with `car`, `pred` and a link column built from
accession; a scatter of `pred` against `car`.
