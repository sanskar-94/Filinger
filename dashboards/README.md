# Dashboards

`flab bi-export` writes a small star schema to `dashboards/data/` (CSV), regenerated from the lake, the
Phase 2 out of fold predictions and the experiment event log. Both BI tools read the same files, and the
summary table is written from the same readout object as `artifacts/phase6/readout.md`, so the numbers in
the dashboard and in the statistical readout cannot drift apart.

| Table | Grain | Used for |
|---|---|---|
| `fact_ab_exposures` | one answered question | A/B metrics, slicing by date, variant, company |
| `agg_ab_daily` | day x variant | trend lines, cumulative thumbs up rate |
| `ab_summary` | metric | difference, intervals, p values, guardrails, decision |
| `fact_filing_events` | one filing | reaction (CAR), model prediction, quintile |
| `agg_model_monthly` | month | IC and long short spread over time |
| `dim_company` | company | ticker, name, sector |

Relationships: `fact_ab_exposures[ticker]` and `fact_filing_events[ticker]` to `dim_company[ticker]`
(many to one). Date tables are optional; `date` and `event_date` are ISO dates.

1. Power BI: `powerbi/README.md` (data model, DAX measures in `powerbi/measures.dax`, page layout).
2. Tableau: `tableau/README.md` (data source, calculated fields, sheets and dashboard layout).
