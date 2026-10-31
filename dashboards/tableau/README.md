# Tableau build

Data source: connect to the text files in `dashboards/data/`; add `dim_company` and relate it to both
facts on `ticker` (relationships, not joins, so each fact keeps its own grain).

Calculated fields (fact_ab_exposures):

```
Thumbs Up Rate      SUM([thumbs_up]) / SUM([rated])
Rate A              SUM(IF [variant] = "A" THEN [thumbs_up] END) / SUM(IF [variant] = "A" THEN [rated] END)
Rate B              SUM(IF [variant] = "B" THEN [thumbs_up] END) / SUM(IF [variant] = "B" THEN [rated] END)
Lift (points)       [Rate B] - [Rate A]
Wilson Low          (([Thumbs Up Rate] + 1.959964^2/(2*SUM([rated]))) - 1.959964*SQRT([Thumbs Up Rate]*(1-[Thumbs Up Rate])/SUM([rated]) + 1.959964^2/(4*SUM([rated])^2))) / (1 + 1.959964^2/SUM([rated]))
Wilson High         (([Thumbs Up Rate] + 1.959964^2/(2*SUM([rated]))) + 1.959964*SQRT([Thumbs Up Rate]*(1-[Thumbs Up Rate])/SUM([rated]) + 1.959964^2/(4*SUM([rated])^2))) / (1 + 1.959964^2/SUM([rated]))
Latency p95 ms      PERCENTILE([latency_s], 0.95) * 1000
```

Calculated fields (fact_filing_events):

```
CAR bps             AVG([car]) * 10000
Correct Direction   ([pred] > 0.5) = ([car] > 0)
```

Sheets: rate by variant with Wilson band (gantt bar from Wilson Low to Wilson High plus a circle for the
rate), cumulative rate by day (from `agg_ab_daily`), summary table (`ab_summary`), IC by month with a zero
reference line, mean CAR by prediction quintile, sector by form heat map.

Dashboard: KPI row on top (decision, lift, users), experiment sheets in the first band, model sheets in the
second, a ticker filter applied to all sheets that use `dim_company`. Publish to Tableau Cloud or Tableau
Public with the CSVs as an extract.
