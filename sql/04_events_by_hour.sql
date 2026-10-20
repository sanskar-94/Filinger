-- Why acceptance time matters: share of filings accepted after the 16:00 ET close, by form.
SELECT form,
       count(*) AS filings,
       round(100.0 * avg(CASE WHEN hour(timezone('America/New_York', CAST(accepted_at AS TIMESTAMPTZ))) >= 16 THEN 1 ELSE 0 END), 1) AS pct_after_close,
       round(100.0 * avg(CASE WHEN hour(timezone('America/New_York', CAST(accepted_at AS TIMESTAMPTZ))) < 9 THEN 1 ELSE 0 END), 1) AS pct_pre_open
FROM read_parquet('{lake}/bronze_index.parquet')
GROUP BY form ORDER BY form;
