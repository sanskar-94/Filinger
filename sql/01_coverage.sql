-- Coverage of the lake: filings, parsed sections and labelled events by form and year.
-- Run with: flab sql sql/01_coverage.sql --lake data/lake
WITH idx AS (
  SELECT form, year(CAST(date_filed AS DATE)) AS yr, count(*) AS filings
  FROM read_parquet('{lake}/bronze_index.parquet') GROUP BY 1, 2
), sec AS (
  SELECT form, year AS yr, count(DISTINCT accession) AS parsed, avg(n_words) AS avg_words
  FROM read_parquet('{lake}/silver/sections/**/*.parquet', hive_partitioning = true) GROUP BY 1, 2
), ev AS (
  SELECT form, year(event_date) AS yr, count(*) AS events, avg(car) AS mean_car, stddev(car) AS sd_car
  FROM read_parquet('{lake}/gold/events/**/*.parquet', hive_partitioning = true) GROUP BY 1, 2
)
SELECT idx.form, idx.yr, filings, parsed, round(100.0 * parsed / filings, 1) AS parse_rate_pct,
       events, round(avg_words) AS avg_words, round(mean_car * 1e4, 1) AS mean_car_bps, round(sd_car * 1e4) AS sd_car_bps
FROM idx LEFT JOIN sec USING (form, yr) LEFT JOIN ev USING (form, yr)
ORDER BY idx.form, idx.yr;
