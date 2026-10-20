-- Average post filing abnormal return by MD&A net tone quintile, per form.
-- A monotone pattern across quintiles is the classic dictionary result (Loughran and McDonald 2011).
WITH f AS (
  SELECT form, car, mdna_lex_net_tone AS tone
  FROM read_parquet('{lake}/gold/features/**/*.parquet', hive_partitioning = true)
  WHERE mdna_lex_net_tone IS NOT NULL
), q AS (
  SELECT *, ntile(5) OVER (PARTITION BY form ORDER BY tone) AS tone_q FROM f
)
SELECT form, tone_q, count(*) AS n, round(avg(tone), 3) AS avg_tone,
       round(avg(car) * 1e4, 1) AS mean_car_bps,
       round(stddev(car) / sqrt(count(*)) * 1e4, 1) AS se_bps
FROM q GROUP BY 1, 2 ORDER BY 1, 2;
