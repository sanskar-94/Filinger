-- Monthly information coefficient (rank correlation of out of sample prediction and realised CAR).
WITH p AS (
  SELECT date_trunc('month', event_date) AS month, pred, car
  FROM read_parquet('{artifacts}/phase2/oof_predictions.parquet')
), r AS (
  SELECT month, rank() OVER (PARTITION BY month ORDER BY pred) AS rp, rank() OVER (PARTITION BY month ORDER BY car) AS rc
  FROM p
)
SELECT month, count(*) AS n, round(corr(rp, rc), 4) AS ic FROM r GROUP BY 1 HAVING count(*) >= 10 ORDER BY 1;
