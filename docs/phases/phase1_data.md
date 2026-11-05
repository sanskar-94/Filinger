# Phase 1: Data at scale

Goal: ten plus years of 10-K, 10-Q and 8-K filings for a few thousand companies, cleaned, split into
sections and paragraphs, aligned to price reactions, and stored as partitioned Parquet on S3.

## Architecture

```
              EDGAR (rate limited, 8 req/s)                     price vendor
                     |                                               |
   flab index  ------+--> bronze/index.parquet                       |
   (submissions API or bulk submissions.zip)                         |
   flab download ----+--> bronze/raw/{cik}/{accession}.txt           +--> bronze/prices.parquet
                     |                                               |
                     +--------------------+--------------------------+
                                          |
                     PySpark job (Databricks or EMR)  filings_lab/pipeline/spark_job.py
                       binaryFile reader -> mapInPandas(clean, sections, paragraphs)
                       prices.groupBy(ticker).cogroup(events) -> applyInPandas(event study)
                       window functions -> labels
                                          |
          silver/sections     silver/paragraphs     silver/prices     gold/events
          (form, year)        (form, year)                            (year)
```

Why downloading is a separate stage: SEC fair access allows 10 requests per second per client. If every
Spark executor fetched from EDGAR the cluster would exceed that within seconds and get blocked. Download
once, politely, resumably; then let Spark use every core on files already in S3.

## Tables

| Table | Grain | Key columns |
|---|---|---|
| `bronze/index` | filing | cik, ticker, form, accession, date_filed, accepted_at (ET), primary_document, items |
| `silver/sections` | filing x section | accession, section (`mdna`, `risk_factors`, `legal`, `8k_*`), text, n_words |
| `silver/paragraphs` | paragraph | paragraph_id (stable hash), accession, section, para_idx, text |
| `gold/events` | filing | event_date, car (market adjusted CAR[0,+2]), car_mm (market model), beta, pre event features, label_up, label_tercile |

## Correctness guarantees, each covered by a test

1. Event day follows the acceptance time rule; date only timestamps are treated conservatively (next day). `tests/test_events.py`
2. Pre event features are invariant to anything that happens on or after day 0. `tests/test_events.py`
3. XBRL facts are joined point in time on the `filed` date. `tests/test_edgar.py`
4. The table of contents never wins section extraction; 10-Q Part I and Part II items are disambiguated. `tests/test_text.py`
5. The Spark job and the single machine runner produce identical tables (same paragraph ids, same CARs to 1e-12, same labels, same event dates) at the same paths on a machine in any time zone (the Spark session is pinned to UTC), and the Spark feature build equals the pandas one column by column. The cluster chain also runs on its own output: Spark lake, then Spark features and the quality gate on that lake, as the Databricks bundle runs them (this caught the Spark job writing no index table and prices under another name). `tests/test_spark_parity.py`
6. The same pipeline, feature build and quality gate produce identical tables on S3 as on local disk. `tests/test_s3_storage.py`
7. Data quality gates pass before anything is trained (below). `tests/test_quality.py`

## Real inputs beyond the filings

| Command | Builds |
|---|---|
| `flab prices --index ... --source stooq` or `--source csv --path vendor/` | daily closes for every ticker in the index plus the market proxy, with a coverage report; refuses to continue without the market proxy |
| `flab facts --index ...` or `--bulk companyfacts.zip` | XBRL fundamentals per company from the API or the nightly bulk archive |
| `flab spark-features --lake s3://...` | the gold feature table in Spark (`groupBy(cik).applyInPandas` for text, `cogroup` for point in time fundamentals, native expressions for the rest) |

The event study also records CAR[+1,+2], the part of the reaction a trader can capture by entering at the
first close after the filing; the backtest in Phase 2 trades that, not CAR[0,+2].

## Data quality gates (`flab dq`)

Twenty two checks with explicit thresholds in one table (`pipeline/quality.py`): required fields and unique
accession numbers in the index, acceptance timestamps that parse, section extraction rates per form, unique
non blank paragraphs, one positive price per ticker and day, event coverage, finite and plausible CARs, no
reaction dated before acceptance, label balance, one feature row per filing, no infinite values, null rates,
and a leakage canary that fails when the lower 95% bound of any single feature's rank correlation with CAR
exceeds 0.30 (a feature that predictive is a timing bug until proven otherwise). A failure exits non zero
and stops the Databricks job and the model gate. All 22 pass on the synthetic lake; tests inject a
look ahead reaction, an implausible CAR, duplicate prices and paragraphs, and a feature built from post event
prices, and each is caught.

## Running it

Synthetic world, one machine:

```bash
flab synth --out data/synthetic --companies 60
flab local --index data/synthetic/index.parquet --prices data/synthetic/prices.parquet --lake data/lake
flab sql sql/01_coverage.sql
```

Real EDGAR, then Spark:

```bash
export SEC_USER_AGENT="Your Org you@example.com"
flab index --tickers universe.txt --start 2014-01-01 --end 2024-12-31 --out data/lake/bronze/index.parquet
flab download --index data/lake/bronze/index.parquet --raw data/lake/bronze/raw
spark-submit -m filings_lab.pipeline.spark_job --index ... --raw ... --prices ... --lake s3://bucket/lake
```

Databricks: `infra/databricks/databricks.yml` (asset bundle: download, transform, features tasks).
EMR: `infra/emr/run_pipeline.sh`. S3 bucket and IAM: `infra/aws/main.tf`.

The single machine path runs on S3 too: every stage takes `s3://` paths (`storage.py`: boto3 for raw filings,
PyArrow's native S3 filesystem for the Parquet partitions, `AWS_ENDPOINT_URL` for S3 compatible stores).
`tests/test_s3_storage.py` starts a moto server, uploads the raw filings, runs the pipeline, the feature build
and the data quality gate against the bucket over HTTP, and requires every table to equal a local run.

## Scale: measured, then estimated

`scripts/bench_pipeline.py` runs both engines on a world five times the default size (300 companies, 2012 to
2024) on four cores and checks that their outputs agree:

| Engine | Filings | Seconds | Filings per second | Paragraphs per second | Raw MB per second |
|---|---:|---:|---:|---:|---:|
| local runner, 4 processes | 34,041 | 76.5 | 445 | 3,788 | 1.86 |
| Spark local[4], session start included | 34,041 | 72.4 | 470 | 4,003 | 1.97 |

Both wrote 289,765 paragraphs, and the row counts of every silver and gold table are identical; the 17 data
quality checks pass on the lake Spark wrote. Spark's advantage is not speed on one machine (parsing is
Python either way) but that the same job spreads over a cluster.

Bytes are the honest basis for real data, since real filings are far longer than synthetic ones: at 0.49 MB
of raw filing per core second, each terabyte needs about 560 core hours, about 9 hours on a 64 core EMR or
Databricks cluster before I/O effects (conservative, because exhibits and XBRL are dropped early). For
3,000 companies over 11 years, roughly 330k filings, the download at 8 requests per second takes about 11.5
hours once (resumable, cached), and bulk `submissions.zip` replaces 3,000 index requests with one download.
