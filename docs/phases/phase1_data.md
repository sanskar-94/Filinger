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
| `flab prices --index ... --source yahoo` or `--source csv --path vendor/` | split and dividend adjusted daily closes for every ticker in the index plus the market proxy, with a coverage report; refuses to continue without the market proxy. Stooq, the first free source, now answers scripts with a browser check and finds nothing |
| `flab download --index ... --raw ... --compress` | raw filings, gzip compressed; an 8-K is fetched as its full submission and pruned to the 8-K and its EX-99 exhibits |
| `flab facts --index ...` or `--bulk companyfacts.zip` | XBRL fundamentals per company from the API or the nightly bulk archive |
| `flab spark-features --lake s3://...` | the gold feature table in Spark (`groupBy(cik).applyInPandas` for text, `cogroup` for point in time fundamentals, native expressions for the rest) |

The event study also records CAR[+1,+2], the part of the reaction a trader can capture by entering at the
first close after the filing; the backtest in Phase 2 trades that, not CAR[0,+2].

## The press release exhibit

An earnings 8-K says almost nothing itself. Item 2.02 is one sentence furnishing a press release as Exhibit 99.1,
and the results, the guidance and management's commentary are all in that exhibit, a separate document of the
submission. The first version downloaded and parsed only the primary document, so on real filings the earnings
section would have held that one sentence and the guidance features, the only group that survives correction in
Phase 2, would have been empty. The synthetic world hid this because its generator writes the release into the
8-K itself.

Now `flab download` fetches the whole submission for an 8-K and keeps the 8-K and its readable EX-99 exhibits
(graphics, XBRL and other exhibits are most of the bytes and nothing reads them). `process_document` appends the
release to the Item 2.02 section, after the item's own sentence so existing paragraph ids hold, or stores it as
`8k_press_release` when the 8-K has no Item 2.02. `tests/test_exhibits.py` builds a submission as EDGAR ships
one (SGML wrapper, cover 8-K, EX-99.1, a uuencoded graphic, an XBRL schema) and checks all of it, including that
both runners read a compressed raw file.

## Run on real EDGAR

On 17 November 2026 `bash scripts/real_run.sh` ran every stage on real filings: 100 of today's large companies
(`configs/universe_large_cap.txt`), 2015 to 2024, on an 8 GB M2 in 70 minutes, 56 of them downloading under the
SEC's rate limit ([summary](../../eval/real/run_summary.json), [quality gates](../../eval/real/dq.md), the model in
[phase 2](phase2_baseline.md#result-on-real-filings-100-companies-2015-to-2024-12216-out-of-fold-events)).

| Stage | Result |
|---|---|
| index | 18,784 filings: 14,870 8-Ks, 2,936 10-Qs, 978 10-Ks, with acceptance times |
| download | all 18,784, 1.1 GB compressed (8-Ks with their press release exhibits) |
| prices | 101 tickers with the market proxy, every filing covered |
| fundamentals | 142,868 XBRL facts for all 100 companies |
| lake | 43,541 sections, 1,403,426 paragraphs, 18,780 events (475 MB) |
| quality gates | 20 pass, 2 warn, none fail |

Sections found: management's discussion in 98.0% of 10-Ks and 10-Qs, risk factors in 94.9% of 10-Ks, the results
section in 99.0% of earnings 8-Ks. The two warnings: 7 empty sections of 43,541, and features of sections an 8-K
does not have (risk factors, management's discussion), which are empty for most rows because 79% of the filings
are 8-Ks. Earnings 8-Ks move prices about twice as much as other 8-Ks (mean absolute 3 day abnormal return 4.2%
against 1.9%; 10-Qs 2.6%, 10-Ks 2.2%).

A 20 company pilot (2019 to 2024, `configs/universe_pilot.txt`) ran first and found five things the synthetic
world could not show, each fixed and tested before the full run:

1. A contact line given as an email alone. The SEC asks for a name and an email; the client now declares an email
   given on its own as `Filinger <email>`.
2. A ticker that moved. In July 2026 XOM passed to ExxonMobil Holdings Corp, a new registrant with no filings
   before then, so today's ticker list dropped Exxon's history. A universe file can now pin `XOM=34088`, and the
   index names any company with no filings in the window instead of leaving it out silently.
3. A network outage. For some minutes the machine could not resolve www.sec.gov and 849 of 2,132 downloads failed
   at once. The client now retries timeouts, resets and DNS failures with backoff, and `flab download` makes up to
   three passes over what failed.
4. Disk. The HTTP cache kept EDGAR's JSON uncompressed; it is now stored gzip compressed (65 MB to 8 MB).
5. Real layouts the section parser missed. P&G's 8-Ks run the item header into its text on one line; ExxonMobil's
   put two headers on one line before the text they share; Bank of America's 10-K puts the company's name between
   "Item 7." and its title; JPMorgan's reports carry no item headers at all and name them only in a cross reference
   index. On the pilot, management's discussion went from 94.0% to 99.2% of periodic reports and the results
   section from 90.3% to 99.6% of earnings 8-Ks.

That universe is today's large companies, so it is survivorship biased: a company that failed or was bought
between 2015 and 2024 is not in it. An earlier check, on 15 November 2026, built the index, prices, events and
XBRL facts for Apple, Microsoft and Nvidia alone (190 events; Nvidia's release of 24 May 2023, accepted after the
close, lands on 25 May with a day 0 return of plus 24.4%, as it should).

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

Real EDGAR on one machine, every stage in one resumable command:

```bash
export SEC_USER_AGENT="Your Org you@example.com"
bash scripts/real_run.sh
```

Real EDGAR, then Spark:

```bash
export SEC_USER_AGENT="Your Org you@example.com"
flab index --tickers universe.txt --start 2014-01-01 --end 2024-12-31 --out data/lake/bronze/index.parquet
flab download --index data/lake/bronze/index.parquet --raw data/lake/bronze/raw --compress
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

Measured on the real run: the 16,533 filings the pilot had not already fetched took 56 minutes to download with
four workers (about 4.9 filings a second), and all 18,784 take 1.1 GB stored compressed, about 60 KB a filing;
the local runner then parsed them into 1.4 million paragraphs in about 8 minutes on four processes. Scaled by
filings, 330k would be about 19 GB compressed and 19 hours of downloading at that pace.
