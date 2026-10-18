# Research notes: Filings Intelligence Lab

This file records the decisions made before any code was written, and the evidence behind them.
Each phase in the build links back to a section here.

## 1. Data sources and their rules

### SEC EDGAR

| Need | Endpoint | Notes |
|------|----------|-------|
| Universe of filings per quarter | `https://www.sec.gov/Archives/edgar/full-index/{YYYY}/QTR{n}/master.idx` | Pipe delimited: CIK, company, form type, date filed, path. Cheapest way to enumerate ten plus years of 10-K, 10-Q and 8-K without hitting per company APIs. |
| Per company filing history | `https://data.sec.gov/submissions/CIK##########.json` | Ten digit zero padded CIK. Recent filings inline, older ones in `files[]` pages. Includes `acceptanceDateTime`, which matters for timing (see section 3). |
| Structured financials | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | XBRL facts by concept (Revenues, NetIncomeLoss, Assets ...) with `filed` and `accn`, so point in time joins are possible. |
| Ticker to CIK | `https://www.sec.gov/files/company_tickers.json` | Current mapping only. Historical tickers change, so the canonical key everywhere in the lab is CIK, never ticker. |
| Nightly bulk | `submissions.zip`, `companyfacts.zip` under `Archives/edgar/daily-index/bulkdata/` | Preferred for full backfills: one download instead of thousands of requests. |
| Filing documents | `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}` | HTML for modern filings, plain text for older ones. |

Fair access policy (sec.gov "Accessing EDGAR Data"):
1. At most 10 requests per second across all machines. The client caps itself at 8 by default to leave headroom.
2. Every request must declare a User-Agent with an organisation name and a contact email. The client refuses to start without one.
3. The SEC can block IPs that ignore this, so the Spark job does not let every executor hit EDGAR. Downloading is a separate, rate limited stage; Spark only processes files already on S3.

### Prices

EDGAR has no prices. The lab treats prices as a pluggable `PriceSource`:
1. `CsvPriceSource` for any vendor export (CRSP via WRDS if you have academic access, Polygon, Tiingo, etc.).
2. `StooqPriceSource` for free daily bars for experimentation.
3. `SyntheticPriceSource` for tests and the offline demo.

Survivorship bias warning: free sources usually drop delisted tickers. Results on free data overstate performance; the README says so and the evaluation reports the coverage rate.

### What we deliberately avoid

Earnings call transcripts from paid aggregators are excluded because redistribution and training rights are unclear. Everything in the lab is derived from EDGAR (public domain US government data) plus a price feed you are licensed to use.

The Loughran McDonald master dictionary is the standard finance lexicon, but its licence for redistribution is not clear from public sources. The repo therefore ships its own small seed lexicon (`filings_lab/features/lexicon.py`) and can load the official CSV from a local path if you download it yourself.

## 2. Text processing decisions

1. Strip HTML and inline XBRL tags, drop tables whose cells are mostly numeric (financial statements are better taken from XBRL), normalise whitespace and unicode.
2. Section extraction by regex on item headers: 10-K Item 1A (Risk Factors) and Item 7 (MD&A); 10-Q Part I Item 2 and Part II Item 1A; 8-K item codes (2.02 results, 5.02 officer changes, 1.01 material agreements ...). The table of contents also contains "Item 1A", so the extractor takes the last header match that is followed by a substantial body, not the first.
3. Paragraph ids are stable hashes of (accession, section, paragraph index). These ids are what the agent cites in Phase 5.
4. Year over year change features: Cohen, Malloy and Nguyen ("Lazy Prices", 2020) found that firms whose 10-K text changes little from the prior year outperform firms whose text changes a lot. The lab computes cosine and Jaccard similarity between a filing and the same company's previous filing of the same form.

## 3. Labels and timing (the part that most often goes wrong)

1. Event time is the EDGAR acceptance timestamp, not the filing date. A 10-K accepted at 17:30 ET can only be traded the next session.
2. Rule: if accepted before 16:00 ET on a trading day, day 0 is that day; otherwise day 0 is the next trading day.
3. Primary label: cumulative abnormal return CAR[0, +2] versus a market proxy (SPY or an equal weighted universe return), and a market model variant estimated over days [-250, -30].
4. Classification target: sign of CAR, plus a three class version (bottom tercile, middle, top tercile within each month) to remove market wide drift.
5. All features must be computable from information available strictly before day 0 open. Point in time XBRL facts are joined on `filed < event_date`.

## 4. Validation and statistics

1. Ordinary K fold leaks because labels span several days and filings cluster in earnings season. The lab uses purged walk forward validation (Lopez de Prado, Advances in Financial Machine Learning, ch. 7): expanding train window, test window rolls forward, training rows whose label window overlaps the test window are purged, and an embargo of several trading days follows each test block.
2. Metrics: AUC, log loss, information coefficient (Spearman rank correlation between prediction and realised CAR) per month, and a long short decile spread.
3. Significance:
   1. Newey West t statistic on the monthly IC series (autocorrelation robust).
   2. Block bootstrap confidence interval for mean IC and for AUC minus 0.5.
   3. Permutation test: shuffle labels within month, refit, to get a null distribution.
   4. Diebold Mariano test to compare two models' loss series.
   5. Benjamini Hochberg correction whenever many feature groups are tested.
4. Expectation setting: published work finds filing text effects are real but small. An out of sample AUC around 0.52 to 0.55 would be a credible result; anything above 0.65 almost certainly means leakage and the test suite has leakage checks for this reason.

## 5. LLM fine tuning

1. Task: structured signal extraction, not return prediction. Input is a section of a filing; output is JSON matching a Pydantic schema (guidance direction, risk factor changes, litigation, liquidity concern, management tone, key quotes with paragraph ids).
2. Base models: Qwen2.5 3B Instruct or Llama 3.2 3B Instruct for single GPU work; 7B or 8B with QLoRA (4 bit NF4) if a 24 GB GPU is available.
3. Labels: a frontier model, through its provider's API, labels a few thousand sections; a human reviews a stratified sample of a few hundred to build the gold test set. The fine tuned model is a distillation of the teacher into a model cheap enough to run over the whole corpus.
4. Recipe (community reference points for Qwen2.5 3B extraction adapters plus TRL defaults): LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections, learning rate 1e-4 to 2e-4, 2 to 3 epochs, completion only loss.
5. Eval harness compares base, fine tuned and frontier model on: JSON validity rate, schema validity rate, field level accuracy and macro F1, quote grounding rate (does the quoted text actually appear in the input), latency and cost per 1k filings.

## 6. Serving and MLOps

1. vLLM serves the base model plus LoRA adapter through its OpenAI compatible API. FastAPI sits in front for auth, schema validation, Prometheus metrics and fallback.
2. Kubernetes on EKS: GPU node group for vLLM, CPU node group for the API, HorizontalPodAutoscaler on CPU for the API and on a custom metric (requests waiting) for vLLM via prometheus adapter.
3. MLflow tracks every training run; the registry stage "Production" is what the deployment pulls.
4. Drift: Population Stability Index on input features and prediction distributions, KS test on text length and lexicon scores, with Prometheus gauges and Grafana alerts.
5. CI gate: GitHub Actions runs the eval harness against a frozen gold set and fails if the candidate model is worse than the production baseline on any gated metric beyond a tolerance.

## 7. Agent

1. Retrieval: BM25 over paragraph chunks (no extra infra), optional dense embeddings, filtered by CIK and date.
2. Tools: `search_filings`, `get_paragraph`, `get_price_history`, `get_event_return`, `get_model_signal`.
3. Every factual claim must carry a citation `[accession#paragraph_id]`; the answer validator rejects citations that were not returned by a tool in the same session.

## 8. Experimentation

1. Unit of randomisation: analyst user id, hashed deterministically into variant A or B (sticky assignment).
2. Primary metric: answer helpfulness rating (thumbs up rate); guardrail metrics: latency p95, citation validity rate.
3. Analysis: two proportion z test, Wilson and Newcombe confidence intervals, bootstrap CI for the difference, and a power analysis that tells you the sample size needed for a given minimum detectable effect before the test starts. Sequential peeking is not allowed unless you switch to the always valid (mSPRT) variant included in the module.

## Sources

1. SEC, Accessing EDGAR Data: https://www.sec.gov/os/accessing-edgar-data
2. Loughran and McDonald (2011), When Is a Liability Not a Liability?, Journal of Finance: https://papers.ssrn.com/abstract=1331573
3. Purged cross validation overview: https://en.wikipedia.org/wiki/Purged_cross-validation
4. Cohen, Malloy, Nguyen (2020), Lazy Prices, Journal of Finance.
5. Qwen2.5 3B LoRA extraction adapter reference: https://huggingface.co/shimilm/qwen2.5-3b-extraction-lora
6. Frankel, Jennings, Lee (2022), Disclosure Sentiment: Machine Learning vs. Dictionary Methods, Management Science.
