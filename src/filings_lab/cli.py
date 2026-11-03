"""``flab``: one command line entry point for every stage of the lab.

Each phase adds subcommands; ``flab --help`` lists them. Heavy dependencies are imported inside the
command functions so that ``flab synth`` works with only the core install.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_synth(a):
    from .synthetic import SyntheticConfig, generate

    cfg = SyntheticConfig(n_companies=a.companies, start=a.start, end=a.end, seed=a.seed)
    world = generate(cfg, a.out)
    print(json.dumps({k: len(v) for k, v in world.items()}))


def cmd_index(a):
    import pandas as pd

    from .edgar.client import EdgarClient
    from .edgar.universe import index_from_api, index_from_bulk, load_company_tickers

    if a.bulk:
        idx = index_from_bulk(a.bulk, a.start, a.end)
    else:
        client = EdgarClient(cache_dir=a.cache)
        tickers = load_company_tickers(client)
        wanted = [t.strip().upper() for t in Path(a.tickers).read_text().split()] if a.tickers else None
        ciks = tickers[tickers["ticker"].isin(wanted)]["cik"] if wanted else tickers["cik"].head(a.limit)
        idx = index_from_api(client, ciks.unique(), a.start, a.end)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(a.out, index=False)
    print(f"{len(idx)} filings from {idx['cik'].nunique()} companies -> {a.out}")
    pd.set_option("display.width", 160)
    print(idx["form"].value_counts())


def cmd_download(a):
    from . import storage
    from .edgar.client import EdgarClient
    from .edgar.download import download_filings

    idx = storage.read_parquet(a.index)
    out = download_filings(EdgarClient(), idx, a.raw, workers=a.workers)
    storage.write_parquet(out, a.index)
    print(out["download_status"].str.split(":").str[0].value_counts())


def cmd_prices(a):
    import pandas as pd

    from .prices.build import build_prices
    from .prices.sources import CsvPriceSource, StooqPriceSource

    idx = pd.read_parquet(a.index)
    source = CsvPriceSource(a.path) if a.source == "csv" else StooqPriceSource()
    prices, report = build_prices(idx, source, a.start, a.end, a.market)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(a.out, index=False)
    print(json.dumps(report, indent=2))


def cmd_facts(a):
    import pandas as pd

    from .edgar.xbrl import facts_from_api, facts_from_bulk

    ciks = set(pd.read_parquet(a.index)["cik"].astype(int))
    if a.bulk:
        facts = facts_from_bulk(a.bulk, ciks)
    else:
        from .edgar.client import EdgarClient

        facts = facts_from_api(EdgarClient(cache_dir=a.cache), sorted(ciks))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(a.out, index=False)
    print(f"{len(facts)} facts for {facts['cik'].nunique() if len(facts) else 0} of {len(ciks)} companies -> {a.out}")


def cmd_local(a):
    from . import storage
    from .pipeline import local

    idx = storage.read_parquet(a.index)
    if "raw_uri" not in idx.columns:
        from .edgar.download import raw_uri

        idx["raw_uri"] = [raw_uri(a.raw, c, acc) for c, acc in zip(idx["cik"], idx["accession"], strict=True)]
    prices = storage.read_parquet(a.prices)
    print(local.run(idx, prices, a.lake, workers=a.workers, market=a.market))


def cmd_spark(a):
    from .pipeline import spark_job

    spark_job.run(a.index, a.raw, a.prices, a.lake, a.market)


def cmd_dq(a):
    from .pipeline.quality import report, run_checks, save

    checks = run_checks(a.lake)
    save(checks, a.out)
    print(report(checks))
    if any(c.status == "fail" for c in checks):
        sys.exit(1)


def cmd_sql(a):
    import pandas as pd

    from .sqlrun import run_sql

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(run_sql(a.file, a.lake, a.artifacts).to_string(index=False))


COMMANDS = {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flab", description="Filings Intelligence Lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate the synthetic EDGAR world")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--companies", type=int, default=60)
    s.add_argument("--start", default="2012-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("index", help="build the bronze filing index from EDGAR")
    s.add_argument("--out", default="data/lake/bronze/index.parquet")
    s.add_argument("--start", default="2014-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--tickers", help="file with whitespace separated tickers")
    s.add_argument("--limit", type=int, default=500, help="first N companies when no ticker file is given")
    s.add_argument("--bulk", help="path to a downloaded submissions.zip")
    s.add_argument("--cache", default="data/http_cache")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("download", help="download raw filings listed in the index")
    s.add_argument("--index", required=True)
    s.add_argument("--raw", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("prices", help="daily prices for every ticker in the index plus the market proxy")
    s.add_argument("--index", required=True)
    s.add_argument("--source", choices=["stooq", "csv"], default="stooq")
    s.add_argument("--path", help="vendor CSV file or directory of TICKER.csv files (source csv)")
    s.add_argument("--start", default="2013-01-01")
    s.add_argument("--end", default="2024-12-31")
    s.add_argument("--market", default="SPY")
    s.add_argument("--out", default="data/lake/bronze/prices.parquet")
    s.set_defaults(fn=cmd_prices)

    s = sub.add_parser("facts", help="XBRL fundamentals for every company in the index")
    s.add_argument("--index", required=True)
    s.add_argument("--bulk", help="path to a downloaded companyfacts.zip")
    s.add_argument("--cache", default="data/http_cache")
    s.add_argument("--out", default="data/lake/bronze/facts.parquet")
    s.set_defaults(fn=cmd_facts)

    for name, fn, help_ in (("local", cmd_local, "run the pipeline on one machine"),
                            ("spark", cmd_spark, "run the pipeline with PySpark")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--index", required=True)
        s.add_argument("--raw", default="data/lake/bronze/raw")
        s.add_argument("--prices", required=True)
        s.add_argument("--lake", default="data/lake")
        s.add_argument("--market", default="SPY")
        s.add_argument("--workers", type=int, default=4)
        s.set_defaults(fn=fn)

    s = sub.add_parser("dq", help="data quality gates over the lake; exits 1 on failure")
    s.add_argument("--lake", default="data/lake")
    s.add_argument("--out", default="artifacts/dq")
    s.set_defaults(fn=cmd_dq)

    s = sub.add_parser("sql", help="run a SQL file from sql/ against the lake with DuckDB")
    s.add_argument("file")
    s.add_argument("--lake", default="data/lake")
    s.add_argument("--artifacts", default="artifacts")
    s.set_defaults(fn=cmd_sql)

    for name, (register, _help) in COMMANDS.items():
        register(sub.add_parser(name, help=_help))
    return p


def main(argv: list[str] | None = None) -> None:
    _load_plugins()
    args = build_parser().parse_args(argv)
    args.fn(args)


def _load_plugins() -> None:
    """Later phases register their subcommands here without making the core CLI import them eagerly."""
    from importlib import import_module

    for mod in ("features.cli", "modeling.cli", "llm.cli", "serving.cli", "agent.cli", "experiment.cli"):
        try:
            m = import_module(f"filings_lab.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("filings_lab"):
                continue
            raise
        COMMANDS.update(getattr(m, "COMMANDS", {}))


if __name__ == "__main__":
    main(sys.argv[1:])
