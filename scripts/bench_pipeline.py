"""Throughput of the Phase 1 pipeline at a larger scale: the single machine runner and the Spark job on the same
synthetic world, with their outputs checked against each other.

    python scripts/bench_pipeline.py --companies 300 --out artifacts/scale            # both engines
    python scripts/bench_pipeline.py --engines spark --world data/synthetic_scale ...  # one engine

Writes ``<out>/bench.json`` and ``<out>/bench.md``. Numbers are for this machine (``cpus`` in the output); the
markdown also turns the Spark rate per core into core hours per terabyte of raw filings, a basis for sizing a cluster.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import pandas as pd


def _count(lake: Path) -> dict:
    out = {}
    for table in ("silver/sections", "silver/paragraphs", "gold/events"):
        p = lake / table
        out[table] = int(len(pd.read_parquet(p, columns=["accession"]))) if p.exists() else None
    return out


def _engine_result(name: str, seconds: float, filings: int, raw_mb: float, counts: dict, cpus: int) -> dict:
    paragraphs = counts["silver/paragraphs"] or 0
    return {"engine": name, "seconds": round(seconds, 1), "filings_per_s": round(filings / seconds, 1),
            "filings_per_core_s": round(filings / seconds / cpus, 1), "paragraphs_per_s": round(paragraphs / seconds),
            "raw_mb_per_s": round(raw_mb / seconds, 2), "rows": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="data/synthetic_scale")
    ap.add_argument("--companies", type=int, default=300)
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--engines", default="local,spark")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="artifacts/scale")
    a = ap.parse_args(argv)

    from filings_lab.synthetic import SyntheticConfig, generate

    world, out = Path(a.world), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    synth_s = None
    if not (world / "index.parquet").exists():
        t0 = time.perf_counter()
        generate(SyntheticConfig(n_companies=a.companies, start=a.start, end=a.end, seed=a.seed), world)
        synth_s = round(time.perf_counter() - t0, 1)
    idx = pd.read_parquet(world / "index.parquet")
    prices = pd.read_parquet(world / "prices.parquet")
    raw_files = list((world / "raw").rglob("*.txt"))
    raw_mb = sum(f.stat().st_size for f in raw_files) / 1e6
    cpus = os.cpu_count() or 1
    prev = json.loads((out / "bench.json").read_text()) if (out / "bench.json").exists() else {}
    res = {"filings": len(idx), "companies": int(idx["cik"].nunique()), "raw_files": len(raw_files),
           "raw_mb": round(raw_mb, 1), "cpus": cpus, "machine": platform.machine(), "synth_seconds": synth_s,
           "engines": prev.get("engines", {}) if prev.get("filings") == len(idx) else {}}

    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    if "local" in engines:
        from filings_lab.pipeline import local

        lake = out / "lake_local"
        t0 = time.perf_counter()
        local.run(idx, prices, str(lake), workers=a.workers)
        res["engines"]["local"] = _engine_result(f"local runner, {a.workers} processes", time.perf_counter() - t0,
                                                 len(idx), raw_mb, _count(lake), cpus)
        print(json.dumps(res["engines"]["local"]))
    if "spark" in engines:
        from filings_lab.pipeline import spark_job

        lake = out / "lake_spark"
        t0 = time.perf_counter()
        spark_job.build_spark().sparkContext.setLogLevel("ERROR")  # session start counted, as on a cluster
        spark_job.run(str(world / "index.parquet"), str(world / "raw"), str(world / "prices.parquet"), str(lake))
        res["engines"]["spark"] = _engine_result(f"Spark local[{cpus}]", time.perf_counter() - t0, len(idx), raw_mb,
                                                 _count(lake), cpus)
        print(json.dumps(res["engines"]["spark"]))

    rows = [e["rows"] for e in res["engines"].values()]
    res["outputs_match"] = all(r == rows[0] for r in rows) if len(rows) > 1 else None
    (out / "bench.json").write_text(json.dumps(res, indent=2))
    (out / "bench.md").write_text(render(res))
    print(render(res))
    return 0 if res["outputs_match"] in (True, None) else 1


def render(r: dict) -> str:
    lines = [f"# Pipeline throughput: {r['filings']:,} filings, {r['companies']} companies, {r['raw_mb']:,} MB raw",
             "", f"Machine: {r['cpus']} cores ({r['machine']}).", "",
             "| engine | seconds | filings/s | filings/core/s | paragraphs/s | raw MB/s | paragraphs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for e in r["engines"].values():
        lines.append(f"| {e['engine']} | {e['seconds']} | {e['filings_per_s']} | {e['filings_per_core_s']} | "
                     f"{e['paragraphs_per_s']:,} | {e['raw_mb_per_s']} | {e['rows']['silver/paragraphs']:,} |")
    if r.get("outputs_match") is not None:
        lines += ["", f"Row counts of silver/sections, silver/paragraphs and gold/events identical across engines: "
                      f"**{r['outputs_match']}**."]
    if "spark" in r["engines"]:
        e = r["engines"]["spark"]
        mb_core = e["raw_mb_per_s"] / r["cpus"]
        tb_core_hours = 1e6 / mb_core / 3600
        lines += ["", f"Parsing runs at {mb_core:.2f} MB of raw filing text per core second. Bytes, not filings, are the "
                      f"guide for real data, whose filings are far longer than the synthetic ones (and carry exhibits "
                      f"and XBRL the cleaner drops early, so this is conservative): each terabyte of raw filings needs "
                      f"about {tb_core_hours:,.0f} core hours, about {tb_core_hours / 64:,.1f} hours on a 64 core EMR or "
                      f"Databricks cluster before I/O effects."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
