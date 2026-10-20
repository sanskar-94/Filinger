"""Thin entry point for spark-submit on EMR (the wheel is shipped with --py-files)."""
from filings_lab.pipeline.spark_job import main

if __name__ == "__main__":
    main()
