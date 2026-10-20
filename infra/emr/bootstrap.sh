#!/usr/bin/env bash
# EMR bootstrap action: Python dependencies for pandas UDFs on every node.
set -euo pipefail
sudo python3 -m pip install "pandas>=2.1" "pyarrow>=14" "numpy>=1.26" "scipy>=1.11" "pydantic>=2.5"
