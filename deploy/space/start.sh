#!/usr/bin/env bash
# One container, two processes: the API on localhost and the Next.js app on the public port. If either stops,
# the script exits so the platform restarts the container. Run from anywhere; works outside Docker too.
set -euo pipefail
cd "$(dirname "$0")/../.."
export FLAB_EVENT_DB="${FLAB_EVENT_DB:-/tmp/experiments.db}" FLAB_API_URL="${FLAB_API_URL:-http://127.0.0.1:8080}"

flab serve --host 127.0.0.1 --port 8080 &
API=$!
for _ in $(seq 1 120); do
  curl -fsS http://127.0.0.1:8080/readyz > /dev/null 2>&1 && break
  kill -0 "$API" 2> /dev/null || { echo "the API stopped while starting"; exit 1; }
  sleep 1
done

(cd web && exec npx next start -H 0.0.0.0 -p "${PORT:-7860}") &
WEB=$!
trap 'kill "$API" "$WEB" 2> /dev/null || true' EXIT INT TERM
while kill -0 "$API" 2> /dev/null && kill -0 "$WEB" 2> /dev/null; do sleep 2; done
exit 1
