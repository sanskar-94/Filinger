# The analyst app and its API in one container, for a free Hugging Face Space (Docker SDK, CPU basic).
# The image clones this repository, builds the synthetic demo data into itself and serves the Next.js app on
# port 7860; the API listens on localhost only. See deploy/space/README.md for the four steps.
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NEXT_TELEMETRY_DISABLED=1
# Node from the official image of the same Debian release, so the Next.js app runs beside the Python API
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
 && apt-get update && apt-get install -y --no-install-recommends git curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
# Spaces run the container as user 1000
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app
ARG REPO=https://github.com/sanskar-94/Filinger
ARG REF=main
RUN git clone --depth 1 --branch "${REF}" "${REPO}" .
RUN pip install --user -e ".[ml,serve,agent]"
# The demo data: the synthetic world, the lake, features, the reaction model and the rule extractor's signals
RUN flab synth --out data/synthetic --companies 60 \
 && flab local --index data/synthetic/index.parquet --prices data/synthetic/prices.parquet --lake data/lake --workers 2 \
 && flab features --lake data/lake --facts data/synthetic/facts.parquet \
 && flab train --lake data/lake --permutations 0 \
 && flab llm-features --lake data/lake --system rules \
 && flab register --lake data/lake --alias champion
RUN cd web && npm ci --no-audit --no-fund && npm run build
# Without a key the agent answers offline. Add NVIDIA_API_KEY as a Space secret to answer with a hosted model.
ENV FLAB_AGENT_MODE=auto FLAB_EVENT_DB=/tmp/experiments.db FLAB_API_URL=http://127.0.0.1:8080 PORT=7860
EXPOSE 7860
CMD ["bash", "deploy/space/start.sh"]
