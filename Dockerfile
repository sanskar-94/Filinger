# API image: FastAPI + reaction model. The LLM runs in its own vLLM deployment (Dockerfile.vllm).
#   docker build -t filings-api .
# Behind a TLS inspecting proxy or a registry mirror (see docs/phases/phase4_serving.md):
#   docker build --network host --build-arg HTTPS_PROXY=... --secret id=proxy_ca,src=proxy.crt \
#                --build-arg BUILD_IMAGE=mirror.gcr.io/library/python:3.12 \
#                --build-arg PYTHON_IMAGE=mirror.gcr.io/library/python:3.12-slim -t filings-api .
# BUILD_IMAGE and PYTHON_IMAGE must share the Python minor version and the Debian release.
ARG BUILD_IMAGE=python:3.12
ARG PYTHON_IMAGE=python:3.12-slim

# The full image has a compiler for any dependency without a wheel, and the OpenMP runtime (libgomp) that the
# LightGBM wheel links against but does not bundle. Copying it here means the runtime stage needs no apt access
# (equivalent to apt-get install libgomp1, without a package manager run in the final image).
FROM ${BUILD_IMAGE} AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
# the proxy CA is mounted for this step only and never written into an image layer
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    pip install --no-cache-dir --upgrade pip build && python -m build --wheel --outdir /dist && \
    pip wheel --no-cache-dir --wheel-dir /wheels "$(ls /dist/*.whl)[ml,serve,agent,aws]" && \
    mkdir -p /runtime-libs && cp -L "$(gcc -print-file-name=libgomp.so.1)" /runtime-libs/libgomp.so.1

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=build /runtime-libs/ /usr/local/lib/
COPY --from=build /wheels /wheels
RUN ldconfig && pip install --no-index --find-links /wheels "filings-lab[ml,serve,agent,aws]" && rm -rf /wheels && \
    python -c "import lightgbm, sklearn; print('lightgbm', lightgbm.__version__, 'sklearn', sklearn.__version__)"
RUN useradd --create-home --uid 10001 app && mkdir -p /state && chown 10001:10001 /state
VOLUME ["/state"]
# numeric, so Kubernetes can verify runAsNonRoot without a runAsUser override
USER 10001:10001
WORKDIR /home/app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"
# One process per container: Prometheus counters, the circuit breaker and the experiment log are per process, so
# several workers behind one /metrics endpoint would report whichever worker answered the scrape. Scale with pods
# (the HPA), not workers. Sync routes still run concurrently on the threadpool.
CMD ["uvicorn", "filings_lab.serving.asgi:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
