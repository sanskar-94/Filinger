# API image: FastAPI + reaction model. The LLM runs in its own vLLM deployment (Dockerfile.vllm).
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir --upgrade pip build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /dist/*.whl /tmp/
RUN pip install "$(ls /tmp/*.whl)[ml,serve]" && rm /tmp/*.whl
RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /home/app
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD curl -fsS http://localhost:8080/healthz || exit 1
CMD ["uvicorn", "filings_lab.serving.asgi:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers"]
