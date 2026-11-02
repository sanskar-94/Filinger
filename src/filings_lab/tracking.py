"""Thin MLflow wrapper. Every training entry point logs through this; it degrades to a JSON run log when
MLflow is not installed, so tests and laptops work without a tracking server."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

try:  # pragma: no cover - exercised only when mlflow is installed
    import mlflow
except Exception:  # noqa: BLE001
    mlflow = None


def mlflow_enabled() -> bool:
    """Use MLflow only when it is installed and a tracking URI is configured, so tests and quick local runs
    never create stray mlruns directories."""
    return mlflow is not None and bool(os.environ.get("MLFLOW_TRACKING_URI"))


@contextmanager
def run(experiment: str, run_name: str, fallback_dir: str = "artifacts/runs"):
    if mlflow_enabled():
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as r:
            yield _MlflowRun(r)
        return
    rec = _JsonRun(Path(fallback_dir) / experiment / f"{run_name}-{int(time.time())}")
    try:
        yield rec
    finally:
        rec.close()


class _MlflowRun:  # pragma: no cover
    def __init__(self, r):
        self.id = r.info.run_id

    def params(self, d: dict):
        mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d: dict, step: int | None = None):
        mlflow.log_metrics({k: float(v) for k, v in d.items() if v is not None}, step=step)

    def artifact(self, path: str):
        mlflow.log_artifact(path)

    def artifacts(self, local_dir: str, artifact_path: str | None = None):
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def tags(self, d: dict):
        mlflow.set_tags(d)


class _JsonRun:
    def __init__(self, path: Path):
        self.path = path
        self.id = path.name
        self.data: dict = {"params": {}, "metrics": {}, "artifacts": [], "tags": {}}

    def params(self, d):
        self.data["params"].update({k: str(v) for k, v in d.items()})

    def metrics(self, d, step=None):
        self.data["metrics"].update({k: float(v) for k, v in d.items() if v is not None})

    def artifact(self, path):
        self.data["artifacts"].append(str(path))

    def artifacts(self, local_dir, artifact_path=None):
        self.data["artifacts"].append(str(local_dir))

    def tags(self, d):
        self.data["tags"].update(d)

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.with_suffix(".json").write_text(json.dumps(self.data, indent=2))
