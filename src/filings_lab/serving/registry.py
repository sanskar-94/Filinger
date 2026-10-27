"""Model registry with aliases ("champion", "challenger").

``MlflowRegistry`` wraps the MLflow Model Registry when a tracking server is configured. ``FileRegistry``
implements the same small interface on a directory (local disk or a synced S3 prefix), so the service, the
CI gate and the tests never depend on a running server.

URIs: ``models:/<name>@<alias>`` or ``models:/<name>/<version>``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

_URI = re.compile(r"^models:/(?P<name>[\w.-]+)(?:@(?P<alias>[\w-]+)|/(?P<version>\d+))$")


class FileRegistry:
    def __init__(self, root: str | Path = "artifacts/registry"):
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def register(self, name: str, artifact_dir: str | Path, metrics: dict | None = None, tags: dict | None = None) -> int:
        d = self._dir(name)
        existing = [int(p.name) for p in d.glob("[0-9]*") if p.is_dir()] if d.exists() else []
        version = max(existing, default=0) + 1
        dest = d / str(version)
        shutil.copytree(artifact_dir, dest)
        meta = {"name": name, "version": version, "created": time.time(), "metrics": metrics or {}, "tags": tags or {}}
        (dest / "registry.json").write_text(json.dumps(meta, indent=2))
        return version

    def _aliases(self, name: str) -> dict:
        f = self._dir(name) / "aliases.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        if not (self._dir(name) / str(version)).exists():
            raise KeyError(f"{name} v{version} not registered")
        al = self._aliases(name)
        al[alias] = int(version)
        (self._dir(name) / "aliases.json").write_text(json.dumps(al, indent=2))

    def get_alias(self, name: str, alias: str) -> int | None:
        return self._aliases(name).get(alias)

    def metadata(self, name: str, version: int) -> dict:
        return json.loads((self._dir(name) / str(version) / "registry.json").read_text())

    def resolve(self, uri: str) -> Path:
        m = _URI.match(uri)
        if not m:
            return Path(uri)  # plain path
        name = m["name"]
        version = int(m["version"]) if m["version"] else self.get_alias(name, m["alias"])
        if version is None:
            raise KeyError(f"alias {m['alias']} not set for {name}")
        return self._dir(name) / str(version)


class MlflowRegistry:  # pragma: no cover - needs a tracking server
    def __init__(self):
        import mlflow
        from mlflow import MlflowClient

        self.mlflow, self.client = mlflow, MlflowClient()

    def register(self, name, artifact_dir, metrics=None, tags=None) -> int:
        with self.mlflow.start_run(run_name=f"register-{name}"):
            self.mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))})
            self.mlflow.log_artifacts(str(artifact_dir), artifact_path="model")
            run_id = self.mlflow.active_run().info.run_id
        mv = self.mlflow.register_model(f"runs:/{run_id}/model", name, tags=tags)
        return int(mv.version)

    def set_alias(self, name, alias, version):
        self.client.set_registered_model_alias(name, alias, str(version))

    def get_alias(self, name, alias):
        try:
            return int(self.client.get_model_version_by_alias(name, alias).version)
        except Exception:  # noqa: BLE001
            return None

    def resolve(self, uri: str) -> Path:
        if not uri.startswith("models:/"):
            return Path(uri)
        return Path(self.mlflow.artifacts.download_artifacts(artifact_uri=uri))


def get_registry():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri.startswith(("http://", "https://", "databricks")):
        return MlflowRegistry()
    return FileRegistry(os.environ.get("FLAB_REGISTRY", "artifacts/registry"))
