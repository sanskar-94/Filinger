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

    def register(self, name: str, artifact_dir: str | Path, metrics: dict | None = None, tags: dict | None = None,
                 kind: str = "reaction", run_id: str | None = None) -> int:
        """Copy ``artifact_dir`` in as the next version. The copy is staged next to its final place and published
        with one rename, so a reader never sees a half written version, and two concurrent registrations cannot
        claim the same number. Permissions are normalised (dirs 755, files 644) because the staging source is
        often a 700 temp dir and the API reads the registry as a different, non root user."""
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        stage = d / f".staging-{os.getpid()}-{time.time_ns()}"
        shutil.copytree(artifact_dir, stage)
        try:
            for path in [stage, *stage.rglob("*")]:
                path.chmod(0o755 if path.is_dir() else 0o644)
            for _ in range(100):
                existing = [int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit()]
                version = max(existing, default=0) + 1
                meta = {"name": name, "version": version, "created": time.time(), "kind": kind, "run_id": run_id,
                        "metrics": metrics or {}, "tags": tags or {}}
                (stage / "registry.json").write_text(json.dumps(meta, indent=2))
                model_meta = stage / "meta.json"
                if model_meta.exists():
                    mm = json.loads(model_meta.read_text())
                    mm["version"] = version
                    model_meta.write_text(json.dumps(mm, indent=2))
                try:
                    os.rename(stage, d / str(version))  # fails if another writer published this number first
                    return version
                except OSError:
                    continue
            raise RuntimeError(f"could not allocate a version for {name}")
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def _aliases(self, name: str) -> dict:
        f = self._dir(name) / "aliases.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def set_alias(self, name: str, alias: str, version: int) -> None:
        if not (self._dir(name) / str(version)).exists():
            raise KeyError(f"{name} v{version} not registered")
        al = self._aliases(name)
        al[alias] = int(version)
        tmp = self._dir(name) / f".aliases-{os.getpid()}-{time.time_ns()}.json"
        tmp.write_text(json.dumps(al, indent=2))
        tmp.chmod(0o644)
        os.replace(tmp, self._dir(name) / "aliases.json")  # atomic: the API never reads a torn alias file

    def get_alias(self, name: str, alias: str) -> int | None:
        return self._aliases(name).get(alias)

    def metadata(self, name: str, version: int) -> dict:
        return json.loads((self._dir(name) / str(version) / "registry.json").read_text())

    def describe(self, uri: str) -> dict:
        m = _URI.match(uri)
        if not m:
            return {"name": None, "version": None}
        version = int(m["version"]) if m["version"] else self.get_alias(m["name"], m["alias"])
        return {"name": m["name"], "version": version}

    def resolve(self, uri: str) -> Path:
        m = _URI.match(uri)
        if not m:
            return Path(uri)  # plain path
        name = m["name"]
        version = int(m["version"]) if m["version"] else self.get_alias(name, m["alias"])
        if version is None:
            raise KeyError(f"alias {m['alias']} not set for {name}")
        return self._dir(name) / str(version)


ARTIFACT_KEY = "reaction_model"
ARTIFACT_KEYS = {"reaction": ARTIFACT_KEY, "adapter": "adapter"}


def _pyfunc_class(kind: str = "reaction"):
    import mlflow.pyfunc

    class ReactionPyfunc(mlflow.pyfunc.PythonModel):
        """MLflow wrapper: ``mlflow models serve`` and ``mlflow.pyfunc.load_model`` can use the model directly."""

        def load_context(self, context):
            from .model import ReactionModel

            self.model = ReactionModel.load(context.artifacts[ARTIFACT_KEY])

        def predict(self, context, model_input, params=None):
            return self.model.predict(model_input)

    class AdapterPyfunc(mlflow.pyfunc.PythonModel):
        """A LoRA adapter for the extraction model. vLLM serves the adapter folder itself; this wrapper makes the
        registered version usable on its own: rows with an ``input`` column (ExtractionInput as JSON) in, the
        extracted signals as JSON out. The base model is the one named in adapter_config.json."""

        def load_context(self, context):
            from ..llm.providers import HFExtractor

            adapter = Path(context.artifacts["adapter"])
            base = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
            self.extractor = HFExtractor(base, str(adapter), style=os.environ.get("FLAB_LLM_PROMPT_STYLE", "compact"))

        def predict(self, context, model_input, params=None):
            from ..llm.prompts import ExtractionInput

            rows = model_input["input"] if hasattr(model_input, "columns") else model_input
            out = []
            for row in rows:
                res = self.extractor.extract(ExtractionInput(**(json.loads(row) if isinstance(row, str) else row)))
                out.append(json.dumps(res.signals) if res.signals is not None else None)
            return out

    return {"reaction": ReactionPyfunc, "adapter": AdapterPyfunc}[kind]


class MlflowRegistry:
    def __init__(self):
        import mlflow
        from mlflow import MlflowClient

        self.mlflow, self.client = mlflow, MlflowClient()

    def register(self, name, artifact_dir, metrics=None, tags=None, kind: str = "reaction", run_id: str | None = None) -> int:
        """Log the artefact as a pyfunc model (MLflow 3 registers logged models, not bare folders) and register it.
        With ``run_id`` the model is logged into that run (the training run), so the version links to it."""
        if run_id:
            ctx = self.mlflow.start_run(run_id=run_id)
        else:
            self.mlflow.set_experiment("model-registry")
            ctx = self.mlflow.start_run(run_name=f"register-{name}")
        with ctx:
            self.mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items()
                                     if isinstance(v, (int, float)) and not isinstance(v, bool)})
            self.mlflow.set_tags(tags or {})
            info = self.mlflow.pyfunc.log_model(name="model", python_model=_pyfunc_class(kind)(),
                                                artifacts={ARTIFACT_KEYS[kind]: str(artifact_dir)})
        mv = self.mlflow.register_model(info.model_uri, name, tags=tags)
        return int(mv.version)

    def set_alias(self, name, alias, version):
        self.client.set_registered_model_alias(name, alias, str(version))

    def get_alias(self, name, alias):
        try:
            return int(self.client.get_model_version_by_alias(name, alias).version)
        except Exception:  # noqa: BLE001
            return None

    def describe(self, uri: str) -> dict:
        m = _URI.match(uri)
        if not m:
            return {"name": None, "version": None}
        version = int(m["version"]) if m["version"] else self.get_alias(m["name"], m["alias"])
        return {"name": m["name"], "version": version}

    def resolve(self, uri: str) -> Path:
        if not uri.startswith("models:/"):
            return Path(uri)
        path = Path(self.mlflow.artifacts.download_artifacts(artifact_uri=uri))
        mlmodel = path / "MLmodel"
        if mlmodel.exists():  # a pyfunc model: the MLmodel file maps the artefact key to its folder
            import yaml

            spec = yaml.safe_load(mlmodel.read_text())
            artifacts = spec.get("flavors", {}).get("python_function", {}).get("artifacts", {})
            for key in ARTIFACT_KEYS.values():
                if rel := artifacts.get(key, {}).get("path"):
                    return path / rel
        return path


def get_registry():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri.startswith(("http://", "https://", "databricks")):
        return MlflowRegistry()
    return FileRegistry(os.environ.get("FLAB_REGISTRY", "artifacts/registry"))
