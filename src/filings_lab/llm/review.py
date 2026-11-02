"""Human review of teacher labels: the step that turns teacher output into a trusted gold set.

A reviewer sees the filing excerpt and the teacher's proposed labels, then accepts, corrects or skips. Every
decision is appended to a JSONL file (one line per review, latest wins), so the process can stop and resume
and nothing is ever overwritten. Agreement between teacher and human per field (accuracy and Cohen's kappa)
says how much the teacher can be trusted on each field, which is the number that justifies distilling it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .schema import LABEL_FIELDS


def cohen_kappa(a: list, b: list) -> float:
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a.tolist()) | set(b.tolist()), key=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


class ReviewStore:
    def __init__(self, candidates: str | Path, reviewed: str | Path):
        self.candidates_path, self.reviewed_path = Path(candidates), Path(reviewed)
        self.lock = threading.Lock()
        self._items = None

    @property
    def items(self) -> list[dict]:
        if self._items is None:
            lines = self.candidates_path.read_text().splitlines() if self.candidates_path.exists() else []
            self._items = [json.loads(x) for x in lines if x.strip()]
        return self._items

    def reviews(self) -> dict[str, dict]:
        if not self.reviewed_path.exists():
            return {}
        out = {}
        for line in self.reviewed_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["accession"]] = r  # latest decision wins
        return out

    def next_item(self) -> dict | None:
        done = self.reviews()
        pending = [it for it in self.items if it["accession"] not in done]
        if not pending:
            return None
        it = pending[0]
        return {"accession": it["accession"], "form": it.get("form"), "company": it.get("company"),
                "paragraphs": it.get("paragraphs", []), "proposed": {k: it["label"][k] for k in LABEL_FIELDS},
                "remaining": len(pending), "reviewed": len(done), "total": len(self.items)}

    def submit(self, accession: str, action: str, label: dict | None, reviewer: str) -> dict:
        item = next((it for it in self.items if it["accession"] == accession), None)
        if item is None:
            raise KeyError(accession)
        if action not in ("accept", "correct", "skip"):
            raise ValueError("action must be accept, correct or skip")
        proposed = {k: item["label"][k] for k in LABEL_FIELDS}
        final = proposed if action == "accept" else ({**proposed, **(label or {})} if action == "correct" else None)
        rec = {"accession": accession, "action": action, "reviewer": reviewer, "ts": time.time(),
               "proposed": proposed, "label": final}
        with self.lock:
            self.reviewed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.reviewed_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    def stats(self) -> dict:
        done = [r for r in self.reviews().values() if r["action"] != "skip"]
        fields = {}
        for f in LABEL_FIELDS:
            t = [r["proposed"][f] for r in done]
            h = [r["label"][f] for r in done]
            fields[f] = {"agreement": float(np.mean([x == y for x, y in zip(t, h, strict=True)])) if done else None,
                         "kappa": cohen_kappa(t, h) if done else None}
        return {"reviewed": len(done), "skipped": sum(r["action"] == "skip" for r in self.reviews().values()),
                "corrected": sum(r["action"] == "correct" and r["label"] != r["proposed"] for r in done),
                "total": len(self.items), "fields": fields}

    def gold(self) -> list[dict]:
        """Reviewed items in gold.jsonl format, human labels replacing the teacher's."""
        by = {it["accession"]: it for it in self.items}
        out = []
        for r in self.reviews().values():
            if r["action"] == "skip":
                continue
            it = dict(by[r["accession"]])
            it["label"] = {**it["label"], **r["label"]}
            out.append(it)
        return out
