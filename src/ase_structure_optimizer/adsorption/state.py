"""Run state (``state.json``) for checkpointing and resuming a workflow.

The state records, per stage, a status and the wall time spent, and per item
(e.g. one candidate configuration) a status plus small metadata. Every change
is written to disk immediately with an atomic replace, so an interrupted run
leaves a consistent file. On resume, items left ``running`` are reset to
``pending`` and restarted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import AdsorptionConfig, result_fingerprint

Status = Literal["pending", "running", "done", "failed"]
STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed")
STATE_FILE = "state.json"
STATE_VERSION = 1


class StateError(RuntimeError):
    """The run directory cannot be resumed (missing state or changed settings)."""


@dataclass
class RunState:
    """Mutable, file-backed state of one workflow run."""

    path: Path
    fingerprint: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    # -- creation / loading -------------------------------------------------

    @classmethod
    def create(cls, run_dir: str | Path, config: AdsorptionConfig) -> RunState:
        """Start a fresh state for `config` in `run_dir` and write it."""

        state = cls(path=Path(run_dir) / STATE_FILE, fingerprint=result_fingerprint(config))
        state.save()
        return state

    @classmethod
    def load(cls, run_dir: str | Path) -> RunState:
        """Read ``state.json`` from `run_dir`."""

        path = Path(run_dir) / STATE_FILE
        if not path.is_file():
            raise StateError(f"no {STATE_FILE} in {run_dir}; is this a run directory?")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != STATE_VERSION:
            raise StateError(f"{path}: unsupported state version {data.get('version')!r}")
        return cls(
            path=path,
            fingerprint=data["fingerprint"],
            stages=data.get("stages", {}),
            items=data.get("items", {}),
            created=data.get("created", time.time()),
        )

    def check_config(self, config: AdsorptionConfig) -> None:
        """Refuse to resume when result-affecting settings changed."""

        if result_fingerprint(config) != self.fingerprint:
            raise StateError(
                "the calculator, surface, sampling or molecule settings (or input files) differ "
                "from the ones this run was started with; start a new run instead of resuming"
            )

    def prepare_resume(self) -> list[tuple[str, str]]:
        """Reset interrupted (``running``) stages and items to ``pending``.

        Returns the `(stage, item)` pairs that were reset (item is ``""`` for
        a stage itself).
        """

        reset = []
        for stage, record in self.stages.items():
            if record.get("status") == "running":
                record["status"] = "pending"
                reset.append((stage, ""))
        for stage, records in self.items.items():
            for item, record in records.items():
                if record.get("status") == "running":
                    record["status"] = "pending"
                    reset.append((stage, item))
        self.save()
        return reset

    # -- stages -------------------------------------------------------------

    def stage_status(self, stage: str) -> Status:
        return self.stages.get(stage, {}).get("status", "pending")

    def set_stage(self, stage: str, status: Status, message: str = "") -> None:
        _check_status(status)
        record = self.stages.setdefault(stage, {"status": "pending", "elapsed": 0.0})
        record["status"] = status
        if message:
            record["message"] = message
        self.save()

    def add_elapsed(self, stage: str, seconds: float) -> None:
        """Accumulate wall time spent in `stage` (survives interruptions)."""

        record = self.stages.setdefault(stage, {"status": "pending", "elapsed": 0.0})
        record["elapsed"] = float(record.get("elapsed", 0.0)) + float(seconds)
        self.save()

    def elapsed(self, stage: str) -> float:
        return float(self.stages.get(stage, {}).get("elapsed", 0.0))

    # -- items --------------------------------------------------------------

    def item_status(self, stage: str, item: str) -> Status:
        return self.items.get(stage, {}).get(item, {}).get("status", "pending")

    def set_item(self, stage: str, item: str, status: Status, **info: Any) -> None:
        """Set an item's status and merge JSON-serializable `info` into its record."""

        _check_status(status)
        record = self.items.setdefault(stage, {}).setdefault(item, {})
        record["status"] = status
        record.update(info)
        self.save()

    def item_record(self, stage: str, item: str) -> dict[str, Any]:
        return dict(self.items.get(stage, {}).get(item, {}))

    def remaining(self, stage: str, items: Iterable[str]) -> list[str]:
        """Items of `stage` that are not ``done`` or ``failed``, in the given order."""

        return [item for item in items if self.item_status(stage, item) not in ("done", "failed")]

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "fingerprint": self.fingerprint,
            "created": self.created,
            "stages": self.stages,
            "items": self.items,
        }

    def save(self) -> None:
        """Atomically write ``state.json``."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)


def _check_status(status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}, got {status!r}")
