"""Tests for the run-state checkpoint file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ase_structure_optimizer.adsorption.config import config_from_dict
from ase_structure_optimizer.adsorption.state import STATE_FILE, RunState, StateError


def _config(**overrides):
    data = {"molecule": "m.xyz", "solid": "s.cif"}
    data.update(overrides)
    return config_from_dict(data)


def test_create_update_and_reload(tmp_path: Path):
    state = RunState.create(tmp_path, _config())
    assert (tmp_path / STATE_FILE).is_file()
    assert state.stage_status("molecule") == "pending"

    state.set_stage("molecule", "done")
    state.add_elapsed("001_t0", 12.5)
    state.add_elapsed("001_t0", 2.5)
    state.set_item("001_t0/relax", "c0001", "done", energy=-1.25, steps=12)
    state.set_item("001_t0/relax", "c0002", "failed", reason="NaN forces")
    state.set_item("001_t0/relax", "c0003", "running")

    loaded = RunState.load(tmp_path)
    assert loaded.stage_status("molecule") == "done"
    assert loaded.elapsed("001_t0") == pytest.approx(15.0)
    assert loaded.item_record("001_t0/relax", "c0001") == {"status": "done", "energy": -1.25, "steps": 12}
    assert loaded.item_status("001_t0/relax", "c0002") == "failed"
    assert loaded.remaining("001_t0/relax", ["c0001", "c0002", "c0003", "c0004"]) == ["c0003", "c0004"]
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_resets_running_items(tmp_path: Path):
    state = RunState.create(tmp_path, _config())
    state.set_stage("surface", "running")
    state.set_item("relax", "a", "running")
    state.set_item("relax", "b", "done")

    loaded = RunState.load(tmp_path)
    assert sorted(loaded.prepare_resume()) == [("relax", "a"), ("surface", "")]
    reloaded = RunState.load(tmp_path)
    assert reloaded.stage_status("surface") == "pending"
    assert reloaded.item_status("relax", "a") == "pending"
    assert reloaded.item_status("relax", "b") == "done"


def test_config_changes_block_resume(tmp_path: Path):
    RunState.create(tmp_path, _config())
    state = RunState.load(tmp_path)
    state.check_config(_config(budget={"wall_time_per_termination": 30}))  # budget may change
    with pytest.raises(StateError, match="new run"):
        state.check_config(_config(sampling={"n_random": 10}))


def test_invalid_inputs(tmp_path: Path):
    with pytest.raises(StateError, match="state.json"):
        RunState.load(tmp_path)
    state = RunState.create(tmp_path, _config())
    with pytest.raises(ValueError):
        state.set_stage("x", "finished")
    data = json.loads((tmp_path / STATE_FILE).read_text())
    data["version"] = 99
    (tmp_path / STATE_FILE).write_text(json.dumps(data))
    with pytest.raises(StateError, match="version"):
        RunState.load(tmp_path)
