"""End-to-end tests of the surface stages (EMT, Cu + CO)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, fcc111, molecule
from ase.io import read, write

pytest.importorskip("pymatgen")
pytest.importorskip("yaml")

from ase_structure_optimizer.adsorption.config import ConfigError, config_from_dict, load_config  # noqa: E402
from ase_structure_optimizer.adsorption.state import RunState, StateError  # noqa: E402
from ase_structure_optimizer.adsorption.workflow import prepare_surfaces  # noqa: E402


def _inputs(tmp_path: Path, solid=None) -> Path:
    write(tmp_path / "CO.xyz", molecule("CO"))
    write(tmp_path / "Cu.cif", solid if solid is not None else bulk("Cu", "fcc", a=3.6))
    return tmp_path


def _config(tmp_path: Path, **overrides):
    data = {
        "molecule": "CO.xyz",
        "solid": "Cu.cif",
        "calculator": {"name": "emt"},
        "surface": {"miller_indices": [[1, 1, 1], [1, 0, 0]], "min_slab_thickness": 6.0, "lateral_buffer": 8.0},
        "budget": {"fmax": 0.1, "max_steps": 50},
    }
    data.update(overrides)
    return config_from_dict(data, base_dir=tmp_path)


def test_bulk_input_produces_relaxed_terminations(tmp_path: Path):
    config = _config(_inputs(tmp_path))
    models = prepare_surfaces(config)
    run_dir = tmp_path / "CO_on_Cu_emt"

    assert [model.term_id for model in models] == ["111_t0", "100_t0"]
    for name in (
        "config.resolved.yaml",
        "state.json",
        "adsorption.log",
        "results.db",
        "molecule/molecule_analysis.json",
        "bulk/bulk_opt.traj",
        "bulk/bulk_opt.extxyz",
        "bulk/bulk_opt_final.extxyz",
        "terminations/terminations.json",
    ):
        assert (run_dir / name).is_file(), name
    for model in models:
        directory = run_dir / "terminations" / model.term_id
        for name in ("slab_opt.traj", "slab_opt.extxyz", "slab_opt_final.extxyz", "unit_slab.extxyz", "sites.json", "slab.json"):
            assert (directory / name).is_file(), name

        slab = model.slab()
        assert len(slab) == model.n_atoms
        assert len(slab.constraints[0].get_indices()) == model.n_fixed > 0
        assert slab.get_potential_energy() == pytest.approx(model.energy_final)
        assert min(model.widths) >= 1.15 + 8.0 - 1e-6  # CO size + lateral_buffer
        vacuum = slab.cell[2, 2] - (slab.positions[:, 2].max() - slab.positions[:, 2].min())
        assert vacuum == pytest.approx(1.15 + 15.0, abs=0.02)  # auto vacuum
        assert model.sites and all(site.position[2] > 1.0 for site in model.sites)
        assert model.surface_energy is not None and model.surface_energy > 0

    fcc111_sites = {site.site_id for site in models[0].sites}
    assert fcc111_sites == {"ontop_Cu", "bridge_Cu-Cu", "hollow_Cu-Cu-Cu_1", "hollow_Cu-Cu-Cu_2"}

    resolved = load_config(run_dir / "config.resolved.yaml")
    assert resolved.molecule == (run_dir / "inputs" / "CO.xyz").resolve()  # copied into the run
    assert (run_dir / "inputs" / "Cu.cif").read_bytes() == (tmp_path / "Cu.cif").read_bytes()
    assert resolved.run_dir == run_dir.resolve()
    assert "molecule: inputs/CO.xyz" in (run_dir / "config.resolved.yaml").read_text()

    from ase.db import connect

    with connect(run_dir / "results.db") as database:
        assert {row.name for row in database.select(kind="slab")} == {"111_t0", "100_t0"}
        assert database.count(kind="bulk") == 1

    selection = json.loads((run_dir / "terminations" / "terminations.json").read_text())
    assert [row["term_id"] for row in selection["selected"]] == ["111_t0", "100_t0"]
    log = (run_dir / "adsorption.log").read_text()
    assert "Miller indices refer to the conventional cell" in log

    # A second start in the same directory is refused.
    with pytest.raises(StateError, match="already contains a run"):
        prepare_surfaces(config)


def test_resume_reuses_finished_terminations(tmp_path: Path):
    config = _config(_inputs(tmp_path))
    first = prepare_surfaces(config)
    run_dir = tmp_path / "CO_on_Cu_emt"

    # Pretend the run was interrupted while relaxing the second termination.
    state = RunState.load(run_dir)
    state.set_item("surface", "100_t0", "running")
    stamp = (run_dir / "terminations" / "111_t0" / "slab_opt.traj").stat().st_mtime_ns

    resumed = prepare_surfaces(config, resume=True)
    assert [m.term_id for m in resumed] == [m.term_id for m in first]
    assert (run_dir / "terminations" / "111_t0" / "slab_opt.traj").stat().st_mtime_ns == stamp
    assert resumed[1].energy_final == pytest.approx(first[1].energy_final)
    assert RunState.load(run_dir).item_status("surface", "100_t0") == "done"
    assert "restarting interrupted items: surface/100_t0" in (run_dir / "adsorption.log").read_text()


def test_resume_refuses_changed_settings(tmp_path: Path):
    config = _config(_inputs(tmp_path), budget={"max_steps": 5})
    prepare_surfaces(config)
    resolved = tmp_path / "CO_on_Cu_emt" / "config.resolved.yaml"
    resolved.write_text(resolved.read_text().replace("fix_fraction: 0.5", "fix_fraction: 0.3"))
    with pytest.raises(StateError, match="new run"):
        prepare_surfaces(config, resume=True)


def test_precut_slab_input_with_vacuum_along_x(tmp_path: Path):
    slab = fcc111("Cu", (2, 2, 4), vacuum=6.0)
    slab.pbc = True
    rotated = slab.copy()
    rotated.set_cell(slab.cell.array[[2, 0, 1]][:, [2, 0, 1]], scale_atoms=False)
    rotated.positions = slab.positions[:, [2, 0, 1]]
    _inputs(tmp_path, solid=rotated)
    models = prepare_surfaces(_config(tmp_path))
    assert [model.term_id for model in models] == ["slab_t0"]
    model = models[0]
    assert model.miller is None and model.surface_energy is None
    relaxed = model.slab()
    assert np.allclose(relaxed.cell[2, :2], 0) and relaxed.cell[2, 2] > 15
    assert not (tmp_path / "CO_on_Cu_emt" / "bulk").exists()
    assert "slab (input_type auto), vacuum along cell axis 0" in (tmp_path / "CO_on_Cu_emt" / "adsorption.log").read_text()


def test_unsupported_elements_are_rejected(tmp_path: Path):
    tisi = bulk("Ti", "hcp", a=2.95, c=4.68)
    _inputs(tmp_path, solid=tisi)
    with pytest.raises(ConfigError, match="Ti"):
        prepare_surfaces(_config(tmp_path))
