"""End-to-end tests of the MLIP workflow up to the full relaxations (EMT, Cu(111) + CO)."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

pytest.importorskip("pymatgen")
pytest.importorskip("yaml")

from ase_structure_optimizer.adsorption import workflow  # noqa: E402
from ase_structure_optimizer.adsorption import relax as relax_module  # noqa: E402
from ase_structure_optimizer.adsorption.cli import main  # noqa: E402
from ase_structure_optimizer.adsorption.config import config_from_dict  # noqa: E402
from ase_structure_optimizer.adsorption.relax import structure_problem  # noqa: E402
from ase_structure_optimizer.adsorption.state import RunState  # noqa: E402
from ase_structure_optimizer.adsorption.workflow import (  # noqa: E402
    resume_adsorption_workflow,
    run_adsorption_workflow,
)


def _inputs(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "CO.xyz", molecule("CO"))
    write(directory / "Cu.cif", bulk("Cu", "fcc", a=3.6))
    return directory


def _config(directory: Path, **overrides):
    data = {
        "molecule": "CO.xyz",
        "solid": "Cu.cif",
        "calculator": {"name": "emt"},
        "surface": {"miller_indices": [[1, 1, 1]], "min_slab_thickness": 6.0, "lateral_buffer": 7.0},
        "sampling": {"n_random": 4, "spins_per_anchor": 1},
        "budget": {"wall_time_per_termination": 0, "max_steps": 120, "min_full_relax": 3},
        "analysis": {"make_gif": False},
    }
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return config_from_dict(data, base_dir=_inputs(directory))


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_full_run_outputs_and_c_down_adsorption(tmp_path: Path):
    started = time.perf_counter()
    result = run_adsorption_workflow(_config(tmp_path))
    assert time.perf_counter() - started < 120
    run_dir = tmp_path / "CO_on_Cu_emt"
    assert result.run_dir == run_dir.resolve()

    for name in ("gas_opt.traj", "gas_opt.extxyz", "gas_opt_final.extxyz", "gas_reference.json", "molecule_analysis.json"):
        assert (run_dir / "molecule" / name).is_file(), name
    term_dir = run_dir / "terminations" / "111_t0"
    for name in ("candidates.json", "prescreen.csv", "results.csv"):
        assert (term_dir / name).is_file(), name

    (termination,) = result.terminations
    candidates = sorted((term_dir / "candidates").glob("*.extxyz"))
    assert len(candidates) == termination.n_candidates
    assert termination.n_prescreened == termination.n_candidates  # unlimited budget: all prescreened
    assert termination.n_relaxed >= 3 and termination.n_failed == 0
    assert result.best is termination

    rows = _rows(term_dir / "results.csv")
    assert len(rows) == termination.n_relaxed
    energies = [float(row["eads"]) for row in rows]
    assert energies == sorted(energies)
    for row in rows:
        assert row["class"] in ("chemisorbed", "physisorbed", "dissociated", "desorbed")
        assert abs(float(row["eads"]) - float(row["eads_screen"])) < 1e-6  # EMT has no dtype
        assert (term_dir / "relax" / f"{row['config_id']}_final.extxyz").is_file()

    # M4 outputs: unique configurations, summary, scan, figures.
    unique = _rows(term_dir / "unique.csv")
    assert len(unique) == termination.n_unique >= 1
    assert unique[0]["config_id"] == termination.best_unique_id
    assert float(unique[0]["eads"]) == pytest.approx(termination.best_eads)
    assert sum(float(row["boltzmann_weight"]) for row in unique) == pytest.approx(1.0, abs=1e-4)
    assert sum(int(row["n_duplicates"]) for row in unique) + len(unique) == len(rows)
    summary = _rows(run_dir / "summary.csv")
    assert [row["config_id"] for row in summary] == [row["config_id"] for row in unique]
    scan = _rows(term_dir / "approach_scan.csv")
    assert len(scan) == 23 and (term_dir / "approach_scan.extxyz").is_file()
    assert len(read(term_dir / "approach_scan.extxyz", index=":")) == 23
    assert read(term_dir / "approach_scan.extxyz", index=0).get_potential_energy() == pytest.approx(
        float(scan[0]["energy_final"])
    )  # scan energies are stored with the structures
    figures = {path.name for path in (run_dir / "figures").glob("*.png")}
    for name in ("eads_ranking_111_t0.png", "eads_vs_tilt_111_t0.png", "eads_site_anchor_heatmap_111_t0.png"):
        assert name in figures
    assert 1 <= len([name for name in figures if name.startswith("summary_111_t0_")]) <= 3
    assert result.summary_csv == run_dir.resolve() / "summary.csv"

    slab = read(term_dir / "slab_opt_final.extxyz")
    n_slab = len(slab)
    c_down = 0
    for row in rows:
        final = read(term_dir / "relax" / f"{row['config_id']}.extxyz", index=-1)
        assert (term_dir / "relax" / f"{row['config_id']}.traj").is_file()
        assert len(final) == n_slab + 2
        fixed = final.constraints[0].get_indices()
        np.testing.assert_allclose(final.positions[fixed], slab.positions[fixed], atol=1e-8)
        oxygen, carbon = final.positions[n_slab], final.positions[n_slab + 1]
        if carbon[2] < oxygen[2] - 0.5 and row["contact"] == "CO":
            c_down += 1
    assert c_down >= 1

    from ase.db import connect

    with connect(run_dir / "results.db") as database:
        assert database.count(kind="adsorbate") == termination.n_relaxed
        assert database.count(kind="molecule") == 1
        row = database.get(name=f"111_t0/{termination.best_unique_id}")
        assert row.eads == pytest.approx(termination.best_eads)
        assert row.adsorption_class == termination.best_class


def test_interrupted_run_resumes_to_the_same_results(tmp_path: Path, monkeypatch):
    reference = run_adsorption_workflow(_config(tmp_path / "reference"))

    calls = {"n": 0}
    original = workflow.optimize_candidate

    def interrupt(*args, **kwargs):
        if args[2].name == "relax":
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, "optimize_candidate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_adsorption_workflow(_config(tmp_path / "interrupted"))
    monkeypatch.setattr(workflow, "optimize_candidate", original)

    run_dir = tmp_path / "interrupted" / "CO_on_Cu_emt"
    state = RunState.load(run_dir)
    assert "running" in {record["status"] for record in state.items["111_t0/relax"].values()}
    resumed = resume_adsorption_workflow(run_dir)

    expected = {row["config_id"]: row for row in _rows(reference.terminations[0].results_csv)}
    actual = {row["config_id"]: row for row in _rows(resumed.terminations[0].results_csv)}
    assert expected.keys() == actual.keys()
    for cid, row in expected.items():
        assert float(actual[cid]["eads"]) == pytest.approx(float(row["eads"]), abs=1e-8)
        assert actual[cid]["contact"] == row["contact"] and actual[cid]["class"] == row["class"]
    assert _rows(reference.summary_csv) and len(_rows(resumed.summary_csv)) == len(_rows(reference.summary_csv))
    log = (run_dir / "adsorption.log").read_text()
    assert "restarting interrupted items" in log
    # A finished run resumes instantly without new calculations.
    again = resume_adsorption_workflow(run_dir)
    assert again.terminations[0].best_config_id == resumed.terminations[0].best_config_id


def test_failures_are_recorded_and_do_not_stop_the_run(tmp_path: Path, monkeypatch):
    original = relax_module.optimize_geometry_atoms

    def flaky(atoms, calculator_config, output_dir, **kwargs):
        if output_dir.name == "prescreen" and kwargs.get("trajectory_name") == "c0001.traj":
            raise RuntimeError("simulated calculator crash")
        return original(atoms, calculator_config, output_dir, **kwargs)

    monkeypatch.setattr(relax_module, "optimize_geometry_atoms", flaky)
    result = run_adsorption_workflow(_config(tmp_path))
    rows = _rows(result.terminations[0].prescreen_csv)
    failed = [row for row in rows if row["status"] == "failed"]
    assert [row["config_id"] for row in failed] == ["c0001"]
    assert "simulated calculator crash" in failed[0]["reason"]
    assert result.terminations[0].n_relaxed >= 3


def test_small_budget_warns(tmp_path: Path):
    result = run_adsorption_workflow(_config(tmp_path, budget={"wall_time_per_termination": 0.5, "min_full_relax": 50}))
    (termination,) = result.terminations
    assert termination.n_relaxed >= 1  # at least one full relaxation always runs
    assert any("increase budget.wall_time_per_termination" in warning for warning in termination.warnings)
    assert "min_full_relax" in (result.log_file).read_text()


def test_structure_problems():
    slab = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 30], pbc=True)

    def system(positions, energy=-1.0):
        atoms = slab + Atoms("CO", positions=positions)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((4, 3)))
        return atoms

    assert structure_problem(system([[0, 0, 2.0], [0, 0, 3.15]]), 2) == ""
    assert "closer than" in structure_problem(system([[0, 0, 0.3], [0, 0, 1.45]]), 2)
    assert "left the surface" in structure_problem(system([[0, 0, 12.0], [0, 0, 13.15]]), 2)
    assert "non-finite" in structure_problem(system([[0, 0, 2.0], [0, 0, 3.15]], energy=float("nan")), 2)
    no_calc = slab + Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.15]])
    assert "no energy" in structure_problem(no_calc, 2)


def test_cli_run_quick_form_and_resume(tmp_path: Path, capsys):
    _inputs(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "molecule: CO.xyz\nsolid: Cu.cif\n"
        "surface: {min_slab_thickness: 6.0, lateral_buffer: 7.0}\n"
        "sampling: {n_random: 2, spins_per_anchor: 1}\n"
        "budget: {max_steps: 60}\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "out"
    code = main(
        [
            "run",
            str(config),
            "--calculator",
            "emt",
            "--miller",
            "1",
            "1",
            "1",
            "--budget",
            "0",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "111_t0" in output and "most stable" in output and "Summary:" in output
    resolved = (run_dir / "config.resolved.yaml").read_text()
    assert "name: emt" in resolved and "wall_time_per_termination: 0.0" in resolved

    assert main(["resume", str(run_dir)]) == 0
    assert main(["run", str(config), "--calculator", "emt", "--run-dir", str(run_dir)]) == 1  # already a run
    assert "already contains a run" in capsys.readouterr().err
    assert main(["run", "--molecule", str(tmp_path / "CO.xyz"), "--solid", str(tmp_path / "Cu.cif"),
                 "--calculator", "maceomol", "--run-dir", str(tmp_path / "x")]) == 1
    assert "molecular model" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["run", "--molecule", str(tmp_path / "CO.xyz")])


def test_gas_bond_change_keeps_input_geometry(tmp_path: Path, monkeypatch, caplog):
    import dataclasses
    import json

    original = workflow.relax_structure

    def slipping(atoms, calculators, output_dir, stem, **kwargs):
        result = original(atoms, calculators, output_dir, stem, **kwargs)
        if stem != "gas_opt":
            return result
        broken = result.atoms.copy()
        broken.positions[1] += [3.0, 0.0, 0.0]  # pull C away from O: "MLIP artifact"
        broken.calc = result.atoms.calc
        return dataclasses.replace(result, atoms=broken)

    monkeypatch.setattr(workflow, "relax_structure", slipping)
    result = run_adsorption_workflow(_config(tmp_path, sampling={"n_random": 0}))
    run_dir = result.run_dir
    reference = json.loads((run_dir / "molecule" / "gas_reference.json").read_text())
    assert reference["bonds_changed"] is True and reference["geometry"] == "input"
    assert "input geometry is used for sampling" in (run_dir / "adsorption.log").read_text()
    # Candidates use the intact input CO (bond length preserved).
    candidate = read(sorted((run_dir / "terminations" / "111_t0" / "candidates").glob("*.extxyz"))[0])
    assert candidate.get_distance(-1, -2) == pytest.approx(molecule("CO").get_distance(0, 1))


def test_facing_label():
    from ase_structure_optimizer.adsorption.analysis import contact_label, facing_label
    from ase_structure_optimizer.adsorption.molecule import analyze_molecule

    slab = Atoms("Cu4", positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]], cell=[5, 5, 30], pbc=True)
    water_co = Atoms("COHOH", positions=[[0, 0, 1.13], [0, 0, 0], [3, 3, 0], [3, 3, 0.96], [3.9, 3, 1.2]])
    analysis = analyze_molecule(water_co)
    assert [f.name for f in analysis.fragments] == ["CO", "H2O"]
    touching = slab + water_co.copy()
    touching.positions[4:6] += [0, 0, 1.9]  # C 1.9 Å above Cu
    touching.positions[6:] += [0, 0, 6.0]  # water far above
    assert contact_label(touching, 4, analysis, 1.25) == "CO"
    assert facing_label(touching, 4, analysis, 1.25) == "CO"
    floating = slab + water_co.copy()
    floating.positions[4:] += [0, 0, 5.0]
    floating.positions[6:] -= [0, 0, 1.5]  # water lower than CO, both out of contact
    assert contact_label(floating, 4, analysis, 1.25) == "none"
    assert facing_label(floating, 4, analysis, 1.25) == "~H2O"


def test_moved_run_directory_can_be_resumed(tmp_path: Path):
    import shutil

    config = _config(tmp_path / "original", sampling={"n_random": 0})
    first = run_adsorption_workflow(config)
    moved = tmp_path / "elsewhere" / "run"
    moved.parent.mkdir()
    shutil.move(str(first.run_dir), moved)
    shutil.rmtree(tmp_path / "original")  # the original inputs are gone too
    state = RunState.load(moved)
    state.set_item("111_t0/final", first.terminations[0].best_unique_id, "running")
    resumed = resume_adsorption_workflow(moved)
    assert resumed.run_dir == moved.resolve()
    assert resumed.terminations[0].best_eads == pytest.approx(first.terminations[0].best_eads)


def test_report_with_animations_and_cli_regeneration(tmp_path: Path, capsys):
    import re

    result = run_adsorption_workflow(_config(tmp_path, analysis={"make_gif": True}, sampling={"n_random": 2}))
    run_dir = result.run_dir
    assert result.report == run_dir / "report.html"
    text = result.report.read_text(encoding="utf-8")
    for section in ("overview", "settings", "timing", "molecule", "angles", "term-111_t0", "warnings", "vasp",
                    "animation"):
        assert f"id='{section}'" in text
    assert "data:image/png;base64," in text and "<script" in text  # embedded images and the jshtml player
    for link in re.findall(r"href='#([^']+)'", text):
        assert f"id='{link}'" in text

    figures = run_dir / "figures"
    best = result.terminations[0].best_unique_id
    for name in (f"relaxation_111_t0_{best}.gif", f"relaxation_111_t0_{best}.png", "approach_111_t0.gif",
                 "approach_111_t0.png", "molecule_axis.png", "angle_definition.png"):
        assert (figures / name).is_file(), name
    from PIL import Image

    with Image.open(figures / "approach_111_t0.gif") as gif:
        assert 1 < gif.n_frames <= 80

    # Regenerate without calculations; figures are redrawn.
    (run_dir / "report.html").unlink()
    (figures / "eads_ranking_111_t0.png").unlink()
    assert main(["report", str(run_dir), "--no-animations"]) == 0
    assert "Report:" in capsys.readouterr().out
    assert (run_dir / "report.html").is_file() and (figures / "eads_ranking_111_t0.png").is_file()
    assert "id='animation'" not in (run_dir / "report.html").read_text(encoding="utf-8")
    assert main(["report", str(tmp_path / "nope")]) == 1
    assert not (tmp_path / "nope").exists()
