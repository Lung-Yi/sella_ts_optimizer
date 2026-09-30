"""Tests for atoms preparation and the optimization runner (shared core, M0)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read

import ase_structure_optimizer as package
from ase_structure_optimizer import CalculatorConfig, optimize_geometry_atoms, optimize_ts_atoms
from ase_structure_optimizer.atoms import apply_charge_and_multiplicity, prepare_atoms
from regression_cases import cu5_ts_guess, water

EMT_CONFIG = CalculatorConfig(name="emt")


# Parameter lists of the public API on main (55b56da). New parameters may only
# be appended, with defaults.
ORIGINAL_SIGNATURES = {
    "optimize_geometry_atoms": [
        ("atoms", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("optimizer", "bfgs"),
        ("trajectory_name", "geometry_min.traj"),
        ("optimized_path_name", "geometry_min_path.xyz"),
        ("final_name", "geometry_min_optimized.xyz"),
    ],
    "run_geometry_optimization": [
        ("xyz_path", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("optimizer", "bfgs"),
        ("trajectory_name", "geometry_min.traj"),
        ("optimized_path_name", "geometry_min_path.xyz"),
        ("final_name", "geometry_min_optimized.xyz"),
    ],
    "optimize_ts_atoms": [
        ("atoms", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("trajectory_name", "sella_ts.traj"),
        ("optimized_path_name", "sella_ts_path.xyz"),
        ("final_name", "sella_ts_optimized.xyz"),
    ],
    "run_ts_optimization": [
        ("xyz_path", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("trajectory_name", "sella_ts.traj"),
        ("optimized_path_name", "sella_ts_path.xyz"),
        ("final_name", "sella_ts_optimized.xyz"),
    ],
    "optimize_irc_atoms": [
        ("atoms", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("dx", 0.1),
        ("direction", "both"),
        ("forward_trajectory_name", "sella_irc_forward.traj"),
        ("reverse_trajectory_name", "sella_irc_reverse.traj"),
        ("forward_path_name", "sella_irc_forward.xyz"),
        ("reverse_path_name", "sella_irc_reverse.xyz"),
        ("full_path_name", "sella_irc_path.xyz"),
    ],
    "run_irc": [
        ("xyz_path", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("fmax", 5e-3),
        ("max_steps", 200),
        ("dx", 0.1),
        ("direction", "both"),
        ("forward_trajectory_name", "sella_irc_forward.traj"),
        ("reverse_trajectory_name", "sella_irc_reverse.traj"),
        ("forward_path_name", "sella_irc_forward.xyz"),
        ("reverse_path_name", "sella_irc_reverse.xyz"),
        ("full_path_name", "sella_irc_path.xyz"),
    ],
    "analyze_frequencies_atoms": [
        ("atoms", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("structure_label", "ASE Atoms"),
        ("delta", 0.01),
        ("temperature", 300.0),
        ("pressure", 1.0),
    ],
    "run_frequency_analysis": [
        ("xyz_path", inspect.Parameter.empty),
        ("calculator_config", inspect.Parameter.empty),
        ("output_dir", inspect.Parameter.empty),
        ("delta", 0.01),
        ("temperature", 300.0),
        ("pressure", 1.0),
    ],
    "build_calculator": [("config", inspect.Parameter.empty)],
    "available_calculators": [],
    "available_minimizers": [],
}

ORIGINAL_ALL = {
    "CalculatorConfig",
    "FrequencyResult",
    "IRCResult",
    "OptimizationResult",
    "analyze_frequencies_atoms",
    "available_calculators",
    "available_minimizers",
    "build_calculator",
    "optimize_geometry_atoms",
    "optimize_irc_atoms",
    "optimize_ts_atoms",
    "run_geometry_optimization",
    "run_frequency_analysis",
    "run_irc",
    "run_ts_optimization",
}


def test_original_public_api_still_exported():
    assert ORIGINAL_ALL <= set(package.__all__)
    for name in package.__all__:
        assert hasattr(package, name), name


@pytest.mark.parametrize("name", sorted(ORIGINAL_SIGNATURES))
def test_public_signatures_only_gain_trailing_defaults(name):
    params = list(inspect.signature(getattr(package, name)).parameters.values())
    original = ORIGINAL_SIGNATURES[name]
    assert [(p.name, p.default) for p in params[: len(original)]] == original
    for extra in params[len(original) :]:
        assert extra.default is not inspect.Parameter.empty, extra.name


# ---------------------------------------------------------------------------
# atoms.py
# ---------------------------------------------------------------------------


def test_charge_and_multiplicity_for_molecules_unchanged():
    atoms = apply_charge_and_multiplicity(water(), charge=-1, multiplicity=3)
    assert atoms.info["charge"] == -1 and atoms.info["spin"] == 3
    assert atoms.get_initial_charges().tolist() == [-1.0, 0.0, 0.0]
    assert atoms.get_initial_magnetic_moments().tolist() == [2.0, 0.0, 0.0]


@pytest.mark.parametrize("pbc", [True, (True, True, False)])
def test_charge_and_multiplicity_for_periodic_systems_only_touch_info(pbc):
    slab = fcc111("Cu", size=(2, 2, 2), vacuum=5.0)
    slab.pbc = pbc
    atoms = apply_charge_and_multiplicity(slab, charge=1, multiplicity=2)
    assert atoms.info["charge"] == 1 and atoms.info["spin"] == 2
    assert "initial_charges" not in atoms.arrays
    assert "initial_magmoms" not in atoms.arrays

    magnetic = bulk("Fe")
    magnetic.set_initial_magnetic_moments([2.2])
    apply_charge_and_multiplicity(magnetic, charge=0, multiplicity=1)
    assert magnetic.get_initial_magnetic_moments().tolist() == [2.2]


def test_prepare_atoms_uses_given_calculator_and_keeps_constraints():
    slab = fcc111("Cu", size=(2, 2, 2), vacuum=5.0)
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    calculator = EMT()
    prepared = prepare_atoms(slab, EMT_CONFIG, calculator=calculator)
    assert prepared.calc is calculator
    assert prepared is not slab and slab.calc is None
    assert prepared.constraints[0].get_indices().tolist() == [0, 1, 2, 3]

    built = prepare_atoms(slab, EMT_CONFIG)
    assert isinstance(built.calc, EMT) and built.calc is not calculator


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------


def _slab_with_adsorbate() -> Atoms:
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.tag == 3]))
    co = molecule("CO")
    co.rotate(180, "x")
    top = slab.positions[:, 2].max()
    co.translate(slab.positions[slab.positions[:, 2].argmax()] + [0.0, 0.0, 2.0] - co.positions[0])
    assert co.positions[:, 2].min() > top
    return slab + co


def test_molecular_outputs_are_plain_xyz_with_energy(tmp_path: Path):
    result = optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path, logfile=None)
    assert result.optimized_xyz == tmp_path / "geometry_min_path.xyz"
    assert result.final_xyz == tmp_path / "geometry_min_optimized.xyz"
    assert "Lattice" not in result.final_xyz.read_text().splitlines()[1]
    assert result.energy == pytest.approx(read(result.trajectory, -1).get_potential_energy())


def test_ts_result_has_energy(tmp_path: Path, capsys):
    result = optimize_ts_atoms(cu5_ts_guess(), EMT_CONFIG, tmp_path, fmax=0.05, max_steps=5)
    capsys.readouterr()
    assert result.energy == pytest.approx(read(result.trajectory, -1).get_potential_energy())


def test_periodic_outputs_keep_cell_pbc_and_fixed_atoms(tmp_path: Path):
    atoms = _slab_with_adsorbate()
    fixed = atoms.constraints[0].get_indices().tolist()
    result = optimize_geometry_atoms(
        atoms, EMT_CONFIG, tmp_path, fmax=0.1, max_steps=30, optimizer="lbfgs", logfile=None
    )

    assert result.final_xyz.name == "geometry_min_optimized.xyz"
    assert result.optimized_xyz.name == "geometry_min_path.xyz"
    assert "Lattice=" in result.final_xyz.read_text().splitlines()[1]

    final = read(result.final_xyz)
    assert final.pbc.tolist() == atoms.pbc.tolist()
    np.testing.assert_allclose(final.cell.array, atoms.cell.array)
    assert final.constraints and final.constraints[0].get_indices().tolist() == fixed
    np.testing.assert_allclose(final.positions[fixed], atoms.positions[fixed])
    assert len(read(result.optimized_xyz, ":")) == result.steps
    assert result.energy == pytest.approx(final.get_potential_energy())
    # Caller's atoms untouched.
    assert atoms.calc is None


def test_cell_filter_relaxes_lattice(tmp_path: Path):
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.rattle(0.01, seed=0)
    result = optimize_geometry_atoms(
        atoms, EMT_CONFIG, tmp_path, fmax=0.01, max_steps=100, cell_filter=True, logfile=None
    )
    images = read(result.trajectory, ":")
    # The trajectory holds the underlying Atoms (not the filter), with its cell.
    assert all(len(image) == len(atoms) for image in images)
    assert images[0].cell.lengths() == pytest.approx([3.8] * 3, abs=1e-6)
    final = read(result.final_xyz)
    assert final.cell.lengths() == pytest.approx(images[-1].cell.lengths())
    assert final.cell.lengths()[0] < 3.7  # EMT Cu lattice constant is ~3.59 Å
    assert result.converged


def test_cell_filter_rejects_molecules(tmp_path: Path):
    with pytest.raises(ValueError, match="periodic"):
        optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path, cell_filter=True)


def test_without_cell_filter_lattice_is_fixed(tmp_path: Path):
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.rattle(0.01, seed=0)
    result = optimize_geometry_atoms(atoms, EMT_CONFIG, tmp_path, fmax=0.01, logfile=None)
    assert read(result.final_xyz).cell.lengths() == pytest.approx([3.8] * 3)


def test_write_outputs_false_writes_only_trajectory(tmp_path: Path):
    result = optimize_geometry_atoms(
        _slab_with_adsorbate(), EMT_CONFIG, tmp_path, fmax=0.2, max_steps=5,
        write_outputs=False, logfile=None,
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["geometry_min.traj"]
    assert result.optimized_xyz is None and result.final_xyz is None and result.path_xyz is None
    assert result.energy is not None and result.steps >= 1


def test_logfile_none_is_silent_and_default_logs(tmp_path: Path, capsys):
    optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path / "quiet", logfile=None)
    assert capsys.readouterr().out == ""
    optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path / "loud")
    assert "BFGS" in capsys.readouterr().out

    logfile = tmp_path / "opt.log"
    optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path / "file", logfile=logfile)
    assert "BFGS" in logfile.read_text()


def test_given_calculator_is_used(tmp_path: Path):
    class CountingEMT(EMT):
        calls = 0

        def calculate(self, *args, **kwargs):
            type(self).calls += 1
            super().calculate(*args, **kwargs)

    calculator = CountingEMT()
    result = optimize_geometry_atoms(water(), EMT_CONFIG, tmp_path, calculator=calculator, logfile=None)
    assert CountingEMT.calls >= result.steps


def test_fixatoms_survive_copy():
    atoms = _slab_with_adsorbate()
    copied = atoms.copy()
    assert copied.constraints[0].get_indices().tolist() == atoms.constraints[0].get_indices().tolist()
