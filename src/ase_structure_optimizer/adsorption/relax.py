"""Calculator selection, relaxations and single points for the workflow.

All optimizations go through the shared `runner.optimize_geometry_atoms()`
and all calculators come from the shared `calculators.get_calculator()`
cache, so each model is loaded once per run.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from ..calculators import CalculatorConfig, get_calculator, model_label
from ..runner import optimize_geometry_atoms
from ..structures import write_trajectory_pair
from .config import AdsorptionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalculatorSet:
    """The two calculator settings of a run; they differ only in dtype."""

    screen: CalculatorConfig
    final: CalculatorConfig

    @property
    def model(self) -> str:
        """Model actually used (model name or file path, else the calculator name)."""

        return model_label(self.final)


@dataclass(frozen=True)
class RelaxResult:
    """Outcome of `relax_structure()`."""

    atoms: Atoms
    energy_screen: float
    energy_final: float
    converged: bool
    steps: int
    trajectory: Path
    extxyz: Path


def resolve_calculators(config: AdsorptionConfig) -> CalculatorSet:
    """Load the run's calculators, trying `fallback_models` for macemp.

    The final-dtype calculator is loaded first; if the configured macemp
    model cannot be loaded (e.g. an unknown model name for the installed
    mace-torch version, or no download possible), each fallback model is
    tried in order. Raises RuntimeError when nothing can be loaded.
    """

    section = config.calculator
    models: list[str | None] = [None]
    if section.name == "macemp":
        models = [section.mace_mp_model, *section.fallback_models]

    errors = []
    for model in models:
        final = config.calculator_config("final", mace_mp_model=model)
        try:
            get_calculator(final)
        except Exception as exc:  # noqa: BLE001 - any loading failure means "try the next model"
            errors.append(f"{model or section.name}: {type(exc).__name__}: {exc}")
            logger.warning("could not load %s model %s: %s", section.name, model, exc)
            continue
        screen = config.calculator_config("screen", mace_mp_model=model)
        if model is not None and model != section.mace_mp_model:
            logger.warning("using fallback %s model %s", section.name, model)
        return CalculatorSet(screen=screen, final=final)
    raise RuntimeError("no calculator model could be loaded:\n  " + "\n  ".join(errors))


def single_point(atoms: Atoms, config: CalculatorConfig) -> Atoms:
    """Copy of `atoms` with energy and forces from `config` stored in a SinglePointCalculator."""

    result = atoms.copy()
    result.calc = get_calculator(config)
    energy = float(result.get_potential_energy())
    forces = result.get_forces()
    _check_finite(energy, forces)
    result.calc = SinglePointCalculator(result, energy=energy, forces=forces)
    return result


def relax_structure(
    atoms: Atoms,
    calculators: CalculatorSet,
    output_dir: Path,
    stem: str,
    optimizer: str,
    fmax: float,
    max_steps: int,
    cell_filter: bool = False,
    logfile: Path | None = None,
) -> RelaxResult:
    """Relax with the screen calculator, then a final-dtype single point.

    Writes ``<stem>.traj`` and ``<stem>.extxyz`` (all optimization frames)
    and ``<stem>_final.extxyz`` (the relaxed structure with its final-dtype
    energy) to `output_dir`.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    result = optimize_geometry_atoms(
        atoms,
        calculators.screen,
        output_dir,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        trajectory_name=f"{stem}.traj",
        calculator=get_calculator(calculators.screen),
        cell_filter=cell_filter,
        logfile=str(logfile) if logfile is not None else None,
        write_outputs=False,
    )
    images = read(result.trajectory, index=":")
    traj_path, extxyz_path = write_trajectory_pair(images, output_dir / stem)

    relaxed = images[-1]
    relaxed.set_constraint(atoms.constraints)  # trajectories keep them, but be explicit
    energy_screen = float(result.energy) if result.energy is not None else math.nan
    final = single_point(relaxed, calculators.final)
    write(output_dir / f"{stem}_final.extxyz", final, format="extxyz")
    return RelaxResult(
        atoms=final,
        energy_screen=energy_screen,
        energy_final=float(final.get_potential_energy()),
        converged=result.converged,
        steps=result.steps,
        trajectory=traj_path,
        extxyz=extxyz_path,
    )


# Failure criteria for adsorption relaxations.
MIN_INTERATOMIC_DISTANCE = 0.5
MAX_DESORPTION_DISTANCE = 10.0


@dataclass(frozen=True)
class CandidateOutcome:
    """Result of optimizing one adsorption configuration."""

    status: str
    reason: str
    atoms: Atoms | None
    energy: float | None
    converged: bool
    steps: int
    elapsed: float
    trajectory: Path | None


def structure_problem(atoms: Atoms, n_slab: int) -> str:
    """Why an optimized adsorption structure is unusable, or ``""``.

    Checks for non-finite energy/forces, atoms closer than 0.5 Å, and a
    molecule more than 10 Å away from the slab.
    """

    from ase.neighborlist import neighbor_list

    from .analysis import min_molecule_slab_distance

    try:
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
    except Exception as exc:  # noqa: BLE001 - missing results are a failure too
        return f"no energy/forces: {exc}"
    if not math.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(np.isfinite(atoms.positions)):
        return "non-finite energy, forces or positions"
    if len(neighbor_list("i", atoms, MIN_INTERATOMIC_DISTANCE)):
        return f"atoms closer than {MIN_INTERATOMIC_DISTANCE} Å"
    distance = min_molecule_slab_distance(atoms, n_slab)
    if distance > MAX_DESORPTION_DISTANCE:
        return f"molecule left the surface ({distance:.1f} Å > {MAX_DESORPTION_DISTANCE} Å)"
    return ""


def optimize_candidate(
    atoms: Atoms,
    calculators: CalculatorSet,
    output_dir: Path,
    name: str,
    optimizer: str,
    fmax: float,
    max_steps: int,
    n_slab: int,
    logfile: Path | None = None,
    write_pair: bool = True,
) -> CandidateOutcome:
    """Optimize one adsorption configuration with the screen calculator.

    Never raises for calculation problems: exceptions and unusable results
    (see `structure_problem()`) give ``status="failed"`` with a reason.
    Writes ``<name>.traj`` (and ``<name>.extxyz`` if `write_pair`).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trajectory = output_dir / f"{name}.traj"
    try:
        result = optimize_geometry_atoms(
            atoms,
            calculators.screen,
            output_dir,
            fmax=fmax,
            max_steps=max_steps,
            optimizer=optimizer,
            trajectory_name=trajectory.name,
            calculator=get_calculator(calculators.screen),
            logfile=str(logfile) if logfile is not None else None,
            write_outputs=False,
        )
        images = read(trajectory, index=":")
    except Exception as exc:  # noqa: BLE001 - one bad configuration must not stop the run
        logger.warning("%s failed: %s: %s", name, type(exc).__name__, exc)
        return CandidateOutcome(
            "failed", f"{type(exc).__name__}: {exc}", None, None, False, 0, time.perf_counter() - started, None
        )

    final = images[-1]
    final.set_constraint(atoms.constraints)
    if write_pair:
        write_trajectory_pair(images, output_dir / name)
    problem = structure_problem(final, n_slab)
    elapsed = time.perf_counter() - started
    if problem:
        return CandidateOutcome("failed", problem, final, None, result.converged, result.steps, elapsed, trajectory)
    return CandidateOutcome(
        "done", "", final, float(final.get_potential_energy()), result.converged, result.steps, elapsed, trajectory
    )


def _check_finite(energy: float, forces: np.ndarray) -> None:
    if not math.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise FloatingPointError("calculator returned a non-finite energy or forces")

