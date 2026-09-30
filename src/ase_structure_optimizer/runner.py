"""Run structure optimizations with ASE-compatible calculators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ase import Atoms
from ase.io import read, write

from .atoms import prepare_atoms
from .calculators import CalculatorConfig


@dataclass(frozen=True)
class OptimizationResult:
    """Paths and metadata produced by a structure optimization."""

    trajectory: Path
    optimized_xyz: Path | None
    final_xyz: Path | None
    converged: bool
    steps: int
    mode: str = "ts"
    optimizer: str = "sella"
    energy: float | None = None

    @property
    def path_xyz(self) -> Path | None:
        """Alias for the XYZ file containing all saved optimization images."""

        return self.optimized_xyz


OptimizationMode = Literal["ts", "min"]


def available_minimizers() -> tuple[str, ...]:
    """Return ASE minimizers supported for ordinary geometry optimization."""

    return ("bfgs", "lbfgs", "fire")


def run_ts_optimization(
    xyz_path: Path,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    trajectory_name: str = "sella_ts.traj",
    optimized_path_name: str = "sella_ts_path.xyz",
    final_name: str = "sella_ts_optimized.xyz",
) -> OptimizationResult:
    """Optimize a transition-state guess from an XYZ file with Sella."""

    return optimize_ts_atoms(
        atoms=read(xyz_path),
        calculator_config=calculator_config,
        output_dir=output_dir,
        fmax=fmax,
        max_steps=max_steps,
        trajectory_name=trajectory_name,
        optimized_path_name=optimized_path_name,
        final_name=final_name,
    )


def optimize_ts_atoms(
    atoms: Atoms,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    trajectory_name: str = "sella_ts.traj",
    optimized_path_name: str = "sella_ts_path.xyz",
    final_name: str = "sella_ts_optimized.xyz",
) -> OptimizationResult:
    """Optimize an ASE Atoms transition-state guess with Sella.

    The input Atoms object is copied before the calculator is attached, so the
    caller's object is not mutated by the optimization setup.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = prepare_atoms(atoms, calculator_config)

    from sella import Sella

    trajectory = output_dir / trajectory_name
    dyn = Sella(atoms, trajectory=str(trajectory))
    converged = dyn.run(fmax, max_steps)

    return _write_optimization_result(
        output_dir=output_dir,
        trajectory=trajectory,
        optimized_path_name=optimized_path_name,
        final_name=final_name,
        converged=bool(converged),
        mode="ts",
        optimizer="sella",
    )


def run_geometry_optimization(
    xyz_path: Path,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    optimizer: str = "bfgs",
    trajectory_name: str = "geometry_min.traj",
    optimized_path_name: str = "geometry_min_path.xyz",
    final_name: str = "geometry_min_optimized.xyz",
) -> OptimizationResult:
    """Optimize an ordinary molecular geometry from an XYZ file."""

    return optimize_geometry_atoms(
        atoms=read(xyz_path),
        calculator_config=calculator_config,
        output_dir=output_dir,
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
        trajectory_name=trajectory_name,
        optimized_path_name=optimized_path_name,
        final_name=final_name,
    )


def optimize_geometry_atoms(
    atoms: Atoms,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    optimizer: str = "bfgs",
    trajectory_name: str = "geometry_min.traj",
    optimized_path_name: str = "geometry_min_path.xyz",
    final_name: str = "geometry_min_optimized.xyz",
    calculator: Any | None = None,
    cell_filter: bool = False,
    logfile: str | Path | None = "-",
    write_outputs: bool = True,
) -> OptimizationResult:
    """Optimize an ASE Atoms object toward a local minimum.

    The input Atoms object is copied before the calculator is attached, so the
    caller's object is not mutated by the optimization setup. Constraints such
    as `FixAtoms` are kept.

    Args:
        calculator: Calculator instance to attach instead of building a new
            one from `calculator_config` (e.g. from `get_calculator()`).
        cell_filter: Also relax the cell of a periodic system, using ASE's
            `FrechetCellFilter` (`ExpCellFilter` on older ASE versions).
        logfile: Optimizer log destination; `"-"` is stdout, `None` disables it.
        write_outputs: When False only the trajectory is written; the path and
            final XYZ files are skipped and reported as `None`.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = prepare_atoms(atoms, calculator_config, calculator=calculator)
    optimizer_class = _get_minimizer(optimizer)

    target = atoms
    if cell_filter:
        if not atoms.pbc.any():
            raise ValueError("cell_filter=True requires a periodic system (atoms.pbc has no True entry)")
        target = _wrap_cell_filter(atoms)

    trajectory = output_dir / trajectory_name
    dyn = optimizer_class(target, trajectory=str(trajectory), logfile=logfile)
    converged = dyn.run(fmax=fmax, steps=max_steps)

    return _write_optimization_result(
        output_dir=output_dir,
        trajectory=trajectory,
        optimized_path_name=optimized_path_name,
        final_name=final_name,
        converged=bool(converged),
        mode="min",
        optimizer=optimizer.lower(),
        write_outputs=write_outputs,
    )


def _get_minimizer(name: str):
    from ase.optimize import BFGS, FIRE, LBFGS

    minimizers = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "fire": FIRE,
    }
    key = name.lower()
    try:
        return minimizers[key]
    except KeyError as exc:
        choices = ", ".join(available_minimizers())
        raise ValueError(
            f"Unknown geometry optimizer '{name}'. Choose one of: {choices}"
        ) from exc


def _wrap_cell_filter(atoms: Atoms):
    """Wrap atoms in a filter that exposes cell degrees of freedom."""

    try:
        from ase.filters import FrechetCellFilter
    except ImportError:
        from ase.constraints import ExpCellFilter

        return ExpCellFilter(atoms)
    return FrechetCellFilter(atoms)


def _final_energy(image: Atoms) -> float | None:
    """Energy stored with a trajectory image, without triggering a calculation."""

    from ase.calculators.calculator import PropertyNotImplementedError

    if image.calc is None:
        return None
    try:
        return float(image.get_potential_energy())
    except PropertyNotImplementedError:
        return None


def _write_optimization_result(
    output_dir: Path,
    trajectory: Path,
    optimized_path_name: str,
    final_name: str,
    converged: bool,
    mode: OptimizationMode,
    optimizer: str,
    write_outputs: bool = True,
) -> OptimizationResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_images = read(trajectory, index=":")
    optimized_xyz: Path | None = None
    final_xyz: Path | None = None
    if write_outputs:
        # Periodic systems need extxyz to keep the cell, pbc and fixed atoms.
        output_format = "extxyz" if optimized_images[-1].pbc.any() else "xyz"
        optimized_xyz = output_dir / optimized_path_name
        final_xyz = output_dir / final_name
        write(optimized_xyz, optimized_images, format=output_format)
        write(final_xyz, optimized_images[-1], format=output_format)

    return OptimizationResult(
        trajectory=trajectory,
        optimized_xyz=optimized_xyz,
        final_xyz=final_xyz,
        converged=converged,
        steps=len(optimized_images),
        mode=mode,
        optimizer=optimizer,
        energy=_final_energy(optimized_images[-1]),
    )
