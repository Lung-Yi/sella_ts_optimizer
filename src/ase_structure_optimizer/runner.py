"""Run structure optimizations with ASE-compatible calculators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ase import Atoms
from ase.io import read, write

from .atoms import apply_charge_and_multiplicity
from .calculators import CalculatorConfig, build_calculator


@dataclass(frozen=True)
class OptimizationResult:
    """Paths and metadata produced by a structure optimization."""

    trajectory: Path
    optimized_xyz: Path
    final_xyz: Path
    converged: bool
    steps: int
    mode: str = "ts"
    optimizer: str = "sella"

    @property
    def path_xyz(self) -> Path:
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
    atoms = _prepare_atoms(atoms, calculator_config)

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
) -> OptimizationResult:
    """Optimize an ASE Atoms object toward a local minimum.

    The input Atoms object is copied before the calculator is attached, so the
    caller's object is not mutated by the optimization setup.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = _prepare_atoms(atoms, calculator_config)
    optimizer_class = _get_minimizer(optimizer)

    trajectory = output_dir / trajectory_name
    dyn = optimizer_class(atoms, trajectory=str(trajectory))
    converged = dyn.run(fmax=fmax, steps=max_steps)

    return _write_optimization_result(
        output_dir=output_dir,
        trajectory=trajectory,
        optimized_path_name=optimized_path_name,
        final_name=final_name,
        converged=bool(converged),
        mode="min",
        optimizer=optimizer.lower(),
    )


def _prepare_atoms(atoms: Atoms, calculator_config: CalculatorConfig) -> Atoms:
    atoms = atoms.copy()
    atoms = apply_charge_and_multiplicity(
        atoms,
        charge=calculator_config.charge,
        multiplicity=calculator_config.multiplicity,
    )
    atoms.calc = build_calculator(calculator_config)
    return atoms


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


def _write_optimization_result(
    output_dir: Path,
    trajectory: Path,
    optimized_path_name: str,
    final_name: str,
    converged: bool,
    mode: OptimizationMode,
    optimizer: str,
) -> OptimizationResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_images = read(trajectory, index=":")
    optimized_xyz = output_dir / optimized_path_name
    final_xyz = output_dir / final_name
    write(optimized_xyz, optimized_images, format="xyz")
    write(final_xyz, optimized_images[-1], format="xyz")

    return OptimizationResult(
        trajectory=trajectory,
        optimized_xyz=optimized_xyz,
        final_xyz=final_xyz,
        converged=converged,
        steps=len(optimized_images),
        mode=mode,
        optimizer=optimizer,
    )
