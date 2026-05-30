"""Run Sella transition-state optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from sella import Sella

from .atoms import apply_charge_and_multiplicity
from .calculators import CalculatorConfig, build_calculator


@dataclass(frozen=True)
class OptimizationResult:
    """Paths produced by a Sella TS optimization."""

    trajectory: Path
    optimized_xyz: Path
    final_xyz: Path
    converged: bool
    steps: int


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

    atoms = atoms.copy()
    atoms = apply_charge_and_multiplicity(
        atoms,
        charge=calculator_config.charge,
        multiplicity=calculator_config.multiplicity,
    )
    atoms.calc = build_calculator(calculator_config)

    trajectory = output_dir / trajectory_name
    dyn = Sella(atoms, trajectory=str(trajectory))
    converged = dyn.run(fmax, max_steps)

    optimized_images = read(trajectory, index=":")
    optimized_xyz = output_dir / optimized_path_name
    final_xyz = output_dir / final_name
    write(optimized_xyz, optimized_images, format="xyz")
    write(final_xyz, optimized_images[-1], format="xyz")

    return OptimizationResult(
        trajectory=trajectory,
        optimized_xyz=optimized_xyz,
        final_xyz=final_xyz,
        converged=bool(converged),
        steps=len(optimized_images),
    )
