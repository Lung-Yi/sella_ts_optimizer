"""Intrinsic reaction coordinate (IRC) path tracing with Sella."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ase import Atoms
from ase.io import read, write

from .atoms import prepare_atoms
from .calculators import CalculatorConfig

IRCDirection = Literal["forward", "reverse", "both"]


@dataclass(frozen=True)
class IRCResult:
    """Paths and metadata produced by an IRC calculation."""

    forward_trajectory: Path | None
    reverse_trajectory: Path | None
    forward_xyz: Path | None
    reverse_xyz: Path | None
    full_path_xyz: Path
    forward_converged: bool | None
    reverse_converged: bool | None
    forward_steps: int | None
    reverse_steps: int | None
    direction: IRCDirection
    optimizer: str = "sella_irc"


def run_irc(
    xyz_path: Path,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    dx: float = 0.1,
    direction: IRCDirection = "both",
    forward_trajectory_name: str = "sella_irc_forward.traj",
    reverse_trajectory_name: str = "sella_irc_reverse.traj",
    forward_path_name: str = "sella_irc_forward.xyz",
    reverse_path_name: str = "sella_irc_reverse.xyz",
    full_path_name: str = "sella_irc_path.xyz",
) -> IRCResult:
    """Trace the intrinsic reaction coordinate from a TS XYZ file with Sella."""

    return optimize_irc_atoms(
        atoms=read(xyz_path),
        calculator_config=calculator_config,
        output_dir=output_dir,
        fmax=fmax,
        max_steps=max_steps,
        dx=dx,
        direction=direction,
        forward_trajectory_name=forward_trajectory_name,
        reverse_trajectory_name=reverse_trajectory_name,
        forward_path_name=forward_path_name,
        reverse_path_name=reverse_path_name,
        full_path_name=full_path_name,
    )


def optimize_irc_atoms(
    atoms: Atoms,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    fmax: float = 5e-3,
    max_steps: int = 200,
    dx: float = 0.1,
    direction: IRCDirection = "both",
    forward_trajectory_name: str = "sella_irc_forward.traj",
    reverse_trajectory_name: str = "sella_irc_reverse.traj",
    forward_path_name: str = "sella_irc_forward.xyz",
    reverse_path_name: str = "sella_irc_reverse.xyz",
    full_path_name: str = "sella_irc_path.xyz",
) -> IRCResult:
    """Trace the IRC from an ASE Atoms transition-state guess with Sella.

    `atoms` should already be converged at (or very near) a first-order
    saddle point, e.g. the output of `optimize_ts_atoms`. Each requested
    direction is run from an independent copy of `atoms`, so the caller's
    object is not mutated and neither run affects the other.
    """

    if direction not in ("forward", "reverse", "both"):
        raise ValueError('direction must be one of "forward", "reverse", "both"')

    output_dir.mkdir(parents=True, exist_ok=True)

    from sella import IRC

    forward_trajectory = forward_xyz = None
    forward_converged: bool | None = None
    forward_steps: int | None = None
    forward_images = None

    if direction in ("forward", "both"):
        forward_atoms = prepare_atoms(atoms, calculator_config)
        forward_trajectory = output_dir / forward_trajectory_name
        dyn = IRC(forward_atoms, trajectory=str(forward_trajectory), dx=dx)
        forward_converged = bool(dyn.run(fmax=fmax, steps=max_steps, direction="forward"))
        forward_images = read(forward_trajectory, index=":")
        forward_steps = len(forward_images)
        forward_xyz = output_dir / forward_path_name
        write(forward_xyz, forward_images, format="xyz")

    reverse_trajectory = reverse_xyz = None
    reverse_converged: bool | None = None
    reverse_steps: int | None = None
    reverse_images = None

    if direction in ("reverse", "both"):
        reverse_atoms = prepare_atoms(atoms, calculator_config)
        reverse_trajectory = output_dir / reverse_trajectory_name
        dyn = IRC(reverse_atoms, trajectory=str(reverse_trajectory), dx=dx)
        reverse_converged = bool(dyn.run(fmax=fmax, steps=max_steps, direction="reverse"))
        reverse_images = read(reverse_trajectory, index=":")
        reverse_steps = len(reverse_images)
        reverse_xyz = output_dir / reverse_path_name
        write(reverse_xyz, reverse_images, format="xyz")

    full_path_xyz = output_dir / full_path_name
    write(full_path_xyz, _stitch_irc_path(forward_images, reverse_images), format="xyz")

    return IRCResult(
        forward_trajectory=forward_trajectory,
        reverse_trajectory=reverse_trajectory,
        forward_xyz=forward_xyz,
        reverse_xyz=reverse_xyz,
        full_path_xyz=full_path_xyz,
        forward_converged=forward_converged,
        reverse_converged=reverse_converged,
        forward_steps=forward_steps,
        reverse_steps=reverse_steps,
        direction=direction,
    )


def _stitch_irc_path(forward_images: list[Atoms] | None, reverse_images: list[Atoms] | None) -> list[Atoms]:
    """Combine forward/reverse IRC images into one reactant->TS->product path.

    Both trajectories start with an identical TS frame (frame 0), since ASE
    writes it before any IRC step is taken. The reverse branch is reversed so
    it ends at the TS, then the forward branch is appended, skipping its own
    duplicate TS frame so the shared TS appears exactly once.
    """

    if forward_images and reverse_images:
        return list(reversed(reverse_images)) + forward_images[1:]
    if reverse_images:
        return list(reversed(reverse_images))
    if forward_images:
        return list(forward_images)
    raise ValueError("At least one IRC direction must be run")
