"""Vibrational frequency analysis for optimized structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.vibrations import Vibrations

from .atoms import apply_charge_and_multiplicity
from .calculators import CalculatorConfig, build_calculator


BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
EV_TO_HARTREE = 0.0367493
HESSIAN_CONVERSION = EV_TO_HARTREE / (ANG_TO_BOHR**2)


@dataclass(frozen=True)
class FrequencyResult:
    """Paths and values produced by vibrational frequency analysis."""

    summary: Path
    detailed_output: Path
    vibration_directory: Path
    frequencies_cm1: tuple[float, ...]

    @property
    def imaginary_count(self) -> int:
        """Number of negative frequencies reported by geomeTRIC."""

        return sum(freq < 0 for freq in self.frequencies_cm1)


def run_frequency_analysis(
    xyz_path: Path,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    delta: float = 0.01,
    temperature: float = 300.0,
    pressure: float = 1.0,
) -> FrequencyResult:
    """Compute vibrational frequencies for a structure in an XYZ file."""

    return analyze_frequencies_atoms(
        atoms=read(xyz_path),
        calculator_config=calculator_config,
        output_dir=output_dir,
        structure_label=xyz_path,
        delta=delta,
        temperature=temperature,
        pressure=pressure,
    )


def analyze_frequencies_atoms(
    atoms: Atoms,
    calculator_config: CalculatorConfig,
    output_dir: Path,
    structure_label: str | Path = "ASE Atoms",
    delta: float = 0.01,
    temperature: float = 300.0,
    pressure: float = 1.0,
) -> FrequencyResult:
    """Compute finite-difference Hessian and vibrational frequencies for Atoms."""

    try:
        from geometric.normal_modes import frequency_analysis
    except ImportError as exc:
        raise RuntimeError(
            "Frequency analysis requires geomeTRIC. Install with "
            "`pip install -e .[vibrations]` or `pip install geometric`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = atoms.copy()
    atoms = apply_charge_and_multiplicity(
        atoms,
        charge=calculator_config.charge,
        multiplicity=calculator_config.multiplicity,
    )
    atoms.calc = build_calculator(calculator_config)

    vibration_directory = output_dir / f"vib_{calculator_config.name}"
    vib = Vibrations(atoms, delta=delta, name=str(vibration_directory))
    vib.run()
    hessian_ase = vib.get_vibrations().get_hessian_2d()

    coords_bohr = atoms.get_positions().flatten() * ANG_TO_BOHR
    hessian_au = hessian_ase * HESSIAN_CONVERSION
    energy_hartree = atoms.get_potential_energy() * EV_TO_HARTREE

    detailed_output = output_dir / f"vibrations_{calculator_config.name}_geometric.txt"
    result = frequency_analysis(
        coords=coords_bohr,
        Hessian=hessian_au,
        elem=atoms.get_chemical_symbols(),
        mass=atoms.get_masses(),
        energy=energy_hartree,
        temperature=temperature,
        pressure=pressure,
        verbose=1,
        outfnm=str(detailed_output),
        note=f"Calculated with {calculator_config.name}",
        normalized=True,
    )
    frequencies = _extract_frequencies(result)

    summary = output_dir / f"frequencies_{calculator_config.name}_summary.txt"
    _write_summary(
        summary=summary,
        structure_label=structure_label,
        calculator_name=calculator_config.name,
        energy_ev=atoms.get_potential_energy(),
        frequencies=frequencies,
        atom_count=len(atoms),
    )

    return FrequencyResult(
        summary=summary,
        detailed_output=detailed_output,
        vibration_directory=vibration_directory,
        frequencies_cm1=tuple(float(freq) for freq in frequencies),
    )


def _extract_frequencies(result) -> np.ndarray:
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=float)
    return np.asarray(result, dtype=float)


def _write_summary(
    summary: Path,
    structure_label: str | Path,
    calculator_name: str,
    energy_ev: float,
    frequencies: np.ndarray,
    atom_count: int,
) -> None:
    with summary.open("w", encoding="utf-8") as handle:
        handle.write(f"# Vibrational frequencies calculated with {calculator_name}\n")
        handle.write("# Finite-difference Hessian from ASE; normal modes from geomeTRIC\n")
        handle.write(f"# Structure: {structure_label}\n")
        handle.write(f"# Number of atoms: {atom_count}\n")
        handle.write(f"# Energy: {energy_ev:.8f} eV\n")
        handle.write("#\n")
        handle.write("# Mode    Frequency (cm^-1)\n")
        handle.write("# ----    -----------------\n")
        for mode, freq in enumerate(frequencies, start=1):
            suffix = "i" if freq < 0 else ""
            handle.write(f"{mode:5d}    {freq:12.2f}{suffix}\n")
