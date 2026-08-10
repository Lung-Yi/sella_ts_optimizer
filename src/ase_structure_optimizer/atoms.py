"""ASE Atoms helpers."""

from __future__ import annotations

from ase import Atoms

from .calculators import CalculatorConfig, build_calculator


def apply_charge_and_multiplicity(
    atoms: Atoms,
    charge: int,
    multiplicity: int,
) -> Atoms:
    """Attach charge and spin metadata to an ASE Atoms object."""

    atoms.info.update({"charge": charge, "spin": multiplicity})

    charges = atoms.get_initial_charges()
    if len(charges):
        charges[0] = charge
        atoms.set_initial_charges(charges)

    magnetic_moments = atoms.get_initial_magnetic_moments()
    if len(magnetic_moments):
        magnetic_moments[0] = multiplicity - 1
        atoms.set_initial_magnetic_moments(magnetic_moments)

    return atoms


def prepare_atoms(atoms: Atoms, calculator_config: CalculatorConfig) -> Atoms:
    """Copy atoms, attach charge/spin metadata, and attach a calculator.

    The input Atoms object is never mutated; a prepared copy is returned.
    """

    atoms = atoms.copy()
    atoms = apply_charge_and_multiplicity(
        atoms,
        charge=calculator_config.charge,
        multiplicity=calculator_config.multiplicity,
    )
    atoms.calc = build_calculator(calculator_config)
    return atoms
