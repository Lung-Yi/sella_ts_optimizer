"""ASE Atoms helpers."""

from __future__ import annotations

from typing import Any

from ase import Atoms

from .calculators import CalculatorConfig, build_calculator


def apply_charge_and_multiplicity(
    atoms: Atoms,
    charge: int,
    multiplicity: int,
) -> Atoms:
    """Attach charge and spin metadata to an ASE Atoms object.

    For isolated molecules (no periodic direction) the total charge and
    `multiplicity - 1` are also written to the initial charge / magnetic
    moment of atom 0, which is how xTB-style calculators read them. Periodic
    systems only get `atoms.info` updated, so no spurious per-atom moment is
    left on the first atom of a slab or bulk cell.
    """

    atoms.info.update({"charge": charge, "spin": multiplicity})

    if atoms.pbc.any():
        return atoms

    charges = atoms.get_initial_charges()
    if len(charges):
        charges[0] = charge
        atoms.set_initial_charges(charges)

    magnetic_moments = atoms.get_initial_magnetic_moments()
    if len(magnetic_moments):
        magnetic_moments[0] = multiplicity - 1
        atoms.set_initial_magnetic_moments(magnetic_moments)

    return atoms


def prepare_atoms(
    atoms: Atoms,
    calculator_config: CalculatorConfig,
    calculator: Any | None = None,
) -> Atoms:
    """Copy atoms, attach charge/spin metadata, and attach a calculator.

    The input Atoms object is never mutated; a prepared copy is returned.
    Constraints such as `FixAtoms` are carried over by the copy. When
    `calculator` is given it is attached as-is (e.g. a cached instance from
    `get_calculator()`); otherwise a new one is built from `calculator_config`.
    """

    atoms = atoms.copy()
    atoms = apply_charge_and_multiplicity(
        atoms,
        charge=calculator_config.charge,
        multiplicity=calculator_config.multiplicity,
    )
    atoms.calc = build_calculator(calculator_config) if calculator is None else calculator
    return atoms
