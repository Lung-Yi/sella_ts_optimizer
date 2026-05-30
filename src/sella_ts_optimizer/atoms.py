"""ASE Atoms helpers."""

from __future__ import annotations

from ase import Atoms


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
