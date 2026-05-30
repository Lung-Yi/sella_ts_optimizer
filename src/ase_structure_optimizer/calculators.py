"""Calculator factory for structure optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalculatorConfig:
    """Configuration needed to construct an ASE calculator."""

    name: str
    charge: int = 0
    multiplicity: int = 1
    threads: int = 32
    qchem_method: str = "wb97x-v"
    qchem_basis: str = "def2-tzvp"
    qchem_label: str = "sella_ts"
    mace_model: str = "extra_large"


def available_calculators() -> tuple[str, ...]:
    """Return calculator names supported by this package."""

    return (
        "qchem",
        "b3lyp",
        "xtb",
        "uma_s",
        "uma_m",
        "eSEN",
        "aimnet2",
        "emt",
        "maceomol",
    )


def build_calculator(config: CalculatorConfig) -> Any:
    """Build an ASE-compatible calculator.

    Optional dependencies are imported lazily so users only need to install the
    backend selected at runtime.
    """

    name = config.name
    charge = config.charge
    multiplicity = config.multiplicity

    if name == "qchem":
        from ase.calculators.qchem import QChem

        return QChem(
            label=config.qchem_label,
            method=config.qchem_method,
            basis=config.qchem_basis,
            charge=charge,
            multiplicity=multiplicity,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=config.threads,
        )

    if name == "b3lyp":
        from ase.calculators.qchem import QChem

        return QChem(
            label=config.qchem_label,
            method="b3lyp",
            basis="def2-svp",
            charge=charge,
            multiplicity=multiplicity,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=config.threads,
        )

    if name == "xtb":
        from xtb.ase.calculator import XTB

        return XTB(method="GFN2-xTB")

    if name in {"uma_s", "uma_m", "eSEN"}:
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        model_names = {
            "uma_s": "uma-s-1p1",
            "uma_m": "uma-m-1p1",
            "eSEN": "esen-sm-conserving-all-omol",
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit(model_names[name], device=device)
        if name == "eSEN":
            return FAIRChemCalculator(predictor)
        return FAIRChemCalculator(predictor, task_name="omol")

    if name == "aimnet2":
        from aimnet2calc import AIMNet2ASE

        return AIMNet2ASE("aimnet2", charge=charge, mult=multiplicity)

    if name == "emt":
        from ase.calculators.emt import EMT

        return EMT()

    if name == "maceomol":
        import torch
        from mace.calculators import mace_omol

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return mace_omol(model=config.mace_model, device=device)

    choices = ", ".join(available_calculators())
    raise ValueError(f"Unknown calculator '{name}'. Choose one of: {choices}")
