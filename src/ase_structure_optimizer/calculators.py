"""Calculator factory for structure optimization."""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ase.calculators.calculator import Calculator, all_changes


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
    device: str = "auto"
    dtype: str = "float64"
    dispersion: bool = False
    mace_mp_model: str = "medium-mpa-0"
    uma_task: str = "omol"
    dispersion_xc: str = "pbe"
    uma_model: str = ""
    uma_merge_mole: bool = False


@dataclass(frozen=True)
class CalculatorCapabilities:
    """What kind of systems a calculator backend can meaningfully describe."""

    periodic: bool
    uses_charge_spin: bool
    elements: frozenset[str] | None
    level_of_theory: str
    stress: bool = True


UMA_TASKS = ("omol", "omat", "oc20", "oc22", "oc25", "odac", "omc")

# Training level of theory of each UMA task head.
UMA_TASK_LEVELS = {
    "omol": "wB97M-V (OMol25)",
    "omat": "PBE/PBE+U (OMat24)",
    "oc20": "RPBE (OC20)",
    "oc22": "PBE+U (OC22)",
    "oc25": "RPBE+D3 (OC25)",
    "odac": "PBE-D3 (ODAC23)",
    "omc": "PBE-D3 (OMC25)",
}

# UMA tasks with a trained stress head; the others cannot relax a cell.
UMA_TASKS_WITH_STRESS = frozenset({"omat", "omc"})

# UMA tasks whose training data already contain D3 dispersion; adding D3 again
# would double count it, so `dispersion=True` is ignored for them.
UMA_TASKS_WITH_D3 = frozenset({"oc25", "odac", "omc"})

# Default pretrained model of each FAIRChem backend (used when uma_model is empty).
FAIRCHEM_DEFAULT_MODELS = {
    "uma_s": "uma-s-1p1",
    "uma_m": "uma-m-1p1",
    "eSEN": "esen-sm-conserving-all-omol",
}

EMT_ELEMENTS = frozenset({"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"})

# ML backends whose model loading is expensive; get_calculator() reuses them.
_CACHEABLE_CALCULATORS = frozenset({"macemp", "maceomol", "uma_s", "uma_m", "eSEN", "aimnet2"})

_CALCULATOR_CACHE: dict[CalculatorConfig, Any] = {}


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
        "macemp",
    )


def build_calculator(config: CalculatorConfig) -> Any:
    """Build an ASE-compatible calculator.

    Optional dependencies are imported lazily so users only need to install the
    backend selected at runtime. A new calculator instance is created on every
    call; use `get_calculator()` to reuse loaded ML models.
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
        _validate_uma_task(config)
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        model = config.uma_model or FAIRCHEM_DEFAULT_MODELS[name]
        device = _resolve_device(config.device)
        if name != "eSEN" and (config.uma_merge_mole or config.uma_task != "omol"):
            calculator = UMATaskCalculator(model, config.uma_task, device, merge_mole=config.uma_merge_mole)
        else:
            predictor = _fairchem_predict_unit(model, device)
            if name == "eSEN":
                return FAIRChemCalculator(predictor)
            calculator = FAIRChemCalculator(predictor, task_name=config.uma_task)
        if _uma_adds_d3(config):
            import torch
            from ase import units
            from ase.calculators.mixing import SumCalculator
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

            # Same D3(BJ) settings as mace_mp(dispersion=True).
            d3 = TorchDFTD3Calculator(
                device=device,
                damping="bj",
                dtype=torch.float32 if config.dtype == "float32" else torch.float64,
                xc=config.dispersion_xc,
                cutoff=40.0 * units.Bohr,
            )
            return SumCalculator([calculator, d3])
        return calculator

    if name == "aimnet2":
        from aimnet2calc import AIMNet2ASE

        return AIMNet2ASE("aimnet2", charge=charge, mult=multiplicity)

    if name == "emt":
        from ase.calculators.emt import EMT

        return EMT()

    if name == "maceomol":
        from mace.calculators import mace_omol

        device = _resolve_device(config.device)
        return mace_omol(model=config.mace_model, device=device, default_dtype=config.dtype)

    if name == "macemp":
        from mace.calculators import mace_mp

        device = _resolve_device(config.device)
        return mace_mp(
            model=config.mace_mp_model,
            device=device,
            default_dtype=config.dtype,
            dispersion=config.dispersion,
            damping="bj",
            dispersion_xc=config.dispersion_xc,
        )

    choices = ", ".join(available_calculators())
    raise ValueError(f"Unknown calculator '{name}'. Choose one of: {choices}")


def get_calculator(config: CalculatorConfig, cache: bool = True) -> Any:
    """Return a calculator for `config`, reusing loaded ML models when possible.

    ML backends (`macemp`, `maceomol`, `uma_s`, `uma_m`, `eSEN`, `aimnet2`) are
    cached with the full `CalculatorConfig` as key, so each distinct setting
    loads its model only once per process. File-based backends (`qchem`,
    `b3lyp`) and all other backends are always built fresh, as is every
    backend when `cache=False`.

    A cached instance may be attached to several Atoms objects in turn; ASE
    calculators recompute whenever the attached structure changes.
    """

    if not cache or config.name not in _CACHEABLE_CALCULATORS:
        return build_calculator(config)
    try:
        return _CALCULATOR_CACHE[config]
    except KeyError:
        calculator = build_calculator(config)
        _CALCULATOR_CACHE[config] = calculator
        return calculator


def clear_calculator_cache() -> None:
    """Drop all calculators cached by `get_calculator()`."""

    _CALCULATOR_CACHE.clear()


def calculator_capabilities(config: CalculatorConfig) -> CalculatorCapabilities:
    """Describe which systems the calculator selected by `config` supports.

    Molecular models (trained on isolated molecules) are reported as
    non-periodic; they must not be used for surfaces or bulk solids.
    """

    name = config.name

    if name == "macemp":
        level = _macemp_level(config.mace_mp_model)
        if config.dispersion:
            level += f" + D3(BJ, {config.dispersion_xc})"
        return CalculatorCapabilities(
            periodic=True, uses_charge_spin=False, elements=None, level_of_theory=level
        )

    if name in {"uma_s", "uma_m"}:
        _validate_uma_task(config)
        level = UMA_TASK_LEVELS[config.uma_task]
        if _uma_adds_d3(config):
            level += f" + D3(BJ, {config.dispersion_xc})"
        molecular = config.uma_task == "omol"
        return CalculatorCapabilities(
            periodic=not molecular,
            uses_charge_spin=molecular,
            elements=None,
            level_of_theory=level,
            stress=config.uma_task in UMA_TASKS_WITH_STRESS,
        )

    if name == "eSEN":
        _validate_uma_task(config)
        return CalculatorCapabilities(
            periodic=False, uses_charge_spin=True, elements=None, level_of_theory="wB97M-V (OMol25)"
        )

    if name == "emt":
        return CalculatorCapabilities(
            periodic=True,
            uses_charge_spin=False,
            elements=EMT_ELEMENTS,
            level_of_theory="EMT (testing only)",
        )

    molecular_levels = {
        "maceomol": "wB97M-V (OMol25)",
        "aimnet2": "wB97M-D3 (AIMNet2)",
        "xtb": "GFN2-xTB",
        "qchem": f"{config.qchem_method}/{config.qchem_basis} (Q-Chem)",
        "b3lyp": "B3LYP/def2-SVP (Q-Chem)",
    }
    if name in molecular_levels:
        return CalculatorCapabilities(
            periodic=False,
            uses_charge_spin=True,
            elements=None,
            level_of_theory=molecular_levels[name],
        )

    choices = ", ".join(available_calculators())
    raise ValueError(f"Unknown calculator '{name}'. Choose one of: {choices}")


def _macemp_level(model: str) -> str:
    """Training level of theory of a MACE-MP family model, guessed from its name."""

    lowered = str(model).lower()
    if "r2scan" in lowered:
        return "r2SCAN (MatPES)"
    if "matpes" in lowered:
        return "PBE (MatPES)"
    return "PBE (MPtrj/MPA)"


class UMATaskCalculator(Calculator):
    """FAIRChem UMA calculator for one task, with consistent charge/spin inputs.

    FAIRChemCalculator reads `atoms.info["charge"]` and `atoms.info["spin"]`
    for every task, and they change the energy even for the periodic tasks,
    which were trained with charge 0 and spin 0. Structures in a workflow
    carry unrelated or inconsistent values there, so for every task except
    `omol` both are fixed to 0; `omol` keeps reading them from `atoms.info`.

    With `merge_mole=True` the mixture-of-linear-experts is merged for each
    reduced composition (and charge/spin) on first use, which gives the same
    energies as the full model with far less memory and compute; one merged
    model is loaded per distinct composition (seconds to a minute each).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model: str, task: str, device: str, merge_mole: bool = False):
        super().__init__()
        self.model = model
        self.task = task
        self.device = device
        self.merge_mole = merge_mole
        self._calculators: dict[Any, Any] = {}
        if merge_mole:
            if _is_model_file(model) and not Path(model).expanduser().is_file():
                raise FileNotFoundError(f"UMA checkpoint not found: {model}")
        else:
            # Load now so that loading errors surface when the calculator is built.
            self._calculators[None] = self._new_calculator()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms.copy()
        if self.task != "omol":
            atoms.info["charge"] = 0
            atoms.info["spin"] = 0
        key = None
        if self.merge_mole:
            key = (_reduced_composition(atoms.numbers), atoms.info.get("charge"), atoms.info.get("spin"))
        if key not in self._calculators:
            self._calculators[key] = self._new_calculator()
        calculator = self._calculators[key]
        atoms.calc = calculator
        for name in properties:
            self.results[name] = calculator.get_property(name, atoms)
        if "energy" in self.results:
            self.results.setdefault("free_energy", self.results["energy"])

    def _new_calculator(self) -> Any:
        from fairchem.core import FAIRChemCalculator

        settings = None
        if self.merge_mole:
            from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

            settings = InferenceSettings(merge_mole=True)
        predictor = _fairchem_predict_unit(self.model, self.device, settings)
        return FAIRChemCalculator(predictor, task_name=self.task)


def _reduced_composition(numbers) -> tuple[tuple[int, int], ...]:
    """Element counts divided by their greatest common divisor, e.g. Ti4Si4 -> ((14, 1), (22, 1))."""

    counts: dict[int, int] = {}
    for number in numbers:
        counts[int(number)] = counts.get(int(number), 0) + 1
    divisor = functools.reduce(math.gcd, counts.values())
    return tuple(sorted((number, count // divisor) for number, count in counts.items()))


def _fairchem_predict_unit(model: str, device: str, inference_settings: Any = None) -> Any:
    """Load a FAIRChem predictor from a registered model name or a checkpoint file."""

    from fairchem.core import pretrained_mlip

    kwargs = {"device": device}
    if inference_settings is not None:
        kwargs["inference_settings"] = inference_settings
    if _is_model_file(model):
        with _mmap_torch_load():
            return pretrained_mlip.load_predict_unit(str(Path(model).expanduser()), **kwargs)
    return pretrained_mlip.get_predict_unit(model, **kwargs)


@contextlib.contextmanager
def _mmap_torch_load() -> Iterator[None]:
    """Memory-map checkpoints while FAIRChem loads them.

    FAIRChem reads the whole checkpoint (model and EMA weights) into RAM; a
    UMA-M checkpoint is 11 GB, which does not fit next to the model on a
    16 GB machine. With mmap the weights are paged in only when copied.
    """

    import torch

    original = getattr(torch, "load", None)
    if original is None:
        yield
        return
    torch.load = functools.partial(original, mmap=True)
    try:
        yield
    finally:
        torch.load = original


def model_label(config: CalculatorConfig) -> str:
    """Name or file path of the ML model behind `config`, else the calculator name."""

    if config.name == "macemp":
        return config.mace_mp_model
    if config.name == "maceomol":
        return config.mace_model
    if config.name in FAIRCHEM_DEFAULT_MODELS:
        return config.uma_model or FAIRCHEM_DEFAULT_MODELS[config.name]
    return config.name


def _is_model_file(model: str) -> bool:
    """True when a FAIRChem model is given as a checkpoint file instead of a registered name."""

    return model.endswith((".pt", ".ckpt")) or "/" in model or "\\" in model or Path(model).expanduser().is_file()


def _uma_adds_d3(config: CalculatorConfig) -> bool:
    """Whether D3(BJ) is added on top of a UMA task (never for tasks trained with D3)."""

    return config.name in {"uma_s", "uma_m"} and config.dispersion and config.uma_task not in UMA_TASKS_WITH_D3


def _validate_uma_task(config: CalculatorConfig) -> None:
    if config.uma_task not in UMA_TASKS:
        choices = ", ".join(UMA_TASKS)
        raise ValueError(f"Unknown uma_task '{config.uma_task}'. Choose one of: {choices}")
    if config.name == "eSEN" and config.uma_task != "omol":
        raise ValueError(
            f"Calculator 'eSEN' only supports uma_task='omol' (got '{config.uma_task}'); "
            "use 'uma_s' or 'uma_m' for the periodic tasks."
        )


def _resolve_device(device: str) -> str:
    """Map `device="auto"` to CUDA when available, otherwise CPU."""

    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
