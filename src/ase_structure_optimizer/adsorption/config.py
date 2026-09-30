"""Adsorption workflow settings: nested frozen dataclasses, YAML I/O and validation.

Only `molecule` and `solid` are required; every other value has a default
(see `default_config_yaml()` for the annotated template). Validation errors
raise `ConfigError` naming the offending field, e.g. ``budget.fmax``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from ..calculators import (
    CalculatorCapabilities,
    CalculatorConfig,
    available_calculators,
    calculator_capabilities,
)

AUTO = "auto"
Auto = Literal["auto"]


class ConfigError(ValueError):
    """Invalid adsorption configuration; the message names the field."""


# ---------------------------------------------------------------------------
# Field parsers. Each takes the raw YAML value and returns the parsed value or
# raises ConfigError with a message that does not yet include the field path.
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _float(minimum: float | None = None, maximum: float | None = None, *, strict_min: bool = False):
    def parse(value: Any) -> float:
        if isinstance(value, str):
            try:
                value = float(value)  # PyYAML reads "1e-5" as a string
            except ValueError:
                raise ConfigError(f"expected a number, got {value!r}") from None
        if not _is_number(value) or not math.isfinite(value):
            raise ConfigError(f"expected a number, got {value!r}")
        value = float(value)
        if minimum is not None and (value <= minimum if strict_min else value < minimum):
            relation = ">" if strict_min else ">="
            raise ConfigError(f"must be {relation} {minimum}, got {value}")
        if maximum is not None and value > maximum:
            raise ConfigError(f"must be <= {maximum}, got {value}")
        return value

    return parse


def _int(minimum: int | None = None, maximum: int | None = None):
    def parse(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"expected an integer, got {value!r}")
        if minimum is not None and value < minimum:
            raise ConfigError(f"must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            raise ConfigError(f"must be <= {maximum}, got {value}")
        return value

    return parse


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"expected true or false, got {value!r}")
    return value


def _str(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"expected a non-empty string, got {value!r}")
    return value


def _choice(*options: Any):
    def parse(value: Any) -> Any:
        if value not in options:
            choices = ", ".join(str(option) for option in options)
            raise ConfigError(f"must be one of {choices}, got {value!r}")
        return value

    return parse


def _auto_or(parser: Callable[[Any], Any]):
    def parse(value: Any) -> Any:
        if value == AUTO:
            return AUTO
        try:
            return parser(value)
        except ConfigError as exc:
            raise ConfigError(f"expected 'auto' or a value: {exc}") from None

    return parse


def _optional(parser: Callable[[Any], Any]):
    def parse(value: Any) -> Any:
        return None if value is None else parser(value)

    return parse


def _list(parser: Callable[[Any], Any], min_length: int = 0, unique: bool = False):
    def parse(value: Any) -> tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"expected a list, got {value!r}")
        if len(value) < min_length:
            raise ConfigError(f"needs at least {min_length} item(s)")
        items = []
        for index, item in enumerate(value):
            try:
                items.append(parser(item))
            except ConfigError as exc:
                raise ConfigError(f"item {index}: {exc}") from None
        if unique and len(set(items)) != len(items):
            raise ConfigError(f"contains duplicates: {list(value)!r}")
        return tuple(items)

    return parse


def _path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(_str(value)).expanduser()


def _miller(value: Any) -> tuple[int, int, int]:
    indices = _list(_int(), min_length=3)(value)
    if len(indices) != 3:
        raise ConfigError(f"a Miller index needs exactly 3 integers, got {list(value)!r}")
    if indices == (0, 0, 0):
        raise ConfigError("Miller index (0, 0, 0) is not a plane")
    return indices


def _bond_override(value: Any) -> tuple[int, int, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"expected [atom_i, atom_j, true|false], got {value!r}")
    i, j = _int(minimum=0)(value[0]), _int(minimum=0)(value[1])
    if i == j:
        raise ConfigError(f"a bond override needs two different atoms, got {value!r}")
    return i, j, _bool(value[2])


def _reference_axis(value: Any) -> Auto | tuple[tuple[int, ...], tuple[int, ...]]:
    if value == AUTO:
        return AUTO
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"expected 'auto' or [[atoms A], [atoms B]], got {value!r}")
    groups = tuple(_list(_int(minimum=0), min_length=1, unique=True)(group) for group in value)
    if set(groups[0]) == set(groups[1]):
        raise ConfigError("the two atom groups of reference_axis must differ")
    return groups  # type: ignore[return-value]


def _scan_heights(value: Any) -> tuple[float, float, float]:
    numbers = _list(_float())(value)
    if len(numbers) != 3:
        raise ConfigError(f"expected [start, stop, step], got {value!r}")
    start, stop, step = numbers
    if start <= 0 or stop <= start or step <= 0:
        raise ConfigError(f"need 0 < start < stop and step > 0, got {list(numbers)}")
    return start, stop, step


_CALCULATOR_CONFIG_FIELDS = {item.name for item in dataclasses.fields(CalculatorConfig)}


def _calculator_name(value: Any) -> str:
    return _choice(*available_calculators())(value)


def _partial_calculator(value: Any) -> CalculatorConfig:
    if not isinstance(value, Mapping):
        raise ConfigError(f"expected a mapping of CalculatorConfig fields, got {value!r}")
    unknown = sorted(set(value) - _CALCULATOR_CONFIG_FIELDS)
    if unknown:
        raise ConfigError(f"unknown CalculatorConfig field(s): {', '.join(unknown)}")
    if "name" not in value:
        raise ConfigError("'name' is required")
    _calculator_name(value["name"])
    config = CalculatorConfig(**value)
    _require_periodic(config)
    return config


def _require_periodic(config: CalculatorConfig) -> CalculatorCapabilities:
    try:
        capabilities = calculator_capabilities(config)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    if not capabilities.periodic:
        raise ConfigError(
            f"calculator '{config.name}'"
            + (f" with uma_task '{config.uma_task}'" if config.name in {"uma_s", "uma_m", "eSEN"} else "")
            + f" is a molecular model ({capabilities.level_of_theory}) and cannot describe "
            "surfaces or bulk solids; use macemp, uma_s/uma_m with uma_task omat or oc20, "
            "or emt (tests only)"
        )
    return capabilities


def _potcar_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"expected a mapping element -> POTCAR name, got {value!r}")
    from ase.data import chemical_symbols

    parsed = {}
    for element, potcar in value.items():
        if element not in chemical_symbols[1:]:
            raise ConfigError(f"unknown element {element!r}")
        parsed[element] = _str(potcar)
    return parsed


def _setting(default: Any, parser: Callable[[Any], Any]) -> Any:
    """Dataclass field with the parser used for its YAML value."""

    metadata = {"parser": parser}
    if isinstance(default, (list, dict, set)):
        return field(default_factory=lambda: type(default)(default), metadata=metadata)
    return field(default=default, metadata=metadata)


def _section(cls: type) -> Any:
    return field(default_factory=cls, metadata={"section": cls})


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleculeProps:
    """Charge, spin and bond-perception settings of the adsorbate."""

    charge: int = _setting(0, _int())
    multiplicity: int = _setting(1, _int(minimum=1))
    bond_scale: float = _setting(1.20, _float(0.0, strict_min=True))
    bond_overrides: tuple[tuple[int, int, bool], ...] = _setting((), _list(_bond_override))
    reference_axis: Auto | tuple[tuple[int, ...], tuple[int, ...]] = _setting(AUTO, _reference_axis)


@dataclass(frozen=True)
class CalculatorSection:
    """Periodic MLIP settings; `to_calculator_config()` builds the shared config."""

    name: str = _setting("macemp", _calculator_name)
    mace_mp_model: str = _setting("medium-mpa-0", _str)
    fallback_models: tuple[str, ...] = _setting(("medium", "small"), _list(_str))
    uma_task: str = _setting("oc20", _choice("omol", "omat", "oc20"))
    dispersion: bool = _setting(True, _bool)
    device: str = _setting("auto", _str)
    dtype_screen: str = _setting("float32", _choice("float32", "float64"))
    dtype_final: str = _setting("float64", _choice("float32", "float64"))

    def to_calculator_config(
        self,
        stage: str = "final",
        charge: int = 0,
        multiplicity: int = 1,
        mace_mp_model: str | None = None,
    ) -> CalculatorConfig:
        """Shared `CalculatorConfig` for the `"screen"` or `"final"` dtype."""

        if stage not in ("screen", "final"):
            raise ValueError(f"stage must be 'screen' or 'final', got {stage!r}")
        return CalculatorConfig(
            name=self.name,
            charge=charge,
            multiplicity=multiplicity,
            device=self.device,
            dtype=self.dtype_screen if stage == "screen" else self.dtype_final,
            dispersion=self.dispersion,
            mace_mp_model=mace_mp_model or self.mace_mp_model,
            uma_task=self.uma_task,
        )


@dataclass(frozen=True)
class SurfaceConfig:
    """Bulk relaxation, slab cutting and supercell construction."""

    input_type: str = _setting(AUTO, _choice(AUTO, "bulk", "slab"))
    relax_bulk: bool = _setting(True, _bool)
    miller_indices: tuple[tuple[int, int, int], ...] = _setting(
        ((0, 0, 1),), _list(_miller, min_length=1, unique=True)
    )
    min_slab_thickness: float = _setting(8.0, _float(0.0, strict_min=True))
    max_terminations: int = _setting(3, _int(minimum=1))
    min_lateral: Auto | float = _setting(AUTO, _auto_or(_float(0.0, strict_min=True)))
    lateral_buffer: float = _setting(10.0, _float(0.0))
    vacuum_above: Auto | float = _setting(AUTO, _auto_or(_float(0.0, strict_min=True)))
    fix_fraction: float = _setting(0.5, _float(0.0, 1.0))


@dataclass(frozen=True)
class SamplingConfig:
    """Initial adsorption configurations."""

    site_types: tuple[str, ...] = _setting(
        ("ontop", "bridge", "hollow"),
        _list(_choice("ontop", "bridge", "hollow"), min_length=1, unique=True),
    )
    spins_per_anchor: int = _setting(3, _int(minimum=1))
    n_random: int = _setting(60, _int(minimum=0))
    contact_gap: Auto | float = _setting(AUTO, _auto_or(_float(0.0, strict_min=True)))
    clash_scale: float = _setting(0.7, _float(0.0, strict_min=True))
    seed: int = _setting(42, _int())
    surface_depth: float = _setting(0.9, _float(0.0, strict_min=True))


@dataclass(frozen=True)
class BudgetConfig:
    """Time budget and optimizer settings per termination."""

    wall_time_per_termination: float = _setting(600.0, _float(0.0))
    prescreen_steps: int = _setting(20, _int(minimum=1))
    prescreen_optimizer: str = _setting("fire", _choice("bfgs", "lbfgs", "fire"))
    relax_optimizer: str = _setting("lbfgs", _choice("bfgs", "lbfgs", "fire"))
    prescreen_fraction: float = _setting(0.40, _float(0.0, 1.0))
    relax_fraction: float = _setting(0.55, _float(0.0, 1.0))
    min_full_relax: int = _setting(5, _int(minimum=0))
    fmax_prescreen: float = _setting(0.15, _float(0.0, strict_min=True))
    fmax: float = _setting(0.05, _float(0.0, strict_min=True))
    max_steps: int = _setting(400, _int(minimum=1))


@dataclass(frozen=True)
class AnalysisConfig:
    """Classification, de-duplication and scans."""

    contact_scale: float = _setting(1.25, _float(0.0, strict_min=True))
    desorbed_distance: float = _setting(4.5, _float(0.0, strict_min=True))
    dedup_energy_tol: float = _setting(0.02, _float(0.0))
    dedup_rmsd_tol: float = _setting(0.30, _float(0.0))
    temperature: float = _setting(298.15, _float(0.0, strict_min=True))
    approach_scan: bool = _setting(True, _bool)
    scan_heights: tuple[float, float, float] = _setting((1.5, 7.0, 0.25), _scan_heights)
    make_gif: bool = _setting(True, _bool)


@dataclass(frozen=True)
class UncertaintyConfig:
    """Single-point comparison with additional models."""

    enabled: bool = _setting(True, _bool)
    calculators: tuple[CalculatorConfig, ...] = _setting(
        (
            CalculatorConfig(name="macemp", mace_mp_model="medium"),
            CalculatorConfig(name="uma_s", uma_task="oc20"),
        ),
        _list(_partial_calculator),
    )
    flag_threshold: float = _setting(0.10, _float(0.0, strict_min=True))


@dataclass(frozen=True)
class DftSelectionConfig:
    """Which configurations get VASP inputs."""

    max_configs: int = _setting(20, _int(minimum=1))
    energy_window: float = _setting(0.30, _float(0.0))
    include_per_class_best: bool = _setting(True, _bool)
    n_uncertain: int = _setting(3, _int(minimum=0))


_DEFAULT_POTCARS = {"Ti": "Ti_pv", "Mo": "Mo_pv", "Si": "Si", "C": "C", "O": "O", "H": "H", "I": "I", "N": "N"}


@dataclass(frozen=True)
class VaspConfig:
    """VASP input parameters."""

    encut: float = _setting(450.0, _float(0.0, strict_min=True))
    ediff: float = _setting(1.0e-5, _float(0.0, strict_min=True))
    ediffg: float = _setting(-0.03, _float())
    ismear: int = _setting(0, _int(minimum=-5))
    sigma: float = _setting(0.05, _float(0.0, strict_min=True))
    ivdw: int = _setting(12, _int(minimum=0))
    kpt_length: float = _setting(25.0, _float(0.0, strict_min=True))
    ispin: Auto | int = _setting(AUTO, _auto_or(_choice(1, 2)))
    potcar_map: dict[str, str] = _setting(_DEFAULT_POTCARS, _potcar_map)
    molecule_box_padding: float = _setting(15.0, _float(0.0, strict_min=True))
    ncore: int | None = _setting(None, _optional(_int(minimum=1)))


@dataclass(frozen=True)
class AdsorptionConfig:
    """Complete adsorption workflow configuration."""

    molecule: Path
    solid: Path
    run_dir: Path | None = _setting(None, _optional(_path))
    molecule_props: MoleculeProps = _section(MoleculeProps)
    calculator: CalculatorSection = _section(CalculatorSection)
    surface: SurfaceConfig = _section(SurfaceConfig)
    sampling: SamplingConfig = _section(SamplingConfig)
    budget: BudgetConfig = _section(BudgetConfig)
    analysis: AnalysisConfig = _section(AnalysisConfig)
    uncertainty: UncertaintyConfig = _section(UncertaintyConfig)
    dft_selection: DftSelectionConfig = _section(DftSelectionConfig)
    vasp: VaspConfig = _section(VaspConfig)

    def calculator_config(self, stage: str = "final", mace_mp_model: str | None = None) -> CalculatorConfig:
        """Shared `CalculatorConfig` with the molecule's charge and multiplicity."""

        return self.calculator.to_calculator_config(
            stage=stage,
            charge=self.molecule_props.charge,
            multiplicity=self.molecule_props.multiplicity,
            mace_mp_model=mace_mp_model,
        )

    def resolved_run_dir(self) -> Path:
        """`run_dir`, or `<molecule>_on_<solid>_<calculator>` next to the molecule file."""

        if self.run_dir is not None:
            return self.run_dir
        name = f"{self.molecule.stem}_on_{self.solid.stem}_{self.calculator.name}"
        return self.molecule.parent / name


# Sections whose values change computed results; `resume` refuses to continue
# a run when any of them differs from the stored configuration.
RESULT_AFFECTING_SECTIONS = ("molecule_props", "calculator", "surface", "sampling")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def config_from_dict(data: Mapping[str, Any], base_dir: str | Path | None = None) -> AdsorptionConfig:
    """Build and validate an `AdsorptionConfig` from plain (YAML) data.

    Relative `molecule`, `solid` and `run_dir` paths are resolved against
    `base_dir` (normally the directory of the YAML file).
    """

    if not isinstance(data, Mapping):
        raise ConfigError(f"configuration must be a mapping, got {type(data).__name__}")
    for key in ("molecule", "solid"):
        if data.get(key) is None:
            raise ConfigError(f"{key}: required field is missing")

    values = _parse_fields(AdsorptionConfig, data, prefix="", required=("molecule", "solid"))
    base = Path(base_dir) if base_dir is not None else None
    for key in ("molecule", "solid", "run_dir"):
        path = values.get(key)
        if path is not None and base is not None and not path.is_absolute():
            values[key] = base / path
    config = AdsorptionConfig(**values)
    _check_consistency(config)
    return config


def _parse_fields(cls: type, data: Mapping[str, Any], prefix: str, required: tuple[str, ...] = ()) -> dict:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{prefix.rstrip('.') or 'config'}: expected a mapping, got {data!r}")
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        known = ", ".join(fields)
        raise ConfigError(f"{prefix}{unknown[0]}: unknown key (known keys: {known})")

    values: dict[str, Any] = {}
    for name, value in data.items():
        item = fields[name]
        where = f"{prefix}{name}"
        if "section" in item.metadata:
            section_cls = item.metadata["section"]
            values[name] = section_cls(**_parse_fields(section_cls, value or {}, prefix=f"{where}."))
        elif name in required:
            try:
                values[name] = _path(value)
            except ConfigError as exc:
                raise ConfigError(f"{where}: {exc}") from None
        else:
            try:
                values[name] = item.metadata["parser"](value)
            except ConfigError as exc:
                raise ConfigError(f"{where}: {exc}") from None
    return values


def _check_consistency(config: AdsorptionConfig) -> None:
    budget = config.budget
    if budget.prescreen_fraction + budget.relax_fraction > 1.0 + 1e-12:
        raise ConfigError(
            "budget.relax_fraction: prescreen_fraction + relax_fraction must not exceed 1 "
            f"(got {budget.prescreen_fraction} + {budget.relax_fraction})"
        )
    try:
        _require_periodic(config.calculator_config())
    except ConfigError as exc:
        raise ConfigError(f"calculator.name: {exc}") from None


def check_calculator_support(config: AdsorptionConfig, symbols: Iterable[str]) -> CalculatorCapabilities:
    """Check that the main calculator is periodic and supports all `symbols`.

    `symbols` should contain the elements of both the molecule and the solid.
    Raises `ConfigError` otherwise; returns the calculator's capabilities.
    """

    capabilities = _require_periodic(config.calculator_config())
    if capabilities.elements is not None:
        missing = sorted(set(symbols) - capabilities.elements)
        if missing:
            raise ConfigError(
                f"calculator '{config.calculator.name}' does not support element(s) "
                f"{', '.join(missing)} (supported: {', '.join(sorted(capabilities.elements))})"
            )
    return capabilities


def load_config(path: str | Path) -> AdsorptionConfig:
    """Read and validate a YAML configuration file."""

    yaml = _require_yaml()
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    try:
        return config_from_dict(data, base_dir=path.parent)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, CalculatorConfig):
        defaults = CalculatorConfig(name=value.name)
        return {
            item.name: getattr(value, item.name)
            for item in dataclasses.fields(value)
            if item.name == "name" or getattr(value, item.name) != getattr(defaults, item.name)
        }
    if dataclasses.is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def config_to_dict(config: AdsorptionConfig) -> dict[str, Any]:
    """Plain-data form of a configuration (round-trips through `config_from_dict`)."""

    return _plain(config)


def dump_config(config: AdsorptionConfig, path: str | Path) -> Path:
    """Write the full configuration, defaults included, as YAML.

    File paths are written as absolute paths so the file can be reloaded
    from any location.
    """

    yaml = _require_yaml()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(config)
    for key in ("molecule", "solid", "run_dir"):
        if data[key] is not None:
            data[key] = str(Path(data[key]).resolve())
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=None)
    return path


def result_fingerprint(config: AdsorptionConfig) -> str:
    """Hash of the settings in `RESULT_AFFECTING_SECTIONS` plus the input files' paths."""

    data = config_to_dict(config)
    relevant = {key: data[key] for key in RESULT_AFFECTING_SECTIONS}
    relevant["molecule"] = data["molecule"]
    relevant["solid"] = data["solid"]
    encoded = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading adsorption configuration files requires PyYAML. Install with "
            '`pip install -e ".[adsorption]"` or `pip install pyyaml`.'
        ) from exc
    return yaml


# ---------------------------------------------------------------------------
# Annotated template (ase-adsorb init-config)
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# ase-adsorb configuration. Only `molecule` and `solid` are required; every
# other value below is the default. Units: Å, eV, degree, s.

run_dir: null                  # null: <molecule>_on_<solid>_<calculator> next to the molecule file
molecule: molecule.xyz         # required: gas-phase molecule (xyz)
solid: surface.cif             # required: bulk crystal or pre-cut slab (cif)

molecule_props:
  charge: 0
  multiplicity: 1
  bond_scale: 1.20             # bonded when d < bond_scale * (r_cov_i + r_cov_j)
  bond_overrides: []           # e.g. [[0, 5, true], [2, 7, false]] forces a bond on / off
  reference_axis: auto         # auto, or [[atom indices A], [atom indices B]]: axis = centroid(A) -> centroid(B)
                               # check the inferred bonds/axis first with: ase-adsorb check-molecule molecule.xyz

calculator:                    # must be a periodic calculator
  # macemp (default): MACE-MP materials model, PBE level (MPtrj/MPA), plus D3(BJ) below.
  # uma_s / uma_m: FAIRChem UMA; uma_task oc20 is trained for adsorbate+surface
  #   (RPBE, no dispersion), omat for bulk materials. Never mix energies from
  #   different models or tasks.
  # emt: ASE EMT, only for tests.
  # Molecular models (maceomol, aimnet2, eSEN, UMA omol, xtb, qchem) are rejected.
  # Periodic MLIPs see almost no isolated molecules in training, so the gas-phase
  # reference energy is an extrapolation: validate final numbers with DFT.
  name: macemp
  mace_mp_model: medium-mpa-0  # or a local model file path (no internet access)
  fallback_models: [medium, small]   # tried in order if the macemp model fails to load
  uma_task: oc20               # uma_s / uma_m only: omat or oc20
  dispersion: true             # macemp D3(BJ), needs torch-dftd; ignored by UMA
  device: auto                 # auto / cuda / cpu
  dtype_screen: float32        # macemp dtype for prescreening and relaxations
  dtype_final: float64         # macemp dtype for the final single points

surface:
  input_type: auto             # auto / bulk / slab
  relax_bulk: true
  miller_indices: [[0, 0, 1]]
  min_slab_thickness: 8.0
  max_terminations: 3          # terminations kept per Miller index
  min_lateral: auto            # auto: largest molecule dimension + lateral_buffer
  lateral_buffer: 10.0
  vacuum_above: auto           # auto: largest molecule dimension + 15 Å
  fix_fraction: 0.5            # fraction of slab atoms fixed at the bottom

sampling:
  site_types: [ontop, bridge, hollow]
  spins_per_anchor: 3          # rotations about the surface normal per anchor (evenly over 360°)
  n_random: 60                 # extra uniformly random orientations
  contact_gap: auto            # anchor-to-nearest-surface-atom distance; auto: 0.9 * sum of vdW radii
  clash_scale: 0.7             # discard if any distance < clash_scale * sum of vdW radii
  seed: 42
  surface_depth: 0.9           # atoms within this depth below the topmost atom count as surface atoms for site finding

budget:
  wall_time_per_termination: 600   # s; 0 = unlimited
  prescreen_steps: 20
  prescreen_optimizer: fire
  relax_optimizer: lbfgs
  prescreen_fraction: 0.40
  relax_fraction: 0.55             # the remaining 5 % is kept in reserve
  min_full_relax: 5
  fmax_prescreen: 0.15
  fmax: 0.05
  max_steps: 400

analysis:
  contact_scale: 1.25          # contact when d < contact_scale * (r_cov_i + r_cov_j)
  desorbed_distance: 4.5       # desorbed when the molecule-surface minimum distance exceeds this
  dedup_energy_tol: 0.02
  dedup_rmsd_tol: 0.30
  temperature: 298.15          # K, Boltzmann weights
  approach_scan: true          # rigid approach scan for the most stable configuration
  scan_heights: [1.5, 7.0, 0.25]   # start, stop, step
  make_gif: true

uncertainty:
  enabled: true
  calculators:                 # extra single-point models (CalculatorConfig fields); skipped if they fail to load
    - {name: macemp, mace_mp_model: medium}
    - {name: uma_s, uma_task: oc20}
  flag_threshold: 0.10         # eV; E_ads standard deviation above this -> low_confidence

dft_selection:
  max_configs: 20
  energy_window: 0.30          # eV above the lowest E_ads of each termination
  include_per_class_best: true
  n_uncertain: 3

vasp:
  encut: 450
  ediff: 1.0e-5
  ediffg: -0.03
  ismear: 0
  sigma: 0.05
  ivdw: 12
  kpt_length: 25.0             # in-plane k-points n_i = max(1, ceil(kpt_length / |a_i|))
  ispin: auto                  # auto: 2 if multiplicity > 1, else 1
  potcar_map: {Ti: Ti_pv, Mo: Mo_pv, Si: Si, C: C, O: O, H: H, I: I, N: N}
  molecule_box_padding: 15.0
  ncore: null
"""


def default_config_yaml() -> str:
    """Annotated YAML template containing every default value."""

    return _TEMPLATE
