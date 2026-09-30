"""Workflow stages of the adsorption run.

Implemented so far (milestones M1-M3): run directory setup, molecule
analysis, the gas-phase reference, bulk relaxation, slab generation and
termination selection, slab supercells, slab relaxation, adsorption sites,
initial configurations, and the time-budgeted prescreening and full
relaxations. `run_adsorption_workflow()` runs everything; `prepare_surfaces()`
stops after the surfaces.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

from ..structures import read_structure
from .analysis import (
    adsorption_height,
    azimuth_angle,
    azimuth_sign_atom,
    boltzmann_weights,
    classify,
    contact_label,
    deduplicate,
    facing_label,
    molecule_part,
    molecule_tilt,
    surface_displacement,
)
from .analysis import SURFACE_DISTORTION_LIMIT
from .config import AUTO, AdsorptionConfig, ConfigError, check_calculator_support, dump_config, load_config
from .molecule import MoleculeAnalysis, analyze_molecule, molecule_in_box
from .relax import (
    CalculatorSet,
    optimize_candidate,
    relax_structure,
    resolve_calculators,
    single_point,
)
from .sampling import Candidate, generate_candidates
from .scheduler import (
    BudgetClock,
    diverse_selection,
    measure_step_time,
    prescreen_count,
    relax_count,
    stratified_sample,
)
from .sites import AdsorptionSite, find_sites
from .state import STATE_FILE, RunState, StateError
from .surface import (
    classify_solid,
    conventional_bulk,
    fix_bottom_layers,
    generate_slabs,
    lateral_repeats,
    molecule_size,
    orient_slab,
    perpendicular_widths,
    slab_thickness,
    surface_energy,
    termination_id,
    top_layer_compositions,
)

logger = logging.getLogger("ase_structure_optimizer.adsorption")

RESOLVED_CONFIG = "config.resolved.yaml"
INPUTS_DIR = "inputs"
LOG_FILE = "adsorption.log"
RESULTS_DB = "results.db"
# Extra vacuum (Å) above the largest molecule dimension when vacuum_above is auto.
AUTO_VACUUM_MARGIN = 15.0


@dataclass
class RunContext:
    """Everything the stages of one run share."""

    config: AdsorptionConfig
    run_dir: Path
    state: RunState
    calculators: CalculatorSet
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceModel:
    """A relaxed slab supercell for one termination, ready for adsorption."""

    term_id: str
    miller: tuple[int, int, int] | None
    shift: float | None
    symmetric: bool | None
    stoichiometric: bool | None
    surface_energy: float | None
    repeats: tuple[int, int]
    n_atoms: int
    n_fixed: int
    thickness: float
    widths: tuple[float, float]
    energy_final: float
    energy_screen: float
    converged: bool
    top_layers: tuple[dict, ...]
    sites: tuple[AdsorptionSite, ...]
    slab_file: Path
    unit_slab_file: Path

    def slab(self) -> Atoms:
        """The relaxed supercell (with FixAtoms and its final-dtype energy)."""

        return read(self.slab_file)

    def unit_slab(self) -> Atoms:
        return read(self.unit_slab_file)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["sites"] = [site.to_dict() for site in self.sites]
        data["slab_file"] = self.slab_file.name
        data["unit_slab_file"] = self.unit_slab_file.name
        data["top_layers"] = list(self.top_layers)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], directory: Path) -> SurfaceModel:
        values = dict(data)
        values["miller"] = tuple(data["miller"]) if data["miller"] is not None else None
        values["repeats"] = tuple(data["repeats"])
        values["widths"] = tuple(data["widths"])
        values["top_layers"] = tuple(data["top_layers"])
        values["sites"] = tuple(AdsorptionSite.from_dict(site) for site in data["sites"])
        values["slab_file"] = directory / data["slab_file"]
        values["unit_slab_file"] = directory / data["unit_slab_file"]
        return cls(**values)


# ---------------------------------------------------------------------------
# Run setup
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def run_logging(run_dir: Path) -> Iterator[None]:
    """Log the workflow to ``<run_dir>/adsorption.log`` and to stderr."""

    run_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(run_dir / LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[ase-adsorb] %(message)s"))
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    # Backends such as MACE configure the root logger; do not log twice.
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    try:
        yield
    finally:
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
        file_handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def start_run(config: AdsorptionConfig) -> RunContext:
    """Create a new run directory, load the calculators and write the resolved config.

    Refuses to reuse a directory that already holds a run (use `resume_run()`).
    """

    run_dir = config.resolved_run_dir().resolve()
    if (run_dir / STATE_FILE).exists():
        raise StateError(f"{run_dir} already contains a run; resume it or choose another run_dir")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    calculators = resolve_calculators(config)
    resolved = config
    if config.calculator.name == "macemp" and calculators.model != config.calculator.mace_mp_model:
        resolved = dataclasses.replace(
            config, calculator=dataclasses.replace(config.calculator, mace_mp_model=calculators.model)
        )
    # Copies of the inputs make the run directory self-contained (movable, resumable).
    inputs = run_dir / INPUTS_DIR
    inputs.mkdir(exist_ok=True)
    copies = {}
    for key in ("molecule", "solid"):
        source = getattr(resolved, key).resolve()
        target = inputs / source.name
        if source != target:
            shutil.copy2(source, target)
        copies[key] = target
    resolved = dataclasses.replace(resolved, run_dir=run_dir, **copies)
    dump_config(resolved, run_dir / RESOLVED_CONFIG)
    state = RunState.create(run_dir, resolved)
    context = RunContext(resolved, run_dir, state, calculators)
    context.timings["load_calculators"] = time.perf_counter() - started
    logger.info("run directory: %s", run_dir)
    logger.info("calculator: %s (model %s)", calculators.final.name, calculators.model)
    return context


def resume_run(run_dir: str | Path) -> RunContext:
    """Reopen a run from its ``config.resolved.yaml`` and ``state.json``."""

    run_dir = Path(run_dir).resolve()
    config_path = run_dir / RESOLVED_CONFIG
    if not config_path.is_file():
        raise StateError(f"no {RESOLVED_CONFIG} in {run_dir}")
    config = dataclasses.replace(load_config(config_path), run_dir=run_dir)
    state = RunState.load(run_dir)
    state.check_config(config)
    reset = state.prepare_resume()
    calculators = resolve_calculators(config)
    if reset:
        logger.info("restarting interrupted items: %s", ", ".join(f"{s}/{i}" if i else s for s, i in reset))
    return RunContext(config, run_dir, state, calculators)


# ---------------------------------------------------------------------------
# Stage: molecule analysis
# ---------------------------------------------------------------------------


def stage_molecule(context: RunContext) -> tuple[Atoms, MoleculeAnalysis]:
    """Read and analyze the molecule; write ``molecule/molecule_analysis.json``."""

    config = context.config
    molecule = read_structure(config.molecule)
    analysis = analyze_molecule(molecule, config.molecule_props)
    directory = context.run_dir / "molecule"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "molecule_analysis.json").write_text(json.dumps(analysis.to_dict(), indent=1), encoding="utf-8")
    logger.info(
        "molecule %s: fragments %s; reference axis %s",
        molecule.get_chemical_formula(),
        ", ".join(fragment.name for fragment in analysis.fragments),
        analysis.reference_axis.method,
    )
    for warning in analysis.warnings:
        logger.warning("molecule: %s", warning)
    return molecule, analysis


# ---------------------------------------------------------------------------
# Stage: surfaces
# ---------------------------------------------------------------------------


def stage_surfaces(context: RunContext, molecule: Atoms) -> list[SurfaceModel]:
    """Bulk relaxation, terminations, slab supercells, slab relaxation and sites."""

    config = context.config
    surface = config.surface
    run_dir = context.run_dir
    state = context.state
    calculators = context.calculators

    solid = read_structure(config.solid)
    try:
        check_calculator_support(config, set(molecule.get_chemical_symbols()) | set(solid.get_chemical_symbols()))
    except ConfigError as exc:
        raise ConfigError(f"calculator.name: {exc}") from None

    size = molecule_size(molecule)
    vacuum = size + AUTO_VACUUM_MARGIN if surface.vacuum_above == AUTO else float(surface.vacuum_above)
    min_lateral = size + surface.lateral_buffer if surface.min_lateral == AUTO else float(surface.min_lateral)
    kind, vacuum_axis = classify_solid(solid, surface.input_type)
    logger.info(
        "solid %s: %s (input_type %s)%s",
        solid.get_chemical_formula(),
        kind,
        surface.input_type,
        f", vacuum along cell axis {vacuum_axis}" if kind == "slab" else "",
    )
    logger.info(
        "molecule size %.2f Å -> vacuum above slab %.2f Å, minimum lateral width %.2f Å", size, vacuum, min_lateral
    )

    terminations_dir = run_dir / "terminations"
    selection_file = terminations_dir / "terminations.json"
    if state.stage_status("terminations") == "done" and selection_file.is_file():
        selected = json.loads(selection_file.read_text(encoding="utf-8"))["selected"]
        units = {item["term_id"]: (item, read(terminations_dir / item["term_id"] / "unit_slab.extxyz")) for item in selected}
    else:
        started = time.perf_counter()
        if kind == "bulk":
            candidates = _bulk_terminations(context, solid, vacuum)
        else:
            unit = orient_slab(solid, vacuum_axis if vacuum_axis is not None else 2, vacuum)
            candidates = [
                {
                    "miller": None,
                    "shift": None,
                    "symmetric": None,
                    "stoichiometric": None,
                    "surface_energy": None,
                    "atoms": unit,
                    "n_atoms": len(unit),
                    "kept": True,
                    "reason": "pre-cut slab from the input file",
                }
            ]
        units = {}
        counters: dict[Any, int] = {}
        for candidate in candidates:
            if not candidate["kept"]:
                continue
            key = candidate["miller"]
            index = counters.get(key, 0)
            counters[key] = index + 1
            term_id = termination_id(candidate["miller"], index)
            candidate["term_id"] = term_id
            directory = terminations_dir / term_id
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / "unit_slab.extxyz", candidate["atoms"], format="extxyz")
            units[term_id] = (candidate, candidate["atoms"])
        _write_selection(selection_file, candidates)
        state.add_elapsed("terminations", time.perf_counter() - started)
        state.set_stage("terminations", "done")

    models = []
    for term_id, (meta, unit) in units.items():
        models.append(_termination_model(context, term_id, meta, unit, min_lateral))
    return models


def _bulk_terminations(context: RunContext, solid: Atoms, vacuum: float) -> list[dict[str, Any]]:
    config = context.config
    surface = config.surface
    budget = config.budget
    calculators = context.calculators
    bulk_dir = context.run_dir / "bulk"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    bulk, converted = conventional_bulk(solid)
    if converted:
        logger.info(
            "input cell is primitive (%d atoms); Miller indices refer to the conventional cell (%d atoms)",
            len(solid),
            len(bulk),
        )
    else:
        logger.info("Miller indices refer to the cell axes of the input file")

    started = time.perf_counter()
    if surface.relax_bulk:
        result = relax_structure(
            bulk,
            calculators,
            bulk_dir,
            "bulk_opt",
            optimizer=budget.relax_optimizer,
            fmax=budget.fmax,
            max_steps=budget.max_steps,
            cell_filter=True,
            logfile=bulk_dir / "bulk_opt.log",
        )
        if not result.converged:
            logger.warning("bulk relaxation did not converge in %d steps", budget.max_steps)
        before, after = bulk.cell.cellpar(), result.atoms.cell.cellpar()
        logger.info(
            "bulk relaxed: a b c %s -> %s Å",
            " ".join(f"{x:.3f}" for x in before[:3]),
            " ".join(f"{x:.3f}" for x in after[:3]),
        )
        bulk = result.atoms
    else:
        bulk = single_point(bulk, calculators.final)
        write(bulk_dir / "bulk_final.extxyz", bulk, format="extxyz")
    bulk_energy = float(bulk.get_potential_energy())
    _db_upsert(context, bulk, kind="bulk", name="bulk", energy_final=bulk_energy)
    context.state.add_elapsed("bulk", time.perf_counter() - started)
    context.state.set_stage("bulk", "done")

    candidates: list[dict[str, Any]] = []
    for miller in surface.miller_indices:
        generated = generate_slabs(bulk, miller, surface.min_slab_thickness, vacuum)
        entries = []
        for order, slab in enumerate(generated):
            gamma = None
            if slab.stoichiometric:
                energy = float(single_point(slab.atoms, calculators.final).get_potential_energy())
                gamma = surface_energy(slab.atoms, energy, bulk, bulk_energy)
            entries.append(
                {
                    "miller": tuple(miller),
                    "shift": slab.shift,
                    "symmetric": slab.symmetric,
                    "stoichiometric": slab.stoichiometric,
                    "surface_energy": gamma,
                    "atoms": slab.atoms,
                    "n_atoms": len(slab.atoms),
                    "order": order,
                }
            )
        # Stoichiometric slabs by surface energy, then the others by symmetry,
        # size and pymatgen's order.
        entries.sort(
            key=lambda e: (
                e["surface_energy"] is None,
                e["surface_energy"] if e["surface_energy"] is not None else 0.0,
                not e["symmetric"],
                e["n_atoms"],
                e["order"],
            )
        )
        for rank, entry in enumerate(entries):
            entry["kept"] = rank < surface.max_terminations
            if entry["surface_energy"] is not None:
                entry["reason"] = f"surface energy rank {rank + 1} of {len(entries)} (MLIP single point, unrelaxed)"
            else:
                entry["reason"] = "non-stoichiometric: no surface energy; ranked by symmetry and atom count"
            if not entry["kept"]:
                entry["reason"] += f"; beyond max_terminations={surface.max_terminations}"
        logger.info(
            "Miller %s: %d termination(s), keeping %d",
            "".join(str(i) for i in miller),
            len(entries),
            sum(e["kept"] for e in entries),
        )
        for entry in entries:
            gamma = entry["surface_energy"]
            logger.info(
                "  shift %.4f: %d atoms, %s, %s, gamma %s -> %s",
                entry["shift"],
                entry["n_atoms"],
                "symmetric" if entry["symmetric"] else "asymmetric",
                "stoichiometric" if entry["stoichiometric"] else "non-stoichiometric",
                f"{gamma * 1000:.1f} meV/Å²" if gamma is not None else "n/a",
                "kept" if entry["kept"] else "dropped",
            )
        candidates.extend(entries)
    return candidates


def _write_selection(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if key not in ("atoms", "order")}
        row["formula"] = candidate["atoms"].get_chemical_formula()
        rows.append(row)
    data = {"candidates": rows, "selected": [row for row in rows if row["kept"]]}
    path.write_text(json.dumps(data, indent=1, default=list), encoding="utf-8")


def _termination_model(
    context: RunContext,
    term_id: str,
    meta: dict[str, Any],
    unit: Atoms,
    min_lateral: float,
) -> SurfaceModel:
    config = context.config
    budget = config.budget
    state = context.state
    directory = context.run_dir / "terminations" / term_id
    model_file = directory / "slab.json"

    if state.item_status("surface", term_id) == "done" and model_file.is_file():
        logger.info("%s: slab already relaxed, reusing", term_id)
        return SurfaceModel.from_dict(json.loads(model_file.read_text(encoding="utf-8")), directory)

    state.set_item("surface", term_id, "running")
    started = time.perf_counter()
    repeats = lateral_repeats(unit, min_lateral)
    supercell = unit.repeat((repeats[0], repeats[1], 1))
    fixed = fix_bottom_layers(supercell, config.surface.fix_fraction)
    logger.info(
        "%s: %dx%d supercell, %d atoms (%d fixed), widths %.2f x %.2f Å",
        term_id,
        repeats[0],
        repeats[1],
        len(supercell),
        len(fixed),
        *perpendicular_widths(supercell),
    )
    result = relax_structure(
        supercell,
        context.calculators,
        directory,
        "slab_opt",
        optimizer=budget.relax_optimizer,
        fmax=budget.fmax,
        max_steps=budget.max_steps,
        logfile=directory / "slab_opt.log",
    )
    if not result.converged:
        logger.warning("%s: slab relaxation did not converge in %d steps", term_id, budget.max_steps)
    slab = result.atoms
    sites = find_sites(unit, slab, config.sampling.site_types, config.sampling.surface_depth)
    (directory / "sites.json").write_text(json.dumps([s.to_dict() for s in sites], indent=1), encoding="utf-8")
    layers = top_layer_compositions(slab)
    logger.info(
        "%s: E_slab %.4f eV, %d sites (%s); top layers %s",
        term_id,
        result.energy_final,
        len(sites),
        ", ".join(site.site_id for site in sites),
        " / ".join(layer["formula"] for layer in layers),
    )

    model = SurfaceModel(
        term_id=term_id,
        miller=tuple(meta["miller"]) if meta["miller"] is not None else None,
        shift=meta["shift"],
        symmetric=meta["symmetric"],
        stoichiometric=meta["stoichiometric"],
        surface_energy=meta["surface_energy"],
        repeats=repeats,
        n_atoms=len(slab),
        n_fixed=len(fixed),
        thickness=slab_thickness(slab),
        widths=perpendicular_widths(slab),
        energy_final=result.energy_final,
        energy_screen=result.energy_screen,
        converged=result.converged,
        top_layers=tuple(layers),
        sites=tuple(sites),
        slab_file=directory / "slab_opt_final.extxyz",
        unit_slab_file=directory / "unit_slab.extxyz",
    )
    model_file.write_text(json.dumps(model.to_dict(), indent=1), encoding="utf-8")
    _db_upsert(context, slab, kind="slab", name=term_id, energy_final=result.energy_final)
    elapsed = time.perf_counter() - started
    state.set_item("surface", term_id, "done", energy_final=result.energy_final, elapsed=elapsed)
    return model


def _db_upsert(context: RunContext, atoms: Atoms, kind: str, name: str, **values: Any) -> None:
    """Store `atoms` in ``results.db``, replacing an earlier row of the same kind and name."""

    from ase.db import connect

    # ASE database key-value pairs cannot hold None.
    values = {key: value for key, value in values.items() if value is not None}
    with connect(context.run_dir / RESULTS_DB) as database:
        stale = [row.id for row in database.select(kind=kind, name=name)]
        if stale:
            database.delete(stale)
        database.write(atoms, kind=kind, name=name, **values)


# ---------------------------------------------------------------------------
# Stage: gas-phase reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GasReference:
    """Gas-phase reference: energies of the MLIP minimum and the sampling geometry.

    `molecule` / `analysis` are the geometry used for sampling and the
    intact-molecule bond graph: the optimized molecule, or the input one if
    the optimization changed the connectivity (`geometry` says which).
    """

    molecule: Atoms
    analysis: MoleculeAnalysis
    energy_screen: float
    energy_final: float
    converged: bool
    bonds_changed: bool
    geometry: str = "optimized"


def stage_gas_reference(context: RunContext, molecule: Atoms, input_analysis: MoleculeAnalysis) -> GasReference:
    """Optimize the molecule in a periodic box and compute E_mol.

    The molecule is relaxed with the screen calculator in the same kind of
    rectangular box as the VASP reference, then a final-dtype single point
    gives E_mol (the MLIP's gas-phase minimum, as the E_ads definition
    requires).

    The optimized geometry and its bond graph are used for sampling and as
    the intact-molecule reference, unless the optimization changed the
    connectivity (e.g. an MLIP slipping an eta5-Cp ring to eta2): then the
    input geometry, whose bonds the user checked with ``check-molecule``, is
    kept for both, and a warning is logged.
    """

    config = context.config
    directory = context.run_dir / "molecule"
    reference_file = directory / "gas_reference.json"
    props = config.molecule_props

    if context.state.stage_status("gas") == "done" and reference_file.is_file():
        data = json.loads(reference_file.read_text(encoding="utf-8"))
        if data.get("geometry", "optimized") == "optimized":
            final = read(directory / "gas_opt_final.extxyz")
            sampled = Atoms(final.get_chemical_symbols(), positions=final.get_positions())
        else:
            sampled = molecule
        return GasReference(
            molecule=sampled,
            analysis=analyze_molecule(sampled, props),
            energy_screen=data["energy_screen"],
            energy_final=data["energy_final"],
            converged=data["converged"],
            bonds_changed=data["bonds_changed"],
            geometry=data.get("geometry", "optimized"),
        )

    started = time.perf_counter()
    boxed = molecule_in_box(molecule, config.vasp.molecule_box_padding)
    result = relax_structure(
        boxed,
        context.calculators,
        directory,
        "gas_opt",
        optimizer=config.budget.relax_optimizer,
        fmax=config.budget.fmax,
        max_steps=config.budget.max_steps,
        logfile=directory / "gas_opt.log",
    )
    optimized = Atoms(result.atoms.get_chemical_symbols(), positions=result.atoms.get_positions())
    analysis = analyze_molecule(optimized, props)
    before = {(i, j) for i, j, _ in input_analysis.bonds}
    after = {(i, j) for i, j, _ in analysis.bonds}
    bonds_changed = before != after
    geometry = "optimized"
    if bonds_changed:
        geometry = "input"
        logger.warning(
            "gas-phase optimization changed the bond graph (broken %s, formed %s); this is an MLIP "
            "artifact to check with DFT. E_mol is still the MLIP gas-phase minimum, but the input "
            "geometry is used for sampling and as the intact-molecule reference",
            sorted(before - after) or "none",
            sorted(after - before) or "none",
        )
        analysis = input_analysis
        sampled = Atoms(molecule.get_chemical_symbols(), positions=molecule.get_positions())
    else:
        sampled = optimized
    if not result.converged:
        logger.warning("gas-phase optimization did not converge in %d steps", config.budget.max_steps)
    (directory / "molecule_analysis.json").write_text(json.dumps(analysis.to_dict(), indent=1), encoding="utf-8")
    reference = {
        "energy_screen": result.energy_screen,
        "energy_final": result.energy_final,
        "converged": result.converged,
        "steps": result.steps,
        "bonds_changed": bonds_changed,
        "geometry": geometry,
        "box": boxed.cell.lengths().tolist(),
    }
    reference_file.write_text(json.dumps(reference, indent=1), encoding="utf-8")
    _db_upsert(context, result.atoms, kind="molecule", name="gas", energy_final=result.energy_final)
    context.state.add_elapsed("gas", time.perf_counter() - started)
    context.state.set_stage("gas", "done")
    logger.info(
        "gas-phase molecule: E_mol %.4f eV (box %s Å, %s after %d steps)",
        result.energy_final,
        " x ".join(f"{x:.1f}" for x in boxed.cell.lengths()),
        "converged" if result.converged else "not converged",
        result.steps,
    )
    return GasReference(
        sampled, analysis, result.energy_screen, result.energy_final, result.converged, bonds_changed, geometry
    )


# ---------------------------------------------------------------------------
# Stage: sampling (initial configurations, prescreening, full relaxations)
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = ("config_id", "site_id", "site_kind", "anchor", "spin", "source", "initial_tilt")
PRESCREEN_COLUMNS = CANDIDATE_COLUMNS + ("status", "energy", "steps", "elapsed", "contact", "facing", "reason")
RESULT_COLUMNS = CANDIDATE_COLUMNS + (
    "status",
    "eads_screen",
    "energy_screen",
    "prescreen_energy",
    "converged",
    "steps",
    "elapsed",
    "contact",
    "reason",
)


@dataclass(frozen=True)
class TerminationResult:
    """Sampling outcome of one termination."""

    term_id: str
    results_csv: Path
    prescreen_csv: Path
    n_candidates: int
    n_prescreened: int
    n_relaxed: int
    n_failed: int
    best_config_id: str | None
    best_eads_screen: float | None
    budget_used: float
    warnings: tuple[str, ...] = ()
    unique_csv: Path | None = None
    n_unique: int = 0
    best_unique_id: str | None = None
    best_eads: float | None = None
    best_class: str | None = None
    class_counts: tuple[tuple[str, int], ...] = ()
    scan_csv: Path | None = None


def stage_sampling(context: RunContext, surface: SurfaceModel, gas: GasReference) -> TerminationResult:
    """Initial configurations, prescreening and full relaxations of one termination."""

    config = context.config
    budget = config.budget
    state = context.state
    term = surface.term_id
    directory = context.run_dir / "terminations" / term
    slab = surface.slab()
    n_slab = len(slab)
    warnings: list[str] = []

    candidates = _candidates(context, surface, slab, gas)
    by_id = {candidate.config_id: candidate for candidate in candidates}

    budget_key = f"sampling/{term}"
    clock = BudgetClock(total=budget.wall_time_per_termination, spent_before=state.elapsed(budget_key))
    checkpoint = {"time": time.perf_counter()}

    def spend() -> None:
        now = time.perf_counter()
        state.add_elapsed(budget_key, now - checkpoint["time"])
        checkpoint["time"] = now

    if state.stage_status(budget_key) == "done":
        logger.info("%s: sampling already finished, reusing results", term)
        return _termination_result(context, surface, candidates, warnings)
    state.set_stage(budget_key, "running")

    calculator = _screen_calculator(context)
    step_time = measure_step_time(candidates[0].atoms(slab, gas.molecule), calculator)
    spend()
    logger.info(
        "%s: %d candidates; %.3f s per force call; budget %s (%.0f s used)",
        term,
        len(candidates),
        step_time,
        "unlimited" if clock.unlimited else f"{budget.wall_time_per_termination:.0f} s",
        clock.elapsed(),
    )

    # -- prescreening -----------------------------------------------------
    pre_stage = f"{term}/prescreen"
    # Stored in selection order (stratum representatives first).
    selected = list(state.items.get(pre_stage, {}))
    if not selected:
        n_pre = prescreen_count(len(candidates), budget, step_time)
        rng = np.random.default_rng(config.sampling.seed)
        chosen = stratified_sample(candidates, n_pre, key=lambda c: (c.site_kind, c.anchor), rng=rng)
        selected = [candidate.config_id for candidate in chosen]
        for cid in selected:
            state.items.setdefault(pre_stage, {})[cid] = {"status": "pending"}
        state.save()
        logger.info(
            "%s: prescreening %d of %d candidates (%d steps, fmax %.2f)",
            term,
            len(selected),
            len(candidates),
            budget.prescreen_steps,
            budget.fmax_prescreen,
        )

    prescreen_dir = directory / "prescreen"
    measured_steps = measured_time = 0.0
    for cid in state.remaining(pre_stage, selected):
        if clock.time_until(budget.prescreen_fraction) <= 0:
            left = len(state.remaining(pre_stage, selected))
            message = f"prescreening budget used up; {left} selected candidate(s) not prescreened"
            logger.warning("%s: %s", term, message)
            warnings.append(message)
            break
        state.set_item(pre_stage, cid, "running")
        candidate = by_id[cid]
        outcome = optimize_candidate(
            candidate.atoms(slab, gas.molecule),
            context.calculators,
            prescreen_dir,
            cid,
            optimizer=budget.prescreen_optimizer,
            fmax=budget.fmax_prescreen,
            max_steps=budget.prescreen_steps,
            n_slab=n_slab,
            write_pair=False,
        )
        measured_steps += max(outcome.steps, 1)
        measured_time += outcome.elapsed
        label = facing = ""
        if outcome.status == "done":
            label = _contact(outcome.atoms, n_slab, gas.analysis, config)
            facing = facing_label(outcome.atoms, n_slab, gas.analysis, config.analysis.contact_scale)
        state.set_item(
            pre_stage,
            cid,
            outcome.status,
            energy=outcome.energy,
            steps=outcome.steps,
            elapsed=round(outcome.elapsed, 3),
            contact=label,
            facing=facing,
            reason=outcome.reason,
        )
        spend()
        _write_prescreen_csv(directory / "prescreen.csv", candidates, state.items.get(pre_stage, {}))
    _write_prescreen_csv(directory / "prescreen.csv", candidates, state.items.get(pre_stage, {}))
    if measured_steps:
        step_time = measured_time / measured_steps  # includes optimizer overhead
        logger.info("%s: measured %.3f s per optimization step during prescreening", term, step_time)

    # -- full relaxations -------------------------------------------------
    relax_stage = f"{term}/relax"
    relax_end = budget.prescreen_fraction + budget.relax_fraction
    queue = [cid for cid in by_id if cid in state.items.get(relax_stage, {})]
    if not queue:
        # Diversity: contact label, or the fragment facing the surface when
        # prescreening ended before contact (see analysis.facing_label).
        records = [
            (cid, record["energy"], record.get("facing") or record.get("contact", ""))
            for cid, record in state.items.get(pre_stage, {}).items()
            if record.get("status") == "done"
        ]
        if not records:
            raise RuntimeError(f"{term}: no candidate survived prescreening")
        n_full = max(1, relax_count(len(records), budget, step_time, clock.time_until(relax_end)))
        queue = diverse_selection(records, n_full)
        for cid in queue:
            state.items.setdefault(relax_stage, {})[cid] = {"status": "pending"}
        state.save()
        logger.info(
            "%s: fully relaxing %d configuration(s) (%d distinct contact/facing labels among prescreened)",
            term,
            len(queue),
            len({record[2] for record in records}),
        )

    relax_dir = directory / "relax"
    finished = [cid for cid in queue if state.item_status(relax_stage, cid) in ("done", "failed")]
    step_counts = [state.item_record(relax_stage, cid).get("steps", 0) for cid in finished]
    relax_seconds = sum(state.item_record(relax_stage, cid).get("elapsed", 0.0) for cid in finished)
    for cid in state.remaining(relax_stage, queue):
        max_steps = budget.max_steps
        if not clock.unlimited:
            left = clock.time_until(relax_end)
            if sum(step_counts) > 0:
                # Relaxation steps near the surface cost more than prescreening steps.
                step_time = relax_seconds / sum(step_counts)
            average_steps = np.mean(step_counts) if step_counts else 0.5 * budget.max_steps
            if finished and left < average_steps * step_time:
                message = (
                    f"relaxation budget used up after {len(finished)} full relaxation(s); "
                    f"{len(state.remaining(relax_stage, queue))} planned configuration(s) skipped"
                )
                logger.warning("%s: %s", term, message)
                warnings.append(message)
                break
            max_steps = max(budget.prescreen_steps, min(budget.max_steps, int(max(left, 0.0) / step_time)))
        state.set_item(relax_stage, cid, "running")
        start_atoms = _prescreened_structure(prescreen_dir / f"{cid}.traj", by_id[cid], slab, gas.molecule)
        outcome = optimize_candidate(
            start_atoms,
            context.calculators,
            relax_dir,
            cid,
            optimizer=budget.relax_optimizer,
            fmax=budget.fmax,
            max_steps=max_steps,
            n_slab=n_slab,
            logfile=relax_dir / f"{cid}.log",
        )
        label = _contact(outcome.atoms, n_slab, gas.analysis, config) if outcome.status == "done" else ""
        eads = None
        if outcome.status == "done":
            eads = outcome.energy - surface.energy_screen - gas.energy_screen
        state.set_item(
            relax_stage,
            cid,
            outcome.status,
            energy_screen=outcome.energy,
            eads_screen=eads,
            converged=outcome.converged,
            steps=outcome.steps,
            elapsed=round(outcome.elapsed, 3),
            contact=label,
            reason=outcome.reason,
        )
        spend()
        finished.append(cid)
        step_counts.append(outcome.steps)
        relax_seconds += outcome.elapsed
        if outcome.atoms is not None:
            system = outcome.atoms
            system.info.update(by_id[cid].metadata())
            _db_upsert(
                context,
                system,
                kind="adsorbate",
                name=f"{term}/{cid}",
                termination=term,
                config_id=cid,
                site_id=by_id[cid].site_id,
                anchor=by_id[cid].anchor,
                status=outcome.status,
                energy_screen=outcome.energy,
                eads_screen=eads,
                converged=outcome.converged,
                steps=outcome.steps,
                elapsed=outcome.elapsed,
                contact=label,
            )
        _write_results_csv(directory / "results.csv", candidates, state, term)
        logger.info(
            "%s: %s %s%s (%d steps, %.1f s)",
            term,
            cid,
            outcome.status,
            f", E_ads(screen) {eads:.3f} eV, contact {label}" if eads is not None else f": {outcome.reason}",
            outcome.steps,
            outcome.elapsed,
        )

    _write_results_csv(directory / "results.csv", candidates, state, term)
    n_done = sum(1 for cid in queue if state.item_status(relax_stage, cid) == "done")
    if n_done < budget.min_full_relax:
        message = (
            f"only {n_done} full relaxation(s) completed (min_full_relax={budget.min_full_relax}); "
            "increase budget.wall_time_per_termination"
        )
        logger.warning("%s: %s", term, message)
        warnings.append(message)
    state.stages.setdefault(budget_key, {})["warnings"] = list(warnings)
    state.set_stage(budget_key, "done")
    return _termination_result(context, surface, candidates, warnings)


def _candidates(context: RunContext, surface: SurfaceModel, slab: Atoms, gas: GasReference) -> list[Candidate]:
    """Generate (or reload) the initial configurations of a termination."""

    term = surface.term_id
    directory = context.run_dir / "terminations" / term
    index_file = directory / "candidates.json"
    stage = f"candidates/{term}"
    if context.state.stage_status(stage) == "done" and index_file.is_file():
        metadata = json.loads(index_file.read_text(encoding="utf-8"))
        candidates = []
        for row in metadata:
            atoms = read(directory / "candidates" / f"{row['config_id']}.extxyz")
            candidates.append(
                Candidate(
                    config_id=row["config_id"],
                    site_id=row["site_id"],
                    site_kind=row["site_kind"],
                    anchor=row["anchor"],
                    spin=row["spin"],
                    source=row["source"],
                    tilt=row["initial_tilt"],
                    molecule_positions=atoms.positions[len(slab) :].copy(),
                )
            )
        return candidates

    started = time.perf_counter()
    candidates, report = generate_candidates(
        slab, surface.sites, gas.molecule, gas.analysis, context.config.sampling
    )
    if not candidates:
        raise RuntimeError(f"{term}: no initial configuration survived the clash checks")
    (directory / "candidates").mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        write(directory / "candidates" / f"{candidate.config_id}.extxyz", candidate.atoms(slab, gas.molecule))
    index_file.write_text(json.dumps([c.metadata() for c in candidates], indent=1), encoding="utf-8")
    context.state.add_elapsed(stage, time.perf_counter() - started)
    context.state.set_stage(stage, "done")
    logger.info(
        "%s: %d initial configurations (%d generated; dropped %d clashing, %d self-image, %d duplicates)",
        term,
        report.kept,
        report.generated,
        report.clash,
        report.self_image,
        report.duplicate,
    )
    return candidates


def _screen_calculator(context: RunContext):
    from ..calculators import get_calculator

    return get_calculator(context.calculators.screen)


def _contact(atoms: Atoms | None, n_slab: int, analysis: MoleculeAnalysis, config: AdsorptionConfig) -> str:
    if atoms is None:
        return ""
    return contact_label(atoms, n_slab, analysis, config.analysis.contact_scale)


def _prescreened_structure(trajectory: Path, candidate: Candidate, slab: Atoms, molecule: Atoms) -> Atoms:
    """Last prescreening frame (with the slab constraints), or the initial structure."""

    initial = candidate.atoms(slab, molecule)
    if not trajectory.is_file():
        return initial
    try:
        last = read(trajectory, index=-1)
    except Exception:  # noqa: BLE001 - an unreadable (interrupted) trajectory: start over
        return initial
    initial.positions = last.positions
    return initial


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})
    temporary.replace(path)


def _write_prescreen_csv(path: Path, candidates: list[Candidate], records: dict[str, dict]) -> None:
    rows = []
    for candidate in candidates:
        if candidate.config_id in records:
            rows.append({**candidate.metadata(), **records[candidate.config_id]})
    _write_csv(path, PRESCREEN_COLUMNS, rows)


def _write_results_csv(path: Path, candidates: list[Candidate], state: RunState, term: str) -> None:
    relax = state.items.get(f"{term}/relax", {})
    prescreen = state.items.get(f"{term}/prescreen", {})
    rows = []
    for candidate in candidates:
        record = relax.get(candidate.config_id)
        if record is None or record.get("status") not in ("done", "failed"):
            continue
        row = {**candidate.metadata(), **record}
        row["prescreen_energy"] = prescreen.get(candidate.config_id, {}).get("energy")
        rows.append(row)
    rows.sort(key=lambda row: (row["status"] != "done", row.get("eads_screen") or 0.0))
    _write_csv(path, RESULT_COLUMNS, rows)


def _termination_result(
    context: RunContext, surface: SurfaceModel, candidates: list[Candidate], warnings: list[str]
) -> TerminationResult:
    return termination_result_from_state(context.state, context.run_dir, surface.term_id, len(candidates), warnings)


def termination_result_from_state(
    state: RunState, run_dir: Path, term: str, n_candidates: int, warnings: Sequence[str] = ()
) -> TerminationResult:
    """Sampling part of a `TerminationResult`, rebuilt from ``state.json``."""

    directory = run_dir / "terminations" / term
    prescreen = state.items.get(f"{term}/prescreen", {})
    relax = state.items.get(f"{term}/relax", {})
    done = {cid: record for cid, record in relax.items() if record.get("status") == "done"}
    best = min(done, key=lambda cid: done[cid]["eads_screen"]) if done else None
    record = state.stages.get(f"sampling/{term}", {})
    # Older runs stored the warnings joined in "message".
    stored = record.get("warnings") or ([record["message"]] if record.get("message") else [])
    return TerminationResult(
        term_id=term,
        results_csv=directory / "results.csv",
        prescreen_csv=directory / "prescreen.csv",
        n_candidates=n_candidates,
        n_prescreened=sum(1 for record in prescreen.values() if record.get("status") in ("done", "failed")),
        n_relaxed=len(done),
        n_failed=sum(1 for record in relax.values() if record.get("status") == "failed"),
        best_config_id=best,
        best_eads_screen=done[best]["eads_screen"] if best else None,
        budget_used=state.elapsed(f"sampling/{term}"),
        warnings=tuple(warnings) if warnings else tuple(stored),
    )


# ---------------------------------------------------------------------------
# Stage: analysis (final single points, classification, angles, scan)
# ---------------------------------------------------------------------------

ANALYSIS_COLUMNS = (
    "config_id",
    "class",
    "eads",
    "eads_screen",
    "contact",
    "tilt",
    "azimuth",
    "height",
    "surface_distorted",
    "max_surface_displacement",
    "energy_final",
    "energy_screen",
    "prescreen_energy",
    "converged",
    "steps",
    "elapsed",
    "site_id",
    "site_kind",
    "anchor",
    "spin",
    "source",
    "initial_tilt",
    "class_reason",
    "status",
    "reason",
)
UNIQUE_COLUMNS = ("config_id", "class", "eads", "boltzmann_weight", "n_duplicates", "duplicates") + tuple(
    column for column in ANALYSIS_COLUMNS if column not in ("config_id", "class", "eads", "status", "reason")
)
SUMMARY_COLUMNS = ("term_id",) + UNIQUE_COLUMNS
SCAN_COLUMNS = ("clearance", "center_height", "energy_final", "eads")


def stage_analysis(
    context: RunContext, surface: SurfaceModel, gas: GasReference, sampled: TerminationResult
) -> TerminationResult:
    """Final-dtype single points and analysis of the fully relaxed configurations.

    ``E_ads = E(slab+mol) - E(slab) - E(mol)`` with all three energies from
    the final-dtype calculator. Writes ``results.csv`` (all relaxed
    configurations) and ``unique.csv`` (de-duplicated, with Boltzmann
    weights) and, if enabled, the rigid approach scan.
    """

    config = context.config
    state = context.state
    term = surface.term_id
    directory = context.run_dir / "terminations" / term
    relax_dir = directory / "relax"
    slab = surface.slab()
    n_slab = len(slab)
    candidates = {c.config_id: c for c in _candidates(context, surface, slab, gas)}
    relax_records = state.items.get(f"{term}/relax", {})
    prescreen_records = state.items.get(f"{term}/prescreen", {})
    final_stage = f"{term}/final"
    budget_key = f"sampling/{term}"

    for cid, record in relax_records.items():
        if record.get("status") != "done" or state.item_status(final_stage, cid) in ("done", "failed"):
            continue
        started = time.perf_counter()
        state.set_item(final_stage, cid, "running")
        relaxed = read(relax_dir / f"{cid}.extxyz", index=-1)
        relaxed.set_constraint(slab.constraints)
        try:
            final = single_point(relaxed, context.calculators.final)
        except Exception as exc:  # noqa: BLE001 - record and continue
            state.set_item(final_stage, cid, "failed", reason=f"{type(exc).__name__}: {exc}")
        else:
            write(relax_dir / f"{cid}_final.extxyz", final, format="extxyz")
            state.set_item(final_stage, cid, "done", energy_final=float(final.get_potential_energy()))
        state.add_elapsed(budget_key, time.perf_counter() - started)

    props = config.molecule_props
    settings = config.analysis
    sign_atom = azimuth_sign_atom(gas.molecule, gas.analysis)
    rows: list[dict[str, Any]] = []
    positions: dict[str, np.ndarray] = {}
    for cid, record in relax_records.items():
        if record.get("status") not in ("done", "failed"):
            continue
        row: dict[str, Any] = {**candidates[cid].metadata(), **record}
        row["prescreen_energy"] = prescreen_records.get(cid, {}).get("energy")
        final_record = state.item_record(final_stage, cid)
        if record.get("status") != "done" or final_record.get("status") != "done":
            row["status"] = "failed"
            row["reason"] = record.get("reason") or final_record.get("reason", "")
            rows.append(row)
            continue
        final = read(relax_dir / f"{cid}_final.extxyz")
        molecule = molecule_part(final, n_slab)
        category, label, why = classify(
            final,
            n_slab,
            gas.analysis,
            props.bond_scale,
            settings.contact_scale,
            settings.desorbed_distance,
            props.bond_overrides,
        )
        displacement = surface_displacement(final, n_slab, slab)
        tilt = molecule_tilt(molecule, gas.analysis)
        azimuth = azimuth_angle(molecule, gas.analysis, sign_atom, final.cell.array)
        row.update(
            {
                "energy_final": final_record["energy_final"],
                "eads": final_record["energy_final"] - surface.energy_final - gas.energy_final,
                "class": category,
                "class_reason": why,
                "contact": label,
                "tilt": None if tilt is None else round(tilt, 2),
                "azimuth": None if azimuth is None else round(azimuth, 2),
                "height": round(adsorption_height(final, n_slab, gas.analysis), 3),
                "max_surface_displacement": round(displacement, 3),
                "surface_distorted": displacement > SURFACE_DISTORTION_LIMIT,
            }
        )
        positions[cid] = molecule.positions
        rows.append(row)
        final.info.update(candidates[cid].metadata())
        _db_upsert(
            context,
            final,
            kind="adsorbate",
            name=f"{term}/{cid}",
            termination=term,
            config_id=cid,
            site_id=row["site_id"],
            anchor=row["anchor"],
            status="done",
            energy_final=row["energy_final"],
            eads=row["eads"],
            eads_screen=row.get("eads_screen"),
            adsorption_class=category,
            contact=label,
            tilt=row["tilt"],
            azimuth=row["azimuth"],
            height=row["height"],
            surface_distorted=row["surface_distorted"],
            converged=row.get("converged"),
            steps=row.get("steps"),
            elapsed=row.get("elapsed"),
        )

    done = [row for row in rows if row.get("status") == "done"]
    done.sort(key=lambda row: row["eads"])
    failed = [row for row in rows if row.get("status") != "done"]
    _write_csv(directory / "results.csv", ANALYSIS_COLUMNS, done + failed)

    unique = deduplicate(done, positions, slab.cell.array, settings.dedup_energy_tol, settings.dedup_rmsd_tol)
    for row, weight in zip(unique, boltzmann_weights([row["eads"] for row in unique], settings.temperature)):
        row["boltzmann_weight"] = round(weight, 6)
        row["duplicates"] = " ".join(row["duplicates"])
    _write_csv(directory / "unique.csv", UNIQUE_COLUMNS, unique)

    counts: dict[str, int] = {}
    for row in unique:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    best = unique[0] if unique else None
    if best is not None:
        logger.info(
            "%s: %d unique configuration(s) (%s); most stable %s: E_ads %.3f eV, %s, contact %s",
            term,
            len(unique),
            ", ".join(f"{count} {name}" for name, count in counts.items()),
            best["config_id"],
            best["eads"],
            best["class"],
            best["contact"],
        )
    distorted = [row["config_id"] for row in done if row["surface_distorted"]]
    if distorted:
        logger.warning("%s: surface distorted (> %.1f Å) in %s", term, SURFACE_DISTORTION_LIMIT, ", ".join(distorted))

    scan_csv = None
    if settings.approach_scan and unique:
        scan_csv = _approach_scan(context, surface, gas, candidates, unique, slab)

    return dataclasses.replace(
        sampled,
        unique_csv=directory / "unique.csv",
        n_unique=len(unique),
        best_unique_id=best["config_id"] if best else None,
        best_eads=best["eads"] if best else None,
        best_class=best["class"] if best else None,
        class_counts=tuple(counts.items()),
        scan_csv=scan_csv,
    )


def scan_target(unique: list[dict]) -> dict:
    """Configuration used for the rigid approach scan.

    The most stable intact (chemisorbed or physisorbed) configuration, since
    the scan places the intact gas-phase molecule; the most stable overall
    if there is none.
    """

    intact = [row for row in unique if row["class"] in ("chemisorbed", "physisorbed")]
    return (intact or unique)[0]


def _approach_scan(
    context: RunContext,
    surface: SurfaceModel,
    gas: GasReference,
    candidates: dict[str, Candidate],
    unique: list[dict],
    slab: Atoms,
) -> Path:
    """Rigid scan of the gas-phase molecule approaching along z (final-dtype single points).

    The molecule keeps the initial orientation and site of the target
    configuration; the scan coordinate is the vertical clearance between its
    lowest atom and the topmost slab atom. Energies are stored in
    ``approach_scan.extxyz`` and ``approach_scan.csv``.
    """

    term = surface.term_id
    directory = context.run_dir / "terminations" / term
    scan_file = directory / "approach_scan.csv"
    stage = f"scan/{term}"
    if context.state.stage_status(stage) == "done" and scan_file.is_file():
        return scan_file

    started = time.perf_counter()
    target = scan_target(unique)
    candidate = candidates[target["config_id"]]
    start, stop, step = context.config.analysis.scan_heights
    clearances = np.arange(start, stop + 0.5 * step, step)
    top = slab.positions[:, 2].max()
    base = candidate.molecule_positions.copy()
    base[:, 2] -= base[:, 2].min() - top
    frames, rows = [], []
    for clearance in clearances:
        placed = base.copy()
        placed[:, 2] += clearance
        system = slab.copy() + Atoms(gas.molecule.get_chemical_symbols(), positions=placed)
        system.cell = slab.cell
        system.pbc = True
        result = single_point(system, context.calculators.final)
        energy = float(result.get_potential_energy())
        result.info.update({"config_id": target["config_id"], "clearance": round(float(clearance), 4)})
        frames.append(result)
        rows.append(
            {
                "clearance": round(float(clearance), 4),
                "center_height": round(adsorption_height(result, len(slab), gas.analysis), 4),
                "energy_final": energy,
                "eads": energy - surface.energy_final - gas.energy_final,
            }
        )
    write(directory / "approach_scan.extxyz", frames, format="extxyz")
    _write_csv(scan_file, SCAN_COLUMNS, rows)
    (directory / "approach_scan.json").write_text(
        json.dumps({"config_id": target["config_id"], "relaxed_eads": target["eads"], "class": target["class"]}),
        encoding="utf-8",
    )
    context.state.add_elapsed(f"sampling/{term}", time.perf_counter() - started)
    context.state.set_stage(stage, "done")
    best_row = min(rows, key=lambda row: row["eads"])
    logger.info(
        "%s: approach scan of %s over %d heights; minimum E_ads %.3f eV at %.2f Å clearance (relaxed %.3f eV)",
        term,
        target["config_id"],
        len(rows),
        best_row["eads"],
        best_row["clearance"],
        target["eads"],
    )
    return scan_file


def write_summary(context: RunContext, results: list[TerminationResult]) -> Path:
    """``summary.csv``: unique configurations of all terminations, sorted by E_ads."""

    rows = []
    for result in results:
        if result.unique_csv is None or not result.unique_csv.is_file():
            continue
        with result.unique_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"term_id": result.term_id, **row})
    rows.sort(key=lambda row: float(row["eads"]))
    path = context.run_dir / "summary.csv"
    _write_csv(path, SUMMARY_COLUMNS, rows)
    return path


def make_figures(context: RunContext, surfaces: list[SurfaceModel], results: list[TerminationResult]) -> Path | None:
    """Static analysis figures in ``figures/`` (skipped with a warning without matplotlib)."""

    try:
        from .plotting import plot_run
    except ImportError as exc:
        logger.warning("figures skipped: %s", exc)
        return None
    directory = context.run_dir / "figures"
    try:
        written = plot_run(context.run_dir, directory, surfaces, results, context.config)
    except Exception as exc:  # noqa: BLE001 - figures must not lose finished results
        logger.warning("figure generation failed: %s: %s", type(exc).__name__, exc)
        return None
    logger.info("%d figure(s) written to %s", len(written), directory)
    return directory


def load_run_summary(run_dir: str | Path) -> tuple[AdsorptionConfig, list[SurfaceModel], list[TerminationResult]]:
    """Configuration, surfaces and termination results of a run, read from its files.

    Nothing is recomputed; used by ``ase-adsorb report``.
    """

    run_dir = Path(run_dir).resolve()
    config = dataclasses.replace(load_config(run_dir / RESOLVED_CONFIG), run_dir=run_dir)
    state = RunState.load(run_dir)
    surfaces, results = [], []
    for term, record in state.items.get("surface", {}).items():
        directory = run_dir / "terminations" / term
        if record.get("status") != "done" or not (directory / "slab.json").is_file():
            continue
        surfaces.append(SurfaceModel.from_dict(json.loads((directory / "slab.json").read_text(encoding="utf-8")), directory))
        candidates_file = directory / "candidates.json"
        n_candidates = len(json.loads(candidates_file.read_text(encoding="utf-8"))) if candidates_file.is_file() else 0
        result = termination_result_from_state(state, run_dir, term, n_candidates)
        results.append(_with_unique(result, directory))
    return config, surfaces, results


def _with_unique(result: TerminationResult, directory: Path) -> TerminationResult:
    """Fill the analysis fields of a `TerminationResult` from ``unique.csv``."""

    unique_csv = directory / "unique.csv"
    if not unique_csv.is_file():
        return result
    with unique_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    best = rows[0] if rows else None
    scan = directory / "approach_scan.csv"
    return dataclasses.replace(
        result,
        unique_csv=unique_csv,
        n_unique=len(rows),
        best_unique_id=best["config_id"] if best else None,
        best_eads=float(best["eads"]) if best else None,
        best_class=best["class"] if best else None,
        class_counts=tuple(counts.items()),
        scan_csv=scan if scan.is_file() else None,
    )


def make_report(context: RunContext, draw_static: bool = False) -> Path | None:
    """Animations and ``report.html`` (failures are logged, never raised)."""

    try:
        from .report import generate_report

        return generate_report(context.run_dir, draw_static=draw_static)
    except Exception as exc:  # noqa: BLE001 - the report must not lose finished results
        logger.warning("report generation failed: %s: %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point for the surface part of the workflow
# ---------------------------------------------------------------------------


def prepare_surfaces(config: AdsorptionConfig, resume: bool = False) -> list[SurfaceModel]:
    """Run setup, molecule analysis and surface preparation.

    With ``resume=True`` the run in `config.resolved_run_dir()` is continued
    and finished terminations are reused.
    """

    run_dir = config.resolved_run_dir().resolve()
    with run_logging(run_dir):
        context = resume_run(run_dir) if resume else start_run(config)
        molecule, _ = stage_molecule(context)
        started = time.perf_counter()
        models = stage_surfaces(context, molecule)
        logger.info("surface preparation finished in %.1f s", time.perf_counter() - started)
        return models


# ---------------------------------------------------------------------------
# Complete workflow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdsorptionResult:
    """Paths and key numbers of an adsorption run (like `OptimizationResult`)."""

    run_dir: Path
    config_file: Path
    log_file: Path
    database: Path
    molecule_energy: float
    terminations: tuple[TerminationResult, ...]
    summary_csv: Path | None = None
    figures_dir: Path | None = None
    report: Path | None = None

    @property
    def best(self) -> TerminationResult | None:
        """Termination holding the lowest final E_ads (screening E_ads if not analyzed)."""

        def energy(t: TerminationResult) -> float | None:
            return t.best_eads if t.best_eads is not None else t.best_eads_screen

        ranked = [t for t in self.terminations if energy(t) is not None]
        return min(ranked, key=energy) if ranked else None


def run_adsorption_workflow(config: AdsorptionConfig) -> AdsorptionResult:
    """Run the adsorption workflow for `config` in a new run directory."""

    return _run(config.resolved_run_dir().resolve(), config)


def resume_adsorption_workflow(run_dir: str | Path) -> AdsorptionResult:
    """Continue an interrupted run from its ``config.resolved.yaml`` and ``state.json``.

    Finished items are skipped, interrupted ones restarted, and the remaining
    sampling budget accounts for the time already spent.
    """

    return _run(Path(run_dir).resolve(), None)


def _run(run_dir: Path, config: AdsorptionConfig | None) -> AdsorptionResult:
    with run_logging(run_dir):
        started = time.perf_counter()
        context = start_run(config) if config is not None else resume_run(run_dir)
        molecule, analysis = stage_molecule(context)
        gas = stage_gas_reference(context, molecule, analysis)
        surfaces = stage_surfaces(context, molecule)
        results = []
        for surface in surfaces:
            sampled = stage_sampling(context, surface, gas)
            results.append(stage_analysis(context, surface, gas, sampled))
        context.state.set_stage("sampling", "done")
        summary_csv = write_summary(context, results)
        figures_dir = make_figures(context, surfaces, results)
        report = make_report(context, draw_static=figures_dir is None)
        for result in results:
            logger.info(
                "%s: %d candidates, %d prescreened, %d relaxed (%d failed), %d unique; most stable %s: E_ads %s (%s); "
                "budget used %.0f s",
                result.term_id,
                result.n_candidates,
                result.n_prescreened,
                result.n_relaxed,
                result.n_failed,
                result.n_unique,
                result.best_unique_id,
                f"{result.best_eads:.3f} eV" if result.best_eads is not None else "n/a",
                result.best_class,
                result.budget_used,
            )
        logger.info("workflow finished in %.1f s (this session)", time.perf_counter() - started)
        return AdsorptionResult(
            run_dir=context.run_dir,
            config_file=context.run_dir / RESOLVED_CONFIG,
            log_file=context.run_dir / LOG_FILE,
            database=context.run_dir / RESULTS_DB,
            molecule_energy=gas.energy_final,
            terminations=tuple(results),
            summary_csv=summary_csv,
            figures_dir=figures_dir,
            report=report,
        )
