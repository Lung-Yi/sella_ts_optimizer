"""Workflow stages of the adsorption run.

Implemented so far (milestones M1-M2): run directory setup, molecule
analysis, bulk relaxation, slab generation and termination selection, slab
supercells, slab relaxation and adsorption sites. `prepare_surfaces()` runs
these stages and returns one `SurfaceModel` per kept termination.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ase import Atoms
from ase.io import read, write

from ..structures import read_structure
from .config import AUTO, AdsorptionConfig, ConfigError, check_calculator_support, dump_config, load_config
from .molecule import MoleculeAnalysis, analyze_molecule
from .relax import CalculatorSet, relax_structure, resolve_calculators, single_point
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
    resolved = dataclasses.replace(
        resolved,
        run_dir=run_dir,
        molecule=resolved.molecule.resolve(),
        solid=resolved.solid.resolve(),
    )
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
    config = load_config(config_path)
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

    with connect(context.run_dir / RESULTS_DB) as database:
        stale = [row.id for row in database.select(kind=kind, name=name)]
        if stale:
            database.delete(stale)
        database.write(atoms, kind=kind, name=name, **values)


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
