"""``report.html``: a single self-contained HTML report of an adsorption run.

Images are embedded as base64, so the file can be copied or mailed on its
own. `generate_report()` only reads the run directory (no calculation): it
(re)draws the figures and animations and writes the HTML; it is what
``ase-adsorb report`` runs.
"""

from __future__ import annotations

import base64
import csv
import datetime
import html
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from ase.io import read

from ..calculators import calculator_capabilities

logger = logging.getLogger("ase_structure_optimizer.adsorption")

REPORT_FILE = "report.html"
MAX_TABLE_ROWS = 40

DISCLAIMERS = (
    "All energies come from a machine-learned interatomic potential (MLIP). MLIP adsorption energies, "
    "and especially dissociation found by an MLIP on reactive surfaces, must be validated with DFT "
    "(see the VASP inputs) before being used.",
    "Periodic MLIPs are trained on almost no isolated molecules, so the gas-phase reference E_mol is an "
    "extrapolation; this affects every E_ads by the same constant.",
    "MLIP accuracy for organometallic molecules is limited (e.g. ring slippage in the gas phase).",
    "E_ads = E(slab+mol) - E(slab) - E(mol), all three from the same calculator, model, task, "
    "dispersion setting and final dtype; negative values are exothermic.",
)

CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3de; --card:#ffffff;
        --accent:#2a78d6; --warn:#fab219; --crit:#d03b3b; }
@media (prefers-color-scheme: dark) {
  :root { color-scheme: dark; --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#383835; --card:#232322; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 12px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 24px 0 8px; }
.meta, .muted { color: var(--ink2); }
nav { margin: 16px 0; display: flex; flex-wrap: wrap; gap: 6px 14px; }
nav a { color: var(--accent); text-decoration: none; }
.notice { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--warn);
          padding: 10px 14px; margin: 12px 0; border-radius: 4px; }
.notice.critical { border-left-color: var(--crit); }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px;
        min-width: 150px; }
.card .value { font-size: 20px; font-weight: 600; }
.card .label { color: var(--ink2); font-size: 12px; }
.table-wrap { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; font-size: 12.5px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: left; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
figure { margin: 12px 0; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; background: #fcfcfb; }
figcaption { color: var(--ink2); font-size: 12px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
@media (max-width: 480px) { .grid2 { grid-template-columns: 1fr; } }
details { margin: 8px 0; }
code { font-size: 12.5px; }
.animation { overflow-x: auto; }
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_report(run_dir: str | Path, draw_static: bool = True, draw_animations: bool = True) -> Path:
    """(Re)draw the figures and animations of a run and write ``report.html``.

    Reads only files of the run directory. `draw_static=False` reuses the
    static figures already in ``figures/``.
    """

    from .plotting import (
        approach_animation,
        plot_angle_definition,
        plot_molecule_axis,
        plot_run,
        relaxation_animation,
        relaxation_jshtml,
    )
    from .workflow import load_run_summary

    run_dir = Path(run_dir).resolve()
    config, surfaces, results = load_run_summary(run_dir)
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    if draw_static:
        plot_run(run_dir, figures, surfaces, results, config)

    molecule_data = _json(run_dir / "molecule" / "molecule_analysis.json")
    gas = _json(run_dir / "molecule" / "gas_reference.json")
    sampled = _sampled_molecule(run_dir, gas)
    analysis = _analysis_from_json(molecule_data)
    plot_molecule_axis(sampled, analysis, figures / "molecule_axis.png")
    directional = molecule_data["reference_axis"]["directional"]
    plot_angle_definition(figures / "angle_definition.png", directional)

    surface_by_id = {surface.term_id: surface for surface in surfaces}
    animations: dict[str, list[tuple[str, Path, Path]]] = {}
    player = ""
    if draw_animations and config.analysis.make_gif:
        for result in results:
            term_dir = run_dir / "terminations" / result.term_id
            surface = surface_by_id[result.term_id]
            items = animations.setdefault(result.term_id, [])
            if result.scan_csv is not None:
                target = _json(term_dir / "approach_scan.json")
                items.append(
                    ("rigid approach scan",)
                    + approach_animation(
                        result.term_id,
                        term_dir,
                        surface.n_atoms,
                        target["relaxed_eads"],
                        figures / f"approach_{result.term_id}.gif",
                        figures / f"approach_{result.term_id}.png",
                    )
                )
            for cid in _animated_configs(result):
                items.append(
                    (f"relaxation of {cid}",)
                    + relaxation_animation(
                        result.term_id,
                        cid,
                        term_dir,
                        surface.n_atoms,
                        surface.energy_screen,
                        gas["energy_screen"],
                        figures / f"relaxation_{result.term_id}_{cid}.gif",
                        figures / f"relaxation_{result.term_id}_{cid}.png",
                    )
                )
        best = _best_overall(results)
        if best is not None:
            surface = surface_by_id[best.term_id]
            player = relaxation_jshtml(
                best.term_id,
                best.best_unique_id,
                run_dir / "terminations" / best.term_id,
                surface.n_atoms,
                surface.energy_screen,
                gas["energy_screen"],
            )

    document = _render(run_dir, config, surfaces, results, molecule_data, gas, animations, player)
    path = run_dir / REPORT_FILE
    path.write_text(document, encoding="utf-8")
    logger.info("report written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(run_dir, config, surfaces, results, molecule_data, gas, animations, player) -> str:
    figures = run_dir / "figures"
    state = _json(run_dir / "state.json")
    capabilities = calculator_capabilities(config.calculator_config())
    title = f"Adsorption of {_formula(molecule_data['symbols'])} on {config.solid.stem}"
    parts: list[str] = []
    add = parts.append

    add(f"<h1>{_e(title)}</h1>")
    add(
        f"<p class='meta'>Run directory <code>{_e(run_dir)}</code> · generated "
        f"{_e(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))} · calculator {_e(config.calculator.name)} "
        f"({_e(capabilities.level_of_theory)})</p>"
    )
    sections = [
        ("overview", "Overview"),
        ("settings", "Settings"),
        ("timing", "Timing and budget"),
        ("molecule", "Molecule"),
        ("angles", "Angle definitions"),
    ]
    sections += [(f"term-{r.term_id}", f"Termination {r.term_id}") for r in results]
    sections += [("comparison", "Terminations compared"), ("warnings", "Warnings"), ("vasp", "VASP selection")]
    if player:
        sections.append(("animation", "Relaxation animation"))
    add("<nav>" + "".join(f"<a href='#{key}'>{_e(label)}</a>" for key, label in sections) + "</nav>")

    for text in DISCLAIMERS[:1]:
        add(f"<div class='notice critical'>{_e(text)}</div>")

    # Overview
    add("<h2 id='overview'>Overview</h2><div class='cards'>")
    for result in results:
        value = f"{result.best_eads:.2f} eV" if result.best_eads is not None else "n/a"
        add(
            f"<div class='card'><div class='label'>{_e(result.term_id)}: most stable</div>"
            f"<div class='value'>{_e(value)}</div><div class='label'>{_e(result.best_unique_id or '')} · "
            f"{_e(result.best_class or '')}</div></div>"
        )
        intact = _best_intact(result)
        if intact is not None and intact["config_id"] != result.best_unique_id:
            add(
                f"<div class='card'><div class='label'>{_e(result.term_id)}: most stable intact</div>"
                f"<div class='value'>{float(intact['eads']):.2f} eV</div><div class='label'>"
                f"{_e(intact['config_id'])} · {_e(intact['class'])}</div></div>"
            )
    add("</div>")
    add(_table(
        ["termination", "candidates", "prescreened", "relaxed", "failed", "unique", "classes", "budget used (s)"],
        [
            [
                r.term_id,
                r.n_candidates,
                r.n_prescreened,
                r.n_relaxed,
                r.n_failed,
                r.n_unique,
                ", ".join(f"{n} {c}" for c, n in r.class_counts),
                f"{r.budget_used:.0f} / {config.budget.wall_time_per_termination:.0f}"
                if config.budget.wall_time_per_termination > 0
                else f"{r.budget_used:.0f} (unlimited)",
            ]
            for r in results
        ],
        numeric={1, 2, 3, 4, 5},
    ))

    # Settings
    add("<h2 id='settings'>Settings</h2>")
    calc = config.calculator
    rows = [
        ["molecule", f"{config.molecule.name} ({_formula(molecule_data['symbols'])}), charge "
         f"{config.molecule_props.charge}, multiplicity {config.molecule_props.multiplicity}"],
        ["solid", config.solid.name],
        ["calculator", f"{calc.name}, model {calc.mace_mp_model if calc.name == 'macemp' else calc.name}"
         + (f", task {calc.uma_task}" if calc.name in ('uma_s', 'uma_m') else "")],
        ["level of theory", capabilities.level_of_theory],
        ["dispersion (macemp)", "D3(BJ)" if calc.dispersion else "none"],
        ["dtype screening / final", f"{calc.dtype_screen} / {calc.dtype_final}"],
        ["Miller indices", ", ".join("(" + " ".join(str(i) for i in m) + ")" for m in config.surface.miller_indices)],
        ["slab", f"min thickness {config.surface.min_slab_thickness} Å, fix fraction {config.surface.fix_fraction}, "
         f"vacuum {config.surface.vacuum_above}, min lateral {config.surface.min_lateral}"],
        ["sampling", f"sites {', '.join(config.sampling.site_types)}; {config.sampling.spins_per_anchor} spins per "
         f"anchor; {config.sampling.n_random} random; contact gap {config.sampling.contact_gap}; seed "
         f"{config.sampling.seed}"],
        ["budget", f"{config.budget.wall_time_per_termination:.0f} s per termination; prescreen "
         f"{config.budget.prescreen_steps} steps ({config.budget.prescreen_optimizer}, fmax "
         f"{config.budget.fmax_prescreen}); relax {config.budget.relax_optimizer}, fmax {config.budget.fmax}, "
         f"max {config.budget.max_steps} steps"],
        ["analysis", f"contact scale {config.analysis.contact_scale}, desorbed > {config.analysis.desorbed_distance} Å, "
         f"dedup {config.analysis.dedup_energy_tol} eV / {config.analysis.dedup_rmsd_tol} Å, T "
         f"{config.analysis.temperature} K"],
    ]
    add(_table(["setting", "value"], rows))
    add("<p class='muted'>Complete settings: <code>config.resolved.yaml</code>.</p>")

    # Timing
    add("<h2 id='timing'>Timing and budget</h2>")
    stages = state.get("stages", {})
    timing_rows = [[name, f"{record.get('elapsed', 0.0):.1f}", record.get("status", "")]
                   for name, record in stages.items()]
    add(_table(["stage", "elapsed (s)", "status"], timing_rows, numeric={1}))
    add("<p class='muted'>Surface and molecule preparation are not part of the sampling budget; "
        "<code>sampling/&lt;termination&gt;</code> is (prescreening, relaxations, final single points, scan).</p>")

    # Molecule
    add("<h2 id='molecule'>Molecule</h2><div class='grid2'><div>")
    add(_table(
        ["fragment", "formula", "atoms"],
        [[f["name"], f["formula"], " ".join(str(i) for i in f["indices"])] for f in molecule_data["fragments"]],
    ))
    axis = molecule_data["reference_axis"]
    add(f"<p><b>Reference axis u:</b> {_e(axis['method'])} — {_e(axis['description'])}"
        + (f"; atoms {axis['start']} → {axis['end']}" if axis["method"] != "undefined" else "") + "</p>")
    add(f"<p><b>Gas-phase reference:</b> E_mol = {gas['energy_final']:.4f} eV "
        f"({'converged' if gas['converged'] else 'not converged'}, {gas.get('steps', '?')} steps).</p>")
    if gas.get("bonds_changed"):
        add("<div class='notice'>The MLIP changed the molecule's bonds during the gas-phase optimization. "
            "E_mol is the MLIP gas-phase minimum, but the input geometry (checked with "
            "<code>ase-adsorb check-molecule</code>) is used for sampling and as the intact-molecule reference. "
            "Rigid scans therefore level off at the strain energy of the input geometry rather than at 0.</div>")
    add("<details><summary>Anchors and bonds</summary>")
    add(_table(["anchor", "kind", "atoms", "merged"],
               [[a["label"], a["kind"], " ".join(map(str, a["atom_indices"])), ", ".join(a["merged"])]
                for a in molecule_data["anchors"]]))
    symbols = molecule_data["symbols"]
    add(_table(["bond", "distance (Å)"],
               [[f"{symbols[i]}{i}–{symbols[j]}{j}", f"{d:.3f}"] for i, j, d in molecule_data["bonds"]], numeric={1}))
    add("</details></div>")
    add(_figure(figures / "molecule_axis.png", "Molecule with the reference axis u (fragments labeled)."))
    add("</div>")

    # Angles
    add("<h2 id='angles'>Angle definitions</h2>")
    add("<p><b>Tilt θ</b> (main measure): angle between the molecular reference axis u and the surface normal +z, "
        "0–180°. θ = 0°: u points away from the surface; θ = 180°: u points toward the surface; θ = 90°: u lies "
        "parallel to the surface. For an axis without head or tail (the long principal axis of a metal-free "
        "molecule, or a planar molecule's normal) θ is folded into 0–90°.</p>")
    add("<p><b>Azimuth φ</b> (secondary): the principal axis perpendicular to u with the smallest moment (the "
        "molecule's longest extent across u), projected onto the surface plane, measured from cell vector a, "
        "0–360°. <b>Contact label</b>: the fragments that touch the surface, e.g. <code>C5H5</code> "
        "or <code>CO_1+H</code>; complements θ.</p>")
    add(_figure(figures / "angle_definition.png", "Tilt θ (side views) and azimuth φ (top view)."))

    # Terminations
    for result in results:
        surface = next(s for s in surfaces if s.term_id == result.term_id)
        term = result.term_id
        add(f"<h2 id='term-{_e(term)}'>Termination {_e(term)}</h2>")
        layers = " / ".join(layer["formula"] for layer in surface.top_layers)
        gamma = f"{surface.surface_energy * 1000:.1f} meV/Å²" if surface.surface_energy is not None else "n/a"
        add(f"<p>{surface.repeats[0]}×{surface.repeats[1]} supercell, {surface.n_atoms} atoms ({surface.n_fixed} "
            f"fixed), thickness {surface.thickness:.2f} Å, lateral widths {surface.widths[0]:.1f} × "
            f"{surface.widths[1]:.1f} Å; surface energy {gamma}; top layers {_e(layers)}; sites: "
            f"{_e(', '.join(site.site_id for site in surface.sites))}.</p>")
        for warning in result.warnings:
            add(f"<div class='notice'>{_e(warning)}</div>")
        unique = _rows(result.unique_csv)
        columns = ["config_id", "class", "eads", "contact", "tilt", "azimuth", "height", "boltzmann_weight",
                   "n_duplicates", "site_id", "anchor", "converged", "class_reason"]
        add(_table(
            ["config", "class", "E_ads (eV)", "contact", "θ (°)", "φ (°)", "height (Å)", "weight", "dup.",
             "initial site", "anchor", "converged", "note"],
            [[_fmt(row.get(c)) for c in columns] for row in unique[:MAX_TABLE_ROWS]],
            numeric={2, 4, 5, 6, 7, 8},
        ))
        if len(unique) > MAX_TABLE_ROWS:
            add(f"<p class='muted'>{len(unique) - MAX_TABLE_ROWS} more rows in <code>unique.csv</code>.</p>")
        add("<div class='grid2'>")
        for name, caption in (
            (f"eads_ranking_{term}.png", "E_ads of the unique configurations by class."),
            (f"eads_vs_tilt_{term}.png", "E_ads against tilt θ; color = contact label, marker = class."),
            (f"eads_site_anchor_heatmap_{term}.png", "Lowest E_ads per initial site and anchor."),
        ):
            add(_figure(figures / name, caption))
        add("</div>")
        for path in sorted(figures.glob(f"summary_{term}_*.png")):
            add(_figure(path, f"{path.stem.split('_')[-1]}: side and top view, relaxation, scan or prescreen comparison."))
        for label, gif, png in animations.get(term, []):
            # Only the static version is embedded (keeps the report small); the GIF is in figures/.
            add(f"<h3>{_e(label.capitalize())}</h3>")
            add(_figure(png, f"Snapshots and energy curve; animation: figures/{gif.name}."))

    add("<h2 id='comparison'>Terminations compared</h2>")
    comparison = figures / "termination_comparison.png"
    add(_figure(comparison, "Most stable configuration of each termination.") if comparison.is_file()
        else "<p class='muted'>Only one termination.</p>")

    # Warnings
    add("<h2 id='warnings'>Warnings</h2>")
    warnings = _log_warnings(run_dir / "adsorption.log")
    add("<ul>" + "".join(f"<li>{_e(w)}</li>" for w in warnings) + "</ul>" if warnings
        else "<p class='muted'>No warnings in <code>adsorption.log</code>.</p>")
    add("<h3>Notes</h3><ul>" + "".join(f"<li>{_e(text)}</li>" for text in DISCLAIMERS) + "</ul>")

    # VASP
    add("<h2 id='vasp'>VASP selection</h2>")
    selection = _rows(run_dir / "vasp" / "selection.csv")
    if selection:
        header = list(selection[0].keys())
        add(_table(header, [[_fmt(row[c]) for c in header] for row in selection]))
    else:
        add("<p class='muted'>Not generated yet (<code>ase-adsorb vasp &lt;run_dir&gt;</code>).</p>")

    if player:
        add("<h2 id='animation'>Relaxation animation of the most stable configuration</h2>")
        add(f"<div class='animation'>{player}</div>")

    body = "\n".join(parts)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if str(value).lower() in ("true", "false"):
        return str(value)
    if number.is_integer() and "." not in str(value):
        return str(int(number))
    return f"{number:.3f}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], numeric: set[int] = frozenset()) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = []
    for row in rows:
        cells = "".join(
            f"<td class='num'>{_e(v)}</td>" if i in numeric else f"<td>{_e(v)}</td>" for i, v in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _figure(path: Path, caption: str) -> str:
    if not path.is_file():
        return ""
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f"<figure><img src='data:{mime};base64,{data}' alt='{_e(caption)}' loading='lazy'>"
            f"<figcaption>{_e(caption)}</figcaption></figure>")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _rows(path: Path | None) -> list[dict]:
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _formula(symbols: Sequence[str]) -> str:
    from ase.formula import Formula

    return Formula.from_list(list(symbols)).format("hill")


def _log_warnings(path: Path) -> list[str]:
    if not path.is_file():
        return []
    seen: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if " WARNING " in line:
            message = line.split(" WARNING ", 1)[1].strip()
            if message not in seen:
                seen.append(message)
    return seen


def _animated_configs(result) -> list[str]:
    """Most stable configuration and most stable intact configuration of a termination."""

    rows = _rows(result.unique_csv)
    chosen = [rows[0]["config_id"]] if rows else []
    intact = _best_intact(result)
    if intact is not None and intact["config_id"] not in chosen:
        chosen.append(intact["config_id"])
    return chosen


def _best_overall(results):
    ranked = [r for r in results if r.best_eads is not None and r.best_unique_id]
    return min(ranked, key=lambda r: r.best_eads) if ranked else None


def _best_intact(result) -> dict | None:
    for row in _rows(result.unique_csv):
        if row["class"] in ("chemisorbed", "physisorbed"):
            return row
    return None


def _sampled_molecule(run_dir: Path, gas: dict):
    """The molecule geometry used for sampling (optimized, or input if its bonds changed)."""

    from ase import Atoms

    if gas.get("geometry", "optimized") == "optimized" and (run_dir / "molecule" / "gas_opt_final.extxyz").is_file():
        final = read(run_dir / "molecule" / "gas_opt_final.extxyz")
    else:
        from .config import load_config

        final = read(load_config(run_dir / "config.resolved.yaml").molecule)
    return Atoms(final.get_chemical_symbols(), positions=final.get_positions())


def _analysis_from_json(data: dict):
    """A `MoleculeAnalysis` rebuilt from ``molecule_analysis.json``."""

    from ..graphs import Fragment
    from .molecule import Anchor, MoleculeAnalysis, ReferenceAxis

    axis = data["reference_axis"]
    return MoleculeAnalysis(
        symbols=tuple(data["symbols"]),
        bonds=tuple((int(i), int(j), float(d)) for i, j, d in data["bonds"]),
        fragments=tuple(Fragment(f["name"], f["formula"], tuple(f["indices"])) for f in data["fragments"]),
        metal_indices=tuple(data["metal_indices"]),
        rings=tuple(tuple(r) for r in data["rings"]),
        center=tuple(data["center"]),
        anchors=tuple(
            Anchor(a["label"], a["kind"], tuple(a["atom_indices"]), tuple(a["position"]), tuple(a["direction"]),
                   tuple(a["merged"]))
            for a in data["anchors"]
        ),
        reference_axis=ReferenceAxis(axis["method"], tuple(axis["start"]), tuple(axis["end"]), axis["directional"],
                                     axis.get("description", "")),
        atom_labels=tuple(data["atom_labels"]),
        graph_hash=data["graph_hash"],
        warnings=tuple(data["warnings"]),
    )
