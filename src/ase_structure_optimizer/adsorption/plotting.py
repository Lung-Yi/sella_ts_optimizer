"""Static analysis figures (PNG, 150 dpi) of an adsorption run.

Colors: adsorption classes use fixed hues (chemisorbed green, physisorbed
blue, dissociated red, desorbed gray); contact labels use the first slots of
a validated categorical palette in a fixed order, and labels beyond the
third are folded into "other" (scatter plots only validate three hues for
color-vision deficiencies); the heat map uses a single-hue blue ramp where a
darker cell means stronger binding. matplotlib is imported lazily.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DPI = 150
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
CLASS_COLORS = {
    "chemisorbed": "#008300",
    "physisorbed": "#2a78d6",
    "dissociated": "#e34948",
    "desorbed": "#8f8e89",
}
CLASS_MARKERS = {"chemisorbed": "o", "physisorbed": "s", "dissociated": "X", "desorbed": "D"}
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
OTHER = "#b5b4ae"
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
MAX_SUMMARY_CONFIGS = 3


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
        }
    )
    return plt


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a results CSV with numeric fields converted."""

    if path is None or not Path(path).is_file():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("eads", "eads_screen", "tilt", "azimuth", "height", "boltzmann_weight", "prescreen_energy"):
                if key in row:
                    row[key] = float(row[key]) if row[key] not in ("", None) else None
            rows.append(row)
    return rows


def contact_colors(rows: Sequence[dict]) -> dict[str, str]:
    """Color per contact label: the three most stable labels get palette slots, the rest "other"."""

    order: list[str] = []
    for row in sorted(rows, key=lambda r: r["eads"]):
        if row["contact"] not in order:
            order.append(row["contact"])
    return {label: (CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER) for i, label in enumerate(order)}


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------


def plot_ranking(rows: Sequence[dict], term: str, path: Path) -> Path:
    """Horizontal E_ads bars of the unique configurations, colored by class."""

    plt = _pyplot()
    rows = sorted(rows, key=lambda r: r["eads"])
    height = max(2.5, 0.32 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.0, height))
    positions = np.arange(len(rows))[::-1]
    for y, row in zip(positions, rows):
        hatch = "////" if str(row.get("low_confidence", "")).lower() == "true" else None
        ax.barh(
            y,
            row["eads"],
            height=0.7,
            color=CLASS_COLORS.get(row["class"], OTHER),
            edgecolor=SURFACE if hatch is None else INK_SECONDARY,
            linewidth=0.8 if hatch is None else 0.4,
            hatch=hatch,
        )
    ax.set_yticks(positions)
    # The class is also written out: red/green alone is not enough under CVD.
    ax.set_yticklabels(
        [f"{row['config_id']}  {row['contact']}  [{row['class']}]" for row in rows], fontsize=7, color=INK
    )
    ax.axvline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("E_ads (eV)")
    ax.set_title(f"{term}: unique configurations")
    _class_legend(ax, {row["class"] for row in rows}, hatch=any(
        str(row.get("low_confidence", "")).lower() == "true" for row in rows
    ))
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_eads_vs_tilt(rows: Sequence[dict], term: str, path: Path, directional: bool = True) -> Path:
    """E_ads against the tilt angle θ; color = contact label, marker = class."""

    plt = _pyplot()
    rows = [row for row in rows if row.get("tilt") is not None]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = contact_colors(rows)
    for row in rows:
        ax.scatter(
            row["tilt"],
            row["eads"],
            s=48,
            marker=CLASS_MARKERS.get(row["class"], "o"),
            color=colors[row["contact"]],
            edgecolors=SURFACE,
            linewidths=1.0,
            zorder=3,
        )
    ax.set_xlim(-5, (180 if directional else 90) + 5)
    ax.set_xticks(np.arange(0, (180 if directional else 90) + 1, 30))
    ax.set_xlabel(
        "tilt angle θ (°)  (0° = reference axis pointing away from the surface)"
        if directional
        else "tilt angle θ (°)  (0° = upright, 90° = lying flat)"
    )
    ax.set_ylabel("E_ads (eV)")
    ax.set_title(f"{term}: E_ads vs tilt")
    from matplotlib.lines import Line2D

    handles = []
    shown = [label for label, color in colors.items() if color != OTHER]
    for label in shown:
        handles.append(Line2D([], [], marker="o", linestyle="", color=colors[label], markersize=7, label=label))
    if any(color == OTHER for color in colors.values()):
        handles.append(
            Line2D([], [], marker="o", linestyle="", color=OTHER, markersize=7, label="other contacts / none")
        )
    for name in sorted({row["class"] for row in rows}):
        handles.append(
            Line2D([], [], marker=CLASS_MARKERS[name], linestyle="", color=INK_SECONDARY, markersize=7, label=name)
        )
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    if not rows:
        ax.text(0.5, 0.5, "no defined tilt angle", transform=ax.transAxes, ha="center", color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_site_anchor_heatmap(rows: Sequence[dict], term: str, path: Path) -> Path:
    """Lowest E_ads per (initial site, anchor) pair; darker = stronger binding."""

    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    sites = sorted({row["site_id"] for row in rows})
    anchors = sorted({row["anchor"] for row in rows}, key=lambda a: (a == "random", a))
    grid = np.full((len(sites), len(anchors)), np.nan)
    for row in rows:
        i, j = sites.index(row["site_id"]), anchors.index(row["anchor"])
        if np.isnan(grid[i, j]) or row["eads"] < grid[i, j]:
            grid[i, j] = row["eads"]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(anchors) + 2.5), max(3.0, 0.45 * len(sites) + 1.5)))
    cmap = LinearSegmentedColormap.from_list("binding", SEQUENTIAL)
    cmap.set_bad("#f0efec")
    finite = grid[np.isfinite(grid)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    if vmax - vmin < 1e-6:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    # Most negative (strongest binding) -> darkest.
    image = ax.imshow(-grid, cmap=cmap, vmin=-vmax, vmax=-vmin, aspect="auto")
    for i in range(len(sites)):
        for j in range(len(anchors)):
            if np.isfinite(grid[i, j]):
                dark = (-grid[i, j] - (-vmax)) / ((-vmin) - (-vmax)) > 0.55
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(anchors)))
    ax.set_xticklabels(anchors, rotation=45, ha="right")
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites)
    ax.grid(False)
    ax.set_xlabel("anchor pointing to the surface")
    ax.set_ylabel("initial site")
    ax.set_title(f"{term}: lowest E_ads (eV) per site and anchor")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("binding strength  -E_ads (eV)", color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_configuration_summary(
    term: str,
    row: dict,
    term_dir: Path,
    n_slab: int,
    e_slab_screen: float,
    e_mol_screen: float,
    path: Path,
    scan: tuple[list[float], list[float]] | None = None,
) -> Path:
    """2x2 panel: side view, top view, E_ads along the relaxation, scan or prescreen/relax comparison."""

    plt = _pyplot()
    from ase.io import read
    from ase.visualize.plot import plot_atoms

    config_id = row["config_id"]
    final = _plottable(read(term_dir / "relax" / f"{config_id}_final.extxyz"))
    trajectory = read(term_dir / "relax" / f"{config_id}.traj", index=":")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    side, top, relax_ax, extra = axes.ravel()
    plot_atoms(_view_region(final, n_slab), side, rotation="-90x", show_unit_cell=0)
    side.set_title("side view")
    plot_atoms(_view_region(final, n_slab, top_only=True), top, rotation="0x", show_unit_cell=2)
    top.set_title("top view")
    for ax in (side, top):
        ax.set_axis_off()

    steps = np.arange(len(trajectory))
    energies = [image.get_potential_energy() - e_slab_screen - e_mol_screen for image in trajectory]
    relax_ax.plot(steps, energies, color=CATEGORICAL[0], linewidth=2)
    relax_ax.set_xlabel("optimization step")
    relax_ax.set_ylabel("E_ads (eV, screening dtype)")
    relax_ax.set_title("relaxation")

    if scan is not None:
        clearance, eads = scan
        shown = np.array(eads) <= max(2.0, row["eads"] + 5.0)
        extra.plot(np.array(clearance)[shown], np.array(eads)[shown], color=CATEGORICAL[0], linewidth=2,
                   marker="o", markersize=4)
        extra.axhline(row["eads"], color=CLASS_COLORS["dissociated"], linestyle="--", linewidth=1.5,
                      label=f"relaxed E_ads {row['eads']:.2f} eV")
        extra.set_xlabel("clearance of the lowest atom above the surface (Å)")
        extra.set_ylabel("E_ads (eV)")
        extra.set_title("rigid approach scan (initial orientation)")
        extra.legend(loc="best")
    else:
        labels = ["prescreen", "relaxed (screen)", "relaxed (final)"]
        prescreen = row.get("prescreen_energy")
        values = [
            prescreen - e_slab_screen - e_mol_screen if prescreen is not None else np.nan,
            row.get("eads_screen") if row.get("eads_screen") is not None else np.nan,
            row["eads"],
        ]
        extra.bar(labels, values, color=[OTHER, CATEGORICAL[0], CLASS_COLORS.get(row["class"], OTHER)],
                  edgecolor=SURFACE, linewidth=2, width=0.6)
        for x, value in enumerate(values):
            if np.isfinite(value):
                extra.text(x, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top",
                           fontsize=8, color=INK)
        extra.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        extra.set_ylabel("E_ads (eV)")
        extra.set_title("prescreening vs full relaxation")
        extra.grid(axis="x", visible=False)

    tilt = "n/a" if row.get("tilt") is None else f"{row['tilt']:.0f}°"
    fig.suptitle(
        f"{term} {config_id}: E_ads {row['eads']:.3f} eV, {row['class']}, contact {row['contact']}, "
        f"tilt {tilt}, site {row['site_id']}",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_termination_comparison(best: Sequence[tuple[str, float, str]], path: Path) -> Path:
    """Lowest E_ads per termination, colored by the class of that configuration."""

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(best) + 2.0), 3.8))
    names = [item[0] for item in best]
    values = [item[1] for item in best]
    ax.bar(names, values, color=[CLASS_COLORS.get(item[2], OTHER) for item in best], edgecolor=SURFACE,
           linewidth=2, width=0.6)
    for x, (value, item) in enumerate(zip(values, best)):
        ax.text(x, value, f"{value:.2f}\n{item[2]}", ha="center", va="top" if value < 0 else "bottom",
                fontsize=8, color=INK)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    low, high = min(0.0, min(values)), max(0.0, max(values))
    span = (high - low) or 1.0
    ax.set_ylim(low - 0.25 * span, high + 0.25 * span if high > 0 else high + 0.05 * span)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("lowest E_ads (eV)")
    ax.set_title("terminations: most stable configuration")
    _class_legend(ax, {item[2] for item in best})
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Whole run
# ---------------------------------------------------------------------------


def plot_run(run_dir: Path, figures_dir: Path, surfaces, results, config) -> list[Path]:
    """All static figures of a run; returns the written files."""

    from .molecule import analyze_molecule
    from ase.io import read

    figures_dir.mkdir(parents=True, exist_ok=True)
    gas = json.loads((run_dir / "molecule" / "gas_reference.json").read_text(encoding="utf-8"))
    analysis_file = run_dir / "molecule" / "molecule_analysis.json"
    directional = json.loads(analysis_file.read_text(encoding="utf-8"))["reference_axis"]["directional"]
    written: list[Path] = []
    best: list[tuple[str, float, str]] = []
    surface_by_id = {surface.term_id: surface for surface in surfaces}
    for result in results:
        term = result.term_id
        term_dir = run_dir / "terminations" / term
        unique = read_rows(result.unique_csv)
        everything = [row for row in read_rows(result.results_csv) if row.get("eads") is not None]
        if not unique:
            continue
        written.append(plot_ranking(unique, term, figures_dir / f"eads_ranking_{term}.png"))
        written.append(plot_eads_vs_tilt(unique, term, figures_dir / f"eads_vs_tilt_{term}.png", directional))
        written.append(plot_site_anchor_heatmap(everything, term, figures_dir / f"eads_site_anchor_heatmap_{term}.png"))

        scan = None
        scan_target_id = None
        if result.scan_csv is not None and Path(result.scan_csv).is_file():
            scan_rows = read_rows(result.scan_csv)
            scan = ([float(r["clearance"]) for r in scan_rows], [float(r["eads"]) for r in scan_rows])
            scan_target_id = json.loads((term_dir / "approach_scan.json").read_text(encoding="utf-8"))["config_id"]
        surface = surface_by_id[term]
        shown = unique[:MAX_SUMMARY_CONFIGS]
        if scan_target_id and scan_target_id not in [row["config_id"] for row in shown]:
            shown = shown[: MAX_SUMMARY_CONFIGS - 1] + [row for row in unique if row["config_id"] == scan_target_id]
        for row in shown:
            written.append(
                plot_configuration_summary(
                    term,
                    row,
                    term_dir,
                    surface.n_atoms,
                    surface.energy_screen,
                    gas["energy_screen"],
                    figures_dir / f"summary_{term}_{row['config_id']}.png",
                    scan=scan if row["config_id"] == scan_target_id else None,
                )
            )
        best.append((term, unique[0]["eads"], unique[0]["class"]))
    if len(best) > 1:
        written.append(plot_termination_comparison(best, figures_dir / "termination_comparison.png"))
    return written


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _class_legend(ax, classes, hatch: bool = False) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=CLASS_COLORS[name], label=name) for name in CLASS_COLORS if name in classes]
    if hatch:
        handles.append(Patch(facecolor=SURFACE, edgecolor=INK_SECONDARY, hatch="////", label="low confidence"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _plottable(atoms):
    """Copy safe for ase.visualize.plot (no calculator, info or tags that break it)."""

    clean = atoms.copy()
    clean.calc = None
    clean.info = {}
    clean.set_tags(0)
    clean.set_constraint()
    return clean


def _view_region(atoms, n_slab: int, top_only: bool = False):
    """Molecule plus the upper slab layers, for readable structure images."""

    z = atoms.positions[:n_slab, 2]
    depth = 3.0 if top_only else 6.0
    keep = [i for i in range(n_slab) if z[i] >= z.max() - depth] + list(range(n_slab, len(atoms)))
    return atoms[keep]


# ---------------------------------------------------------------------------
# Animations (with static versions; IDEs often cannot show GIFs)
# ---------------------------------------------------------------------------

MAX_ANIMATION_FRAMES = 80
STATIC_PANELS = 4


def frame_indices(n_frames: int, max_frames: int = MAX_ANIMATION_FRAMES) -> list[int]:
    """Evenly spaced frame indices (first and last included), at most `max_frames`."""

    if n_frames <= max_frames:
        return list(range(n_frames))
    return sorted({int(round(i)) for i in np.linspace(0, n_frames - 1, max_frames)})


def _fixed_region(frames, n_slab: int):
    """Same atom selection (molecule + upper slab layers of the first frame) for every frame."""

    z = frames[0].positions[:n_slab, 2]
    keep = [i for i in range(n_slab) if z[i] >= z.max() - 6.0] + list(range(n_slab, len(frames[0])))
    return [_plottable(frame)[keep] for frame in frames]


def _view_limits(ax, frames, rotation: str):
    """Axis limits covering the first and last frame, so the view does not jump."""

    from ase.visualize.plot import plot_atoms

    limits = []
    for frame in (frames[0], frames[-1]):
        ax.clear()
        plot_atoms(frame, ax, rotation=rotation, show_unit_cell=0)
        limits.append((ax.get_xlim(), ax.get_ylim()))
    ax.clear()
    x = (min(l[0][0] for l in limits), max(l[0][1] for l in limits))
    y = (min(l[1][0] for l in limits), max(l[1][1] for l in limits))
    return x, y


def _animation(frames, x_values, y_values, x_label, y_label, title, reference=None, dpi: int = 90):
    """Figure and FuncAnimation: structure (side view) left, energy curve right."""

    plt = _pyplot()
    from matplotlib.animation import FuncAnimation
    from ase.visualize.plot import plot_atoms

    rotation = "-90x"
    fig, (structure_ax, curve_ax) = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=dpi,
                                                 gridspec_kw={"width_ratios": [1.2, 1.0]})
    x_limits, y_limits = _view_limits(structure_ax, frames, rotation)
    curve_ax.plot(x_values, y_values, color=GRID, linewidth=2, zorder=1)
    if reference is not None:
        curve_ax.axhline(reference[0], color=CLASS_COLORS["dissociated"], linestyle="--", linewidth=1.5,
                         label=reference[1])
        curve_ax.legend(loc="best")
    (trace,) = curve_ax.plot([], [], color=CATEGORICAL[0], linewidth=2, zorder=2)
    (marker,) = curve_ax.plot([], [], "o", color=CATEGORICAL[0], markersize=8, markeredgecolor=SURFACE, zorder=3)
    curve_ax.set_xlabel(x_label)
    curve_ax.set_ylabel(y_label)
    fig.suptitle(title, fontsize=11, color=INK)

    def update(index):
        structure_ax.clear()
        plot_atoms(frames[index], structure_ax, rotation=rotation, show_unit_cell=0)
        structure_ax.set_xlim(*x_limits)
        structure_ax.set_ylim(*y_limits)
        structure_ax.set_axis_off()
        trace.set_data(x_values[: index + 1], y_values[: index + 1])
        marker.set_data([x_values[index]], [y_values[index]])
        return trace, marker

    fig.tight_layout()
    animation = FuncAnimation(fig, update, frames=len(frames), interval=120, blit=False)
    return fig, animation


def _static_strip(frames, x_values, y_values, x_label, y_label, title, path: Path, reference=None) -> Path:
    """Static stand-in for an animation: a few snapshots above the full energy curve."""

    plt = _pyplot()
    from ase.visualize.plot import plot_atoms

    picks = sorted({int(round(i)) for i in np.linspace(0, len(frames) - 1, min(STATIC_PANELS, len(frames)))})
    fig = plt.figure(figsize=(11.0, 6.4))
    grid = fig.add_gridspec(2, len(picks), height_ratios=[1.0, 1.0])
    probe = fig.add_subplot(grid[0, 0])
    x_limits, y_limits = _view_limits(probe, frames, "-90x")
    probe.remove()
    for column, index in enumerate(picks):
        ax = fig.add_subplot(grid[0, column])
        plot_atoms(frames[index], ax, rotation="-90x", show_unit_cell=0)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_axis_off()
        ax.set_title(f"{x_label.split(' (')[0]} {x_values[index]:.3g}", fontsize=9, color=INK_SECONDARY)
    curve = fig.add_subplot(grid[1, :])
    curve.plot(x_values, y_values, color=CATEGORICAL[0], linewidth=2)
    curve.plot([x_values[i] for i in picks], [y_values[i] for i in picks], "o", color=CATEGORICAL[0],
               markersize=8, markeredgecolor=SURFACE)
    if reference is not None:
        curve.axhline(reference[0], color=CLASS_COLORS["dissociated"], linestyle="--", linewidth=1.5,
                      label=reference[1])
        curve.legend(loc="best")
    curve.set_xlabel(x_label)
    curve.set_ylabel(y_label)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _relaxation_data(term_dir: Path, config_id: str, n_slab: int, e_slab_screen: float, e_mol_screen: float,
                     max_frames: int = MAX_ANIMATION_FRAMES):
    from ase.io import read

    trajectory = read(term_dir / "relax" / f"{config_id}.traj", index=":")
    picks = frame_indices(len(trajectory), max_frames)
    frames = _fixed_region([trajectory[i] for i in picks], n_slab)
    energies = [trajectory[i].get_potential_energy() - e_slab_screen - e_mol_screen for i in picks]
    return frames, [float(i) for i in picks], energies


def relaxation_animation(term: str, config_id: str, term_dir: Path, n_slab: int, e_slab_screen: float,
                         e_mol_screen: float, gif_path: Path, static_path: Path) -> tuple[Path, Path]:
    """``relaxation_<term>_<config>.gif`` and its static PNG."""

    plt = _pyplot()
    from matplotlib.animation import PillowWriter

    frames, steps, energies = _relaxation_data(term_dir, config_id, n_slab, e_slab_screen, e_mol_screen)
    title = f"{term} {config_id}: relaxation"
    labels = ("optimization step", "E_ads (eV, screening dtype)")
    fig, animation = _animation(frames, steps, energies, *labels, title)
    animation.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    _static_strip(frames, steps, energies, *labels, title, static_path)
    return gif_path, static_path


def approach_animation(term: str, term_dir: Path, n_slab: int, relaxed_eads: float, gif_path: Path,
                       static_path: Path) -> tuple[Path, Path]:
    """``approach_<term>.gif`` (rigid scan, farthest to closest) and its static PNG."""

    plt = _pyplot()
    from ase.io import read
    from matplotlib.animation import PillowWriter

    scan = read(term_dir / "approach_scan.extxyz", index=":")[::-1]
    clearance = [float(frame.info["clearance"]) for frame in scan]
    rows = {round(float(r["clearance"]), 4): float(r["eads"]) for r in read_rows(term_dir / "approach_scan.csv")}
    eads = [rows[round(c, 4)] for c in clearance]
    frames = _fixed_region(scan, n_slab)
    reference = (relaxed_eads, f"relaxed E_ads {relaxed_eads:.2f} eV")
    title = f"{term}: rigid approach of the gas-phase molecule"
    labels = ("clearance (Å)", "E_ads (eV)")
    fig, animation = _animation(frames, clearance, eads, *labels, title, reference=reference)
    fig.axes[1].invert_xaxis()  # the molecule approaches: clearance decreases left to right
    animation.save(gif_path, writer=PillowWriter(fps=6))
    plt.close(fig)
    _static_strip(frames, clearance, eads, *labels, title, static_path, reference=reference)
    return gif_path, static_path


def relaxation_jshtml(term: str, config_id: str, term_dir: Path, n_slab: int, e_slab_screen: float,
                      e_mol_screen: float, max_frames: int = 40) -> str:
    """Self-contained HTML/JavaScript player of a relaxation (``FuncAnimation.to_jshtml``)."""

    plt = _pyplot()

    frames, steps, energies = _relaxation_data(term_dir, config_id, n_slab, e_slab_screen, e_mol_screen, max_frames)
    fig, animation = _animation(frames, steps, energies, "optimization step", "E_ads (eV, screening dtype)",
                                f"{term} {config_id}: relaxation", dpi=72)
    html = animation.to_jshtml(fps=6, default_mode="once")
    plt.close(fig)
    return html


# ---------------------------------------------------------------------------
# Schematics for the report
# ---------------------------------------------------------------------------


def plot_molecule_axis(molecule, analysis, path: Path) -> Path:
    """Molecule drawn with the reference axis u pointing up, fragments labeled."""

    plt = _pyplot()
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors

    positions = molecule.get_positions() - molecule.get_positions().mean(axis=0)
    axis = analysis.reference_axis.vector(molecule)
    if axis is not None:
        # Rotate so that u points along +y; x is any perpendicular direction.
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(axis, helper)
        x /= np.linalg.norm(x)
        depth = np.cross(x, axis)
        basis = np.vstack([x, axis, depth])
        positions = positions @ basis.T
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_facecolor(SURFACE)
    for i, j, _ in analysis.bonds:
        ax.plot(positions[[i, j], 0], positions[[i, j], 1], color=INK_SECONDARY, linewidth=1.2, zorder=1)
    order = np.argsort(positions[:, 2])
    for index in order:
        number = molecule.numbers[index]
        ax.scatter(positions[index, 0], positions[index, 1], s=(covalent_radii[number] * 38) ** 2 / 4,
                   color=jmol_colors[number], edgecolors=INK, linewidths=0.6, zorder=2)
    for fragment in analysis.fragments:
        center = positions[list(fragment.indices)].mean(axis=0)
        ax.annotate(fragment.name, center[:2], xytext=(10, -12), textcoords="offset points", fontsize=8,
                    color=INK, zorder=4)
    for index in analysis.metal_indices:
        ax.annotate(f"{analysis.symbols[index]}{index}", positions[index, :2], xytext=(8, 6),
                    textcoords="offset points", fontsize=8, color=INK, zorder=4)
    if axis is not None:
        method = analysis.reference_axis.method
        if method in ("principal", "plane_normal"):
            start = positions.mean(axis=0)
            length = 0.6 * (np.ptp(positions[:, 1]) or 1.0)
        else:
            start = positions[list(analysis.reference_axis.start)].mean(axis=0)
            end = positions[list(analysis.reference_axis.end)].mean(axis=0)
            length = 1.4 * np.linalg.norm((end - start)[:2])
        tip = (start[0], start[1] + length)
        ax.annotate("", xy=tip, xytext=(start[0], start[1]),
                    arrowprops={"arrowstyle": "-|>", "color": CLASS_COLORS["dissociated"], "linewidth": 2.5},
                    zorder=5)
        ax.text(tip[0] + 0.2, tip[1], "u", color=CLASS_COLORS["dissociated"], fontsize=13, fontweight="bold")
        ax.update_datalim([tip, (tip[0] + 0.6, tip[1] + 0.3)])
        ax.autoscale_view()
    ax.set_title(f"reference axis: {analysis.reference_axis.method}")
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_angle_definition(path: Path, directional: bool = True) -> Path:
    """Schematic of the tilt angle θ (side views) and the azimuth φ (top view)."""

    plt = _pyplot()
    angles = (0, 90, 180) if directional else (0, 45, 90)
    notes = {0: "u away from surface", 45: "tilted", 90: "u parallel to surface", 180: "u toward surface"}
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    for ax, angle in zip(axes[:3], angles):
        ax.add_patch(plt.Rectangle((-1.2, -0.6), 2.4, 0.4, color="#d9d7cf"))
        ax.annotate("", xy=(-0.9, 1.1), xytext=(-0.9, -0.2),
                    arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 1.2})
        ax.text(-0.85, 1.05, "z", color=INK_SECONDARY, fontsize=9)
        radians = np.radians(angle)
        start = np.array([0.2, 0.3 if angle < 150 else 1.0])
        tip = start + 0.7 * np.array([np.sin(radians), np.cos(radians)])
        ax.annotate("", xy=tip, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": CLASS_COLORS["dissociated"], "linewidth": 2.2})
        ax.text(*(tip + [0.05, 0.05]), "u", color=CLASS_COLORS["dissociated"], fontsize=11, fontweight="bold")
        ax.set_title(f"θ = {angle}°\n{notes[angle]}", fontsize=9)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.7, 1.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
    ax = axes[3]
    ax.annotate("", xy=(1.0, 0.0), xytext=(-0.2, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 1.2})
    ax.text(1.0, -0.2, "a", color=INK_SECONDARY, fontsize=10)
    phi = np.radians(55)
    ax.annotate("", xy=(0.9 * np.cos(phi), 0.9 * np.sin(phi)), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": CATEGORICAL[0], "linewidth": 2.2})
    arc = np.linspace(0, phi, 30)
    ax.plot(0.35 * np.cos(arc), 0.35 * np.sin(arc), color=INK_SECONDARY, linewidth=1)
    ax.text(0.42, 0.18, "φ", fontsize=11, color=INK)
    ax.set_title("top view: φ = second axis\nprojected, from cell vector a", fontsize=9)
    ax.set_xlim(-0.4, 1.3)
    ax.set_ylim(-0.4, 1.1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
