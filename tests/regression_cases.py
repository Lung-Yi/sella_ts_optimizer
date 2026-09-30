"""Reference cases for the core regression tests.

Each case runs an existing public workflow (geometry minimization, Sella TS,
Sella IRC, frequency analysis, and the ``ase-structure-opt`` CLI) on small
deterministic systems and reduces the outputs to a JSON-serializable summary.

The summaries were recorded on the unmodified ``main`` branch with
``python tests/record_baseline.py`` and are compared against fresh runs by
``tests/test_core_regression.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms
from ase.io import read, write

BASELINE_PATH = Path(__file__).parent / "data" / "regression_baseline.json"

# Lines printed by the CLI itself (optimizer logs are excluded because they
# contain wall-clock timestamps).
CLI_PREFIXES = (
    "BFGS ",
    "LBFGS ",
    "FIRE ",
    "Sella TS",
    "Final ",
    "Optimization path",
    "Trajectory",
    "IRC ",
    "Full IRC",
    "Forward ",
    "Reverse ",
    "Frequency ",
    "Detailed ",
    "Warning",
)


def water() -> Atoms:
    """Slightly distorted water molecule (EMT supports H and O)."""

    return Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.05, -0.80, -0.45]],
    )


def carbon_monoxide() -> Atoms:
    """Stretched CO molecule."""

    return Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.1, 1.30]])


def cu5_ts_guess() -> Atoms:
    """Cu5 cluster guess that Sella converges to a first-order saddle with EMT."""

    return Atoms(
        "Cu5",
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.17, 0.0],
            [1.25, 0.72, 2.0],
            [3.7, 2.2, 0.3],
        ],
    )


def eclipsed_ethane() -> Atoms:
    """Ethane with one methyl rotated to the eclipsed conformation (rotation TS guess)."""

    from ase.build import molecule

    atoms = molecule("C2H6")
    symbols = atoms.get_chemical_symbols()
    carbon = [i for i, s in enumerate(symbols) if s == "C"]
    top = max(carbon, key=lambda i: atoms.positions[i, 2])
    methyl = [
        i
        for i, s in enumerate(symbols)
        if s == "H" and atoms.positions[i, 2] * atoms.positions[top, 2] > 0
    ]
    axis = atoms.positions[top] - atoms.positions[[c for c in carbon if c != top][0]]
    rotated = atoms.copy()
    rotated.rotate(60.0, axis, center=atoms.positions[top])
    atoms.positions[methyl] = rotated.positions[methyl]
    return atoms


def _xyz_summary(path: Path) -> dict[str, Any]:
    images = read(path, index=":")
    text = path.read_text(encoding="utf-8").splitlines()
    return {
        "frames": len(images),
        "comment_line": text[1],
        "symbols": images[-1].get_chemical_symbols(),
        "final_positions": images[-1].get_positions().round(8).tolist(),
        "pbc": images[-1].pbc.tolist(),
    }


def _trajectory_summary(path: Path) -> dict[str, Any]:
    images = read(path, index=":")
    energies = [float(image.get_potential_energy()) for image in images]
    return {
        "frames": len(images),
        "first_energy": energies[0],
        "final_energy": energies[-1],
        "min_energy": min(energies),
    }


def _listing(directory: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    )


def _result_summary(result: Any, output_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"result_type": type(result).__name__}
    fields = [field.name for field in dataclasses.fields(result)]
    summary["fields"] = fields
    for name in fields:
        value = getattr(result, name)
        if isinstance(value, Path):
            summary[name] = str(value.relative_to(output_dir))
        elif isinstance(value, (bool, int, str)) or value is None:
            summary[name] = value
    return summary


def _optimization_summary(result: Any, output_dir: Path) -> dict[str, Any]:
    summary = _result_summary(result, output_dir)
    summary["files"] = _listing(output_dir)
    summary["trajectory_data"] = _trajectory_summary(result.trajectory)
    summary["path_xyz"] = _xyz_summary(result.optimized_xyz)
    summary["final_xyz_data"] = _xyz_summary(result.final_xyz)
    return summary


def _irc_summary(result: Any, output_dir: Path) -> dict[str, Any]:
    summary = _result_summary(result, output_dir)
    summary["files"] = _listing(output_dir)
    for branch in ("forward", "reverse"):
        trajectory = getattr(result, f"{branch}_trajectory")
        if trajectory is not None:
            summary[f"{branch}_trajectory_data"] = _trajectory_summary(trajectory)
            summary[f"{branch}_xyz_data"] = _xyz_summary(getattr(result, f"{branch}_xyz"))
    summary["full_path_data"] = _xyz_summary(result.full_path_xyz)
    return summary


def _quiet(func: Callable[[], Any]) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return func()


# ---------------------------------------------------------------------------
# Library-level cases
# ---------------------------------------------------------------------------


def case_min_emt(workdir: Path, optimizer: str) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, optimize_geometry_atoms

    output_dir = workdir / f"min_emt_{optimizer}"
    result = _quiet(
        lambda: optimize_geometry_atoms(
            atoms=water(),
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
            fmax=5e-3,
            max_steps=200,
            optimizer=optimizer,
        )
    )
    return _optimization_summary(result, output_dir)


def case_min_emt_file(workdir: Path) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, run_geometry_optimization

    xyz = workdir / "co.xyz"
    write(xyz, carbon_monoxide(), format="xyz")
    output_dir = workdir / "min_emt_file"
    result = _quiet(
        lambda: run_geometry_optimization(
            xyz_path=xyz,
            calculator_config=CalculatorConfig(name="emt", charge=0, multiplicity=1),
            output_dir=output_dir,
        )
    )
    return _optimization_summary(result, output_dir)


def case_min_emt_charged(workdir: Path) -> dict[str, Any]:
    """Non-default charge/multiplicity: checks the per-atom metadata handling."""

    from ase_structure_optimizer.atoms import prepare_atoms
    from ase_structure_optimizer import CalculatorConfig

    config = CalculatorConfig(name="emt", charge=-1, multiplicity=2)
    prepared = prepare_atoms(water(), config)
    return {
        "info": {key: prepared.info[key] for key in sorted(prepared.info)},
        "initial_charges": prepared.get_initial_charges().tolist(),
        "initial_magmoms": prepared.get_initial_magnetic_moments().tolist(),
        "calculator": type(prepared.calc).__name__,
    }


def case_ts_emt(workdir: Path) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, optimize_ts_atoms

    output_dir = workdir / "ts_emt"
    result = _quiet(
        lambda: optimize_ts_atoms(
            atoms=cu5_ts_guess(),
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
        )
    )
    return _optimization_summary(result, output_dir)


def case_ts_emt_file(workdir: Path) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, run_ts_optimization

    xyz = workdir / "cu5_guess.xyz"
    write(xyz, cu5_ts_guess(), format="xyz")
    output_dir = workdir / "ts_emt_file"
    result = _quiet(
        lambda: run_ts_optimization(
            xyz_path=xyz,
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
            fmax=1e-2,
            max_steps=50,
            trajectory_name="custom.traj",
            optimized_path_name="custom_path.xyz",
            final_name="custom_final.xyz",
        )
    )
    return _optimization_summary(result, output_dir)


def _cu5_ts(workdir: Path) -> Path:
    """Write a converged Cu5 TS xyz (via the public TS API) and return its path."""

    from ase_structure_optimizer import CalculatorConfig, optimize_ts_atoms

    output_dir = workdir / "cu5_ts_for_irc"
    result = _quiet(
        lambda: optimize_ts_atoms(
            atoms=cu5_ts_guess(),
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
        )
    )
    return result.final_xyz


def case_irc_emt(workdir: Path, direction: str) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, run_irc

    ts_xyz = _cu5_ts(workdir)
    output_dir = workdir / f"irc_emt_{direction}"
    result = _quiet(
        lambda: run_irc(
            xyz_path=ts_xyz,
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
            direction=direction,
        )
    )
    return _irc_summary(result, output_dir)


def case_irc_emt_atoms(workdir: Path) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, optimize_irc_atoms

    ts_atoms = read(_cu5_ts(workdir))
    before = ts_atoms.get_positions().copy()
    output_dir = workdir / "irc_emt_atoms"
    result = _quiet(
        lambda: optimize_irc_atoms(
            atoms=ts_atoms,
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
            fmax=2e-2,
            max_steps=20,
            dx=0.05,
            direction="both",
        )
    )
    summary = _irc_summary(result, output_dir)
    summary["input_not_mutated"] = bool(np.array_equal(before, ts_atoms.get_positions()))
    summary["input_has_calc"] = ts_atoms.calc is not None
    return summary


def case_freq_emt(workdir: Path) -> dict[str, Any]:
    from ase_structure_optimizer import CalculatorConfig, run_frequency_analysis

    ts_xyz = _cu5_ts(workdir)
    output_dir = workdir / "freq_emt"
    result = _quiet(
        lambda: run_frequency_analysis(
            xyz_path=ts_xyz,
            calculator_config=CalculatorConfig(name="emt"),
            output_dir=output_dir,
        )
    )
    return {
        "files": _listing(output_dir),
        "summary": str(result.summary.relative_to(output_dir)),
        "detailed_output": str(result.detailed_output.relative_to(output_dir)),
        "vibration_directory": str(result.vibration_directory.relative_to(output_dir)),
        "frequencies_cm1": list(result.frequencies_cm1),
        "imaginary_count": result.imaginary_count,
    }


# ---------------------------------------------------------------------------
# xTB cases (optional backend)
# ---------------------------------------------------------------------------


def case_ts_irc_xtb(workdir: Path) -> dict[str, Any]:
    """Ethane methyl-rotation TS -> IRC -> minimum with xTB.

    xTB is only bitwise reproducible single-threaded, so unless the current
    process already runs with ``OMP_NUM_THREADS=1`` the case re-runs itself in
    a single-threaded subprocess.
    """

    if os.environ.get("OMP_NUM_THREADS") != "1":
        import json
        import subprocess
        import sys

        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        tests_dir = Path(__file__).resolve().parent
        code = (
            "import json, sys; from pathlib import Path; "
            f"sys.path[:0] = [{str(tests_dir.parent / 'src')!r}, {str(tests_dir)!r}]; "
            "from regression_cases import case_ts_irc_xtb; "
            f"print('@@JSON@@' + json.dumps(case_ts_irc_xtb(Path({str(workdir)!r}))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        payload = completed.stdout.split("@@JSON@@", 1)[1]
        return json.loads(payload)

    from ase_structure_optimizer import (
        CalculatorConfig,
        optimize_geometry_atoms,
        optimize_ts_atoms,
        run_irc,
    )

    config = CalculatorConfig(name="xtb")
    ts_dir = workdir / "ts_xtb"
    ts = _quiet(lambda: optimize_ts_atoms(eclipsed_ethane(), config, ts_dir))
    irc_dir = workdir / "irc_xtb"
    irc = _quiet(lambda: run_irc(ts.final_xyz, config, irc_dir, max_steps=100))
    min_dir = workdir / "min_xtb"
    minimum = _quiet(
        lambda: optimize_geometry_atoms(read(irc.full_path_xyz, index=0), config, min_dir)
    )
    return {
        "ts": _optimization_summary(ts, ts_dir),
        "irc": _irc_summary(irc, irc_dir),
        "min": _optimization_summary(minimum, min_dir),
    }


# ---------------------------------------------------------------------------
# CLI cases
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str], cwd: Path) -> list[str]:
    from ase_structure_optimizer.cli import main

    buffer = io.StringIO()
    old_cwd = Path.cwd()
    os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(buffer):
            code = main(argv)
    finally:
        os.chdir(old_cwd)
    assert code == 0
    lines = []
    for line in buffer.getvalue().splitlines():
        if line.startswith(CLI_PREFIXES):
            lines.append(line.replace(str(cwd), "<WORKDIR>"))
    return lines


def _cli_dirs(cwd: Path) -> dict[str, list[str]]:
    return {
        path.name: _listing(path)
        for path in sorted(cwd.iterdir())
        if path.is_dir()
    }


def case_cli(workdir: Path, name: str, extra: list[str]) -> dict[str, Any]:
    cwd = workdir / f"cli_{name}"
    cwd.mkdir(parents=True)
    if "--mode" in extra and extra[extra.index("--mode") + 1] == "irc":
        ts_xyz = _cu5_ts(workdir)
        xyz = cwd / "cu5_ts.xyz"
        xyz.write_text(ts_xyz.read_text(encoding="utf-8"), encoding="utf-8")
    elif "--mode" in extra and extra[extra.index("--mode") + 1] == "ts":
        xyz = cwd / "cu5_guess.xyz"
        write(xyz, cu5_ts_guess(), format="xyz")
    else:
        xyz = cwd / "water.xyz"
        write(xyz, water(), format="xyz")
    lines = _run_cli([str(xyz), "--calculator", "emt", *extra], cwd)
    dirs = _cli_dirs(cwd)
    finals = {}
    for dirname in dirs:
        for candidate in (cwd / dirname).glob("*optimized.xyz"):
            finals[f"{dirname}/{candidate.name}"] = _xyz_summary(candidate)
        for candidate in (cwd / dirname).glob("sella_irc_path.xyz"):
            finals[f"{dirname}/{candidate.name}"] = _xyz_summary(candidate)
    return {"stdout": lines, "dirs": dirs, "xyz": finals}


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def all_cases() -> dict[str, tuple[Callable[[Path], dict[str, Any]], str | None]]:
    """Return ``{case_name: (runner, required_module_or_None)}``."""

    cases: dict[str, tuple[Callable[[Path], dict[str, Any]], str | None]] = {}
    for optimizer in ("bfgs", "lbfgs", "fire"):
        cases[f"min_emt_{optimizer}"] = (
            lambda wd, opt=optimizer: case_min_emt(wd, opt),
            None,
        )
    cases["min_emt_file"] = (case_min_emt_file, None)
    cases["min_emt_charged_metadata"] = (case_min_emt_charged, None)
    cases["ts_emt"] = (case_ts_emt, None)
    cases["ts_emt_file_custom_names"] = (case_ts_emt_file, None)
    for direction in ("both", "forward", "reverse"):
        cases[f"irc_emt_{direction}"] = (
            lambda wd, d=direction: case_irc_emt(wd, d),
            None,
        )
    cases["irc_emt_atoms"] = (case_irc_emt_atoms, None)
    cases["freq_emt"] = (case_freq_emt, "geometric")
    cases["cli_min_default"] = (lambda wd: case_cli(wd, "min_default", []), None)
    cases["cli_min_fire"] = (
        lambda wd: case_cli(wd, "min_fire", ["--mode", "minimum", "--optimizer", "fire", "--fmax", "0.01"]),
        None,
    )
    cases["cli_min_outdir"] = (
        lambda wd: case_cli(wd, "min_outdir", ["--output-dir", "custom_out", "--charge", "1", "--multiplicity", "2"]),
        None,
    )
    cases["cli_ts"] = (lambda wd: case_cli(wd, "ts", ["--mode", "ts"]), None)
    cases["cli_irc"] = (lambda wd: case_cli(wd, "irc", ["--mode", "irc"]), None)
    cases["cli_irc_forward"] = (
        lambda wd: case_cli(wd, "irc_forward", ["--mode", "irc", "--irc-direction", "forward", "--irc-dx", "0.05"]),
        None,
    )
    cases["cli_irc_freq_warning"] = (
        lambda wd: case_cli(wd, "irc_freq", ["--mode", "irc", "--irc-direction", "reverse", "--max-steps", "10", "--frequencies"]),
        None,
    )
    cases["cli_ts_freq"] = (
        lambda wd: case_cli(wd, "ts_freq", ["--mode", "ts", "--frequencies"]),
        "geometric",
    )
    cases["ts_irc_min_xtb"] = (case_ts_irc_xtb, "xtb")
    return cases


def run_case(name: str, workdir: Path) -> dict[str, Any] | None:
    """Run one case; return ``None`` if its optional dependency is missing."""

    runner, requirement = all_cases()[name]
    if requirement is not None and not _has_module(requirement):
        return None
    workdir.mkdir(parents=True, exist_ok=True)
    return runner(workdir)
