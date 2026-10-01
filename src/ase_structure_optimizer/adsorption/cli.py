"""Command line interface ``ase-adsorb``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Subcommands implemented in later milestones.
_PLANNED = {
    "vasp": ("Select configurations and write VASP inputs.", "M6"),
    "dft-collect": ("Read VASP results and compare with the MLIP.", "M7"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``ase-adsorb`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="ase-adsorb",
        description="Automated molecule-on-surface adsorption sampling with periodic MLIPs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-molecule",
        help="Print the inferred bonds, fragments, anchors and reference axis of a molecule.",
    )
    check.add_argument("molecule", type=Path, help="Gas-phase molecule (xyz).")
    check.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file whose molecule_props (bond_scale, bond_overrides, reference_axis) are used.",
    )
    check.add_argument("--bond-scale", type=float, default=None, help="Override molecule_props.bond_scale.")
    check.add_argument("--json", type=Path, default=None, help="Also write the analysis as JSON to this file.")

    init = subparsers.add_parser("init-config", help="Print an annotated configuration template.")
    init.add_argument("-o", "--output", type=Path, default=None, help="Write to this file instead of stdout.")

    run = subparsers.add_parser(
        "run",
        help="Run the MLIP workflow (options override the configuration file).",
        description=(
            "Run the adsorption workflow. Either give a configuration file, or at least "
            "--molecule and --solid; options override the file's values."
        ),
    )
    run.add_argument("config", type=Path, nargs="?", default=None, help="Configuration file (YAML).")
    run.add_argument("--molecule", type=Path, default=None, help="Gas-phase molecule (xyz).")
    run.add_argument("--solid", type=Path, default=None, help="Bulk crystal or slab (cif).")
    run.add_argument("--calculator", default=None, help="calculator.name, e.g. macemp.")
    run.add_argument("--mace-mp-model", default=None, help="calculator.mace_mp_model (name or file path).")
    run.add_argument("--uma-model", default=None, help="calculator.uma_model (name or checkpoint file path).")
    run.add_argument("--uma-task", default=None, help="calculator.uma_task (oc20, omat, oc22, oc25, odac or omc).")
    run.add_argument("--device", default=None, help="calculator.device (auto, cuda, cpu).")
    run.add_argument(
        "--miller",
        type=int,
        nargs=3,
        action="append",
        metavar=("H", "K", "L"),
        default=None,
        help="Miller index; repeat for several (replaces surface.miller_indices).",
    )
    run.add_argument("--budget", type=float, default=None, help="budget.wall_time_per_termination in s (0 = unlimited).")
    run.add_argument("--run-dir", type=Path, default=None, help="Output directory.")

    resume = subparsers.add_parser("resume", help="Continue an interrupted run.")
    resume.add_argument("run_dir", type=Path, help="Run directory containing state.json.")

    report = subparsers.add_parser("report", help="Redraw the figures and write report.html (no calculations).")
    report.add_argument("run_dir", type=Path, help="Run directory containing state.json.")
    report.add_argument("--no-animations", action="store_true", help="Skip the GIF and HTML animations.")

    for name, (description, milestone) in _PLANNED.items():
        planned = subparsers.add_parser(name, help=f"{description} (not implemented yet, {milestone})")
        planned.add_argument("args", nargs="*", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``ase-adsorb`` command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-molecule":
        return _check_molecule(parser, args)
    if args.command == "init-config":
        from .config import default_config_yaml

        text = default_config_yaml()
        if args.output is None:
            sys.stdout.write(text)
        else:
            if args.output.exists():
                parser.error(f"{args.output} already exists; not overwriting")
            args.output.write_text(text, encoding="utf-8")
            print(f"Configuration template written to {args.output}")
        return 0
    if args.command in ("run", "resume"):
        return _run(parser, args)
    if args.command == "report":
        return _report(args)

    description, milestone = _PLANNED[args.command]
    print(f"ase-adsorb {args.command}: not implemented yet (planned for milestone {milestone}).", file=sys.stderr)
    return 2


def run_config_from_args(args: argparse.Namespace):
    """Build the `AdsorptionConfig` of ``ase-adsorb run`` from a file and options."""

    from .config import _require_yaml, config_from_dict

    data: dict = {}
    base_dir = Path.cwd()
    if args.config is not None:
        yaml = _require_yaml()
        data = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        base_dir = args.config.resolve().parent
    # Paths given on the command line are relative to the current directory.
    for key in ("molecule", "solid", "run_dir"):
        value = getattr(args, key)
        if value is not None:
            data[key] = str(value.expanduser().resolve())
    calculator = dict(data.get("calculator") or {})
    for option, key in (
        ("calculator", "name"),
        ("mace_mp_model", "mace_mp_model"),
        ("uma_model", "uma_model"),
        ("uma_task", "uma_task"),
        ("device", "device"),
    ):
        if getattr(args, option) is not None:
            calculator[key] = getattr(args, option)
    if calculator:
        data["calculator"] = calculator
    if args.miller is not None:
        data["surface"] = {**(data.get("surface") or {}), "miller_indices": [list(m) for m in args.miller]}
    if args.budget is not None:
        data["budget"] = {**(data.get("budget") or {}), "wall_time_per_termination": args.budget}
    return config_from_dict(data, base_dir=base_dir)


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .state import StateError
    from .workflow import resume_adsorption_workflow, run_adsorption_workflow

    try:
        if args.command == "run":
            if args.config is None and (args.molecule is None or args.solid is None):
                parser.error("give a configuration file or both --molecule and --solid")
            config = run_config_from_args(args)
            for label, path in (("molecule", config.molecule), ("solid", config.solid)):
                if not path.is_file():
                    parser.error(f"{label} file does not exist: {path}")
            result = run_adsorption_workflow(config)
        else:
            result = resume_adsorption_workflow(args.run_dir)
    except (ConfigError, StateError) as exc:
        print(f"ase-adsorb {args.command}: {exc}", file=sys.stderr)
        return 1

    print(f"\nRun directory: {result.run_dir}")
    for termination in result.terminations:
        best = (
            f"most stable {termination.best_unique_id}, E_ads {termination.best_eads:.3f} eV ({termination.best_class})"
            if termination.best_eads is not None
            else "no successful relaxation"
        )
        print(
            f"  {termination.term_id}: {termination.n_relaxed} relaxed, {termination.n_failed} failed, "
            f"{termination.n_unique} unique; {best}; {termination.unique_csv}"
        )
        for warning in termination.warnings:
            print(f"    warning: {warning}")
    if result.summary_csv is not None:
        print(f"Summary: {result.summary_csv}")
    if result.figures_dir is not None:
        print(f"Figures: {result.figures_dir}")
    if result.report is not None:
        print(f"Report: {result.report}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .report import generate_report
    from .state import StateError
    from .workflow import run_logging

    run_dir = args.run_dir.expanduser().resolve()
    if not (run_dir / "state.json").is_file():
        print(f"ase-adsorb report: {run_dir} is not a run directory (no state.json)", file=sys.stderr)
        return 1
    try:
        with run_logging(run_dir):
            path = generate_report(run_dir, draw_animations=not args.no_animations)
    except (ConfigError, StateError, FileNotFoundError) as exc:
        print(f"ase-adsorb report: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {path}")
    print(f"Figures: {run_dir / 'figures'}")
    return 0


def _check_molecule(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from ..structures import read_structure
    from .config import ConfigError, MoleculeProps, load_config
    from .molecule import analyze_molecule, format_analysis

    path = args.molecule.expanduser()
    if not path.is_file():
        parser.error(f"molecule file does not exist: {path}")

    props = MoleculeProps()
    if args.config is not None:
        try:
            props = load_config(args.config).molecule_props
        except (ConfigError, OSError) as exc:
            parser.error(str(exc))
    if args.bond_scale is not None:
        if args.bond_scale <= 0:
            parser.error("--bond-scale must be positive")
        props = dataclasses.replace(props, bond_scale=args.bond_scale)

    atoms = read_structure(path)
    try:
        analysis = analyze_molecule(atoms, props)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Molecule: {path}  ({atoms.get_chemical_formula()}, bond_scale {props.bond_scale})\n")
    print(format_analysis(analysis, atoms.get_positions()))
    print(
        "\nCheck that the bonds above are chemically correct before running the workflow; "
        "fix them with molecule_props.bond_scale or bond_overrides."
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(analysis.to_dict(), indent=1), encoding="utf-8")
        print(f"Analysis written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
