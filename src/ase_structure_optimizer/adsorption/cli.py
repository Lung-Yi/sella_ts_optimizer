"""Command line interface ``ase-adsorb``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Subcommands implemented in later milestones.
_PLANNED = {
    "run": ("Run the full MLIP workflow.", "M3"),
    "resume": ("Continue an interrupted run.", "M3"),
    "report": ("Regenerate figures and report.html.", "M5"),
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

    description, milestone = _PLANNED[args.command]
    print(f"ase-adsorb {args.command}: not implemented yet (planned for milestone {milestone}).", file=sys.stderr)
    return 2


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
