"""Command line interface for structure optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calculators import CalculatorConfig, available_calculators
from .runner import available_minimizers


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Optimize an XYZ structure as a transition state or local minimum.",
    )
    parser.add_argument("xyz", type=Path, help="Initial structure in XYZ format.")
    parser.add_argument(
        "--mode",
        choices=("ts", "min", "minimum"),
        default="min",
        help="Optimization target: ordinary local minimum or Sella transition state.",
    )
    parser.add_argument(
        "--calculator",
        default="xtb",
        choices=available_calculators(),
        help="ASE calculator backend used for energy and forces.",
    )
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge.")
    parser.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for output files.")
    parser.add_argument("--fmax", type=float, default=5e-3, help="Force convergence threshold.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum optimization steps.")
    parser.add_argument(
        "--optimizer",
        default="bfgs",
        choices=available_minimizers(),
        help="ASE optimizer used when --mode min.",
    )
    parser.add_argument(
        "--frequencies",
        action="store_true",
        help="Run vibrational frequency analysis on the optimized structure.",
    )
    parser.add_argument(
        "--freq-delta",
        type=float,
        default=0.01,
        help="Finite-difference displacement in Angstrom for frequency analysis.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in Kelvin for geomeTRIC frequency analysis.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Pressure in atm for geomeTRIC frequency analysis.",
    )
    parser.add_argument("--threads", type=int, default=32, help="Thread count for Q-Chem calculators.")
    parser.add_argument(
        "--qchem-method",
        default="wb97x-v",
        help="Q-Chem method for --calculator qchem.",
    )
    parser.add_argument(
        "--qchem-basis",
        default="def2-tzvp",
        help="Q-Chem basis for --calculator qchem.",
    )
    parser.add_argument(
        "--qchem-label",
        default=None,
        help="Q-Chem calculation label/prefix.",
    )
    parser.add_argument(
        "--mace-model",
        default="extra_large",
        help="MACE-OMOL model size/name for --calculator maceomol.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    xyz = args.xyz.expanduser().resolve()
    if not xyz.is_file():
        parser.error(f"XYZ file does not exist: {xyz}")

    output_dir = args.output_dir
    mode = "min" if args.mode == "minimum" else args.mode

    if output_dir is None:
        suffix = "sella_ts" if mode == "ts" else "min"
        output_dir = xyz.with_suffix("").parent / f"{xyz.stem}_{suffix}_{args.calculator}"
    output_dir = output_dir.expanduser().resolve()

    config = CalculatorConfig(
        name=args.calculator,
        charge=args.charge,
        multiplicity=args.multiplicity,
        threads=args.threads,
        qchem_method=args.qchem_method,
        qchem_basis=args.qchem_basis,
        qchem_label=args.qchem_label or ("sella_ts" if mode == "ts" else "geometry_min"),
        mace_model=args.mace_model,
    )

    if mode == "ts":
        from .runner import run_ts_optimization

        result = run_ts_optimization(
            xyz_path=xyz,
            calculator_config=config,
            output_dir=output_dir,
            fmax=args.fmax,
            max_steps=args.max_steps,
        )
    else:
        from .runner import run_geometry_optimization

        result = run_geometry_optimization(
            xyz_path=xyz,
            calculator_config=config,
            output_dir=output_dir,
            fmax=args.fmax,
            max_steps=args.max_steps,
            optimizer=args.optimizer,
        )

    status = "converged" if result.converged else "stopped"
    label = (
        "Sella TS optimization"
        if mode == "ts"
        else f"{args.optimizer.upper()} geometry optimization"
    )
    final_label = "Final TS geometry" if mode == "ts" else "Final optimized geometry"
    print(f"{label} {status} after {result.steps} saved step(s).")
    print(f"{final_label}: {result.final_xyz}")
    print(f"Optimization path: {result.optimized_xyz}")
    print(f"Trajectory: {result.trajectory}")

    if args.frequencies:
        from .frequencies import run_frequency_analysis

        freq_result = run_frequency_analysis(
            xyz_path=result.final_xyz,
            calculator_config=config,
            output_dir=output_dir,
            delta=args.freq_delta,
            temperature=args.temperature,
            pressure=args.pressure,
        )
        print(
            "Frequency analysis complete: "
            f"{freq_result.imaginary_count} imaginary mode(s)."
        )
        print(f"Frequency summary: {freq_result.summary}")
        print(f"Detailed output: {freq_result.detailed_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
