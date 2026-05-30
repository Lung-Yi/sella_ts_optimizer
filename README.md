# Sella TS Optimizer

Sella TS Optimizer is a small Python package for transition-state optimization
from a single initial XYZ guess. It can be used in two ways:

- as a command line tool: `sella-ts-opt input.xyz --calculator xtb`
- as a Python library inside another project

The package uses ASE as the molecular structure/calculator interface and Sella
as the saddle-point optimizer. Optional vibrational analysis uses ASE finite
differences plus geomeTRIC normal-mode analysis.

## Workflow

```text
initial TS guess XYZ or ASE Atoms
-> attach charge and spin metadata
-> build selected ASE calculator
-> Sella saddle-point optimization
-> optimized TS XYZ, optimization path, ASE trajectory
```

With frequency analysis enabled:

```text
optimized TS structure
-> ASE finite-difference Hessian
-> geomeTRIC frequency analysis
-> frequency summary and detailed normal-mode output
```

A successful Sella run means the optimizer converged according to the selected
calculator. It does not prove the structure is the chemically desired transition
state. For a first-order transition state, the optimized structure should
normally have exactly one imaginary frequency.

## Installation

From this repository:

```bash
pip install -e .
```

This installs only the core package dependencies: `ase`, `sella`, and `numpy`.
Calculator backends are imported only when selected, so install the backend you
plan to use.

For MACE-OMOL:

```bash
pip install -e ".[mace]"
```

For frequency analysis:

```bash
pip install -e ".[vibrations]"
```

For both:

```bash
pip install -e ".[all]"
```

Other calculator backends, such as xTB, Q-Chem, FAIRChem, and AIMNet2, are
environment-specific and must be installed/configured separately.

## Quick Start: CLI

Run with the default calculator, `xtb`:

```bash
sella-ts-opt path/to/ts_guess.xyz
```

Choose a calculator:

```bash
sella-ts-opt path/to/ts_guess.xyz --calculator maceomol
```

Set charge and multiplicity:

```bash
sella-ts-opt path/to/ts_guess.xyz \
  --calculator aimnet2 \
  --charge 1 \
  --multiplicity 2
```

Write outputs to a specific directory:

```bash
sella-ts-opt path/to/ts_guess.xyz \
  --calculator maceomol \
  --output-dir runs/my_ts
```

Run optimization plus vibrational analysis:

```bash
sella-ts-opt path/to/ts_guess.xyz \
  --calculator maceomol \
  --frequencies
```

If you do not want to install the package, use the local wrapper:

```bash
python run_sella_ts.py path/to/ts_guess.xyz --calculator xtb
```

## Quick Start: Python Library

Use an XYZ file:

```python
from pathlib import Path

from sella_ts_optimizer import CalculatorConfig, run_ts_optimization

result = run_ts_optimization(
    xyz_path=Path("ts_guess.xyz"),
    calculator_config=CalculatorConfig(name="maceomol"),
    output_dir=Path("runs/my_ts"),
)

print(result.converged)
print(result.final_xyz)
```

Use an existing ASE `Atoms` object:

```python
from pathlib import Path

from ase.io import read

from sella_ts_optimizer import CalculatorConfig, optimize_ts_atoms

atoms = read("ts_guess.xyz")

result = optimize_ts_atoms(
    atoms=atoms,
    calculator_config=CalculatorConfig(name="maceomol", charge=0, multiplicity=1),
    output_dir=Path("runs/my_ts"),
)

print(result.final_xyz)
```

Run frequency analysis from Python:

```python
from pathlib import Path

from sella_ts_optimizer import CalculatorConfig, run_frequency_analysis

freq_result = run_frequency_analysis(
    xyz_path=Path("runs/my_ts/sella_ts_optimized.xyz"),
    calculator_config=CalculatorConfig(name="maceomol"),
    output_dir=Path("runs/my_ts"),
)

print(freq_result.imaginary_count)
print(freq_result.frequencies_cm1)
```

The main public imports are:

```python
from sella_ts_optimizer import (
    CalculatorConfig,
    FrequencyResult,
    OptimizationResult,
    analyze_frequencies_atoms,
    available_calculators,
    build_calculator,
    optimize_ts_atoms,
    run_frequency_analysis,
    run_ts_optimization,
)
```

## Input XYZ

The input should be a single-frame XYZ file containing an initial
transition-state guess:

```xyz
3
initial TS guess
C  0.5991509178  0.1139063154  0.0000000000
N -0.5792138678  0.0650815375  0.0000000000
H  0.3259086002 -1.0308725400  0.0000000000
```

No separate `chg` or `mult` files are required. Charge and multiplicity default
to neutral singlet, `0` and `1`.

## Supported Calculators

The CLI accepts these calculator names:

- `xtb`: GFN2-xTB
- `maceomol`: MACE-OMOL
- `uma_s`: FAIRChem UMA-S
- `uma_m`: FAIRChem UMA-M
- `eSEN`: FAIRChem eSEN
- `aimnet2`: AIMNet2
- `qchem`: Q-Chem with `wb97x-v/def2-tzvp` by default
- `b3lyp`: Q-Chem with `b3lyp/def2-svp`
- `emt`: ASE EMT, useful only for smoke tests

Examples:

```bash
sella-ts-opt ts_guess.xyz --calculator xtb
sella-ts-opt ts_guess.xyz --calculator maceomol --mace-model extra_large
sella-ts-opt ts_guess.xyz --calculator qchem --threads 32
```

MACE-OMOL downloads pretrained model files through the MACE package. By default,
the cache is:

```text
~/.cache/mace/
```

You can also pass a local model path:

```bash
sella-ts-opt ts_guess.xyz \
  --calculator maceomol \
  --mace-model /path/to/MACE-omol-0-extra-large-1024.model
```

## Outputs

By default, outputs are written next to the input XYZ in:

```text
<xyz_stem>_sella_ts_<calculator>/
```

Optimization outputs:

- `sella_ts_optimized.xyz`: final optimized transition-state geometry
- `sella_ts_path.xyz`: all saved optimization images
- `sella_ts.traj`: ASE trajectory written by Sella

Frequency outputs, when `--frequencies` is used:

- `frequencies_<calculator>_summary.txt`: compact frequency table
- `vibrations_<calculator>_geometric.txt`: detailed geomeTRIC output
- `vib_<calculator>/`: ASE finite-difference displacement cache

The CLI prints whether Sella reported convergence, where files were written,
and how many imaginary frequencies were found when frequency analysis is run.

## API Design Notes

Use `run_ts_optimization()` when your workflow starts from an XYZ file. Use
`optimize_ts_atoms()` when another package already created an ASE `Atoms`
object.

Both functions return `OptimizationResult`:

```python
OptimizationResult(
    trajectory=...,
    optimized_xyz=...,
    final_xyz=...,
    converged=...,
    steps=...,
)
```

Use `run_frequency_analysis()` for an optimized XYZ file. Use
`analyze_frequencies_atoms()` for an ASE `Atoms` object. Both return
`FrequencyResult`, including `frequencies_cm1` and `imaginary_count`.

## Practical Notes

- Sella optimizes on the potential-energy surface provided by the selected
  calculator. Results depend strongly on the calculator.
- xTB and ML potentials are usually practical for screening.
- Q-Chem/DFT can be expensive, especially for frequency analysis.
- Classical force fields and general-purpose ML potentials may be unreliable
  for reactive geometries outside their training domain.
- A common workflow is to optimize cheaply with xTB or an ML potential, then
  verify or refine with DFT.
