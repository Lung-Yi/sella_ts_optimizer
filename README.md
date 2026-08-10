# ASE Structure Optimizer

ASE Structure Optimizer is a small Python package for structure optimization from a
single initial XYZ geometry. It supports both ordinary local-minimum geometry
optimization and transition-state optimization. It can be used in two ways:

- as a command line tool: `ase-structure-opt input.xyz --calculator xtb`
- as a Python library inside another project

The package uses ASE as the molecular structure/calculator interface. Local
minimum searches use ASE optimizers, and transition-state searches use Sella as
the saddle-point optimizer. Optional vibrational analysis uses ASE finite
differences plus geomeTRIC normal-mode analysis.

## Workflow

```text
initial XYZ or ASE Atoms
-> attach charge and spin metadata
-> build selected ASE calculator
-> ASE local-minimum optimization or Sella saddle-point optimization
-> final optimized XYZ, optimization path, ASE trajectory
```

With frequency analysis enabled:

```text
optimized structure
-> ASE finite-difference Hessian
-> geomeTRIC frequency analysis
-> frequency summary and detailed normal-mode output
```

With IRC path tracing:

```text
converged transition-state XYZ (e.g. output of --mode ts)
-> attach charge and spin metadata
-> build selected ASE calculator
-> Sella IRC, walked forward and/or reverse from the saddle point
-> forward path XYZ, reverse path XYZ, stitched reactant->TS->product path XYZ
```

A successful optimization means the selected optimizer converged according to
the selected calculator. It does not prove the structure is chemically desired.
For a local minimum, the optimized structure should normally have no imaginary
frequencies. For a first-order transition state, it should normally have exactly
one imaginary frequency.

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

Run ordinary local-minimum geometry optimization with the default calculator, `xtb`:

```bash
ase-structure-opt path/to/structure.xyz
```

Run transition-state optimization explicitly:

```bash
ase-structure-opt path/to/ts_guess.xyz --mode ts
```

Choose a calculator:

```bash
ase-structure-opt path/to/structure.xyz --calculator maceomol
```

Set charge and multiplicity:

```bash
ase-structure-opt path/to/structure.xyz \
  --calculator aimnet2 \
  --charge 1 \
  --multiplicity 2
```

Write outputs to a specific directory:

```bash
ase-structure-opt path/to/structure.xyz \
  --calculator maceomol \
  --output-dir runs/my_min
```

Run optimization plus vibrational analysis:

```bash
ase-structure-opt path/to/structure.xyz \
  --calculator maceomol \
  --frequencies
```

Choose an ASE optimizer for local-minimum optimization:

```bash
ase-structure-opt path/to/structure.xyz \
  --mode min \
  --optimizer lbfgs
```

Trace the IRC path from a converged transition state:

```bash
ase-structure-opt path/to/sella_ts_optimized.xyz \
  --mode irc \
  --calculator maceomol
```

The input to `--mode irc` should already be a converged transition state, for
example the `sella_ts_optimized.xyz` produced by `--mode ts`. IRC starts by
diagonalizing the Hessian at the input geometry to find the reaction
direction, so starting from a geometry that is not close to a genuine
first-order saddle point will produce a poor or invalid path. IRC is
calculator-agnostic, exactly like `--mode ts`, so `--calculator maceomol`
works the same way here as it does for TS or local-minimum optimization.

By default both directions away from the transition state are traced. Trace
only one direction, or change the IRC step size:

```bash
ase-structure-opt path/to/sella_ts_optimized.xyz \
  --mode irc \
  --calculator maceomol \
  --irc-direction forward \
  --irc-dx 0.05
```

If you do not want to install the package, use the local wrapper:

```bash
python run_ase_structure_opt.py path/to/structure.xyz --calculator xtb
```

## Quick Start: Python Library

Use an XYZ file for transition-state optimization:

```python
from pathlib import Path

from ase_structure_optimizer import CalculatorConfig, run_ts_optimization

result = run_ts_optimization(
    xyz_path=Path("ts_guess.xyz"),
    calculator_config=CalculatorConfig(name="maceomol"),
    output_dir=Path("runs/my_ts"),
)

print(result.converged)
print(result.final_xyz)
```

Use an XYZ file for local-minimum geometry optimization:

```python
from pathlib import Path

from ase_structure_optimizer import CalculatorConfig, run_geometry_optimization

result = run_geometry_optimization(
    xyz_path=Path("structure.xyz"),
    calculator_config=CalculatorConfig(name="maceomol"),
    output_dir=Path("runs/my_minimum"),
    optimizer="bfgs",
)

print(result.converged)
print(result.final_xyz)
```

Use an existing ASE `Atoms` object:

```python
from pathlib import Path

from ase.io import read

from ase_structure_optimizer import CalculatorConfig, optimize_ts_atoms

atoms = read("ts_guess.xyz")

result = optimize_ts_atoms(
    atoms=atoms,
    calculator_config=CalculatorConfig(name="maceomol", charge=0, multiplicity=1),
    output_dir=Path("runs/my_ts"),
)

print(result.final_xyz)
```

Trace an IRC path from Python:

```python
from pathlib import Path

from ase_structure_optimizer import CalculatorConfig, run_irc

irc_result = run_irc(
    xyz_path=Path("runs/my_ts/sella_ts_optimized.xyz"),
    calculator_config=CalculatorConfig(name="maceomol"),
    output_dir=Path("runs/my_irc"),
)

print(irc_result.forward_converged, irc_result.reverse_converged)
print(irc_result.full_path_xyz)
```

Run frequency analysis from Python:

```python
from pathlib import Path

from ase_structure_optimizer import CalculatorConfig, run_frequency_analysis

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
from ase_structure_optimizer import (
    CalculatorConfig,
    FrequencyResult,
    IRCResult,
    OptimizationResult,
    analyze_frequencies_atoms,
    available_calculators,
    available_minimizers,
    build_calculator,
    optimize_geometry_atoms,
    optimize_irc_atoms,
    optimize_ts_atoms,
    run_geometry_optimization,
    run_frequency_analysis,
    run_irc,
    run_ts_optimization,
)
```

## Input XYZ

The input should be a single-frame XYZ file containing an initial structure or
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

IRC path tracing (`--mode irc`) uses the same Sella machinery as `--mode ts`
and imposes no extra requirements on the calculator, so every calculator
listed above works with `--mode irc` exactly as it does with `--mode ts`.

Examples:

```bash
ase-structure-opt ts_guess.xyz --mode ts --calculator xtb
ase-structure-opt structure.xyz --calculator xtb
ase-structure-opt structure.xyz --mode min --optimizer fire --calculator maceomol
ase-structure-opt structure.xyz --calculator maceomol --mace-model extra_large
ase-structure-opt ts_guess.xyz --mode ts --calculator qchem --threads 32
ase-structure-opt sella_ts_optimized.xyz --mode irc --calculator maceomol
```

## MACE-OMOL Model Location

When `--calculator maceomol` is used with the default model name,
`--mace-model extra_large`, the MACE package looks for this model file:

```text
MACE-omol-0-extra-large-1024.model
```

If the file is not already cached, MACE downloads it automatically from the
official MACE foundations release and saves it in the local MACE cache.

By default, the cache directory is:

```text
~/.cache/mace/
```

So the default full path is usually:

```text
~/.cache/mace/MACE-omol-0-extra-large-1024.model
```

On this machine, the model was found at:

```text
/home/lungyi/.cache/mace/MACE-omol-0-extra-large-1024.model
```

The cache location follows `XDG_CACHE_HOME`. If `XDG_CACHE_HOME` is set, MACE
uses:

```text
$XDG_CACHE_HOME/mace/MACE-omol-0-extra-large-1024.model
```

For example, this command makes MACE read/write the model under
`/data/model_cache/mace/`:

```bash
XDG_CACHE_HOME=/data/model_cache \
  ase-structure-opt structure.xyz --calculator maceomol
```

You can also bypass the cache lookup and pass a local model path explicitly:

```bash
ase-structure-opt structure.xyz \
  --calculator maceomol \
  --mace-model /path/to/MACE-omol-0-extra-large-1024.model
```

The same setting works from Python:

```python
from pathlib import Path

from ase_structure_optimizer import CalculatorConfig, run_geometry_optimization

config = CalculatorConfig(
    name="maceomol",
    mace_model="/home/lungyi/.cache/mace/MACE-omol-0-extra-large-1024.model",
)

result = run_geometry_optimization(
    xyz_path=Path("structure.xyz"),
    calculator_config=config,
    output_dir=Path("runs/my_min"),
)
```

## Outputs

By default, local-minimum outputs are written next to the input XYZ in:

```text
<xyz_stem>_min_<calculator>/
```

Transition-state outputs, when `--mode ts` is used, are written in:

```text
<xyz_stem>_sella_ts_<calculator>/
```

Transition-state optimization outputs:

- `sella_ts_optimized.xyz`: final optimized transition-state geometry
- `sella_ts_path.xyz`: all saved optimization images
- `sella_ts.traj`: ASE trajectory written by Sella

Local-minimum optimization outputs:

- `geometry_min_optimized.xyz`: final optimized geometry
- `geometry_min_path.xyz`: all saved optimization images
- `geometry_min.traj`: ASE trajectory written by the selected ASE optimizer

IRC outputs, when `--mode irc` is used, are written in:

```text
<xyz_stem>_irc_<calculator>/
```

- `sella_irc_path.xyz`: the full stitched IRC path — reverse-branch images
  reversed, then forward-branch images, so the shared transition-state frame
  appears exactly once and the path reads continuously from one branch's
  endpoint through the TS to the other branch's endpoint. This is the main
  deliverable for visualizing the reaction path.
- `sella_irc_forward.xyz` / `sella_irc_reverse.xyz`: all saved images for each
  individually traced direction (only written for directions actually run,
  per `--irc-direction`)
- `sella_irc_forward.traj` / `sella_irc_reverse.traj`: ASE trajectories
  written by Sella's IRC optimizer for each direction

Frequency outputs, when `--frequencies` is used:

- `frequencies_<calculator>_summary.txt`: compact frequency table
- `vibrations_<calculator>_geometric.txt`: detailed geomeTRIC output
- `vib_<calculator>/`: ASE finite-difference displacement cache

The CLI prints whether the optimizer reported convergence, where files were
written, and how many imaginary frequencies were found when frequency analysis
is run. For `--mode irc`, it prints convergence and step count separately for
the forward and reverse branches, plus the path to the stitched full path XYZ.

## API Design Notes

Use `run_geometry_optimization()` or `run_ts_optimization()` when your workflow
starts from an XYZ file. Use `optimize_geometry_atoms()` or `optimize_ts_atoms()`
when another package already created an ASE `Atoms` object.

These functions return `OptimizationResult`:

```python
OptimizationResult(
    trajectory=...,
    optimized_xyz=...,
    final_xyz=...,
    converged=...,
    steps=...,
    mode=...,
    optimizer=...,
)
```

Use `run_frequency_analysis()` for an optimized XYZ file. Use
`analyze_frequencies_atoms()` for an ASE `Atoms` object. Both return
`FrequencyResult`, including `frequencies_cm1` and `imaginary_count`.

Use `run_irc()` for a converged transition-state XYZ file. Use
`optimize_irc_atoms()` for an ASE `Atoms` object already at a saddle point.
Both return `IRCResult`:

```python
IRCResult(
    forward_trajectory=...,
    reverse_trajectory=...,
    forward_xyz=...,
    reverse_xyz=...,
    full_path_xyz=...,
    forward_converged=...,
    reverse_converged=...,
    forward_steps=...,
    reverse_steps=...,
    direction=...,
    optimizer=...,
)
```

Fields for a direction that was not run (via `direction="forward"` or
`direction="reverse"`) are `None`.

## Practical Notes

- All optimizers use the potential-energy surface provided by the selected
  calculator. Results depend strongly on the calculator.
- xTB and ML potentials are usually practical for screening.
- Q-Chem/DFT can be expensive, especially for frequency analysis.
- IRC assumes the input geometry is already a converged first-order saddle
  point; a direction that reaches `--max-steps` without the local Hessian's
  lowest eigenvalue turning positive is reported as `converged=False`
  ("stopped"), which is a normal, expected outcome for IRC and not
  necessarily a failure. `--irc-dx` controls the IRC step size; other Sella
  IRC tuning parameters are left at their library defaults.
- IRC computes an initial Hessian diagonalization independently for each
  direction it runs (`--irc-direction both` runs it twice). This is
  negligible for xTB and ML potentials, but doubles the up-front Hessian cost
  for Q-Chem/DFT calculators.
- Classical force fields and general-purpose ML potentials may be unreliable
  for reactive geometries outside their training domain.
- A common workflow is to optimize cheaply with xTB or an ML potential, then
  verify or refine with DFT.
