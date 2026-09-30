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

For the shared tools used by the (upcoming) adsorption workflow, such as bond
graphs (`networkx`) and MACE-MP with D3 dispersion (`torch-dftd`):

```bash
pip install -e ".[adsorption]"
```

For running the test suite:

```bash
pip install -e ".[test]"
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
- `macemp`: MACE-MP materials foundation model (`medium-mpa-0` by default),
  for periodic systems such as bulk solids and surfaces

The FAIRChem UMA models (`uma_s`, `uma_m`) use the `omol` task by default,
which is the molecular task used in all existing workflows. From Python, set
`CalculatorConfig(uma_task=...)` to `omat` (bulk materials) or `oc20`
(adsorbate + surface systems) for periodic systems. `eSEN` only supports
`omol`. Energies from different tasks or models must never be mixed.

ML backends run on CUDA when available and otherwise on CPU. From Python,
`CalculatorConfig(device="cpu")` or `device="cuda"` overrides this.

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

## MACE-MP Model Location

`macemp` reads its model name from `CalculatorConfig.mace_mp_model`
(default `medium-mpa-0`). This setting is separate from `mace_model`, which
only affects `maceomol`. Like MACE-OMOL, MACE-MP models are downloaded on
first use into the MACE cache, `~/.cache/mace/` or
`$XDG_CACHE_HOME/mace/`. Available model names depend on the installed
`mace-torch` version. On machines without internet access, download the model
file elsewhere and pass its path:

```python
from ase_structure_optimizer import CalculatorConfig

config = CalculatorConfig(
    name="macemp",
    mace_mp_model="/data/models/mace-mpa-0-medium.model",
    dispersion=True,     # add D3(BJ), requires torch-dftd
    dtype="float64",     # MACE default_dtype; float32 is faster for screening
)
```

FAIRChem UMA models are downloaded from Hugging Face on first use and need an
account with access to the UMA model repository (`huggingface-cli login`).

## Periodic Systems

The core optimization functions also accept periodic ASE `Atoms` (bulk
crystals and slabs, with `atoms.pbc` set). Molecular inputs (all `pbc`
False), including every XYZ input to `ase-structure-opt`, behave exactly as
before.

For periodic systems:

- Use a periodic calculator. Check with `calculator_capabilities()`: the
  molecular models (`maceomol`, `aimnet2`, `eSEN`, UMA with `uma_task="omol"`,
  `xtb`, `qchem`, `b3lyp`) report `periodic=False` and must not be used for
  surfaces or solids. `macemp`, UMA with `omat`/`oc20`, and `emt` (tests
  only) report `periodic=True`.
- Charge and multiplicity are stored only in `atoms.info`; no initial charge
  or magnetic moment is written to atom 0, unlike for molecules.
- `path` and `final` output files keep their `.xyz` names but are written in
  extended XYZ format, so the cell, pbc and fixed atoms (`FixAtoms`) are
  preserved.
- `optimize_geometry_atoms(..., cell_filter=True)` also relaxes the cell
  (`FrechetCellFilter`, or `ExpCellFilter` on older ASE).
- `FixAtoms` constraints on the input are kept during optimization.

```python
from pathlib import Path

from ase.build import fcc111
from ase.constraints import FixAtoms

from ase_structure_optimizer import (
    CalculatorConfig,
    calculator_capabilities,
    get_calculator,
    optimize_geometry_atoms,
)

slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0)
slab.pbc = True
slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.tag >= 3]))

config = CalculatorConfig(name="macemp", dispersion=True)
assert calculator_capabilities(config).periodic

result = optimize_geometry_atoms(
    atoms=slab,
    calculator_config=config,
    output_dir=Path("runs/cu111"),
    fmax=0.05,
    optimizer="lbfgs",
    calculator=get_calculator(config),  # cached: the model is loaded only once
)
print(result.energy, result.final_xyz)
```

Structure helpers in `ase_structure_optimizer.structures`:

- `read_structure(path)`: reads xyz / extxyz / cif / traj (last frame). CIF
  occupancy info and tags are cleared so `ase.visualize.plot.plot_atoms`
  works; plain XYZ files are returned with `pbc=False`.
- `write_trajectory_pair(images, path_stem)`: writes `<path_stem>.traj` and a
  text `<path_stem>.extxyz` copy, keeping stored energies.
- `is_slab(atoms, min_gap=8.0)`: returns `(True, axis)` when exactly one cell
  axis has a vacuum gap of at least `min_gap` Å, otherwise `(False, None)`.

Bond-graph helpers in `ase_structure_optimizer.graphs` (need `networkx`):

- `build_bond_graph(atoms, indices=None, scale=1.2, overrides=())`: atoms are
  bonded when `d < scale * (r_cov_i + r_cov_j)` (minimum image along periodic
  directions); `overrides` forces `(i, j, True/False)` bonds.
- `split_fragments(graph, symbols)`: ligand fragments after removing
  transition-metal atoms, named by formula (`C5H5`, `CO_1`, `CO_2`, `H`).
- `same_connectivity(g1, g2)` and `graph_hash(graph, symbols)`: compare bond
  graphs index-wise or independent of atom order.

## Adsorption Workflow (in development)

The `ase_structure_optimizer.adsorption` subpackage and the `ase-adsorb`
command (also `python run_ase_adsorption.py`) automate molecule-on-surface
adsorption sampling with periodic MLIPs. They need the `[adsorption]` extra.
Currently available:

```bash
# Inspect the inferred bonds, ligand fragments, anchor points and reference
# axis of the adsorbate before running anything.
ase-adsorb check-molecule molecule.xyz [--config config.yaml] [--bond-scale 1.2] [--json analysis.json]

# Print (or write) the annotated configuration template with all defaults.
ase-adsorb init-config > config.yaml
```

Only `molecule` and `solid` are required in the configuration file. Molecular
calculators (`maceomol`, `aimnet2`, `eSEN`, UMA `omol`, `xtb`, Q-Chem) are
rejected when the file is loaded. From Python:

```python
from ase_structure_optimizer.adsorption import analyze_molecule, load_config
from ase_structure_optimizer.structures import read_structure

config = load_config("config.yaml")
analysis = analyze_molecule(read_structure(config.molecule), config.molecule_props)
print([fragment.name for fragment in analysis.fragments])
print(analysis.reference_axis.method, [anchor.label for anchor in analysis.anchors])
```

Run the MLIP part of the workflow (gas-phase reference, surfaces, initial
configurations, time-budgeted prescreening and full relaxations):

```bash
ase-adsorb run config.yaml
# or without a configuration file; options override the file's values
ase-adsorb run --molecule mol.xyz --solid TiSi.cif --calculator macemp --miller 0 0 1
ase-adsorb run config.yaml --mace-mp-model /path/to/mace-mp-0b3-medium.model --budget 600

# continue after an interruption (finished items are skipped, the remaining
# time budget accounts for the time already spent)
ase-adsorb resume <run_dir>
```

From Python:

```python
from ase_structure_optimizer.adsorption import load_config, run_adsorption_workflow

result = run_adsorption_workflow(load_config("config.yaml"))
for termination in result.terminations:
    print(termination.term_id, termination.n_relaxed, termination.best_config_id, termination.best_eads_screen)
```

How the sampling works:

- Surfaces: a bulk input is relaxed with a cell filter, cut with pymatgen
  (terminations ranked by MLIP surface energy, at most `max_terminations`),
  repeated laterally to `min_lateral`, and relaxed with its bottom layers
  fixed; ontop / bridge / hollow sites are found per termination
  (`bulk/`, `terminations/<miller>_t<i>/`). Miller indices refer to the axes
  of the input CIF when it is a conventional cell; a primitive input cell
  (e.g. `ase.build.bulk("Cu")`) is converted to the conventional cell first.
- The molecule is relaxed in a periodic box (`molecule/gas_opt.*`) for E_mol.
  If the MLIP changes its bonds there (e.g. an eta5-Cp ring slipping), the
  input geometry is used for sampling and as the intact-molecule reference,
  and a warning is logged.
- Each (site x anchor x spin) and `n_random` random orientations give an
  initial configuration. The molecule is lowered until its closest atom is at
  the contact gap (`0.9 x` the sum of van der Waals radii by default: Bondi,
  with Alvarez 2013 values for elements Bondi lacks). Clashing, self-image
  and duplicate configurations are dropped (`candidates/`).
- The step time is measured, then as many candidates as fit into
  `prescreen_fraction` of `wall_time_per_termination` are prescreened
  (stratified over site type x anchor; `prescreen.csv`).
- The best configuration of each contact label (or, if not yet in contact,
  of each fragment facing the surface) is fully relaxed first, then the rest
  by energy, until `prescreen_fraction + relax_fraction` of the budget is
  used (`relax/<config_id>.traj/.extxyz`, `results.csv`, `results.db`).
- Failed configurations (exceptions, NaN, atoms closer than 0.5 Å, molecule
  more than 10 Å from the surface) are recorded with a reason and skipped.

Analysis of the fully relaxed configurations (per termination):

- Final energies: a single point with `calculator.dtype_final` for every
  relaxed configuration; `E_ads = E(slab+mol) - E(slab) - E(mol)` uses only
  final-dtype energies. `eads_screen` keeps the screening-dtype value.
- Class, in this order: `dissociated` (the molecule's bonds changed; with a
  metal center this is judged per ligand: bonds inside ligands unchanged and
  every ligand still bonded to its metal, so a hapticity change such as
  eta5 -> eta3 Cp is not a dissociation), `desorbed` (farther than
  `desorbed_distance` from the slab), `chemisorbed` (at least one contact
  atom), `physisorbed`. `surface_distorted` flags free slab atoms moved by
  more than 1 Å.
- Contact label: fragments touching the surface (`CO_1+H`, `C5H5`, `Mo` for
  the metal atom), `none` without contact.
- Angles: tilt θ between the reference axis u and the surface normal (0° = u
  points away from the surface; axes without head/tail fold into 0-90°),
  azimuth φ of the second principal axis against cell vector a; `height` is
  the molecule center (metal, else center of mass) above the topmost slab atom.
- `results.csv`: all relaxed configurations; `unique.csv`: de-duplicated
  (same class and contact, |ΔE_ads| < `dedup_energy_tol`, RMSD up to lattice
  translations < `dedup_rmsd_tol`) with Boltzmann weights at `temperature`;
  `summary.csv`: unique configurations of all terminations.
- Rigid approach scan (`approach_scan.csv/.extxyz`): the intact molecule in
  the initial orientation of the most stable intact configuration, lowered
  over `scan_heights` (clearance of its lowest atom above the surface).
- Figures in `figures/`: `eads_ranking_<term>.png`, `eads_vs_tilt_<term>.png`,
  `eads_site_anchor_heatmap_<term>.png`, `summary_<term>_<config>.png`
  (top 3) and `termination_comparison.png`.

The `report`, `vasp` and `dft-collect` subcommands are not implemented yet.

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
    energy=...,
)
```

`energy` is the potential energy (eV) of the last trajectory frame, read from
the trajectory without an extra calculation. `optimize_geometry_atoms()` also
accepts these keyword arguments:

- `calculator=None`: attach this calculator instance instead of building one
  from `calculator_config`
- `cell_filter=False`: also relax the cell of a periodic system
- `logfile="-"`: optimizer log destination; `None` disables the log
- `write_outputs=True`: when `False`, only the trajectory is written and
  `optimized_xyz` / `final_xyz` are `None` (for large screening runs)

Calculator helpers:

- `get_calculator(config, cache=True)`: like `build_calculator()`, but ML
  backends (`macemp`, `maceomol`, `uma_s`, `uma_m`, `eSEN`, `aimnet2`) are
  cached per `CalculatorConfig`, so the model is loaded only once per process.
  `build_calculator()` still creates a new instance on every call.
  `clear_calculator_cache()` drops cached models.
- `calculator_capabilities(config)` returns `CalculatorCapabilities`:

```python
CalculatorCapabilities(
    periodic=...,           # can describe bulk solids / surfaces
    uses_charge_spin=...,   # uses charge / multiplicity
    elements=...,           # frozenset of supported elements, or None
    level_of_theory=...,    # e.g. "PBE (MPtrj/MPA)", "RPBE (OC20)"
)
```

`CalculatorConfig` fields added for periodic systems, all with defaults that
keep the existing backends unchanged: `device="auto"`, `dtype="float64"`
(MACE `default_dtype`), `dispersion=False` (D3(BJ) for `macemp`),
`mace_mp_model="medium-mpa-0"`, `uma_task="omol"`.

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

## Running Tests

```bash
pip install -e ".[test]"
python -m pytest
```

`tests/test_core_regression.py` re-runs the minimization, TS, IRC, frequency
and CLI workflows (EMT, plus xTB when installed) and compares file names,
step counts, energies and coordinates with the reference recorded on the
original code in `tests/data/regression_baseline.json`. Regenerate the
reference with `python tests/record_baseline.py` only after an intentional,
reviewed behavior change.
