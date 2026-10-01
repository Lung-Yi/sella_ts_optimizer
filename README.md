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

For the adsorption workflow (`ase-adsorb`, see
[Adsorption Workflow](#adsorption-workflow)): pymatgen, bond graphs (`networkx`),
PyYAML, plotting, and MACE-MP with D3 dispersion (`torch-dftd`):

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
`CalculatorConfig(uma_task=...)` to a periodic task: `omat` (bulk
materials), `oc20` (adsorbate + surface), `oc22` (oxide surfaces), `oc25`
(solid-liquid interfaces), `odac` (MOFs) or `omc` (molecular crystals); the
tasks available depend on the checkpoint. `eSEN` only supports `omol`.
Energies from different tasks or models must never be mixed.

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
    dispersion_xc="pbe", # D3(BJ) damping parameters; "r2scan" for r2SCAN-trained models
    dtype="float64",     # MACE default_dtype; float32 is faster for screening
)
```

FAIRChem UMA models are downloaded from Hugging Face on first use and need an
account with access to the UMA model repository (`huggingface-cli login`).
`CalculatorConfig.uma_model` selects another registered model (e.g.
`"uma-s-1p2"`) or a local checkpoint file, which needs no login; it is used
by `uma_s`, `uma_m` and `eSEN` (empty = the default `uma-s-1p1`,
`uma-m-1p1`, `esen-sm-conserving-all-omol`). `dispersion=True` adds D3(BJ)
to UMA as well, except for the tasks whose training data already include D3
(`oc25`, `odac`, `omc`):

```python
config = CalculatorConfig(
    name="uma_s",
    uma_model="~/.cache/uma/uma-s-1p2p1.pt",  # local checkpoint
    uma_task="oc20",
    dispersion=True,
    dispersion_xc="rpbe",                      # OC20 is RPBE
)
```

For every task except `omol`, the UMA calculator passes charge 0 and spin 0
to the model (the training convention of the periodic datasets): FAIRChem
otherwise reads `atoms.info["charge"]` / `["spin"]`, which shift periodic
energies by eV. `uma_merge_mole=True` merges UMA's mixture of experts once
per reduced composition. This gives the same energies (within 1e-5 eV) with
much less memory and compute, and is what makes UMA-M usable on an 8 GB
GPU. Local checkpoints are memory-mapped while loading, so the 11 GB UMA-M
file loads on a 16 GB machine.

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

## Adsorption Workflow

`ase-adsorb` (Python package `ase_structure_optimizer.adsorption`) finds how
a molecule adsorbs on a solid surface. You give it a gas-phase molecule
(`.xyz`) and a crystal or pre-cut slab (`.cif`); it builds the surfaces,
generates hundreds of initial adsorption configurations, screens and relaxes
them with a periodic machine-learned interatomic potential (MLIP) within a
time budget, classifies the results, and writes CSV tables, figures and a
self-contained `report.html`.

```text
molecule.xyz + solid.cif
-> molecule analysis: bonds, ligand fragments, anchor points, reference axis u
-> gas-phase reference: molecule relaxed in a periodic box -> E_mol
-> bulk relaxation (cell + atoms) -> slab terminations (pymatgen)
-> slab supercells, bottom layers fixed, relaxed -> E_slab, adsorption sites
-> initial configurations: site x anchor x rotation, plus random orientations
-> prescreening (short relaxations) within the time budget
-> full relaxations of the most promising, diverse configurations
-> final single points: E_ads = E(slab+mol) - E(slab) - E(mol)
-> classification, angles, de-duplication, Boltzmann weights, approach scan
-> results.csv / unique.csv / summary.csv, figures/, report.html
```

The energies come from an MLIP. They are a screening tool: validate the
configurations you care about with DFT (VASP input generation is the next
development milestone).

### Step 0: environment and model files

Install the package with the adsorption extra (once, from the repository):

```bash
pip install -e ".[adsorption]"
```

Every new terminal needs the Python environment in which the package is
installed. With conda:

```bash
conda activate transformervae
```

If this prints `CondaError: Run 'conda init' before 'conda activate'`, the
shell has not been set up for conda. Either run `conda init bash` once and
open a new terminal, or load conda into the current shell:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformervae
```

Check that the command is found:

```bash
ase-adsorb --help
```

Without installing, `python run_ase_adsorption.py ...` from the repository
root does the same as `ase-adsorb ...`.

**MLIP model files.** The default calculator `macemp` (MACE-MP family,
periodic materials models) downloads `medium-mpa-0` on first use, which can
be very slow. Download model files once and point the configuration at the
local file instead (`calculator.mace_mp_model: /path/to/file.model`). Models
tested with this workflow:

| model file | `dispersion_xc` | notes |
|---|---|---|
| `~/.cache/mace/MACE-matpes-r2scan-omat-ft.model` | `r2scan` | r2SCAN-level (MatPES); keeps eta5-Cp intact; recommended for organometallic adsorbates |
| `~/.cache/mace/mace-mp-0b3-medium.model` | `pbe` | PBE-level (MPtrj); slips eta5-Cp to eta2 in the gas phase and decomposes CpMo(CO)3H on TiSi |

`dispersion_xc` must match the functional the model was trained on, because
it selects the D3(BJ) damping parameters.

UMA models also work: set `calculator.name: uma_s` (or `uma_m`),
`calculator.uma_model` to a local checkpoint (or a registered name such as
`uma-s-1p2`; `null` downloads `uma-s-1p1`, which needs a Hugging Face login),
and a periodic `uma_task`:

```yaml
calculator:
  name: uma_s
  uma_model: /home/lungyi/.cache/uma/uma-s-1p2p1.pt   # or uma-m-1p1.pt
  uma_merge_mole: false   # true for uma-m on small GPUs (one merged model per composition)
  uma_task: oc20          # adsorbate + surface (RPBE)
  dispersion: true        # D3(BJ) added on top; skipped for oc25 / odac / omc
  dispersion_xc: rpbe
```

Only the `omat` and `omc` tasks have a trained stress head. With any other
task (e.g. `oc20`) the bulk keeps the lattice of the input file and only the
atomic positions are relaxed; a warning says so in the log. UMA ignores
`dtype_screen` / `dtype_final`, so one model instance serves both stages.

UMA-M (`uma-m-1p1.pt`, 11 GB) needs `uma_merge_mole: true` on an 8 GB GPU:
unmerged it fills the GPU and takes about 5.9 s per force call for the
258-atom TiSi system; merged it takes about 3.0 s and 2.6 GB. UMA-S 1.2 takes
about 0.25 s. Raise `budget.wall_time_per_termination` accordingly.

A CUDA GPU is used automatically when available (`calculator.device: auto`).
For the TiSi example (240-atom slab, 18-atom molecule) one force call takes
about 0.13 s on a laptop RTX 3070 Ti.

### Step 1: check the molecule

The workflow decides "which parts of the molecule can point at the surface"
and "is the molecule still intact" from its bond graph. Check the graph
before running anything, especially for metal complexes:

```bash
ase-adsorb check-molecule CpMo_CO3H.xyz
```

It prints the atoms, all bonds with lengths, the ligand fragments (for a
metal complex the connected pieces after removing the metal, e.g. `C5H5`,
`CO_1`, `CO_2`, `CO_3`, `H`), the anchor points, and the reference axis u.
Check that:

- every chemical bond is listed, and no bond is listed that should not be
  (e.g. all five Mo-C bonds of an eta5-Cp ring must appear);
- the fragments are the ligands you expect;
- no warning such as "the bond graph has N disconnected parts" appears.

Atoms are bonded when `distance < bond_scale * (r_cov,i + r_cov,j)` (covalent
radii; `bond_scale` defaults to 1.2). Fix wrong bonds in the configuration
file and check again:

```yaml
molecule_props:
  bond_scale: 1.2
  bond_overrides: [[0, 8, true], [2, 7, false]]   # force atoms 0-8 bonded, 2-7 not bonded
  reference_axis: [[0], [8, 9, 10, 11, 12]]       # optional: u = centroid(atoms A) -> centroid(atoms B)
```

```bash
ase-adsorb check-molecule CpMo_CO3H.xyz --config config.yaml
ase-adsorb check-molecule CpMo_CO3H.xyz --bond-scale 1.3 --json analysis.json
```

Atom indices start at 0, in the order of the `.xyz` file.

### Step 2: write the configuration file

```bash
ase-adsorb init-config -o config.yaml
```

writes a commented template containing every setting with its default. Only
`molecule` and `solid` are required; delete everything you do not want to
change. A typical file:

```yaml
molecule: CpMo_CO3H.xyz        # relative paths are relative to this file
solid: TiSi.cif
calculator:
  name: macemp
  mace_mp_model: /home/lungyi/.cache/mace/MACE-matpes-r2scan-omat-ft.model
  fallback_models: []          # do not fall back to (downloaded) default models
  dispersion: true
  dispersion_xc: r2scan
surface:
  miller_indices: [[0, 0, 1]]  # several: [[0, 0, 1], [1, 0, 0]]
budget:
  wall_time_per_termination: 600   # seconds of sampling per termination
```

Miller indices refer to the cell axes of the input CIF when it is a
conventional cell (e.g. the TiSi CIF with a = 6.54, b = 3.64, c = 5.00 Å);
a primitive cell such as `ase.build.bulk("Cu")` is first converted to the
conventional cell. The log says which convention was used.

Molecular calculators (`maceomol`, `aimnet2`, `eSEN`, UMA with `omol`, `xtb`,
Q-Chem) cannot describe surfaces and are rejected when the file is loaded.
All settings are listed in [Configuration reference](#configuration-reference).

### Step 3: run

```bash
ase-adsorb run config.yaml 2>&1 | tee run.log
```

Command-line options override the file, or replace it for quick runs:

```bash
ase-adsorb run config.yaml --budget 1200 --miller 1 0 0
ase-adsorb run --molecule CpMo_CO3H.xyz --solid TiSi.cif \
  --calculator macemp --mace-mp-model ~/.cache/mace/MACE-matpes-r2scan-omat-ft.model \
  --miller 0 0 1 --budget 600 --run-dir runs/cpmo_tisi
```

Options: `--molecule`, `--solid`, `--calculator`, `--mace-mp-model`,
`--uma-model`, `--uma-task`, `--device`, `--miller H K L` (repeatable), `--budget`
(seconds per termination, 0 = unlimited), `--run-dir`. `dispersion_xc` and
all other settings are set in the configuration file.

The results go to `<molecule>_on_<solid>_<calculator>/` next to the molecule
file (for example `CpMo_CO3H_on_TiSi_macemp/`), or to `run_dir`. A run
directory is never reused: to repeat a calculation, rename the old directory
or give another `--run-dir`.

How long it takes: surface preparation takes about a minute; then each
termination takes about `wall_time_per_termination` (default 600 s) plus the
analysis, figures and report (a few minutes). The TiSi(001) example with two
terminations finishes in about 25 minutes.

Follow the progress in another terminal:

```bash
tail -f CpMo_CO3H_on_TiSi_macemp/adsorption.log
```

### Step 4: if the run was interrupted

`Ctrl+C`, a closed terminal or a crash leave a consistent `state.json`.
Continue with:

```bash
ase-adsorb resume CpMo_CO3H_on_TiSi_macemp
```

Finished structures are reused, the configuration that was running is
restarted, and the time budget of a termination counts the time already
spent. `resume` reads the run's own `config.resolved.yaml` and the copies of
the input files in `inputs/`, so the run directory can be moved or copied
before resuming. `resume` refuses to continue if the calculator, surface,
sampling or molecule settings in `config.resolved.yaml` were edited (the
results would be mixed); start a new run instead.

### Step 5: read the results

Open `report.html` in a web browser. It is a single file (all images
embedded) with:

- the most stable configuration of each termination, and the most stable
  intact one when the overall best is a dissociated structure;
- the settings, time spent per stage and budget use;
- the molecule: fragments, bonds, anchors, and a sketch of the reference axis;
- the definitions of the angles;
- per termination: slab details, the table of unique configurations, the
  figures, and snapshots of the relaxation and approach animations;
- all warnings of the run, and notes on MLIP limitations;
- a player that animates the relaxation of the most stable configuration.

The tables are also CSV files for your own analysis:

| file | content |
|---|---|
| `summary.csv` | unique configurations of all terminations, sorted by `eads` |
| `terminations/<term>/unique.csv` | unique configurations of one termination, with Boltzmann weights |
| `terminations/<term>/results.csv` | every fully relaxed configuration (duplicates included) |
| `terminations/<term>/prescreen.csv` | the prescreened configurations |

Important columns:

- `eads`: adsorption energy in eV (final dtype); negative = exothermic.
  `eads_screen` is the same with the faster screening dtype.
- `class`: `chemisorbed` (intact, touching the surface), `physisorbed`
  (intact, not touching), `dissociated` (bonds of the molecule changed), or
  `desorbed` (left the surface). `class_reason` explains a dissociation,
  e.g. `H detached from Mo0` or `broken C1-O2`.
- `contact`: fragments touching the surface, e.g. `CO_1+CO_2`, `C5H5`, `Mo`
  (the metal atom itself), `none`.
- `tilt` (θ, degree): angle between the reference axis u and the surface
  normal. 0° = u points away from the surface, 180° = toward it, 90° =
  parallel. For CpMo(CO)3H u points from Mo to the Cp ring, so θ ≈ 0° means
  "Cp up, CO legs down" and θ ≈ 180° means "Cp down".
- `azimuth` (φ, degree): in-plane orientation, measured from cell vector a.
- `height` (Å): molecule center (metal atom, else center of mass) above the
  topmost slab atom.
- `boltzmann_weight`: relative population at `analysis.temperature`.
- `n_duplicates` / `duplicates`: equivalent configurations that were merged.
- `converged`, `steps`: whether the relaxation reached `fmax` within `max_steps`.
- `surface_distorted`: a free slab atom moved by more than 1 Å.
- `site_id`, `anchor`, `initial_tilt`: how the configuration started.

Structures:

| path | content |
|---|---|
| `terminations/<term>/relax/<config>.extxyz` | relaxation trajectory (text; `.traj` is the binary ASE version) |
| `terminations/<term>/relax/<config>_final.extxyz` | relaxed structure with its final-dtype energy |
| `terminations/<term>/candidates/<config>.extxyz` | initial configuration |
| `terminations/<term>/slab_opt_final.extxyz` | relaxed clean slab |
| `molecule/gas_opt_final.extxyz` | relaxed gas-phase molecule |

Open `.extxyz` files with ASE (`ase gui file.extxyz`), OVITO or VESTA. All
structures and energies are also in the ASE database `results.db`:

```bash
ase db results.db kind=adsorbate adsorption_class=chemisorbed -c name,eads,contact,tilt -s eads
```

```python
from ase.db import connect

with connect("results.db") as db:
    for row in db.select(kind="adsorbate", sort="eads"):
        atoms = row.toatoms()
        print(row.name, row.eads, row.adsorption_class, row.contact)
```

Figures in `figures/`: `eads_ranking_<term>.png` (E_ads of the unique
configurations colored by class), `eads_vs_tilt_<term>.png`,
`eads_site_anchor_heatmap_<term>.png` (lowest E_ads per initial site and
anchor), `summary_<term>_<config>.png` (side and top view, relaxation
curve, approach scan), `termination_comparison.png`, the molecule and angle
sketches, and `approach_<term>.gif` / `relaxation_<term>_<config>.gif` with a
static `.png` version each.

Redraw all figures and the report from the files of a run, without any
calculation (e.g. after updating the package):

```bash
ase-adsorb report CpMo_CO3H_on_TiSi_macemp
ase-adsorb report CpMo_CO3H_on_TiSi_macemp --no-animations   # faster
```

### How the workflow works

- **Molecule.** Bonds from covalent radii; ligand fragments are the pieces
  left after removing transition-metal atoms. Anchors (the points that can
  be turned toward the surface) are fragment and ring centroids, the metal
  atom (turned with its least-coordinated side down), hydrides, heteroatoms
  and terminal atoms. The reference axis u is user-defined, or metal ->
  largest ring bonded to it, or metal -> heaviest ligand, or the long axis
  of a metal-free molecule (the plane normal for planar symmetric molecules).
- **Gas-phase reference.** The molecule is relaxed in a periodic box (edges =
  size + `vasp.molecule_box_padding`) with the same calculator; its final
  single-point energy is E_mol. If the MLIP changes the molecule's bonds
  there (e.g. MACE-MP-0b3 slipping an eta5-Cp ring), E_mol is still the
  MLIP minimum, but the input geometry is used for sampling and as the
  intact reference, and the report shows a warning.
- **Surfaces.** A bulk input is relaxed together with its cell, cut into all
  terminations with pymatgen, and the terminations are ranked by MLIP surface
  energy (at most `max_terminations` kept, named `<miller>_t<i>` with `t0` the
  most stable). Each slab is at least `min_slab_thickness` thick, repeated
  laterally until both in-plane widths reach `min_lateral` (auto: molecule
  size + `lateral_buffer`), given `vacuum_above` of vacuum, relaxed with
  `fix_fraction` of its atoms (whole bottom layers) fixed. Ontop, bridge and
  hollow sites are found per termination, symmetry-inequivalent, near the
  cell center.
- **Initial configurations.** For every site x anchor x
  `spins_per_anchor` rotations the anchor is turned toward the surface and
  placed above the site; `n_random` random orientations are added. The
  molecule is lowered until its closest atom is at `contact_gap`
  (auto: 0.9 x the sum of van der Waals radii). Clashes, contact with the
  molecule's periodic images and duplicates are removed.
- **Time budget.** The step time is measured; prescreening (short
  `prescreen_steps` relaxations) uses `prescreen_fraction` of
  `wall_time_per_termination`, choosing at least one configuration per site
  type x anchor. The best of each contact type is then fully relaxed first,
  then the rest by energy, until `prescreen_fraction + relax_fraction` of the
  budget is used. Configurations that crash, give NaN, collapse (atoms
  < 0.5 Å apart) or fly away (> 10 Å) are marked failed with a reason.
- **Analysis.** Final-dtype single points, classification, angles, merging of
  equivalent configurations (same class and contact, |ΔE_ads| <
  `dedup_energy_tol`, RMSD up to lattice translations < `dedup_rmsd_tol`),
  Boltzmann weights, and a rigid approach scan: the intact molecule in the
  initial orientation of the most stable intact configuration, lowered from
  7 Å to 1.5 Å above the surface. The scan should level off near 0 eV far
  from the surface; a different plateau means E_mol belongs to a different
  molecular geometry (see the gas-phase warning).

A metal complex counts as intact as long as the bonds inside every ligand
are unchanged and every ligand is still bonded to its metal atom, so a
change of hapticity (eta5 -> eta3 Cp) is not a dissociation, but a hydride
or CO leaving the metal is.

### Choosing and comparing calculators

- Use a periodic model trained at a level of theory you trust for both the
  surface and the molecule. Compare models on the same system: run once per
  model (each run gets its own directory) and compare `summary.csv`, the
  fraction of intact configurations, the adsorption heights and whether the
  approach scan levels off at 0 eV.
- Energies from different models, tasks (UMA `oc20` vs `omat`) or D3
  settings must never be combined into one E_ads. Absolute E_ads from
  different models are not directly comparable; compare the ranking of
  binding modes and their geometry.
- Periodic MLIPs see almost no isolated molecules in training, so E_mol is an
  extrapolation and shifts all E_ads of a run by the same amount.
- Example (CpMo(CO)3H on TiSi(001), 600 s per termination, D3 on): with
  MACE-MP-0b3 the Cp ring slips in the gas phase and 8 of 13 relaxed
  structures decompose with E_ads down to -9.5 eV; with the r2SCAN MatPES
  model the gas-phase molecule stays eta5 and 24 of 25 relaxed structures
  stay intact, binding through CO legs or the Cp ring with E_ads of -0.4 to
  -3.0 eV. With UMA-S 1.2 (`oc20` + D3(BJ, rpbe), experimental lattice) the
  gas-phase molecule stays eta5, but each step is about 4x slower, so only 7
  structures were relaxed; 2 of them decompose (one embedded in the surface
  at -11 eV).
- Check the approach scan: at the largest clearance E_ads should be close to
  0 eV. UMA energies are not additive between systems of different
  composition (most likely because UMA mixes its experts according to the
  composition of the whole system): on TiSi the scan levels off at about
  -0.6 eV even beyond the 6 Å model cutoff, so every E_ads of that run is
  shifted by roughly this amount (rankings within the run are unaffected).

### Configuration reference

All settings with their defaults (`ase-adsorb init-config` prints the same
with comments). Lengths in Å, energies in eV, angles in degrees, times in s.

| setting | default | meaning |
|---|---|---|
| `molecule` | required | gas-phase molecule (`.xyz`) |
| `solid` | required | bulk crystal or pre-cut slab (`.cif`) |
| `run_dir` | `null` | output directory; `null` = `<molecule>_on_<solid>_<calculator>` next to the molecule |
| **molecule_props** | | |
| `charge`, `multiplicity` | `0`, `1` | total charge and spin multiplicity of the molecule |
| `bond_scale` | `1.2` | bonded if `d < bond_scale * (r_cov,i + r_cov,j)` |
| `bond_overrides` | `[]` | `[[i, j, true/false], ...]` force bonds on/off |
| `reference_axis` | `auto` | or `[[atoms A], [atoms B]]`: u = centroid(A) -> centroid(B) |
| **calculator** | | must be periodic |
| `name` | `macemp` | `macemp`, `uma_s`, `uma_m`, `emt` (tests only) |
| `mace_mp_model` | `medium-mpa-0` | macemp: model name or local model file |
| `fallback_models` | `[medium, small]` | macemp: tried if the model cannot be loaded; `[]` to disable |
| `uma_model` | `null` | UMA: `null` (= `uma-s-1p1` / `uma-m-1p1`), a registered name, or a local `.pt` checkpoint |
| `uma_merge_mole` | `false` | UMA: merge the experts once per composition (same energies, much less memory/compute) |
| `uma_task` | `oc20` | UMA: `oc20` (adsorbate + surface), `omat` (bulk), `oc22`, `oc25`, `odac`, `omc` |
| `dispersion` | `true` | add D3(BJ); not added for UMA `oc25` / `odac` / `omc` (trained with D3) |
| `dispersion_xc` | `pbe` | D3(BJ) parameters: `pbe`; `r2scan` for r2SCAN models; `rpbe` for UMA `oc20` |
| `device` | `auto` | `auto`, `cuda`, `cpu` |
| `dtype_screen`, `dtype_final` | `float32`, `float64` | MACE precision for relaxations / final energies |
| **surface** | | |
| `input_type` | `auto` | `auto`, `bulk` or `slab` |
| `relax_bulk` | `true` | relax the bulk cell before cutting (positions only for UMA tasks without stress) |
| `miller_indices` | `[[0, 0, 1]]` | surfaces to build |
| `min_slab_thickness` | `8.0` | atom-to-atom slab thickness |
| `max_terminations` | `3` | terminations kept per Miller index |
| `min_lateral` | `auto` | minimum in-plane width; auto = molecule size + `lateral_buffer` |
| `lateral_buffer` | `10.0` | see `min_lateral` |
| `vacuum_above` | `auto` | vacuum above the slab; auto = molecule size + 15 |
| `fix_fraction` | `0.5` | fraction of slab atoms fixed (whole bottom layers) |
| **sampling** | | |
| `site_types` | `[ontop, bridge, hollow]` | site types to use |
| `spins_per_anchor` | `3` | rotations about the surface normal per anchor |
| `n_random` | `60` | extra random orientations |
| `contact_gap` | `auto` | initial distance to the surface; auto = 0.9 x vdW radii sum |
| `clash_scale` | `0.7` | drop if any distance < `clash_scale` x vdW radii sum |
| `seed` | `42` | random seed |
| `surface_depth` | `0.9` | depth below the top atom counted as surface for sites |
| **budget** | | per termination |
| `wall_time_per_termination` | `600` | seconds; `0` = unlimited (all prescreened, top 30 % relaxed) |
| `prescreen_steps`, `prescreen_optimizer`, `fmax_prescreen` | `20`, `fire`, `0.15` | prescreening |
| `relax_optimizer`, `fmax`, `max_steps` | `lbfgs`, `0.05`, `400` | full relaxations (also bulk, slab, molecule) |
| `prescreen_fraction`, `relax_fraction` | `0.40`, `0.55` | budget shares; the rest is reserve |
| `min_full_relax` | `5` | warn if fewer full relaxations fit in the budget |
| **analysis** | | |
| `contact_scale` | `1.25` | contact if `d < contact_scale * (r_cov,i + r_cov,j)` |
| `desorbed_distance` | `4.5` | desorbed if farther from the slab |
| `dedup_energy_tol`, `dedup_rmsd_tol` | `0.02`, `0.30` | merging of equivalent configurations |
| `temperature` | `298.15` | K, for Boltzmann weights |
| `approach_scan`, `scan_heights` | `true`, `[1.5, 7.0, 0.25]` | rigid scan: start, stop, step |
| `make_gif` | `true` | write GIF animations |
| **uncertainty**, **dft_selection**, **vasp** | | used by the VASP and uncertainty steps (in development) |

### Output directory

```text
CpMo_CO3H_on_TiSi_macemp/
├── report.html              # open in a browser
├── summary.csv              # all terminations, unique configurations
├── adsorption.log           # progress, warnings
├── config.resolved.yaml     # all settings actually used
├── state.json               # checkpoint for resume
├── results.db               # ASE database: all structures and energies
├── inputs/                  # copies of the input files
├── molecule/                # gas-phase optimization, molecule_analysis.json, gas_reference.json
├── bulk/                    # bulk relaxation (bulk input only)
├── figures/                 # PNG figures, GIF animations
└── terminations/
    ├── terminations.json    # all terminations and why they were kept
    └── 001_t0/
        ├── unit_slab.extxyz, slab_opt.*, slab_opt_final.extxyz, slab.json, sites.json
        ├── candidates/      # initial configurations
        ├── prescreen/       # prescreening trajectories
        ├── relax/           # full relaxations and final structures
        ├── prescreen.csv, results.csv, unique.csv
        └── approach_scan.csv, approach_scan.extxyz
```

### Python API

```python
from ase_structure_optimizer.adsorption import (
    analyze_molecule,
    generate_report,
    load_config,
    resume_adsorption_workflow,
    run_adsorption_workflow,
)

config = load_config("config.yaml")
result = run_adsorption_workflow(config)          # or resume_adsorption_workflow("run_dir")

print(result.run_dir, result.summary_csv, result.report)
for termination in result.terminations:
    print(
        termination.term_id,
        termination.n_relaxed,
        termination.best_unique_id,
        termination.best_eads,
        termination.best_class,
    )
```

`run_adsorption_workflow()` returns `AdsorptionResult` (`run_dir`,
`config_file`, `log_file`, `database`, `molecule_energy`, `terminations`,
`summary_csv`, `figures_dir`, `report`, and the property `best`); each
`TerminationResult` holds the counts, CSV paths, warnings and the most stable
configuration. `prepare_surfaces(config)` runs only the setup and surface
stages; `analyze_molecule(atoms, config.molecule_props)` is what
`check-molecule` prints.

### Troubleshooting

| message or symptom | what to do |
|---|---|
| `CondaError: Run 'conda init' before 'conda activate'` | `source ~/miniconda3/etc/profile.d/conda.sh`, then `conda activate <env>`; or `conda init bash` once |
| `ase-adsorb: command not found` | activate the environment, or `pip install -e ".[adsorption]"`, or use `python run_ase_adsorption.py` |
| model download hangs or fails | download the model file once and set `calculator.mace_mp_model` to its path, `fallback_models: []` |
| `... is a molecular model ... cannot describe surfaces` | use `macemp`, or `uma_s`/`uma_m` with a periodic `uma_task` (e.g. `oc20`, `omat`) |
| UMA asks for a Hugging Face login | set `calculator.uma_model` to a local `.pt` checkpoint |
| UMA-M killed while loading / CUDA out of memory | set `calculator.uma_merge_mole: true` |
| `uma_task ... has no trained stress head` warning | expected for `oc20` etc.; the input lattice is kept. Use experimental lattice constants in the CIF |
| `... does not support element(s)` | the calculator (e.g. `emt`) lacks an element; use `macemp` or UMA |
| `already contains a run` | rename the old run directory, set `run_dir`, or `ase-adsorb resume` it |
| `resume` refuses: settings differ | the run's settings were edited; start a new run |
| warning `gas-phase optimization changed the bond graph` | the MLIP breaks the molecule in the gas phase; try another model, check E_ads with DFT |
| warning `prescreening budget used up` / `relaxation budget used up` / `only N full relaxation(s)` | increase `budget.wall_time_per_termination` |
| many `dissociated` results with very negative E_ads | check the structures; this is often an MLIP artifact on reactive surfaces (compare models, validate with DFT) |
| CUDA out of memory | `calculator.device: cpu`, or a smaller `min_lateral` / `lateral_buffer` |
| `.traj` files look broken in an editor | they are binary; open the `.extxyz` copies or use `ase gui` |

The `vasp` and `dft-collect` subcommands (DFT input generation and comparison)
and the multi-model uncertainty estimate are not implemented yet.

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
`mace_mp_model="medium-mpa-0"`, `uma_task="omol"`, `dispersion_xc="pbe"` (the
functional whose D3(BJ) parameters are used; use `"r2scan"` with r2SCAN-trained
models such as `MACE-matpes-r2scan-omat-ft.model`), `uma_model=""` (registered
FAIRChem model name or checkpoint path; empty keeps the previous defaults).

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
