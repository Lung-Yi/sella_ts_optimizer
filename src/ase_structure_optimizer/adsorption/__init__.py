"""Automated molecule-on-surface adsorption workflow.

Optional dependencies (``pip install -e ".[adsorption]"``) are imported only
when needed; this subpackage is never imported by ``ase_structure_optimizer``
itself.
"""

from .analysis import boltzmann_weights, classify, deduplicate, intact_check
from .config import (
    AdsorptionConfig,
    ConfigError,
    check_calculator_support,
    config_from_dict,
    config_to_dict,
    default_config_yaml,
    dump_config,
    load_config,
)
from .report import generate_report
from .molecule import Anchor, MoleculeAnalysis, ReferenceAxis, analyze_molecule, generate_conformers
from .sites import AdsorptionSite, find_sites
from .workflow import (
    AdsorptionResult,
    SurfaceModel,
    TerminationResult,
    prepare_surfaces,
    resume_adsorption_workflow,
    run_adsorption_workflow,
)

__all__ = [
    "AdsorptionConfig",
    "AdsorptionResult",
    "AdsorptionSite",
    "Anchor",
    "ConfigError",
    "MoleculeAnalysis",
    "ReferenceAxis",
    "SurfaceModel",
    "TerminationResult",
    "analyze_molecule",
    "boltzmann_weights",
    "classify",
    "deduplicate",
    "intact_check",
    "check_calculator_support",
    "config_from_dict",
    "config_to_dict",
    "default_config_yaml",
    "dump_config",
    "find_sites",
    "generate_conformers",
    "generate_report",
    "load_config",
    "prepare_surfaces",
    "resume_adsorption_workflow",
    "run_adsorption_workflow",
]
