"""Automated molecule-on-surface adsorption workflow.

Optional dependencies (``pip install -e ".[adsorption]"``) are imported only
when needed; this subpackage is never imported by ``ase_structure_optimizer``
itself.
"""

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
from .molecule import Anchor, MoleculeAnalysis, ReferenceAxis, analyze_molecule, generate_conformers

__all__ = [
    "AdsorptionConfig",
    "Anchor",
    "ConfigError",
    "MoleculeAnalysis",
    "ReferenceAxis",
    "analyze_molecule",
    "check_calculator_support",
    "config_from_dict",
    "config_to_dict",
    "default_config_yaml",
    "dump_config",
    "generate_conformers",
    "load_config",
]
