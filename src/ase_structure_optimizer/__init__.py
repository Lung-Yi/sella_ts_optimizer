"""Structure optimization utilities with ASE-compatible calculators."""

from .calculators import (
    CalculatorCapabilities,
    CalculatorConfig,
    available_calculators,
    build_calculator,
    calculator_capabilities,
    clear_calculator_cache,
    get_calculator,
)
from .frequencies import (
    FrequencyResult,
    analyze_frequencies_atoms,
    run_frequency_analysis,
)
from .graphs import (
    Fragment,
    build_bond_graph,
    graph_hash,
    same_connectivity,
    split_fragments,
)
from .irc import IRCResult, optimize_irc_atoms, run_irc
from .runner import (
    OptimizationResult,
    available_minimizers,
    optimize_geometry_atoms,
    optimize_ts_atoms,
    run_geometry_optimization,
    run_ts_optimization,
)
from .structures import is_slab, read_structure, write_trajectory_pair

__all__ = [
    "CalculatorCapabilities",
    "CalculatorConfig",
    "Fragment",
    "FrequencyResult",
    "IRCResult",
    "OptimizationResult",
    "analyze_frequencies_atoms",
    "available_calculators",
    "available_minimizers",
    "build_bond_graph",
    "build_calculator",
    "calculator_capabilities",
    "clear_calculator_cache",
    "get_calculator",
    "graph_hash",
    "is_slab",
    "optimize_geometry_atoms",
    "optimize_irc_atoms",
    "optimize_ts_atoms",
    "read_structure",
    "run_geometry_optimization",
    "run_frequency_analysis",
    "run_irc",
    "run_ts_optimization",
    "same_connectivity",
    "split_fragments",
    "write_trajectory_pair",
]
