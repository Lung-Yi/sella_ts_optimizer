"""The package and the existing CLI must work with only the core dependencies."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Optional packages (the [mace] / [vibrations] / [adsorption] extras and
# calculator backends) that are hidden from the subprocess. matplotlib, pillow
# and scipy are not listed: ASE and Sella themselves require them.
BLOCKED = [
    "networkx",
    "yaml",
    "pymatgen",
    "pandas",
    "torch",
    "torch_dftd",
    "mace",
    "fairchem",
    "xtb",
    "aimnet2calc",
    "geometric",
]


def _run_blocked(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    prelude = textwrap.dedent(
        f"""
        import importlib.abc, sys
        BLOCKED = {BLOCKED!r}

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ModuleNotFoundError(f"No module named {{name!r}} (blocked by test)")
                return None

        sys.meta_path.insert(0, _Blocker())
        sys.path.insert(0, {str(ROOT / "src")!r})
        """
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_import_and_cli_without_optional_packages(tmp_path: Path):
    completed = _run_blocked(
        """
        import ase_structure_optimizer as package
        from ase_structure_optimizer.cli import main
        from ase import Atoms
        from ase.io import write

        assert not any(name.startswith("ase_structure_optimizer.adsorption") for name in sys.modules)
        for name in BLOCKED:
            assert name not in sys.modules, name

        write("h2o.xyz", Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.48], [0.05, -0.8, -0.45]]))
        assert main(["h2o.xyz", "--calculator", "emt"]) == 0

        try:
            package.build_bond_graph(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]))
        except RuntimeError as exc:
            assert "[adsorption]" in str(exc)
        else:
            raise AssertionError("expected RuntimeError without networkx")

        # The adsorption subpackage imports without its optional packages and
        # explains what to install when they are needed.
        from ase_structure_optimizer.adsorption import config_from_dict, load_config
        config_from_dict({"molecule": "m.xyz", "solid": "s.cif"})
        try:
            load_config("missing.yaml")
        except RuntimeError as exc:
            assert "[adsorption]" in str(exc)
        else:
            raise AssertionError("expected RuntimeError without PyYAML")
        print("OK")
        """,
        tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("OK")
    assert (tmp_path / "h2o_min_emt" / "geometry_min_optimized.xyz").is_file()
