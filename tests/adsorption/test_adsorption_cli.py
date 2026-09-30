"""Tests for the ase-adsorb command line interface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("networkx")
yaml = pytest.importorskip("yaml")

from ase_structure_optimizer.adsorption.cli import main  # noqa: E402
from ase_structure_optimizer.adsorption.config import default_config_yaml, load_config  # noqa: E402

CPMO_XYZ = Path(__file__).resolve().parents[1] / "data" / "CpMo_CO3H.xyz"


def test_check_molecule_prints_analysis(tmp_path: Path, capsys):
    out_json = tmp_path / "analysis.json"
    assert main(["check-molecule", str(CPMO_XYZ), "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    for expected in ("Bonds (22)", "C5H5", "CO_3", "Anchors", "metal_ring", "Mo0   - C12"):
        assert expected in text
    assert "Warnings" not in text
    assert json.loads(out_json.read_text())["fragments"][4]["name"] == "C5H5"


def test_check_molecule_uses_config_and_bond_scale(tmp_path: Path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "molecule: m.xyz\nsolid: s.cif\nmolecule_props:\n  bond_overrides: [[0, 7, false]]\n",
        encoding="utf-8",
    )
    assert main(["check-molecule", str(CPMO_XYZ), "--config", str(config)]) == 0
    assert "Mo0   - H7" not in capsys.readouterr().out

    assert main(["check-molecule", str(CPMO_XYZ), "--bond-scale", "1.0"]) == 0
    assert "disconnected" in capsys.readouterr().out


def test_check_molecule_errors(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["check-molecule", str(tmp_path / "missing.xyz")])
    xyz = tmp_path / "m.xyz"
    shutil.copy(CPMO_XYZ, xyz)
    with pytest.raises(SystemExit):
        main(["check-molecule", str(xyz), "--bond-scale", "-1"])


def test_init_config(tmp_path: Path, capsys):
    assert main(["init-config"]) == 0
    assert capsys.readouterr().out == default_config_yaml()

    target = tmp_path / "config.yaml"
    assert main(["init-config", "-o", str(target)]) == 0
    assert load_config(target).calculator.name == "macemp"
    with pytest.raises(SystemExit):
        main(["init-config", "-o", str(target)])  # refuses to overwrite


@pytest.mark.parametrize("command", ["vasp", "dft-collect"])
def test_planned_commands_report_not_implemented(command, capsys):
    assert main([command, "anything"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
