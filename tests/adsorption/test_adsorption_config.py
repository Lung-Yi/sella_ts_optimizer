"""Tests for the adsorption configuration schema."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from ase_structure_optimizer import CalculatorConfig  # noqa: E402
from ase_structure_optimizer.adsorption.config import (  # noqa: E402
    AdsorptionConfig,
    ConfigError,
    check_calculator_support,
    config_from_dict,
    config_to_dict,
    default_config_yaml,
    dump_config,
    load_config,
    result_fingerprint,
)

MINIMAL = {"molecule": "mol.xyz", "solid": "bulk.cif"}


def _config(**overrides) -> AdsorptionConfig:
    data = dict(MINIMAL)
    data.update(overrides)
    return config_from_dict(data)


def test_template_matches_defaults():
    parsed = config_from_dict(yaml.safe_load(default_config_yaml()))
    assert parsed == AdsorptionConfig(molecule=Path("molecule.xyz"), solid=Path("surface.cif"))


def test_minimal_config_uses_spec_defaults():
    config = _config()
    assert config.run_dir is None
    assert config.molecule_props.bond_scale == 1.2
    assert config.calculator.name == "macemp" and config.calculator.dispersion is True
    assert config.surface.miller_indices == ((0, 0, 1),)
    assert config.sampling.site_types == ("ontop", "bridge", "hollow")
    assert config.budget.wall_time_per_termination == 600
    assert config.analysis.scan_heights == (1.5, 7.0, 0.25)
    assert config.uncertainty.calculators == (
        CalculatorConfig(name="macemp", mace_mp_model="medium"),
        CalculatorConfig(name="uma_s", uma_task="oc20"),
    )
    assert config.vasp.potcar_map["Ti"] == "Ti_pv"
    assert config.vasp.ediff == 1e-5


def test_required_fields():
    with pytest.raises(ConfigError, match="molecule: required"):
        config_from_dict({"solid": "a.cif"})
    with pytest.raises(ConfigError, match="solid: required"):
        config_from_dict({"molecule": "a.xyz"})


@pytest.mark.parametrize(
    ("overrides", "where"),
    [
        ({"budget": {"fmax": -1}}, "budget.fmax"),
        ({"budget": {"max_steps": 1.5}}, "budget.max_steps"),
        ({"budget": {"prescreen_optimizer": "sella"}}, "budget.prescreen_optimizer"),
        ({"budget": {"prescreen_fraction": 0.6, "relax_fraction": 0.6}}, "budget.relax_fraction"),
        ({"surface": {"miller_indices": [[0, 0, 0]]}}, "surface.miller_indices"),
        ({"surface": {"miller_indices": [[1, 1]]}}, "surface.miller_indices"),
        ({"surface": {"fix_fraction": 1.5}}, "surface.fix_fraction"),
        ({"surface": {"min_lateral": "big"}}, "surface.min_lateral"),
        ({"surface": {"thickness": 3}}, "surface.thickness"),
        ({"sampling": {"site_types": ["ontop", "fourfold"]}}, "sampling.site_types"),
        ({"sampling": {"n_random": True}}, "sampling.n_random"),
        ({"molecule_props": {"bond_overrides": [[0, 0, True]]}}, "molecule_props.bond_overrides"),
        ({"molecule_props": {"reference_axis": [[0], [0]]}}, "molecule_props.reference_axis"),
        ({"analysis": {"scan_heights": [7.0, 1.5, 0.25]}}, "analysis.scan_heights"),
        ({"vasp": {"ispin": 3}}, "vasp.ispin"),
        ({"vasp": {"potcar_map": {"Xx": "Xx"}}}, "vasp.potcar_map"),
        ({"calculator": {"name": "gaussian"}}, "calculator.name"),
        ({"calculator": {"dtype_final": "float16"}}, "calculator.dtype_final"),
        ({"uncertainty": {"calculators": [{"name": "macemp", "model": "x"}]}}, "uncertainty.calculators"),
        ({"unknown_section": {}}, "unknown_section"),
        ({"budget": 5}, "budget"),
    ],
)
def test_validation_errors_name_the_field(overrides, where):
    with pytest.raises(ConfigError) as info:
        _config(**overrides)
    assert str(info.value).startswith(where), str(info.value)


@pytest.mark.parametrize(
    "calculator",
    [
        {"name": "maceomol"},
        {"name": "aimnet2"},
        {"name": "xtb"},
        {"name": "eSEN", "uma_task": "omol"},
        {"name": "uma_s", "uma_task": "omol"},
    ],
)
def test_molecular_calculators_are_rejected(calculator):
    with pytest.raises(ConfigError, match="molecular model"):
        _config(calculator=calculator)


def test_molecular_uncertainty_calculators_are_rejected():
    with pytest.raises(ConfigError, match="uncertainty.calculators.*molecular model"):
        _config(uncertainty={"calculators": [{"name": "maceomol"}]})


def test_periodic_calculators_are_accepted():
    assert _config(calculator={"name": "uma_m", "uma_task": "omat"}).calculator.uma_task == "omat"
    assert _config(calculator={"name": "emt"}).calculator.name == "emt"


def test_values_are_parsed():
    config = _config(
        molecule_props={"bond_overrides": [[0, 5, True]], "reference_axis": [[0], [8, 9, 10]]},
        surface={"min_lateral": 12, "miller_indices": [[1, 1, 1], [1, 0, 0]]},
        vasp={"ediff": "1e-6", "ispin": 2, "ncore": 4},
    )
    assert config.molecule_props.bond_overrides == ((0, 5, True),)
    assert config.molecule_props.reference_axis == ((0,), (8, 9, 10))
    assert config.surface.min_lateral == 12.0
    assert config.surface.miller_indices == ((1, 1, 1), (1, 0, 0))
    assert config.vasp.ediff == 1e-6 and config.vasp.ispin == 2 and config.vasp.ncore == 4


def test_calculator_configs_for_screen_and_final():
    config = _config(molecule_props={"charge": 1, "multiplicity": 2})
    screen = config.calculator_config("screen")
    final = config.calculator_config("final")
    assert screen == dataclasses.replace(final, dtype="float32")
    assert final.dtype == "float64" and final.charge == 1 and final.multiplicity == 2
    assert final.dispersion is True and final.uma_task == "oc20" and final.name == "macemp"
    assert config.calculator_config(mace_mp_model="medium").mace_mp_model == "medium"
    with pytest.raises(ValueError):
        config.calculator_config("prescreen")


def test_uma_model_and_shared_dtype():
    config = _config(calculator={"name": "uma_s", "uma_task": "oc25", "uma_model": "/m/uma-s-1p2p1.pt"})
    screen = config.calculator_config("screen")
    final = config.calculator_config("final")
    assert final.uma_model == "/m/uma-s-1p2p1.pt" and final.uma_task == "oc25"
    # Both stages map to one CalculatorConfig, so the model is loaded only once.
    assert screen == final
    assert _config(calculator={"name": "uma_s"}).calculator_config().uma_model == ""
    with pytest.raises(ConfigError, match="uma_task"):
        _config(calculator={"name": "uma_s", "uma_task": "bogus"})


def test_load_resolves_paths_relative_to_file(tmp_path: Path):
    path = tmp_path / "sub" / "config.yaml"
    path.parent.mkdir()
    path.write_text("molecule: mol.xyz\nsolid: /abs/bulk.cif\nrun_dir: runs/a\n", encoding="utf-8")
    config = load_config(path)
    assert config.molecule == path.parent / "mol.xyz"
    assert config.solid == Path("/abs/bulk.cif")
    assert config.run_dir == path.parent / "runs" / "a"


def test_load_error_mentions_file(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("molecule: m.xyz\nsolid: s.cif\nbudget: {fmax: 0}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"bad\.yaml: budget\.fmax"):
        load_config(path)


def test_dump_and_reload_roundtrip(tmp_path: Path):
    config = _config(
        molecule_props={"bond_overrides": [[0, 5, True]], "reference_axis": [[0], [8, 9]]},
        uncertainty={"calculators": [{"name": "uma_m", "uma_task": "omat", "device": "cpu"}]},
        vasp={"ncore": 8},
    )
    path = dump_config(config, tmp_path / "config.resolved.yaml")
    assert load_config(path) == dataclasses.replace(
        config, molecule=config.molecule.resolve(), solid=config.solid.resolve()
    )
    assert config_to_dict(config)["uncertainty"]["calculators"] == [
        {"name": "uma_m", "device": "cpu", "uma_task": "omat"}
    ]


def test_resolved_run_dir(tmp_path: Path):
    config = config_from_dict({"molecule": "CpMo.xyz", "solid": "TiSi.cif"}, base_dir=tmp_path)
    assert config.resolved_run_dir() == tmp_path / "CpMo_on_TiSi_macemp"
    assert _config(run_dir="/x/y").resolved_run_dir() == Path("/x/y")


def test_fingerprint_tracks_only_result_affecting_settings():
    base = result_fingerprint(_config())
    assert result_fingerprint(_config(budget={"wall_time_per_termination": 60})) == base
    assert result_fingerprint(_config(vasp={"encut": 500})) == base
    assert result_fingerprint(_config(sampling={"seed": 1})) != base
    assert result_fingerprint(_config(calculator={"dispersion": False})) != base
    assert result_fingerprint(_config(surface={"fix_fraction": 0.3})) != base
    assert result_fingerprint(_config(molecule="other.xyz")) != base


def test_check_calculator_support_elements():
    emt = _config(calculator={"name": "emt"})
    assert check_calculator_support(emt, ["Cu", "C", "O"]).periodic
    with pytest.raises(ConfigError, match="Si, Ti"):
        check_calculator_support(emt, ["Ti", "Si", "C"])
    assert check_calculator_support(_config(), ["Mo", "Ti", "Si"]).elements is None
