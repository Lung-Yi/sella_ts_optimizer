"""Tests for calculator configuration, capabilities and caching."""

from __future__ import annotations

import dataclasses
import inspect
import sys
import types

import pytest

from ase_structure_optimizer import calculators
from ase_structure_optimizer.calculators import (
    CalculatorConfig,
    available_calculators,
    build_calculator,
    calculator_capabilities,
    clear_calculator_cache,
    get_calculator,
)


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_calculator_cache()
    yield
    clear_calculator_cache()


class _Recorder:
    """Callable that records its calls and returns a fresh object each time."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return types.SimpleNamespace(source=self.name, args=args, kwargs=kwargs)


@pytest.fixture
def fake_backends(monkeypatch):
    """Install fake torch / mace / fairchem modules and return their recorders."""

    cuda_available = {"value": True}
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available["value"])
    monkeypatch.setitem(sys.modules, "torch", torch)

    mace_omol = _Recorder("mace_omol")
    mace_mp = _Recorder("mace_mp")
    mace = types.ModuleType("mace")
    mace_calculators = types.ModuleType("mace.calculators")
    mace_calculators.mace_omol = mace_omol
    mace_calculators.mace_mp = mace_mp
    mace.calculators = mace_calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", mace_calculators)

    get_predict_unit = _Recorder("predict_unit")
    fairchem_calculator = _Recorder("FAIRChemCalculator")
    fairchem = types.ModuleType("fairchem")
    fairchem_core = types.ModuleType("fairchem.core")
    fairchem_core.FAIRChemCalculator = fairchem_calculator
    fairchem_core.pretrained_mlip = types.SimpleNamespace(get_predict_unit=get_predict_unit)
    fairchem.core = fairchem_core
    monkeypatch.setitem(sys.modules, "fairchem", fairchem)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)

    return types.SimpleNamespace(
        cuda=cuda_available,
        mace_omol=mace_omol,
        mace_mp=mace_mp,
        get_predict_unit=get_predict_unit,
        fairchem_calculator=fairchem_calculator,
    )


def test_config_fields_extend_the_original_ones():
    names = [field.name for field in dataclasses.fields(CalculatorConfig)]
    assert names[:8] == [
        "name",
        "charge",
        "multiplicity",
        "threads",
        "qchem_method",
        "qchem_basis",
        "qchem_label",
        "mace_model",
    ]
    config = CalculatorConfig(name="emt")
    assert (config.device, config.dtype, config.dispersion, config.mace_mp_model, config.uma_task) == (
        "auto",
        "float64",
        False,
        "medium-mpa-0",
        "omol",
    )
    # Original positional construction still works.
    legacy = CalculatorConfig("qchem", 1, 2, 8, "b3lyp", "sto-3g", "lbl", "small")
    assert legacy.mace_model == "small" and legacy.device == "auto"


def test_available_calculators_keeps_original_order():
    original = ("qchem", "b3lyp", "xtb", "uma_s", "uma_m", "eSEN", "aimnet2", "emt", "maceomol")
    names = available_calculators()
    assert names[: len(original)] == original
    assert "macemp" in names


def test_default_uma_and_esen_calls_are_unchanged(fake_backends):
    build_calculator(CalculatorConfig(name="uma_s"))
    build_calculator(CalculatorConfig(name="uma_m"))
    build_calculator(CalculatorConfig(name="eSEN"))

    assert [call[0][0] for call in fake_backends.get_predict_unit.calls] == [
        "uma-s-1p1",
        "uma-m-1p1",
        "esen-sm-conserving-all-omol",
    ]
    assert all(call[1] == {"device": "cuda"} for call in fake_backends.get_predict_unit.calls)
    kwargs = [call[1] for call in fake_backends.fairchem_calculator.calls]
    assert kwargs == [{"task_name": "omol"}, {"task_name": "omol"}, {}]


def test_uma_task_is_passed_through(fake_backends):
    build_calculator(CalculatorConfig(name="uma_s", uma_task="oc20"))
    build_calculator(CalculatorConfig(name="uma_m", uma_task="omat"))
    kwargs = [call[1] for call in fake_backends.fairchem_calculator.calls]
    assert kwargs == [{"task_name": "oc20"}, {"task_name": "omat"}]


def test_esen_rejects_periodic_tasks(fake_backends):
    with pytest.raises(ValueError, match="eSEN"):
        build_calculator(CalculatorConfig(name="eSEN", uma_task="oc20"))
    with pytest.raises(ValueError, match="eSEN"):
        calculator_capabilities(CalculatorConfig(name="eSEN", uma_task="omat"))
    assert fake_backends.get_predict_unit.calls == []


def test_unknown_uma_task_is_rejected(fake_backends):
    with pytest.raises(ValueError, match="uma_task"):
        build_calculator(CalculatorConfig(name="uma_s", uma_task="bogus"))


def test_default_maceomol_call_is_unchanged(fake_backends):
    build_calculator(CalculatorConfig(name="maceomol"))
    fake_backends.cuda["value"] = False
    build_calculator(CalculatorConfig(name="maceomol", mace_model="/models/m.model"))
    assert fake_backends.mace_omol.calls == [
        ((), {"model": "extra_large", "device": "cuda", "default_dtype": "float64"}),
        ((), {"model": "/models/m.model", "device": "cpu", "default_dtype": "float64"}),
    ]


def test_maceomol_default_dtype_matches_mace_default():
    """Passing dtype='float64' explicitly must equal mace_omol()'s own default."""

    mace_calculators = pytest.importorskip("mace.calculators")
    default = inspect.signature(mace_calculators.mace_omol).parameters["default_dtype"].default
    assert default == CalculatorConfig(name="maceomol").dtype


def test_explicit_device_skips_autodetection(fake_backends):
    build_calculator(CalculatorConfig(name="maceomol", device="cpu"))
    assert fake_backends.mace_omol.calls[0][1]["device"] == "cpu"


def test_macemp_arguments(fake_backends):
    build_calculator(
        CalculatorConfig(
            name="macemp", mace_mp_model="medium", dtype="float32", dispersion=True, device="cpu"
        )
    )
    assert fake_backends.mace_mp.calls == [
        (
            (),
            {
                "model": "medium",
                "device": "cpu",
                "default_dtype": "float32",
                "dispersion": True,
                "damping": "bj",
                "dispersion_xc": "pbe",
            },
        )
    ]


def test_unknown_calculator():
    with pytest.raises(ValueError, match="Unknown calculator"):
        build_calculator(CalculatorConfig(name="nope"))
    with pytest.raises(ValueError, match="Unknown calculator"):
        calculator_capabilities(CalculatorConfig(name="nope"))


@pytest.mark.parametrize(
    ("config", "periodic"),
    [
        (CalculatorConfig(name="macemp"), True),
        (CalculatorConfig(name="emt"), True),
        (CalculatorConfig(name="uma_s", uma_task="oc20"), True),
        (CalculatorConfig(name="uma_m", uma_task="omat"), True),
        (CalculatorConfig(name="uma_s"), False),
        (CalculatorConfig(name="uma_m", uma_task="omol"), False),
        (CalculatorConfig(name="eSEN"), False),
        (CalculatorConfig(name="maceomol"), False),
        (CalculatorConfig(name="aimnet2"), False),
        (CalculatorConfig(name="xtb"), False),
        (CalculatorConfig(name="qchem"), False),
        (CalculatorConfig(name="b3lyp"), False),
    ],
)
def test_capabilities_periodicity(config, periodic):
    capabilities = calculator_capabilities(config)
    assert capabilities.periodic is periodic
    assert capabilities.level_of_theory


def test_every_calculator_has_capabilities():
    for name in available_calculators():
        calculator_capabilities(CalculatorConfig(name=name))


def test_capability_details():
    emt = calculator_capabilities(CalculatorConfig(name="emt"))
    assert emt.elements == frozenset({"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt", "H", "C", "N", "O"})
    assert emt.uses_charge_spin is False
    assert calculator_capabilities(CalculatorConfig(name="macemp")).elements is None
    assert "D3" in calculator_capabilities(CalculatorConfig(name="macemp", dispersion=True)).level_of_theory
    assert "D3" not in calculator_capabilities(CalculatorConfig(name="macemp")).level_of_theory
    r2scan = CalculatorConfig(name="macemp", mace_mp_model="/m/MACE-matpes-r2scan-omat-ft.model", dispersion=True,
                              dispersion_xc="r2scan")
    assert calculator_capabilities(r2scan).level_of_theory == "r2SCAN (MatPES) + D3(BJ, r2scan)"
    assert "OC20" in calculator_capabilities(CalculatorConfig(name="uma_s", uma_task="oc20")).level_of_theory
    assert calculator_capabilities(CalculatorConfig(name="uma_s")).uses_charge_spin is True


def test_cache_returns_same_instance_for_same_config(fake_backends):
    config = CalculatorConfig(name="macemp", device="cpu")
    first = get_calculator(config)
    assert get_calculator(config) is first
    assert get_calculator(CalculatorConfig(name="macemp", device="cpu")) is first
    assert len(fake_backends.mace_mp.calls) == 1

    other = get_calculator(dataclasses.replace(config, dtype="float32"))
    assert other is not first
    assert len(fake_backends.mace_mp.calls) == 2


def test_cache_can_be_bypassed_and_cleared(fake_backends):
    config = CalculatorConfig(name="uma_s", uma_task="oc20")
    first = get_calculator(config)
    assert get_calculator(config, cache=False) is not first
    clear_calculator_cache()
    assert get_calculator(config) is not first


def test_file_based_and_cheap_backends_are_not_cached():
    qchem = CalculatorConfig(name="qchem")
    assert get_calculator(qchem) is not get_calculator(qchem)
    emt = CalculatorConfig(name="emt")
    assert get_calculator(emt) is not get_calculator(emt)
    assert calculators._CALCULATOR_CACHE == {}


def test_build_calculator_never_caches(fake_backends):
    config = CalculatorConfig(name="macemp")
    assert build_calculator(config) is not build_calculator(config)
    assert calculators._CALCULATOR_CACHE == {}


def test_dispersion_xc_is_passed(fake_backends):
    build_calculator(CalculatorConfig(name="macemp", dispersion=True, dispersion_xc="r2scan", device="cpu"))
    assert fake_backends.mace_mp.calls[0][1]["dispersion_xc"] == "r2scan"
