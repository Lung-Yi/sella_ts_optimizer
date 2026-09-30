"""Regression tests: existing min / TS / IRC / frequency / CLI behavior is unchanged.

Every case in ``regression_cases.py`` is re-run and compared with the summary
recorded on the unmodified ``main`` branch (``tests/data/regression_baseline.json``).
Output file names, directory layout, step counts, convergence flags and CLI
messages must match exactly; energies, coordinates and frequencies must match
to tight numerical tolerances. Result dataclasses may only gain new fields
appended after the existing ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from regression_cases import BASELINE_PATH, all_cases, run_case

FLOAT_ABS_TOL = 1e-6
FREQUENCY_ABS_TOL = 1e-3

BASELINE: dict[str, Any] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _compare(expected: Any, actual: Any, where: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        for key, value in expected.items():
            assert key in actual, f"{where}: missing key {key!r}"
            if key == "fields":
                # Existing dataclass fields keep their order; new ones only appended.
                assert actual[key][: len(value)] == value, f"{where}.fields: {actual[key]}"
                continue
            _compare(value, actual[key], f"{where}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), where
        assert len(actual) == len(expected), f"{where}: length {len(actual)} != {len(expected)}"
        for index, (exp, act) in enumerate(zip(expected, actual)):
            _compare(exp, act, f"{where}[{index}]")
        return
    if isinstance(expected, float) and not isinstance(expected, bool):
        tol = FREQUENCY_ABS_TOL if "frequencies" in where else FLOAT_ABS_TOL
        assert isinstance(actual, (int, float)), where
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol), (
            f"{where}: {actual!r} != {expected!r}"
        )
        return
    assert actual == expected, f"{where}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("name", [name for name in all_cases()])
def test_regression_case(name: str, tmp_path: Path) -> None:
    if name not in BASELINE:
        pytest.skip(f"no baseline recorded for {name} (optional dependency missing when recorded)")
    summary = run_case(name, tmp_path)
    if summary is None:
        pytest.skip(f"optional dependency for {name} is not installed")
    _compare(BASELINE[name], summary, name)
