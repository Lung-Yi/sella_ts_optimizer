"""Tests for the time-budget scheduler."""

from __future__ import annotations

import time

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT

from ase_structure_optimizer.adsorption.config import BudgetConfig
from ase_structure_optimizer.adsorption.scheduler import (
    BudgetClock,
    diverse_selection,
    measure_step_time,
    prescreen_count,
    relax_count,
    stratified_sample,
)


def test_budget_clock():
    clock = BudgetClock(total=100.0, spent_before=40.0)
    assert not clock.unlimited
    assert clock.elapsed() >= 40.0
    assert clock.time_until(0.4) == pytest.approx(0.0, abs=0.1)
    assert clock.time_until(0.95) == pytest.approx(55.0, abs=0.1)
    time.sleep(0.01)
    assert clock.session_elapsed() > 0
    unlimited = BudgetClock(total=0.0)
    assert unlimited.unlimited and unlimited.time_until(0.4) == float("inf")


def test_counts_follow_the_formulas():
    budget = BudgetConfig(wall_time_per_termination=600, prescreen_steps=20, prescreen_fraction=0.4, max_steps=400)
    # N_pre = floor(0.4 * 600 / (20 * 0.1)) = 120
    assert prescreen_count(500, budget, 0.1) == 120
    assert prescreen_count(50, budget, 0.1) == 50
    # N_full = max(5, floor(330 / (0.5 * 400 * 0.1))) = 16
    assert relax_count(100, budget, 0.1, 330.0) == 16
    assert relax_count(100, budget, 10.0, 330.0) == 5  # min_full_relax
    assert relax_count(3, budget, 0.1, 330.0) == 3
    assert relax_count(100, budget, 0.1, -5.0) == 5

    unlimited = BudgetConfig(wall_time_per_termination=0)
    assert prescreen_count(500, unlimited, 0.1) == 500
    assert relax_count(100, unlimited, 0.1, float("inf")) == 30  # top 30 %
    assert relax_count(10, unlimited, 0.1, float("inf")) == 5  # but at least min_full_relax


def test_stratified_sample_covers_strata():
    items = [(kind, anchor, i) for i, (kind, anchor) in enumerate(
        [(k, a) for k in ("ontop", "bridge", "hollow") for a in ("A", "B", "C", "D") for _ in range(5)]
    )]
    rng = np.random.default_rng(1)
    chosen = stratified_sample(items, 15, key=lambda item: item[:2], rng=rng)
    assert len(chosen) == 15 and len({item[:2] for item in chosen}) == 12
    assert len({item[:2] for item in chosen[:12]}) == 12  # representatives first
    assert len(set(chosen)) == 15

    few = stratified_sample(items, 5, key=lambda item: item[:2], rng=np.random.default_rng(2))
    assert len({item[:2] for item in few}) == 5
    assert stratified_sample(items, 100, key=lambda item: item[:2], rng=rng) == items
    again = stratified_sample(items, 15, key=lambda item: item[:2], rng=np.random.default_rng(1))
    assert again == chosen


def test_diverse_selection():
    records = [
        ("a", -1.0, "CO"),
        ("b", -0.9, "CO"),
        ("c", -0.5, "C5H5"),
        ("d", -0.8, "CO"),
        ("e", -0.1, "none"),
        ("f", -0.6, "C5H5"),
    ]
    # Best of each label first (CO: a, C5H5: f, none: e), then by energy.
    assert diverse_selection(records, 3) == ["a", "f", "e"]
    assert diverse_selection(records, 5) == ["a", "f", "e", "b", "d"]
    assert diverse_selection(records, 2) == ["a", "f"]
    assert diverse_selection(records, 10) == ["a", "f", "e", "b", "d", "c"]


def test_measure_step_time():
    atoms = molecule("H2O")
    assert measure_step_time(atoms, EMT(), repeats=3) > 0
    with pytest.raises(ValueError):
        measure_step_time(atoms, EMT(), repeats=1)
