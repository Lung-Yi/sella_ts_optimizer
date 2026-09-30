"""Time budget for the adsorption sampling of one termination.

The budget (`budget.wall_time_per_termination`, 0 = unlimited) covers the
prescreening and full relaxations of this milestone plus the final single
points of the analysis. Its phases end at fractions of the budget:

* prescreening until ``prescreen_fraction`` of the budget,
* full relaxations until ``prescreen_fraction + relax_fraction``,
* the rest (5 % by default) is reserve for the final single points.

Time already spent before an interruption is carried over on resume.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence, TypeVar

import numpy as np
from ase import Atoms

from .config import BudgetConfig

T = TypeVar("T")

# Fraction of the budget used when it is unlimited: top 30 % of the
# prescreened configurations are fully relaxed.
UNLIMITED_RELAX_FRACTION = 0.30


@dataclass
class BudgetClock:
    """Wall clock of one termination's sampling budget."""

    total: float
    spent_before: float = 0.0
    _started: float = field(default_factory=time.perf_counter)

    @property
    def unlimited(self) -> bool:
        return self.total <= 0

    def elapsed(self) -> float:
        """Budget time used so far, including previous sessions."""

        return self.spent_before + time.perf_counter() - self._started

    def session_elapsed(self) -> float:
        return time.perf_counter() - self._started

    def time_until(self, fraction: float) -> float:
        """Seconds left until `fraction` of the budget is used (inf if unlimited)."""

        if self.unlimited:
            return math.inf
        return fraction * self.total - self.elapsed()


def measure_step_time(atoms: Atoms, calculator, repeats: int = 3) -> float:
    """Average wall time (s) of one force evaluation.

    `repeats` evaluations on slightly displaced copies of `atoms`; the first
    is not counted because it includes model initialization and compilation.
    """

    if repeats < 2:
        raise ValueError("repeats must be >= 2 (the first evaluation is discarded)")
    times = []
    for index in range(repeats):
        trial = atoms.copy()
        trial.positions[-1, 0] += 1e-4 * (index + 1)
        trial.calc = calculator
        started = time.perf_counter()
        trial.get_forces()
        times.append(time.perf_counter() - started)
    return float(np.mean(times[1:]))


def prescreen_count(n_candidates: int, budget: BudgetConfig, step_time: float) -> int:
    """Number of candidates that fit in the prescreening phase."""

    if budget.wall_time_per_termination <= 0:
        return n_candidates
    cost = budget.prescreen_steps * max(step_time, 1e-9)
    return min(n_candidates, int(budget.prescreen_fraction * budget.wall_time_per_termination // cost))


def relax_count(n_available: int, budget: BudgetConfig, step_time: float, seconds_left: float) -> int:
    """Planned number of full relaxations.

    ``max(min_full_relax, floor(seconds_left / (0.5 * max_steps * step_time)))``
    (an average relaxation is assumed to use half of `max_steps`), capped at
    `n_available`. With an unlimited budget: the top 30 % of prescreened
    configurations (at least `min_full_relax`).
    """

    if budget.wall_time_per_termination <= 0:
        planned = max(budget.min_full_relax, math.ceil(UNLIMITED_RELAX_FRACTION * n_available))
    else:
        cost = 0.5 * budget.max_steps * max(step_time, 1e-9)
        planned = max(budget.min_full_relax, int(max(seconds_left, 0.0) // cost))
    return min(n_available, planned)


def stratified_sample(
    items: Sequence[T],
    n: int,
    key: Callable[[T], Hashable],
    rng: np.random.Generator,
) -> list[T]:
    """Pick `n` items covering as many `key` strata as possible.

    One random item per stratum first (strata in random order if there are
    more strata than `n`), then random items from the rest. The result is in
    selection order: stratum representatives first, so that if the budget
    runs out before all items are processed, coverage is kept.
    """

    if n >= len(items):
        return list(items)
    strata: dict[Hashable, list[int]] = {}
    for index, item in enumerate(items):
        strata.setdefault(key(item), []).append(index)
    chosen: list[int] = []
    for stratum in rng.permutation(len(strata)):
        members = list(strata.values())[stratum]
        chosen.append(int(members[rng.integers(len(members))]))
        if len(chosen) == n:
            break
    taken = set(chosen)
    rest = [index for index in range(len(items)) if index not in taken]
    extra = rng.choice(rest, size=n - len(chosen), replace=False) if n > len(chosen) else []
    return [items[index] for index in chosen + [int(i) for i in extra]]


def diverse_selection(records: Sequence[tuple[str, float, str]], n: int) -> list[str]:
    """Choose `n` ids from ``(id, energy, contact_label)`` records.

    The lowest-energy record of every distinct contact label is taken first
    (labels in order of their best energy), then the remaining records by
    energy. Returned in selection order.
    """

    ordered = sorted(records, key=lambda record: (record[1], record[0]))
    selected: list[str] = []
    seen: set[str] = set()
    for identifier, _, label in ordered:
        if label not in seen:
            seen.add(label)
            selected.append(identifier)
    for identifier, _, _ in ordered:
        if identifier not in selected:
            selected.append(identifier)
    return selected[:n]
