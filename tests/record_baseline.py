"""Record reference outputs for the core regression tests.

Run this on the unmodified ``main`` branch (or whenever an intentional,
reviewed behavior change requires a new baseline)::

    python tests/record_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from regression_cases import BASELINE_PATH, all_cases, run_case  # noqa: E402


def main() -> int:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    baseline: dict[str, object] = {"_meta": {"recorded_at_commit": commit}}
    with tempfile.TemporaryDirectory() as tmp:
        for name in all_cases():
            print(f"recording {name} ...", flush=True)
            summary = run_case(name, Path(tmp) / name)
            if summary is None:
                print(f"  skipped {name}: optional dependency missing")
                continue
            baseline[name] = summary
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
