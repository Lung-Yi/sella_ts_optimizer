"""Run the structure optimizer without installing the package."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sella_ts_optimizer.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
