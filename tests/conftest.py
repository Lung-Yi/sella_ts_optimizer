"""Shared pytest configuration.

The package source directory is put on ``sys.path`` so the tests run against
this checkout even when the package is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
