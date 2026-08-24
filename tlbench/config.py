"""Paths and tunables.

Nothing here is specific to any one brokerage. Every input path is overridable by
environment variable so the pipeline can point at a real TMS export without the
paths ever entering version control.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("TLBENCH_DATA", ROOT / "data"))
REFERENCE = ROOT / "data" / "reference"

# Source export. Unset in the demo, which generates its own.
SOURCE_EXPORT = os.getenv("TLBENCH_SOURCE_EXPORT")
SOURCE_SHEET = os.getenv("TLBENCH_SOURCE_SHEET", "Sheet1")
# Header row index as pandas sees it (0-based). TMS exports commonly carry two
# banner rows above the real header.
SOURCE_HEADER_ROW = int(os.getenv("TLBENCH_SOURCE_HEADER_ROW", "2"))

# Great-circle -> practical road miles. Calibrated against the market provider's
# own mileage in stage 09; 1.17 is the long-standing industry stand-in.
CIRCUITY = float(os.getenv("TLBENCH_CIRCUITY", "1.17"))

# Local moves still involve real driving. Floor them so $/mile stays finite.
MIN_MILES = float(os.getenv("TLBENCH_MIN_MILES", "25"))

# Benchmark window. Defaults to the 12 complete months of the demo dataset.
WINDOW_START = os.getenv("TLBENCH_WINDOW_START", "2025-04-01")
WINDOW_END = os.getenv("TLBENCH_WINDOW_END", "2026-04-01")

# A lane needs repeat volume from several carriers before its own percentiles
# mean anything.
MIN_LANE_LOADS = int(os.getenv("TLBENCH_MIN_LANE_LOADS", "10"))
MIN_LANE_CARRIERS = int(os.getenv("TLBENCH_MIN_LANE_CARRIERS", "3"))

# Below this share of our loads appearing in the provider's own report count, a
# lane comparison is unmeasured rather than merely uncertain. See docs/METHOD.md.
MIN_MATCH_RATE = float(os.getenv("TLBENCH_MIN_MATCH_RATE", "0.60"))

# National spot averages are published for hauls at or above this distance.
NATIONAL_MIN_MILES = float(os.getenv("TLBENCH_NATIONAL_MIN_MILES", "250"))


def data(*parts: str) -> Path:
    """Resolve a path under the data directory, creating parents as needed."""
    p = DATA.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
