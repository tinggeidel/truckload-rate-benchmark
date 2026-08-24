"""Assemble the deliverable workbook.

The workbook is what actually gets read. It is built so that a reader can start
at the summary, disagree with a number, and follow it all the way down to the
loads it came from without asking anyone for help -- which is the only version
of this that survives being questioned in a meeting.
"""
from __future__ import annotations

import pandas as pd

DETAIL_COLUMNS = [
    "id", "custom_id", "start_date", "customer_name", "carrier_name", "mode",
    "mode_derived", "mode_basis", "tl_confidence", "equipment_type",
    "origin_city", "origin_state", "origin_zip", "dest_city", "dest_state",
    "dest_zip", "est_miles", "revenue", "carrier_cost", "cost_per_mile",
]


def build_summary(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def carrier_scorecard(universe: pd.DataFrame, active_within_days: int = 60) -> pd.DataFrame:
    """Per-carrier view of the truckload book."""
    universe = universe.copy()
    universe["start_date"] = pd.to_datetime(universe["start_date"])

    scorecard = universe.groupby("carrier_name").agg(
        loads=("id", "size"), spend=("carrier_cost", "sum"),
        avg_cost=("carrier_cost", "mean"), avg_cpm=("cost_per_mile", "mean"),
        first_load=("start_date", "min"), last_load=("start_date", "max"),
        lanes=("lane", "nunique"), customers=("customer_name", "nunique"),
    ).reset_index().sort_values("spend", ascending=False)

    scorecard["days_active"] = (
        scorecard["last_load"] - scorecard["first_load"]).dt.days
    cutoff = universe["start_date"].max() - pd.Timedelta(days=active_within_days)
    scorecard["still_active"] = scorecard["last_load"] >= cutoff
    return scorecard


def write_workbook(path, sheets: dict[str, pd.DataFrame]) -> list[tuple[str, int]]:
    """Write each frame to its own sheet and report what landed where."""
    written = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            # Excel caps sheet names at 31 characters and rejects several
            # punctuation marks; truncating here beats an exception at the end
            # of a long pipeline run.
            safe = name[:31]
            frame.to_excel(writer, sheet_name=safe, index=False)
            written.append((safe, len(frame)))
    return written
