"""Shared fixtures.

Every test runs offline: no network, no market-data subscription, no TMS export.
The synthetic generator is seeded, so anything derived from it is reproducible.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tlbench import classify, geo, lanes as lanes_mod, synth  # noqa: E402


@pytest.fixture(scope="session")
def resolver():
    return geo.TableResolver.from_csv()


@pytest.fixture(scope="session")
def raw_loads():
    """A small but structurally complete generated book."""
    return synth.generate_loads(seed=11, n_loads=4000)


@pytest.fixture(scope="session")
def completed_loads(raw_loads, resolver):
    df = raw_loads[(raw_loads["status_description"] == "Completed")
                   & (raw_loads["carrier_cost"] > 0)].copy()
    return geo.attach_mileage(df, resolver)


@pytest.fixture(scope="session")
def classified(completed_loads):
    labelled, boundary = classify.derive_mode(completed_loads)
    return labelled


@pytest.fixture(scope="session")
def truckload_universe(classified):
    universe = classify.benchmark_universe(classified, "2025-04-01", "2026-04-01")
    return lanes_mod.add_lane_keys(universe)


@pytest.fixture(scope="session")
def lane_table(truckload_universe):
    return lanes_mod.recoverable_vs_own_p25(
        truckload_universe, lanes_mod.summarize_lanes(truckload_universe))


def make_loads(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal load frame with sensible defaults for unset columns."""
    defaults = {
        "id": 0, "carrier_name": "Generic Trucking Co", "equipment_type": "Van - standard",
        "carrier_cost": 1000.0, "revenue": 1200.0, "est_miles": 500.0,
        "cost_per_mile": 2.0, "origin_zip": "46803", "dest_zip": "60601",
        "origin_city": "Fort Wayne", "origin_state": "IN",
        "dest_city": "Chicago", "dest_state": "IL",
        "customer_name": "Acme Industries", "start_date": pd.Timestamp("2025-06-01"),
        "mode": "LTL", "status_description": "Completed",
    }
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record["id"] = index + 1
        record.update(row)
        if "cost_per_mile" not in row and record["est_miles"]:
            record["cost_per_mile"] = record["carrier_cost"] / record["est_miles"]
        records.append(record)
    return pd.DataFrame(records)
