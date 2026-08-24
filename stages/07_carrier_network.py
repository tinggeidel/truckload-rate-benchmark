"""Stage 07 -- carrier-network health.

Sourcing questions answered from the shipment file alone: no accounting system,
no carrier scorecard, no survey. That constraint is deliberate -- these are the
questions a broker can answer today with what the TMS already holds.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import config, network, report  # noqa: E402

report.configure_pandas()
report.heading("stage 07 -- carrier network health")

universe = pd.read_csv(config.data("loads_truckload.csv"), low_memory=False,
                       dtype={"origin_zip": str, "dest_zip": str},
                       parse_dates=["start_date"])
print(f"  truckload universe: {len(universe):,} loads, "
      f"{report.money(universe['carrier_cost'].sum())}, "
      f"{universe['carrier_name'].nunique():,} carriers")

report.section("network by month")
monthly = network.monthly_network(universe)
report.table(monthly)
report.note(
    "Arrivals and departures moving together means the network is churning "
    "rather than growing. Headcount alone hides that completely."
)

report.section("usage concentration")
report.table(network.usage_concentration(universe).reset_index(), floats=",.1f")
report.note(
    "The one-and-done bucket is the number that matters. Every load in it was "
    "priced by a carrier with no relationship to protect and no expectation of "
    "the next load."
)

report.section("first-load premium, measured within lane")
report.table(network.first_load_premium(universe).reset_index())
report.note(
    "Compared within lane on purpose. New carriers get called on the hard "
    "freight, so an unconditional average would measure lane mix and report it "
    "as onboarding cost."
)

report.section("carrier lifespan and retention")
life = network.carrier_lifespan(universe)
once = life[life["loads"] == 1]
print(f"  used exactly once  : {len(once):,} of {len(life):,} carriers "
      f"({100 * len(once) / len(life):.0f}%), {report.money(once['spend'].sum())}")
print(f"  active in last 60d : {int(life['still_active'].sum()):,} "
      f"({100 * life['still_active'].mean():.0f}%)")
repeat = life[life["loads"] > 1]
print(f"  median lifespan    : {repeat['days_active'].median():.0f} days (repeat carriers)")
report.table(life.groupby("bucket", observed=True).agg(
    carriers=("loads", "size"), median_days=("days_active", "median"),
    pct_still_active=("still_active", lambda s: 100 * s.mean()),
    spend=("spend", "sum")).reset_index(), floats=",.1f")

report.section("matched-lane price trend")
split = pd.Timestamp(config.WINDOW_START) + pd.DateOffset(months=6)
trend = network.matched_lane_trend(universe, split.strftime("%Y-%m-%d"))
if trend["lanes"]:
    print(f"  {trend['lanes']} lanes present in both halves, {trend['loads']:,} loads")
    print(f"  volume-weighted average cost: {report.money(trend['before'])} -> "
          f"{report.money(trend['after'])}   {report.pct(trend['change_pct'])}")
    report.note(
        "The lane basket is held fixed across the two halves. Without that, a "
        "book that simply hauled further this half reads as a rate increase, and "
        "the difference between reporting a rate change and a mix change is the "
        "whole value of the number."
    )
else:
    report.note("No lanes carry enough volume in both halves to support a matched trend.")

monthly.to_csv(config.data("carrier_network_monthly.csv"), index=False)
life.reset_index().to_csv(config.data("carrier_lifespan.csv"), index=False)
report.wrote(config.data("carrier_lifespan.csv"))
print()
