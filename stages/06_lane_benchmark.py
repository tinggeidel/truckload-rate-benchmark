"""Stage 06 -- benchmark the book against itself.

Deliberately before any external market data is bought. On a lane where we buy
repeatedly from several carriers, the 25th percentile is a price we have already
paid for this freight, recently. The gap to it is recoverable without knowing
anything about the wider market, and it cannot be wrong about the market because
it never refers to one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import classify, config, lanes as lanes_mod, report  # noqa: E402

report.configure_pandas()
report.heading("stage 06 -- internal lane benchmark")

df = pd.read_csv(config.data("loads_classified.csv"), low_memory=False,
                 dtype={"origin_zip": str, "dest_zip": str},
                 parse_dates=["start_date"])

universe = classify.benchmark_universe(df, config.WINDOW_START, config.WINDOW_END)
universe = lanes_mod.add_lane_keys(universe)
print(f"  window: {config.WINDOW_START} .. {config.WINDOW_END}")
print(f"  high-confidence truckload: {len(universe):,} loads  "
      f"{report.money(universe['carrier_cost'].sum())}")
report.note(
    "Complete months only. A partial month at either end biases the average "
    "toward whatever mix happened to move in it."
)

lanes = lanes_mod.recoverable_vs_own_p25(universe, lanes_mod.summarize_lanes(universe))
eligible = lanes_mod.benchmarkable(lanes)

report.section("lane structure")
print(f"  lanes total                      : {len(lanes):,}")
print(f"  median loads per lane            : {lanes['loads'].median():.0f}")
print(f"  benchmarkable (>={config.MIN_LANE_LOADS} loads, "
      f">={config.MIN_LANE_CARRIERS} carriers) : {len(eligible):,} lanes, "
      f"{eligible['loads'].sum():,} loads, {report.money(eligible['spend'].sum())} "
      f"({100 * eligible['spend'].sum() / lanes['spend'].sum():.0f}% of truckload spend)")
n80 = lanes_mod.lane_concentration(lanes, 80.0)
print(f"  lanes carrying 80% of spend      : {n80:,} "
      f"({100 * n80 / len(lanes):.1f}% of lanes)")
report.note(
    "This concentration decides how much external market data is worth buying. "
    "A book where a few dozen lanes carry most of the spend can be benchmarked "
    "properly for the price of a few dozen lookups; the long tail cannot be "
    "benchmarked by any tool at any price, and saying so up front is more useful "
    "than quietly averaging over it."
)

report.section("recoverable against each lane's own 25th percentile")
print(f"  {report.money(eligible['recoverable'].sum())} across "
      f"{len(eligible):,} benchmarkable lanes")
report.note(
    "Not a forecast. Every dollar of this was priced by a carrier we already use, "
    "on this lane, inside the window."
)

report.section("top 15 lanes by spend")
report.table(lanes.head(15)[[
    "lane", "top_city", "loads", "miles", "spend", "avg_cost", "p25_cost",
    "avg_cpm", "carriers", "recoverable", "cum_spend_pct"]])

report.section("top 10 lanes by recoverable spend")
report.table(eligible.nlargest(10, "recoverable")[[
    "lane", "top_city", "loads", "miles", "avg_cost", "p25_cost", "avg_cpm",
    "carriers", "recoverable"]])

biggest = lanes.iloc[0]
report.section(f"inside the largest lane: {biggest['lane']} ({biggest['top_city']})")
spread = lanes_mod.price_percentiles(
    universe.loc[universe["lane"] == biggest["lane"], "carrier_cost"])
print(f"  {int(biggest['loads']):,} loads, {report.money(biggest['spend'])}, "
      f"{int(biggest['carriers'])} carriers")
print("  cost spread: " + "  ".join(f"{k}=${v:,.0f}" for k, v in spread.items()))
report.table(lanes_mod.carrier_spread(universe, biggest["lane"]).head(10).reset_index())
report.note(
    "A wide spread on a single lane is the cheapest finding in the whole "
    "analysis: same lane, same period, same freight, and a materially different "
    "price depending on who answered the phone."
)

universe.to_csv(config.data("loads_truckload.csv"), index=False)
lanes.to_csv(config.data("lane_summary.csv"), index=False)
report.wrote(config.data("lane_summary.csv"))
print()
