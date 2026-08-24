"""Stage 08 -- build the market-data worklist.

Lane-level market data is bought or pulled one lane at a time, so the worklist is
a budget decision: which lanes are worth looking up, and which cannot be looked
up at all. Saying which lanes are unbenchmarkable is part of the deliverable --
those loads are not evidence of anything, and they must not be quietly averaged
into a headline as though they were.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import config, lanes as lanes_mod, report  # noqa: E402

report.configure_pandas()
report.heading("stage 08 -- market lookup worklist")

lanes = pd.read_csv(config.data("lane_summary.csv"))
universe = pd.read_csv(config.data("loads_truckload.csv"), low_memory=False,
                       dtype={"origin_zip": str, "dest_zip": str})
eligible = lanes_mod.benchmarkable(lanes)

# Equipment the provider prices, mapped from what the TMS recorded. Everything
# specialised collapses to flatbed; anything unrecognised falls back to van,
# which is what the overwhelming majority of the book actually moves on.
EQUIPMENT_MAP = {
    "Flatbed": "Flatbed", "Step deck": "Flatbed", "Conestoga": "Flatbed",
    "Stepdeck Conestoga": "Flatbed", "Low boy": "Flatbed",
    "Double drop (low boy)": "Flatbed", "Hotshot Flatbed": "Flatbed",
    "Reefer": "Reefer", "Tanker": "Reefer",
}

equipment = universe.groupby("lane")["equipment_type"].agg(
    lambda s: s.mode().iloc[0] if s.notna().any() else "Van - standard")
worklist = eligible.merge(equipment.rename("equipment_raw"), on="lane", how="left")
worklist["provider_equipment"] = worklist["equipment_raw"].map(EQUIPMENT_MAP).fillna("Van")
worklist["origin"] = worklist["top_city"].str.split(" > ").str[0]
worklist["destination"] = worklist["top_city"].str.split(" > ").str[-1]

US_STATES = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
    .split())

is_domestic = (worklist["o_state"].isin(US_STATES) & worklist["d_state"].isin(US_STATES))
is_lane = worklist["miles"] > 50
pullable = worklist[is_domestic & is_lane].copy()
pullable = pullable.sort_values("spend", ascending=False)
pullable["cum_pct"] = 100 * pullable["spend"].cumsum() / pullable["spend"].sum()

total_spend = lanes["spend"].sum()
print(f"  truckload spend in window        : {report.money(total_spend)}")
print(f"  benchmarkable lanes              : {len(eligible):,}  "
      f"{report.money(eligible['spend'].sum())}")
print(f"  pullable (US domestic, >50 mi)   : {len(pullable):,}  "
      f"{report.money(pullable['spend'].sum())} "
      f"({100 * pullable['spend'].sum() / total_spend:.0f}% of truckload spend)")

report.section("excluded from the market pull")
excluded = worklist[~(is_domestic & is_lane)]
cross_border = worklist[~is_domestic]
local = worklist[is_domestic & ~is_lane]
print(f"  cross-border lanes : {len(cross_border):>4}  "
      f"{report.money(cross_border['spend'].sum())}")
print(f"  local (<=50 mi)    : {len(local):>4}  {report.money(local['spend'].sum())}")
thin = lanes[~lanes["lane"].isin(eligible["lane"])]
print(f"  too thin to price  : {len(thin):>4}  {report.money(thin['spend'].sum())} "
      f"across {thin['loads'].sum():,} loads, median "
      f"{thin['loads'].median():.0f} load(s) per lane")
report.note(
    "The thin tail is the honest limitation of the whole exercise. Those lanes "
    "cannot be benchmarked by this method or by a rate service, and the right "
    "thing to do is report their size rather than fold them into an average that "
    "would then carry a precision it has not earned."
)

report.section(f"worklist -- top 25 lanes by spend")
report.table(pullable.head(25)[[
    "lane", "origin", "destination", "provider_equipment", "loads", "miles",
    "avg_cost", "avg_cpm", "spend", "carriers", "cum_pct"]])

out = config.data("market_worklist.csv")
pullable.to_csv(out, index=False)
report.wrote(out)
print()
