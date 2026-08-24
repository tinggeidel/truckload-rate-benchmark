"""Stage 04 -- derive the truckload / LTL label.

The central stage. Everything downstream depends on this being right, and there
is no field in the export that answers it. See `tlbench/classify.py` for the
method and the reasoning behind each rule.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import classify, coerce, config, report  # noqa: E402

report.configure_pandas()
report.heading("stage 04 -- derive the truckload / ltl label")

df = pd.read_csv(config.data("loads_miles.csv"), low_memory=False,
                 dtype={"origin_zip": str, "dest_zip": str})
completed = coerce.completed(df)

flagged = classify.flag_anchors(completed)
anchor_ltl = flagged[flagged["ltl_common_carrier"]]
anchor_tl = flagged[flagged["tl_equipment"]]

report.section("anchor sets")
print(f"  LTL network carriers : {len(anchor_ltl):>6,} loads  "
      f"{report.money(anchor_ltl['carrier_cost'].sum())}")
print(f"  truckload equipment  : {len(anchor_tl):>6,} loads  "
      f"{report.money(anchor_tl['carrier_cost'].sum())}")
report.note(
    "Neither anchor is inferred. A terminal-network carrier does not haul full "
    "truckloads for a brokerage, and nothing but a truckload moves on a lowboy. "
    "The boundary below is fitted to separate these two populations, then "
    "applied to everything in between."
)

separation = classify.band_separation(flagged)
report.section("where the anchors separate, by distance band")
report.table(separation, floats=",.0f")

boundary = classify.fit_boundary(separation)
print(f"\n  fitted boundary: {boundary.describe()}")
for miles in (100, 300, 600, 1000, 2000):
    print(f"      {miles:>5} mi -> {report.money(boundary.cut(miles))}")

labelled = classify.assign_confidence(classify.classify(flagged, boundary))

report.section("derived mode vs the TMS mode column")
print(pd.crosstab(labelled["mode"], labelled["mode_derived"], margins=True).to_string())
says_tl = int((labelled["mode"] == "TL").sum())
derived_tl = int((labelled["mode_derived"] == "TL").sum())
report.note(
    f"The TMS calls {says_tl:,} loads truckload. The derivation finds "
    f"{derived_tl:,}. The flag is not noisy around the right answer -- it is "
    "wrong in one direction, almost everywhere."
)

report.section("spend by derived mode")
spend = labelled.groupby("mode_derived").agg(
    loads=("id", "size"), spend=("carrier_cost", "sum"), revenue=("revenue", "sum"))
spend["gross_margin_pct"] = 100 * (spend["revenue"] - spend["spend"]) / spend["revenue"]
spend["pct_of_spend"] = 100 * spend["spend"] / spend["spend"].sum()
report.table(spend.reset_index(), floats=",.1f")

report.section("which rule decided each load")
basis = labelled.groupby(["mode_derived", "mode_basis"]).agg(
    loads=("id", "size"), spend=("carrier_cost", "sum"))
report.table(basis.reset_index(), floats=",.0f")
fitted_share = 100 * basis.loc[("TL", "cost_vs_miles"), "spend"] / \
    labelled.loc[labelled["mode_derived"] == "TL", "carrier_cost"].sum()
report.note(
    f"{fitted_share:.0f}% of truckload spend is decided by the fitted boundary "
    "rather than by an anchor. That is the portion of the result that rests on "
    "inference, and it is the number to quote when someone asks how much of this "
    "is a guess."
)

report.section("confidence: loads that price below a plausible truckload floor")
truckload = labelled[labelled["mode_derived"] == "TL"]
confidence = truckload.groupby("tl_confidence").agg(
    loads=("id", "size"), spend=("carrier_cost", "sum"),
    median_cpm=("cost_per_mile", "median"))
report.table(confidence.reset_index(), floats=",.2f")
held = confidence.loc["low"] if "low" in confidence.index else None
if held is not None:
    report.note(
        f"{int(held['loads']):,} loads held out of the benchmark. They are "
        "probably partials or volume LTL that the boundary swept up. Including "
        "them would drag every lane average down and manufacture a finding that "
        "we buy below market, so this fails toward exclusion."
    )

out = config.data("loads_classified.csv")
labelled.to_csv(out, index=False)
report.wrote(out)
print()
