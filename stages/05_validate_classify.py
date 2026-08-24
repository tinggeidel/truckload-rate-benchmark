"""Stage 05 -- validate the derived label.

Two checks, and they are different in kind.

The first is available on any dataset: does the freight labelled truckload
actually price like truckload, band by band? That is a consistency check. It can
only catch a label that contradicts physics; it cannot confirm one that is
merely wrong.

The second needs ground truth, which only the synthetic dataset has. A real
export has no answer key -- that absence is the entire reason `classify.py`
exists. Running the classifier against a dataset where the answer *is* known is
the only way to put a number on how well it works, and it is what separates this
from an untested heuristic that happens to produce plausible output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import classify, config, report  # noqa: E402

report.configure_pandas()
report.heading("stage 05 -- validate the derived label")

df = pd.read_csv(config.data("loads_classified.csv"), low_memory=False,
                 dtype={"origin_zip": str, "dest_zip": str})
priced = df[df["est_miles"].notna()].copy()
priced["band"] = pd.cut(priced["est_miles"], classify.DISTANCE_BANDS)

report.section("cost per mile by derived mode and distance band")
for mode in ("LTL", "TL"):
    subset = priced[priced["mode_derived"] == mode]
    stats = subset.groupby("band", observed=True)["cost_per_mile"].agg(
        n="size", p10=lambda x: x.quantile(0.10), p25=lambda x: x.quantile(0.25),
        median="median", p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90))
    print(f"\n  --- {mode} ---")
    report.table(stats.reset_index(), floats=",.2f")
report.note(
    "The two populations should barely overlap, and LTL cost per mile should "
    "fall away with distance far faster than truckload does. Overlap in a band "
    "is where the boundary is doing the most guessing."
)

report.section("derived-truckload freight priced below a plausible floor")
truckload = priced[priced["mode_derived"] == "TL"]
for lo, hi, floor in classify.TRUCKLOAD_FLOORS:
    band = truckload[(truckload["est_miles"] > lo) & (truckload["est_miles"] <= hi)]
    below = band[band["cost_per_mile"] < floor]
    if not len(band):
        continue
    label = f"{lo:.0f}-{hi:.0f} mi" if hi != float("inf") else f"{lo:.0f}+ mi"
    print(f"  {label:<12} floor ${floor:.2f}/mi: {len(below):>5,} of {len(band):>5,} "
          f"loads ({100 * len(below) / len(band):>4.1f}%)  "
          f"{report.money(below['carrier_cost'].sum())}")

report.section("derived-LTL freight that prices like truckload")
ltl = priced[priced["mode_derived"] == "LTL"]
odd = ltl[(ltl["carrier_cost"] > 1500) & (ltl["cost_per_mile"] > 1.20)]
print(f"  {len(odd):,} loads, {report.money(odd['carrier_cost'].sum())}")
if len(odd):
    report.table(odd.groupby("carrier_name").agg(
        loads=("id", "size"), spend=("carrier_cost", "sum")
    ).nlargest(6, "spend").reset_index(), floats=",.0f")
report.note(
    "The mirror-image error. A small residue here is expected -- expedited and "
    "guaranteed LTL genuinely prices this high -- but a large one means the "
    "boundary sits too high and truckload is leaking out of the benchmark."
)

truth_path = config.data("truth_labels.csv")
if not truth_path.exists():
    report.section("ground truth")
    report.note(
        "No truth labels present, which is the normal case for a real export. "
        "The consistency checks above are all that is available; treat the "
        "classifier's accuracy as unmeasured rather than as good."
    )
    print()
    raise SystemExit(0)

truth = pd.read_csv(truth_path)
scored = df.merge(truth[["id", "true_mode"]], on="id", how="inner")

report.section("scored against ground truth")
overall = classify.score_against_truth(scored)
print(f"  loads scored          : {overall['n']:,}")
print(f"  accuracy              : {overall['accuracy']:.3f}")
print(f"  truckload precision   : {overall['tl_precision']:.3f}")
print(f"  truckload recall      : {overall['tl_recall']:.3f}")
print(f"  truckload F1          : {overall['tl_f1']:.3f}")

report.section("what the confidence gate buys")
truckload_rows = scored[scored["mode_derived"] == "TL"]
high = truckload_rows[truckload_rows["tl_confidence"] == "high"]
held = truckload_rows[truckload_rows["tl_confidence"] == "low"]
purity_all = (truckload_rows["true_mode"] == "TL").mean()
purity_high = (high["true_mode"] == "TL").mean()
print(f"  all derived truckload : {len(truckload_rows):>6,} loads, "
      f"{100 * purity_all:.1f}% genuinely truckload")
print(f"  high confidence only  : {len(high):>6,} loads, "
      f"{100 * purity_high:.1f}% genuinely truckload")
print(f"  held out              : {len(held):>6,} loads, "
      f"{100 * (held['true_mode'] == 'LTL').mean():.1f}% of them genuinely LTL")
report.note(
    f"The gate removes {len(held):,} loads and lifts purity from "
    f"{100 * purity_all:.1f}% to {100 * purity_high:.1f}%. Nearly everything it "
    "removes is freight that does not belong in a truckload rate benchmark, "
    "which is what the gate is for."
)

report.section("where the classifier is wrong")
wrong = scored[scored["mode_derived"] != scored["true_mode"]]
if len(wrong):
    report.table(wrong.groupby(["true_mode", "mode_derived", "mode_basis"]).agg(
        loads=("id", "size"), spend=("carrier_cost", "sum"),
        median_cpm=("cost_per_mile", "median")).reset_index(), floats=",.2f")
    report.note(
        "Errors concentrate on the fitted boundary rather than on the anchors, "
        "which is the expected shape: the anchors are evidence and the boundary "
        "is inference. Freight that is genuinely ambiguous in price is genuinely "
        "ambiguous to this method too."
    )
print()
