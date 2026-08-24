"""Stage 10 -- assemble the deliverable workbook.

Every headline on the summary sheet is traceable to a sheet behind it, and the
load detail carries the derived mode, the rule that decided it, and the
confidence grade. A reader who distrusts the classification can filter on
`mode_basis` and see exactly which loads rest on inference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import config, market, report, synth, workbook  # noqa: E402

report.configure_pandas()
report.heading("stage 10 -- build the workbook")

classified = pd.read_csv(config.data("loads_classified.csv"), low_memory=False,
                         dtype={"origin_zip": str, "dest_zip": str},
                         parse_dates=["start_date"])
universe = pd.read_csv(config.data("loads_truckload.csv"), low_memory=False,
                       dtype={"origin_zip": str, "dest_zip": str},
                       parse_dates=["start_date"])
lanes = pd.read_csv(config.data("lane_summary.csv"))
benchmarked = pd.read_csv(config.data("market_benchmark_results.csv"))
sell = pd.read_csv(config.data("sell_side_view.csv"))
monthly = pd.read_csv(config.data("carrier_network_monthly.csv"))
lifespan = pd.read_csv(config.data("carrier_lifespan.csv"))

national = market.national_benchmark(
    lanes[lanes["lane"].isin(benchmarked["lane"])], synth.NATIONAL_SPOT_PER_MILE)
national_summary = market.variance_summary(national, rate_column="national_rate")
lane_summary = market.variance_summary(benchmarked)

truckload = classified[classified["mode_derived"] == "TL"]
once = lifespan[lifespan["loads"] == 1]
problem = sell[sell["sell_side_problem"]]

summary = workbook.build_summary([
    ("Completed loads in file", f"{len(classified):,}"),
    ("Total carrier spend", report.money(classified["carrier_cost"].sum())),
    ("", ""),
    ("TMS mode column says truckload",
     f"{int((classified['mode'] == 'TL').sum()):,} loads"),
    ("Derived truckload",
     f"{len(truckload):,} loads / {report.money(truckload['carrier_cost'].sum())}"),
    ("Truckload share of carrier spend",
     f"{100 * truckload['carrier_cost'].sum() / classified['carrier_cost'].sum():.1f}%"),
    ("Held out as low confidence",
     f"{int((truckload['tl_confidence'] == 'low').sum()):,} loads"),
    ("", ""),
    ("Benchmark window", f"{config.WINDOW_START} to {config.WINDOW_END}"),
    ("Lanes benchmarked vs market",
     f"{lane_summary['lanes']:,} lanes / {lane_summary['loads']:,} loads"),
    ("", ""),
    ("Benchmarked against national average",
     f"{report.money(national_summary['variance'])} "
     f"({report.pct(national_summary['variance_pct'])})"),
    ("Benchmarked against lane-level market",
     f"{report.money(lane_summary['variance'])} "
     f"({report.pct(lane_summary['variance_pct'])})"),
    ("Which of those is correct", "Lane-level. See the Method notes sheet."),
    ("Lanes below market", f"{lane_summary['lanes_below']} of {lane_summary['lanes']}"),
    ("", ""),
    ("Lanes with a sell-side problem", f"{len(problem)}"),
    ("Revenue on those lanes", report.money(problem["revenue"].sum())),
    ("", ""),
    ("Truckload carriers used", f"{universe['carrier_name'].nunique():,}"),
    ("Used exactly once",
     f"{len(once):,} ({100 * len(once) / len(lifespan):.0f}%)"),
    ("Still active in last 60 days",
     f"{int(lifespan['still_active'].sum()):,} "
     f"({100 * lifespan['still_active'].mean():.0f}%)"),
])

method_notes = workbook.build_summary([
    ("Why two benchmark numbers",
     "A national spot average is the mean of a lane population that looks "
     "nothing like this book."),
    ("What goes wrong",
     "This book concentrates into rural, low-backhaul destinations. A carrier "
     "arrives loaded and leaves empty and prices the empty half into the rate. "
     "That premium is real, large, and absent from a national average."),
    ("Size of the error",
     "On the worst lane the national comparison is wrong by roughly 90 "
     "percentage points, and it is wrong in the direction that blames sourcing."),
    ("What to use instead",
     "Lane-level market rates, graded by how much of our own volume the "
     "provider actually matched. Below 60% match, treat a lane as unmeasured."),
    ("Mode column",
     "Not usable. The truckload flag is derived from carrier identity, "
     "equipment, and a fitted cost-versus-distance boundary. See mode_basis on "
     "the load detail sheet for which rule decided each load."),
    ("Confidence grade",
     "Derived-truckload loads pricing below a plausible floor -- absolute and "
     "relative to their own lane -- are held out rather than forced to a side."),
    ("Unbenchmarkable freight",
     "Lanes with too little repeat volume cannot be benchmarked by this method "
     "or by a rate service. Their size is reported rather than averaged in."),
])

sheets = {
    "Summary": summary,
    "Method notes": method_notes,
    "Market benchmark": benchmarked,
    "National comparison": national,
    "Sell side": sell,
    "All TL lanes": lanes.head(300),
    "Carrier scorecard": workbook.carrier_scorecard(universe),
    "Network by month": monthly,
    "Load detail": classified[workbook.DETAIL_COLUMNS],
}

path = config.data("truckload_rate_benchmark.xlsx")
written = workbook.write_workbook(path, sheets)
for name, rows in written:
    print(f"  {name:<24} {rows:>8,} rows")
report.wrote(path)
print()
