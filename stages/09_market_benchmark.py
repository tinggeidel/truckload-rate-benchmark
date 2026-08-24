"""Stage 09 -- benchmark against the market, correctly and incorrectly.

This stage runs the benchmark twice on purpose.

The first pass uses a published national spot average, because that number is
free, immediate, and the obvious thing to reach for. In the engagement this
pipeline came from, it returned roughly +21% above market and about $1.5M of
annual overpay -- a finding that was large, plausible, entirely wrong, and
pointed at a sourcing team that had done nothing wrong.

The second pass uses lane-level market rates and inverts the result.

Both are printed because the gap between them is the finding. An analysis that
only showed the correct number would be less useful: the wrong number is what
anyone reaching for free data will get, and the size of the error is the argument
for not doing that.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import config, geo, lanes as lanes_mod, market, report, synth  # noqa: E402

report.configure_pandas()
report.heading("stage 09 -- market benchmark")

lanes = pd.read_csv(config.data("lane_summary.csv"))
universe = pd.read_csv(config.data("loads_truckload.csv"), low_memory=False,
                       dtype={"origin_zip": str, "dest_zip": str})
worklist = pd.read_csv(config.data("market_worklist.csv"))
eligible = lanes[lanes["lane"].isin(worklist["lane"])].copy()

# A real pull is dropped in as a CSV. The synthetic one is derived from the same
# ground truth the loads were priced against, so the demo stays self-consistent.
pull_path = config.data("market_pull.csv")
truth_path = config.data("truth_labels.csv")
if pull_path.exists():
    pull = pd.read_csv(pull_path)
    print(f"  market pull: {pull_path} ({len(pull)} lanes)")
elif truth_path.exists():
    # The truth columns are never written into the export, so they are joined
    # back here purely to synthesise the provider's answer. Nothing downstream
    # of this join reads them.
    truth = pd.read_csv(truth_path)
    pull = synth.market_pull(universe.merge(truth, on="id", how="inner"), eligible)
    pull.to_csv(pull_path, index=False)
    print(f"  market pull: synthetic, {len(pull)} lanes")
else:
    raise SystemExit(
        f"no market data. Provide {pull_path} with columns: lane, provider_miles, "
        "market_rate, market_rate_90d, provider_reports_total, provider_reports_ours."
    )

# ---------------------------------------------------------------------------
# Pass 1 -- the national average. Kept to be refuted.
# ---------------------------------------------------------------------------
report.section("pass 1: benchmarked against a national spot average")
national = market.national_benchmark(eligible, synth.NATIONAL_SPOT_PER_MILE)
national_summary = market.variance_summary(national, rate_column="national_rate")
print(f"  national spot assumption : ${synth.NATIONAL_SPOT_PER_MILE:.2f}/mile, "
      f"hauls {config.NATIONAL_MIN_MILES:.0f}+ miles")
print(f"  lanes compared           : {national_summary['lanes']:,}  "
      f"({national_summary['loads']:,} loads)")
print(f"  our spend                : {report.money(national_summary['spend'])}")
print(f"  same loads at national   : {report.money(national_summary['at_market'])}")
print(f"  APPARENT VARIANCE        : {report.money(national_summary['variance'])}  "
      f"{report.pct(national_summary['variance_pct'])}")
report.note(
    "Read on its own this says the book is badly overpaying and that sourcing is "
    "the problem. It is wrong. The next pass shows why."
)

# ---------------------------------------------------------------------------
# Pass 2 -- lane-level market rates.
# ---------------------------------------------------------------------------
benchmarked = market.lane_benchmark(eligible, pull)
benchmarked = market.dedupe_market_pairs(market.grade_match_rate(benchmarked))

report.section("mileage estimator calibration")
calibration = benchmarked.dropna(subset=["provider_miles", "miles"])
error = geo.calibration_error(calibration["miles"], calibration["provider_miles"])
print(f"  n={len(calibration)}   mean error {error.mean():+.1f}%   "
      f"median {error.median():+.1f}%   range {error.min():+.1f}% to {error.max():+.1f}%")
report.note(
    "The centroid estimate runs short of real road miles, which matters more than "
    "it sounds: a mileage error moves every derived rate per mile by the same "
    "proportion in the same direction. Provider mileage is used wherever it exists."
)

report.section("provider data quality")
unmeasured = benchmarked[~benchmarked["measured"]]
duplicated = benchmarked[benchmarked["market_pair_dupe"]]
print(f"  lanes where the provider matched <{100 * config.MIN_MATCH_RATE:.0f}% of "
      f"our loads : {len(unmeasured)}")
if len(unmeasured):
    report.table(unmeasured[["lane", "origin", "destination", "loads",
                             "provider_reports_ours", "match_rate", "var_pct"]])
    report.note(
        "These are unmeasured, not merely uncertain. The provider is describing a "
        "handful of our loads and presenting it with the same confidence as a "
        "lane built from hundreds. Do not act on these."
    )
print(f"  lanes silently aggregated to a shared market pair          : {len(duplicated)}")
if len(duplicated):
    report.note(
        "Identical rate, report count, and mileage across two distinct lanes is "
        "the provider answering a thin lane with the surrounding market average. "
        "Summed alongside its twin it double-counts."
    )

report.section("pass 2: benchmarked against lane-level market rates")
lane_summary = market.variance_summary(benchmarked)
print(f"  lanes compared         : {lane_summary['lanes']:,}  "
      f"({lane_summary['loads']:,} loads)")
print(f"  our spend              : {report.money(lane_summary['spend'])}")
print(f"  same loads at market   : {report.money(lane_summary['at_market'])}")
print(f"  TRUE VARIANCE          : {report.money(lane_summary['variance'])}  "
      f"{report.pct(lane_summary['variance_pct'])}")
print(f"  lanes above market     : {lane_summary['lanes_above']}")
print(f"  lanes below market     : {lane_summary['lanes_below']}")

report.section("robustness")
report.table(market.robustness_cuts(benchmarked), floats=",.2f")
report.note(
    "A finding that survives every cut is a finding. The national-average result "
    "does not survive contact with lane-level data at all, which is the test the "
    "first pass failed."
)

report.section("where the national average goes wrong, lane by lane")
comparison = national.merge(
    benchmarked[["lane", "market_rate", "provider_miles"]], on="lane", how="inner")
comparison["national_var_pct"] = comparison["var_pct"]
comparison["true_var_pct"] = 100 * (comparison["avg_cost"] - comparison["market_rate"]) \
    / comparison["market_rate"]
comparison["error_pts"] = comparison["national_var_pct"] - comparison["true_var_pct"]
report.table(comparison.nlargest(10, "error_pts")[[
    "lane", "top_city", "loads", "miles", "avg_cost", "national_rate",
    "national_var_pct", "market_rate", "true_var_pct", "error_pts"]], floats=",.1f")
report.note(
    "Every one of these is a rural, low-backhaul destination. A carrier reaches "
    "them loaded and leaves empty, and prices the empty half into the rate. That "
    "premium is real and it is large, and a national average -- which is mostly "
    "made of lanes between dense markets -- knows nothing about it."
)

report.section("lanes genuinely above market")
above = benchmarked[benchmarked["var_per_load"] > 0].nlargest(8, "annual_var")
if len(above):
    report.table(above[["lane", "origin", "destination", "loads", "avg_cost",
                        "market_rate", "var_per_load", "var_pct", "annual_var",
                        "grade"]], floats=",.0f")
    actionable = above[above["measured"]]
    print(f"\n  of which measured well enough to act on: {len(actionable)} lanes, "
          f"{report.money(actionable['annual_var'].sum())}")
report.note(
    "This is the real overpay, and it is a fraction of what the national average "
    "claimed. It is also a list of specific lanes rather than a headline, which "
    "is the difference between a finding and a number."
)

if "market_rate_90d" in benchmarked.columns:
    report.section("against the current market")
    recent = benchmarked.dropna(subset=["market_rate_90d"])
    current = market.variance_summary(recent, rate_column="market_rate_90d")
    print(f"  {current['lanes']} lanes, {current['loads']:,} loads: "
          f"{report.money(current['variance'])}  {report.pct(current['variance_pct'])} "
          "vs the trailing 90-day market")
    rose = int((recent["market_rate_90d"] > recent["market_rate"]).sum())
    report.note(
        f"The 90-day rate sits above the 1-year rate on {rose} of {len(recent)} "
        "lanes. The market rose across the window. Whether that is a problem "
        "depends entirely on whether sell rates moved with it -- which is the "
        "next section, and the point at which this analysis stopped being about "
        "sourcing."
    )

report.section("the sell side")
sell = market.sell_side_view(universe, benchmarked)
problem = sell[sell["sell_side_problem"]]
print(f"  lanes buying at or below market whose margin is collapsing: {len(problem)}")
if len(problem):
    report.table(problem.head(10)[[
        "lane", "origin", "destination", "loads", "var_pct", "margin_pct",
        "margin_pct_earlier", "margin_pct_recent", "margin_change_pts",
        "gross_profit"]], floats=",.1f")
    exposure = problem["revenue"].sum()
    print(f"\n  revenue on those lanes: {report.money(exposure)} "
          f"({100 * exposure / sell['revenue'].sum():.0f}% of benchmarked revenue)")
report.note(
    "This is the finding the first pass hid completely. These lanes are sourced "
    "at or below market and are still losing margin, so no amount of carrier "
    "negotiation fixes them -- the sell rate is what is wrong. Note that whole- "
    "window margin looks unremarkable on most of them; the collapse only shows up "
    "when the recent months are separated from the earlier ones."
)
report.note(
    "Sourcing and pricing are usually owned by different people, which is how a "
    "lane bleeds a point of margin a month for most of a year without either of "
    "them escalating it."
)

benchmarked.to_csv(config.data("market_benchmark_results.csv"), index=False)
sell.to_csv(config.data("sell_side_view.csv"), index=False)
report.wrote(config.data("market_benchmark_results.csv"))
print()
