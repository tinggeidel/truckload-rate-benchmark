"""End-to-end behaviour of the pipeline.

These are the tests that pin the repo's actual claims. Everything else checks a
function; these check that running the whole thing on data whose answer is known
produces the conclusions the README describes -- including the wrong one, which
has to keep being wrong for the comparison to mean anything.
"""
import pandas as pd
import pytest

from tlbench import classify, lanes as lanes_mod, market, synth


@pytest.fixture(scope="module")
def benchmarked(truckload_universe, lane_table):
    eligible = lanes_mod.benchmarkable(lane_table)
    pull = synth.market_pull(truckload_universe, eligible)
    joined = market.lane_benchmark(eligible, pull)
    return market.dedupe_market_pairs(market.grade_match_rate(joined))


@pytest.fixture(scope="module")
def national(lane_table, benchmarked):
    eligible = lane_table[lane_table["lane"].isin(benchmarked["lane"])]
    return market.national_benchmark(eligible, synth.NATIONAL_SPOT_PER_MILE)


class TestTheTrap:
    def test_national_average_says_the_book_is_badly_overpaying(self, national):
        """The wrong answer, reproduced.

        This is what anyone reaching for free published data gets. It has to
        keep coming out wrong, because the size of the error is the argument
        for not doing this.
        """
        summary = market.variance_summary(national, rate_column="national_rate")
        assert summary["variance_pct"] > 15

    def test_lane_level_data_says_the_book_buys_below_market(self, benchmarked):
        """The right answer, and the opposite sign."""
        summary = market.variance_summary(benchmarked)
        assert summary["variance_pct"] < 0

    def test_the_two_methods_disagree_by_more_than_twenty_points(
            self, national, benchmarked):
        national_pct = market.variance_summary(
            national, rate_column="national_rate")["variance_pct"]
        lane_pct = market.variance_summary(benchmarked)["variance_pct"]
        assert national_pct - lane_pct > 20

    def test_the_error_concentrates_on_premium_destinations(
            self, national, benchmarked):
        """Not spread evenly -- it lands on exactly the rural, low-backhaul
        lanes the book is concentrated into, which is why it is so convincing."""
        comparison = national.merge(
            benchmarked[["lane", "market_rate"]], on="lane", how="inner")
        comparison["true_var_pct"] = 100 * (
            comparison["avg_cost"] - comparison["market_rate"]) / comparison["market_rate"]
        comparison["error_pts"] = comparison["var_pct"] - comparison["true_var_pct"]

        premium_destinations = {
            zip_code[:3] for zip_code, *_rest, premium in synth.DESTINATIONS
            if premium > 1.5
        }
        worst = comparison.nlargest(5, "error_pts")
        assert worst["error_pts"].min() > 40
        assert all(lane.split(">")[1] in premium_destinations
                   for lane in worst["lane"])


class TestRobustness:
    def test_the_finding_survives_every_cut(self, benchmarked):
        """A result that only holds on the full set is an artifact of whatever
        the weakest lanes contributed. The national-average headline failed
        exactly this test."""
        cuts = market.robustness_cuts(benchmarked)
        assert (cuts["variance_pct"] < 0).all()
        assert cuts["variance_pct"].max() - cuts["variance_pct"].min() < 5

    def test_unmeasured_lanes_are_identified(self, benchmarked):
        """The provider matched only a fraction of our loads on some lanes.
        Those must be flagged rather than silently averaged in."""
        assert (~benchmarked["measured"]).any()

    def test_market_pair_duplicates_are_identified(self, benchmarked):
        assert benchmarked["market_pair_dupe"].any()


class TestClassifierEndToEnd:
    def test_derived_label_is_accurate_against_ground_truth(self, classified):
        assert classify.score_against_truth(classified)["accuracy"] > 0.90

    def test_derived_label_beats_the_column_it_replaces(self, classified):
        truth = classified["true_mode"]
        derived = (classified["mode_derived"] == truth).mean()
        flag = (classified["mode"].where(classified["mode"] == "TL", "LTL") == truth).mean()
        assert derived > flag + 0.30

    def test_most_truckload_spend_rests_on_evidence_not_inference(self, classified):
        """The fitted boundary decides the ambiguous middle, but the anchors
        should carry a meaningful share -- otherwise the result is one
        regression line wearing a methodology."""
        truckload = classified[classified["mode_derived"] == "TL"]
        assert (truckload["mode_basis"] == "tl_equipment").any()
        assert truckload["mode_basis"].nunique() >= 2

    def test_benchmark_universe_is_cleaner_than_the_raw_labels(self, classified):
        truckload = classified[classified["mode_derived"] == "TL"]
        universe = classify.benchmark_universe(classified, "2025-04-01", "2026-04-01")
        assert (universe["true_mode"] == "TL").mean() > (truckload["true_mode"] == "TL").mean()


class TestInternalBenchmark:
    def test_finds_recoverable_spend_without_any_market_data(self, lane_table):
        """It needs no subscription and cannot be wrong about the market,
        because it never refers to one."""
        eligible = lanes_mod.benchmarkable(lane_table)
        assert eligible["recoverable"].sum() > 0

    def test_recoverable_is_a_fraction_of_spend_not_a_multiple_of_it(self, lane_table):
        eligible = lanes_mod.benchmarkable(lane_table)
        assert eligible["recoverable"].sum() < eligible["spend"].sum() * 0.5


class TestSellSide:
    def test_finds_lanes_sourced_well_that_are_still_losing_margin(
            self, truckload_universe, benchmarked):
        """The finding the national-average pass hid completely."""
        sell = market.sell_side_view(truckload_universe, benchmarked)
        flagged = sell[sell["sell_side_problem"]]

        assert len(flagged) > 0
        assert (flagged["buys_below_market"]).all()
