"""Market benchmarking -- and the trap the module exists to encode."""
import numpy as np
import pandas as pd
import pytest

from tlbench import market


def make_lanes(rows: list[dict]) -> pd.DataFrame:
    defaults = {"lane": "468>606", "loads": 50, "spend": 75000.0,
                "avg_cost": 1500.0, "med_cost": 1450.0, "miles": 500.0,
                "carriers": 6}
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record["lane"] = row.get("lane", f"L{index:03d}>D{index:03d}")
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


def make_pull(rows: list[dict]) -> pd.DataFrame:
    defaults = {"lane": "468>606", "origin": "Fort Wayne, IN",
                "destination": "Chicago, IL", "equipment": "Van",
                "provider_miles": 500.0, "market_rate": 1500.0,
                "market_rate_90d": 1650.0, "provider_reports_total": 800,
                "provider_reports_ours": 45, "rate_strength": 92}
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record["lane"] = row.get("lane", f"L{index:03d}>D{index:03d}")
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


class TestNationalBenchmark:
    def test_short_hauls_are_excluded(self):
        """Published national averages cover hauls at or above a distance floor.

        Comparing a 100-mile lane against them measures nothing.
        """
        lanes = make_lanes([{"miles": 100.0}, {"miles": 800.0}])
        assert len(market.national_benchmark(lanes, 2.30, min_miles=250)) == 1

    def test_variance_is_priced_off_the_flat_rate(self):
        lanes = make_lanes([{"miles": 1000.0, "avg_cost": 2500.0, "loads": 10}])
        out = market.national_benchmark(lanes, 2.00, min_miles=250).iloc[0]

        assert out["national_rate"] == pytest.approx(2000.0)
        assert out["var_per_load"] == pytest.approx(500.0)
        assert out["annual_var"] == pytest.approx(5000.0)

    def test_a_premium_lane_reads_as_massive_overpay(self):
        """The trap, in one assertion.

        A rural destination whose own market clears $4.00/mile is compared
        against a $2.30 national mean and reads as +74% overpay. It is not
        overpaying; the national average simply does not describe this lane.
        """
        lanes = make_lanes([{"miles": 400.0, "avg_cost": 1600.0}])
        out = market.national_benchmark(lanes, 2.30, min_miles=250).iloc[0]
        assert out["var_pct"] > 70


class TestMatchRate:
    def test_low_match_is_graded_unmeasured(self):
        """The provider saw 7 of our 63 loads. That is not a 12% variance --
        it is an absence of data wearing a variance's clothes."""
        joined = make_pull([{"provider_reports_ours": 7}]).assign(loads=63)
        graded = market.grade_match_rate(joined, min_match_rate=0.60)

        assert not graded["measured"].iloc[0]
        assert graded["grade"].iloc[0] == "unmeasured"

    def test_high_match_is_measured(self):
        joined = make_pull([{"provider_reports_ours": 55}]).assign(loads=60)
        assert market.grade_match_rate(joined, min_match_rate=0.60)["measured"].iloc[0]

    def test_threshold_is_inclusive(self):
        joined = make_pull([{"provider_reports_ours": 60}]).assign(loads=100)
        assert market.grade_match_rate(joined, min_match_rate=0.60)["measured"].iloc[0]


class TestMarketPairDedupe:
    def test_identical_figures_flag_the_second_lane(self):
        """Two distinct lanes returning byte-identical rate, report count, and
        mileage is the provider answering a thin lane with a market average."""
        pull = make_pull([
            {"lane": "A>B", "market_rate": 2500.0, "provider_reports_total": 1137},
            {"lane": "C>D", "market_rate": 2500.0, "provider_reports_total": 1137},
        ])
        out = market.dedupe_market_pairs(pull)
        assert out["market_pair_dupe"].tolist() == [False, True]

    def test_distinct_lanes_are_not_flagged(self):
        pull = make_pull([
            {"lane": "A>B", "market_rate": 2500.0},
            {"lane": "C>D", "market_rate": 2600.0},
        ])
        assert not market.dedupe_market_pairs(pull)["market_pair_dupe"].any()

    def test_missing_signature_columns_are_tolerated(self):
        pull = make_pull([{"lane": "A>B"}]).drop(columns=["market_rate_90d"])
        assert "market_pair_dupe" in market.dedupe_market_pairs(pull).columns


class TestLaneBenchmark:
    def test_provider_mileage_replaces_the_estimate(self):
        """A mileage error moves every derived rate per mile proportionally."""
        lanes = make_lanes([{"lane": "A>B", "miles": 480.0, "avg_cost": 1500.0}])
        pull = make_pull([{"lane": "A>B", "provider_miles": 500.0}])
        out = market.lane_benchmark(lanes, pull).iloc[0]

        assert out["use_miles"] == 500.0
        assert out["our_cpm"] == pytest.approx(3.0)

    def test_falls_back_to_the_estimate_when_provider_mileage_is_missing(self):
        lanes = make_lanes([{"lane": "A>B", "miles": 480.0}])
        pull = make_pull([{"lane": "A>B", "provider_miles": np.nan}])
        assert market.lane_benchmark(lanes, pull)["use_miles"].iloc[0] == 480.0

    def test_variance_sign_follows_the_market(self):
        lanes = make_lanes([{"lane": "A>B", "avg_cost": 1400.0}])
        pull = make_pull([{"lane": "A>B", "market_rate": 1500.0}])
        out = market.lane_benchmark(lanes, pull).iloc[0]
        assert out["var_per_load"] < 0  # buying below market


class TestVarianceSummary:
    def test_weighted_by_loads_not_by_lane(self):
        """An unweighted mean of per-lane percentages lets a 4-load lane
        outvote a 400-load one, which on a concentrated book is how a rounding
        error on the tail overturns the headline."""
        benchmarked = pd.DataFrame({
            "loads": [400, 4],
            "avg_cost": [1000.0, 1000.0],
            "market_rate": [1000.0, 500.0],  # tiny lane is 100% over
        })
        summary = market.variance_summary(benchmarked)

        unweighted = np.mean([0.0, 100.0])
        assert summary["variance_pct"] < unweighted / 10

    def test_counts_lanes_above_and_below(self):
        benchmarked = pd.DataFrame({
            "loads": [10, 10, 10],
            "avg_cost": [1100.0, 900.0, 950.0],
            "market_rate": [1000.0, 1000.0, 1000.0],
        })
        summary = market.variance_summary(benchmarked)
        assert summary["lanes_above"] == 1
        assert summary["lanes_below"] == 2

    def test_empty_input_returns_zeros_rather_than_raising(self):
        summary = market.variance_summary(
            pd.DataFrame(columns=["loads", "avg_cost", "market_rate"]))
        assert summary["lanes"] == 0

    def test_rows_missing_a_market_rate_are_dropped(self):
        benchmarked = pd.DataFrame({
            "loads": [10, 10],
            "avg_cost": [1000.0, 1000.0],
            "market_rate": [1000.0, np.nan],
        })
        assert market.variance_summary(benchmarked)["lanes"] == 1


class TestRobustness:
    def test_reports_every_cut(self):
        benchmarked = pd.DataFrame({
            "loads": [10, 10],
            "avg_cost": [1000.0, 1000.0],
            "market_rate": [1000.0, 1000.0],
            "measured": [True, False],
            "market_pair_dupe": [False, True],
        })
        cuts = market.robustness_cuts(benchmarked)
        assert cuts["cut"].tolist() == [
            "all verified lanes", "market-pair deduped", "measured only"]
        assert cuts.loc[cuts["cut"] == "measured only", "lanes"].iloc[0] == 1

    def test_works_without_the_optional_grading_columns(self):
        benchmarked = pd.DataFrame({
            "loads": [10], "avg_cost": [1000.0], "market_rate": [1000.0]})
        assert len(market.robustness_cuts(benchmarked)) == 3


class TestSellSide:
    def _book(self):
        """A committed lane whose sell rate was agreed once and never revisited,
        while the buy side kept tracking a rising market."""
        rows = []
        for month in range(1, 13):
            cost = 1000.0 + 25.0 * month          # buy tracks the market up
            rows.append({
                "lane": "A>B", "id": month,
                "start_date": pd.Timestamp("2025-04-01") + pd.DateOffset(months=month - 1),
                "carrier_cost": cost, "revenue": 1300.0,  # sell held flat
            })
        return pd.DataFrame(rows)

    def test_detects_collapsing_margin_on_a_lane_bought_below_market(self):
        benchmarked = pd.DataFrame({
            "lane": ["A>B"], "loads": [12], "avg_cost": [1150.0],
            "market_rate": [1300.0], "var_per_load": [-150.0],
        })
        out = market.sell_side_view(self._book(), benchmarked).iloc[0]

        assert out["buys_below_market"]
        assert out["margin_change_pts"] < 0
        assert out["sell_side_problem"]

    def test_whole_window_margin_hides_the_collapse(self):
        """Why the trajectory comparison exists: the average of a healthy first
        half and an underwater second half looks unremarkable."""
        benchmarked = pd.DataFrame({
            "lane": ["A>B"], "loads": [12], "avg_cost": [1150.0],
            "market_rate": [1300.0], "var_per_load": [-150.0],
        })
        out = market.sell_side_view(self._book(), benchmarked).iloc[0]
        assert out["margin_pct"] > out["margin_pct_recent"]

    def test_a_lane_buying_above_market_is_not_a_sell_side_problem(self):
        """That one is a sourcing problem, and it belongs in a different list."""
        benchmarked = pd.DataFrame({
            "lane": ["A>B"], "loads": [12], "avg_cost": [1500.0],
            "market_rate": [1300.0], "var_per_load": [200.0],
        })
        out = market.sell_side_view(self._book(), benchmarked).iloc[0]
        assert not out["buys_below_market"]
        assert not out["sell_side_problem"]

    def test_healthy_stable_lane_is_not_flagged(self):
        stable = pd.DataFrame({
            "lane": ["A>B"] * 12, "id": range(12),
            "start_date": pd.date_range("2025-04-01", periods=12, freq="MS"),
            "carrier_cost": [1000.0] * 12, "revenue": [1300.0] * 12,
        })
        benchmarked = pd.DataFrame({
            "lane": ["A>B"], "loads": [12], "avg_cost": [1000.0],
            "market_rate": [1100.0], "var_per_load": [-100.0],
        })
        out = market.sell_side_view(stable, benchmarked).iloc[0]
        assert not out["sell_side_problem"]
