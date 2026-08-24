"""Lane aggregation and the internal benchmark."""
import pandas as pd
import pytest

from tlbench import lanes as lanes_mod
from conftest import make_loads


class TestLaneKeys:
    def test_lane_key_uses_three_digit_prefixes(self):
        out = lanes_mod.add_lane_keys(make_loads([
            {"origin_zip": "46803", "dest_zip": "60601"}]))
        assert out["lane"].iloc[0] == "468>606"

    def test_canadian_points_key_on_the_fsa(self):
        out = lanes_mod.add_lane_keys(make_loads([
            {"origin_zip": "46803", "dest_zip": "N1H4G8"}]))
        assert out["lane"].iloc[0] == "468>N1H"

    def test_lane_is_directional(self):
        """A backhaul is a different market and must not merge with the fronthaul."""
        out = lanes_mod.add_lane_keys(make_loads([
            {"origin_zip": "46803", "dest_zip": "60601"},
            {"origin_zip": "60601", "dest_zip": "46803"},
        ]))
        assert out["lane"].nunique() == 2

    def test_lane_name_is_readable(self):
        out = lanes_mod.add_lane_keys(make_loads([{}]))
        assert out["lane_name"].iloc[0] == "Fort Wayne, IN > Chicago, IL"


class TestSummarize:
    def test_one_row_per_lane(self):
        loads = lanes_mod.add_lane_keys(make_loads([
            {"origin_zip": "46803", "dest_zip": "60601"},
            {"origin_zip": "46803", "dest_zip": "60601"},
            {"origin_zip": "46803", "dest_zip": "75201"},
        ]))
        summary = lanes_mod.summarize_lanes(loads)
        assert len(summary) == 2
        assert summary.set_index("lane").loc["468>606", "loads"] == 2

    def test_percentiles_are_computed_per_lane(self):
        loads = lanes_mod.add_lane_keys(make_loads(
            [{"carrier_cost": cost} for cost in (1000, 2000, 3000, 4000)]))
        summary = lanes_mod.summarize_lanes(loads).iloc[0]
        assert summary["p25_cost"] == pytest.approx(1750.0)
        assert summary["med_cost"] == pytest.approx(2500.0)

    def test_cumulative_share_reaches_one_hundred(self, lane_table):
        assert lane_table["cum_spend_pct"].max() == pytest.approx(100.0, abs=0.01)

    def test_sorted_by_spend_descending(self, lane_table):
        assert lane_table["spend"].is_monotonic_decreasing


class TestRecoverable:
    def test_computed_per_load_not_from_lane_averages(self):
        """Averaging first hides the lanes worth a phone call.

        A lane with a few very expensive loads and one that is uniformly
        slightly expensive can share an average, and only the first is
        actionable.
        """
        loads = lanes_mod.add_lane_keys(make_loads(
            [{"carrier_cost": cost} for cost in (1000, 1000, 1000, 4000)]))
        summary = lanes_mod.summarize_lanes(loads)
        out = lanes_mod.recoverable_vs_own_p25(loads, summary)

        # p25 is 1000, so only the 4000 load contributes.
        assert out["recoverable"].iloc[0] == pytest.approx(3000.0)

    def test_never_negative(self):
        """Buying below a lane's own p25 is not a liability."""
        loads = lanes_mod.add_lane_keys(make_loads(
            [{"carrier_cost": cost} for cost in (500, 5000, 5000, 5000)]))
        summary = lanes_mod.summarize_lanes(loads)
        out = lanes_mod.recoverable_vs_own_p25(loads, summary)
        assert (out["recoverable"] >= 0).all()

    def test_uniform_lane_recovers_little(self):
        loads = lanes_mod.add_lane_keys(make_loads([{"carrier_cost": 1000.0}] * 8))
        summary = lanes_mod.summarize_lanes(loads)
        out = lanes_mod.recoverable_vs_own_p25(loads, summary)
        assert out["recoverable"].iloc[0] == pytest.approx(0.0)


class TestBenchmarkable:
    def test_requires_both_load_and_carrier_thresholds(self):
        """Ten loads from one carrier is one negotiated rate repeated ten times."""
        summary = pd.DataFrame({
            "lane": ["enough", "thin", "one_carrier"],
            "loads": [20, 3, 20],
            "carriers": [5, 5, 1],
        })
        eligible = lanes_mod.benchmarkable(summary, min_loads=10, min_carriers=3)
        assert eligible["lane"].tolist() == ["enough"]

    def test_boundary_values_are_inclusive(self):
        summary = pd.DataFrame({"lane": ["edge"], "loads": [10], "carriers": [3]})
        assert len(lanes_mod.benchmarkable(summary, min_loads=10, min_carriers=3)) == 1


class TestConcentration:
    def test_counts_lanes_to_the_share_threshold(self):
        summary = pd.DataFrame({"cum_spend_pct": [40.0, 70.0, 85.0, 100.0]})
        assert lanes_mod.lane_concentration(summary, 80.0) == 3

    def test_real_book_is_concentrated(self, lane_table):
        """The property that decides how much market data is worth buying."""
        n80 = lanes_mod.lane_concentration(lane_table, 80.0)
        assert n80 < len(lane_table) * 0.6


class TestCarrierSpread:
    def test_reports_per_carrier_cost_on_one_lane(self):
        loads = lanes_mod.add_lane_keys(make_loads([
            {"carrier_name": "A Co", "carrier_cost": 1000.0},
            {"carrier_name": "A Co", "carrier_cost": 1200.0},
            {"carrier_name": "B Co", "carrier_cost": 2000.0},
        ]))
        spread = lanes_mod.carrier_spread(loads, "468>606")
        assert spread.loc["A Co", "loads"] == 2
        assert spread.loc["B Co", "avg_cost"] == pytest.approx(2000.0)

    def test_percentiles_summary_keys(self):
        result = lanes_mod.price_percentiles(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        assert set(result) == {"p10", "p25", "p50", "p75", "p90"}
