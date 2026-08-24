"""Carrier-network health."""
import pandas as pd
import pytest

from tlbench import network


def book(rows: list[dict]) -> pd.DataFrame:
    defaults = {"id": 0, "carrier_name": "A Co", "carrier_cost": 1000.0,
                "cost_per_mile": 2.0, "lane": "468>606",
                "start_date": pd.Timestamp("2025-04-01")}
    records = []
    for index, row in enumerate(rows):
        record = dict(defaults)
        record["id"] = index + 1
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


class TestMonthlyNetwork:
    def test_counts_arrivals_and_departures(self):
        df = book([
            {"carrier_name": "A", "start_date": pd.Timestamp("2025-04-05")},
            {"carrier_name": "B", "start_date": pd.Timestamp("2025-04-20")},
            {"carrier_name": "A", "start_date": pd.Timestamp("2025-05-05")},
            {"carrier_name": "C", "start_date": pd.Timestamp("2025-05-10")},
        ])
        monthly = network.monthly_network(df).set_index("month")

        assert monthly.loc["2025-04", "new"] == 2
        assert monthly.loc["2025-04", "last_time_seen"] == 1   # B never returns
        assert monthly.loc["2025-05", "active_carriers"] == 2

    def test_churn_is_a_percentage_of_active_carriers(self):
        df = book([
            {"carrier_name": "A", "start_date": pd.Timestamp("2025-04-05")},
            {"carrier_name": "B", "start_date": pd.Timestamp("2025-04-06")},
        ])
        monthly = network.monthly_network(df)
        assert monthly["churn_pct"].iloc[0] == pytest.approx(100.0)


class TestUsageConcentration:
    def test_buckets_carriers_by_load_count(self):
        df = book(
            [{"carrier_name": "once"}]
            + [{"carrier_name": "twice"}] * 2
            + [{"carrier_name": "many"}] * 30)
        out = network.usage_concentration(df)

        assert out.loc["1 load", "carriers"] == 1
        assert out.loc["2", "carriers"] == 1
        assert out.loc["21-100", "carriers"] == 1

    def test_shares_sum_to_one_hundred(self):
        df = book([{"carrier_name": f"C{i % 7}"} for i in range(40)])
        out = network.usage_concentration(df)
        assert out["pct_spend"].sum() == pytest.approx(100.0)


class TestFirstLoadPremium:
    def test_measured_within_lane(self):
        """New carriers get called on the hard freight, so an unconditional
        average would measure lane mix and report it as onboarding cost."""
        rows = [{"carrier_name": "incumbent", "cost_per_mile": 2.0,
                 "start_date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=d)}
                for d in range(25)]
        rows.append({"carrier_name": "newcomer", "cost_per_mile": 2.4,
                     "start_date": pd.Timestamp("2025-06-01")})
        out = network.first_load_premium(book(rows), min_lane_loads=20)

        assert out.loc["1st load", "median_vs_lane_pct"] > 0

    def test_thin_lanes_are_excluded(self):
        out = network.first_load_premium(book([{"carrier_name": "A"}] * 3),
                                         min_lane_loads=20)
        assert out.empty or out["loads"].sum() == 0


class TestLifespan:
    def test_reports_first_last_and_span(self):
        df = book([
            {"carrier_name": "A", "start_date": pd.Timestamp("2025-04-01")},
            {"carrier_name": "A", "start_date": pd.Timestamp("2025-07-01")},
        ])
        life = network.carrier_lifespan(df)
        assert life.loc["A", "days_active"] == 91

    def test_still_active_uses_the_books_own_end_date(self):
        """Not today's date -- the analysis has to be reproducible next month."""
        df = book([
            {"carrier_name": "recent", "start_date": pd.Timestamp("2026-03-01")},
            {"carrier_name": "gone", "start_date": pd.Timestamp("2025-04-01")},
        ])
        life = network.carrier_lifespan(df, active_within_days=60)
        assert life.loc["recent", "still_active"]
        assert not life.loc["gone", "still_active"]


class TestMatchedLaneTrend:
    def test_holds_the_lane_basket_fixed(self):
        """A book that simply hauled further this half must not read as a rate
        increase."""
        rows = []
        for i in range(12):
            rows.append({"lane": "short", "carrier_cost": 1000.0,
                         "start_date": pd.Timestamp("2025-04-01")})
            rows.append({"lane": "short", "carrier_cost": 1100.0,
                         "start_date": pd.Timestamp("2025-11-01")})
        # A long lane that only appears in the second half; it must be excluded.
        for i in range(12):
            rows.append({"lane": "long", "carrier_cost": 9000.0,
                         "start_date": pd.Timestamp("2025-11-01")})

        trend = network.matched_lane_trend(book(rows), "2025-10-01")
        assert trend["lanes"] == 1
        assert trend["change_pct"] == pytest.approx(10.0, abs=0.1)

    def test_no_matched_lanes_returns_empty_result(self):
        rows = [{"lane": "a", "start_date": pd.Timestamp("2025-04-01")}] * 12
        trend = network.matched_lane_trend(book(rows), "2025-10-01")
        assert trend["lanes"] == 0

    def test_requires_volume_on_both_sides(self):
        rows = [{"lane": "a", "start_date": pd.Timestamp("2025-04-01")}] * 12
        rows += [{"lane": "a", "start_date": pd.Timestamp("2025-11-01")}] * 2
        assert network.matched_lane_trend(
            book(rows), "2025-10-01", min_loads_each_side=10)["lanes"] == 0
