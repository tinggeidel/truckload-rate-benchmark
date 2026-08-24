"""The synthetic generator.

The generator has to be trustworthy in two ways: reproducible, so a result can be
checked, and genuinely dirty, so the pipeline is exercised rather than flattered.
"""
import numpy as np
import pandas as pd
import pytest

from tlbench import coerce, synth


class TestDeterminism:
    def test_same_seed_gives_the_same_book(self):
        a = synth.generate_loads(seed=3, n_loads=500)
        b = synth.generate_loads(seed=3, n_loads=500)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self):
        a = synth.generate_loads(seed=3, n_loads=500)
        b = synth.generate_loads(seed=4, n_loads=500)
        assert not a["carrier_cost"].equals(b["carrier_cost"])


@pytest.fixture(scope="module")
def loads():
    return synth.generate_loads(seed=5, n_loads=6000)


@pytest.fixture(scope="module")
def dirty():
    return synth.dirty_export(synth.generate_loads(seed=5, n_loads=1500), seed=5)


class TestShape:
    def test_has_every_export_column(self, loads):
        for column in ("id", "carrier_name", "carrier_cost", "origin_zip",
                       "dest_zip", "start_date", "mode", "equipment_type"):
            assert column in loads.columns

    def test_truckload_is_a_minority_of_loads_but_most_of_spend(self, loads):
        """The usual shape of a brokerage book, and the reason a bad mode flag
        is expensive: it misroutes most of the money."""
        completed = coerce.completed(loads)
        truckload = completed[completed["true_mode"] == "TL"]

        by_count = len(truckload) / len(completed)
        by_spend = truckload["carrier_cost"].sum() / completed["carrier_cost"].sum()
        assert 0.35 < by_count < 0.50
        assert by_spend > 0.75

    def test_mode_column_is_nearly_useless(self, loads):
        completed = coerce.completed(loads)
        truckload = completed[completed["true_mode"] == "TL"]
        assert (truckload["mode"] == "TL").mean() < 0.10

    def test_open_loads_carry_no_cost(self, loads):
        open_loads = loads[loads["status_description"] != "Completed"]
        assert len(open_loads) > 0
        assert (open_loads["carrier_cost"] == 0).all()

    def test_weight_is_sparse_on_truckload(self, loads):
        """Tempting as a truckload discriminator, and unusable in practice."""
        truckload = loads[loads["true_mode"] == "TL"]
        assert truckload["total_weight"].notna().mean() < 0.20

    def test_some_loads_pick_up_and_deliver_in_the_same_city(self, loads):
        assert (loads["origin_zip"] == loads["dest_zip"]).any()

    def test_rural_destinations_carry_a_real_premium(self, loads):
        """The premium has to be in the data, not just in the docstring --
        otherwise the national-average trap cannot reproduce."""
        premiums = {d[0]: d[4] for d in synth.DESTINATIONS}
        assert premiums["26505"] > 1.5   # Morgantown WV, rural
        assert premiums["60601"] == 1.0  # Chicago, dense


class TestDirtyExport:
    def test_ground_truth_never_reaches_the_export(self, dirty):
        """A real export has no answer key, and the pipeline must not come to
        depend on one."""
        for column in ("true_mode", "true_miles", "true_market_cpm", "dest_premium"):
            assert column not in dirty.columns

    def test_dates_are_numeric_not_timestamps(self, dirty):
        assert pd.api.types.is_numeric_dtype(
            pd.to_numeric(dirty["start_date"], errors="coerce"))

    def test_both_date_encodings_are_present(self, dirty):
        values = pd.to_numeric(dirty["start_date"], errors="coerce").dropna()
        assert (values.between(20_000, 80_000)).any()      # Excel serials
        assert (values > 1e14).any()                        # epoch microseconds

    def test_repeated_header_rows_are_injected(self, dirty):
        assert (dirty["id"].astype(str) == "id").sum() == 6

    def test_some_postal_codes_are_degraded_to_floats(self, dirty):
        assert dirty["dest_zip"].astype(str).str.contains(r"\.0$").any()

    def test_the_pipeline_recovers_everything(self, dirty):
        """End to end: the coercions undo every degradation."""
        cleaned = coerce.coerce_export(dirty)

        assert len(cleaned) == len(dirty) - 6
        assert cleaned["start_date"].notna().all()
        assert not cleaned["dest_zip"].str.contains(r"\.", na=False).any()
        assert cleaned["dest_zip"].str.len().isin([3, 5, 6]).all()

    def test_leading_zeros_survive_the_round_trip(self, dirty):
        """Presque Isle is 04769 and is the case that catches this."""
        cleaned = coerce.coerce_export(dirty)
        if (cleaned["dest_city"] == "Presque Isle").any():
            zips = cleaned.loc[cleaned["dest_city"] == "Presque Isle", "dest_zip"]
            assert (zips == "04769").all()


class TestMarketPull:
    def test_produces_the_columns_the_benchmark_needs(self, lane_table,
                                                      truckload_universe):
        from tlbench import lanes as lanes_mod

        eligible = lanes_mod.benchmarkable(lane_table)
        pull = synth.market_pull(truckload_universe, eligible)

        for column in ("lane", "provider_miles", "market_rate", "market_rate_90d",
                       "provider_reports_total", "provider_reports_ours"):
            assert column in pull.columns

    def test_excludes_cross_border_lanes(self, lane_table, truckload_universe):
        """The provider does not price them reliably, so they are not pulled."""
        from tlbench import lanes as lanes_mod

        eligible = lanes_mod.benchmarkable(lane_table)
        pull = synth.market_pull(truckload_universe, eligible)
        assert not pull["destination"].str.contains(", ON").any()

    def test_current_market_sits_above_the_trailing_year(self, lane_table,
                                                         truckload_universe):
        from tlbench import lanes as lanes_mod

        eligible = lanes_mod.benchmarkable(lane_table)
        pull = synth.market_pull(truckload_universe, eligible)
        assert (pull["market_rate_90d"] > pull["market_rate"]).mean() > 0.9
