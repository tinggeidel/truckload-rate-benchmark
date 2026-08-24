"""The truckload / LTL derivation.

The most consequential logic in the repo: every downstream number is computed on
whatever this decides is truckload.
"""
import numpy as np
import pandas as pd
import pytest

from tlbench import classify
from conftest import make_loads


class TestAnchors:
    @pytest.mark.parametrize("name", [
        "ESTES EXPRESS LINES", "Old Dominion Freight Line", "r+l carriers",
        "SAIA MOTOR FREIGHT", "XPO LOGISTICS FREIGHT", "A DUIE PYLE INC",
        "PITT-OHIO EXPRESS", "Pitt Ohio Express",
    ])
    def test_ltl_networks_are_recognised(self, name):
        out = classify.flag_anchors(make_loads([{"carrier_name": name}]))
        assert out["ltl_common_carrier"].iloc[0]

    @pytest.mark.parametrize("name", [
        "Ridgeline Transport LLC", "Copper Creek Carriers Inc", "Blackwater Trucking Co",
    ])
    def test_truckload_carriers_are_not_flagged_as_ltl(self, name):
        out = classify.flag_anchors(make_loads([{"carrier_name": name}]))
        assert not out["ltl_common_carrier"].iloc[0]

    def test_word_boundaries_prevent_false_matches(self):
        """`SAIA` must not match a carrier that merely contains those letters.

        Substring matching here would silently reclassify a truckload carrier as
        an LTL network and remove its loads from the benchmark entirely.
        """
        out = classify.flag_anchors(make_loads([
            {"carrier_name": "ISAIAH TRANSPORT LLC"},
            {"carrier_name": "HOLLANDER FREIGHT SYSTEMS"},
        ]))
        assert not out["ltl_common_carrier"].any()

    @pytest.mark.parametrize("equipment", [
        "Flatbed", "Step deck", "Low boy", "Conestoga", "Tanker", "Power only",
    ])
    def test_truckload_equipment_is_recognised(self, equipment):
        out = classify.flag_anchors(make_loads([{"equipment_type": equipment}]))
        assert out["tl_equipment"].iloc[0]

    def test_standard_van_is_not_an_anchor(self):
        """Dry van is the ambiguous middle -- it proves nothing either way."""
        out = classify.flag_anchors(make_loads([{"equipment_type": "Van - standard"}]))
        assert not out["tl_equipment"].iloc[0]

    def test_missing_carrier_name_does_not_raise(self):
        out = classify.flag_anchors(make_loads([{"carrier_name": None}]))
        assert not out["ltl_common_carrier"].iloc[0]


class TestBoundary:
    def test_cut_rises_with_distance(self):
        boundary = classify.Boundary(intercept=300.0, slope=0.5)
        assert boundary.cut(1000) > boundary.cut(100)
        assert boundary.cut(100) == 350.0

    def test_fit_uses_the_geometric_midpoint(self):
        """Not the arithmetic mean.

        Cost is roughly log-distributed, so an arithmetic midpoint of a $600 LTL
        ceiling and a $2,400 truckload floor sits far too close to the truckload
        side and sweeps heavy LTL into the benchmark.
        """
        separation = pd.DataFrame({
            "band": ["a", "b"], "mid_miles": [200.0, 800.0],
            "ltl_n": [100, 100], "ltl_p90_cost": [600.0, 600.0],
            "tl_n": [100, 100], "tl_p10_cost": [2400.0, 2400.0],
        })
        boundary = classify.fit_boundary(separation)

        geometric = np.sqrt(600.0 * 2400.0)  # 1200
        arithmetic = (600.0 + 2400.0) / 2    # 1500
        assert boundary.cut(200) == pytest.approx(geometric, abs=1.0)
        assert boundary.cut(200) < arithmetic

    def test_thin_bands_are_excluded_from_the_fit(self):
        thin = pd.DataFrame({
            "band": ["a", "b", "c"], "mid_miles": [200.0, 500.0, 800.0],
            "ltl_n": [100, 2, 100], "ltl_p90_cost": [600.0, 5000.0, 600.0],
            "tl_n": [100, 2, 100], "tl_p10_cost": [2400.0, 5000.0, 2400.0],
        })
        boundary = classify.fit_boundary(thin)
        # The outlier band would have bent the line badly had it been included.
        assert boundary.cut(500) == pytest.approx(np.sqrt(600.0 * 2400.0), abs=5.0)

    def test_too_few_usable_bands_raises_a_clear_error(self):
        """Failing loudly beats fitting a line to one point."""
        separation = pd.DataFrame({
            "band": ["a"], "mid_miles": [200.0], "ltl_n": [100],
            "ltl_p90_cost": [600.0], "tl_n": [100], "tl_p10_cost": [2400.0],
        })
        with pytest.raises(ValueError, match="cannot fit a boundary"):
            classify.fit_boundary(separation)

    def test_band_separation_reports_both_anchor_counts(self):
        loads = make_loads(
            [{"carrier_name": "ESTES EXPRESS LINES", "est_miles": 300.0,
              "carrier_cost": 400.0}] * 5
            + [{"equipment_type": "Flatbed", "est_miles": 300.0,
                "carrier_cost": 2000.0}] * 5)
        separation = classify.band_separation(classify.flag_anchors(loads))
        band = separation[separation["band"] == "(250, 500]"].iloc[0]
        assert band["ltl_n"] == 5 and band["tl_n"] == 5


class TestClassify:
    @pytest.fixture
    def boundary(self):
        return classify.Boundary(intercept=350.0, slope=0.45)

    def test_cheap_van_load_is_ltl(self, boundary):
        loads = classify.flag_anchors(make_loads([
            {"carrier_cost": 400.0, "est_miles": 500.0}]))
        out = classify.classify(loads, boundary)
        assert out["mode_derived"].iloc[0] == "LTL"

    def test_expensive_van_load_is_truckload(self, boundary):
        loads = classify.flag_anchors(make_loads([
            {"carrier_cost": 2000.0, "est_miles": 500.0}]))
        out = classify.classify(loads, boundary)
        assert out["mode_derived"].iloc[0] == "TL"
        assert out["mode_basis"].iloc[0] == "cost_vs_miles"

    def test_ltl_network_overrides_an_expensive_price(self, boundary):
        """Carrier identity is evidence; the fitted boundary is inference."""
        loads = classify.flag_anchors(make_loads([
            {"carrier_name": "ESTES EXPRESS LINES", "carrier_cost": 5000.0,
             "est_miles": 500.0}]))
        out = classify.classify(loads, boundary)
        assert out["mode_derived"].iloc[0] == "LTL"
        assert out["mode_basis"].iloc[0] == "ltl_common_carrier"

    def test_truckload_equipment_overrides_a_cheap_price(self, boundary):
        loads = classify.flag_anchors(make_loads([
            {"equipment_type": "Low boy", "carrier_cost": 200.0, "est_miles": 500.0}]))
        out = classify.classify(loads, boundary)
        assert out["mode_derived"].iloc[0] == "TL"
        assert out["mode_basis"].iloc[0] == "tl_equipment"

    def test_missing_mileage_falls_back_to_absolute_cost(self, boundary):
        loads = classify.flag_anchors(make_loads([
            {"carrier_cost": 2000.0, "est_miles": np.nan},
            {"carrier_cost": 300.0, "est_miles": np.nan},
        ]))
        out = classify.classify(loads, boundary)
        assert out["mode_derived"].tolist() == ["TL", "LTL"]
        assert set(out["mode_basis"]) == {"cost_only"}

    def test_the_tms_mode_column_is_never_consulted(self, boundary):
        """The entire point: the flag is not an input at any stage."""
        mislabelled = classify.flag_anchors(make_loads([
            {"mode": "TL", "carrier_cost": 300.0, "est_miles": 500.0},
            {"mode": "LTL", "carrier_cost": 3000.0, "est_miles": 500.0},
        ]))
        out = classify.classify(mislabelled, boundary)
        assert out["mode_derived"].tolist() == ["LTL", "TL"]


class TestConfidence:
    @pytest.fixture
    def boundary(self):
        return classify.Boundary(intercept=350.0, slope=0.45)

    def _grade(self, rows, boundary, **kwargs):
        loads = classify.classify(classify.flag_anchors(make_loads(rows)), boundary)
        return classify.assign_confidence(loads, **kwargs)

    def test_normal_truckload_rate_is_high_confidence(self, boundary):
        out = self._grade([{"carrier_cost": 1500.0, "est_miles": 600.0}], boundary)
        assert out["tl_confidence"].iloc[0] == "high"

    def test_below_the_absolute_floor_is_held_out(self, boundary):
        """$0.90/mile on a 1,000-mile haul is not a full truckload."""
        out = self._grade([{"carrier_cost": 900.0, "est_miles": 1000.0}], boundary)
        assert out["mode_derived"].iloc[0] == "TL"
        assert out["tl_confidence"].iloc[0] == "low"

    def test_equipment_beats_the_floor(self, boundary):
        out = self._grade(
            [{"equipment_type": "Flatbed", "carrier_cost": 500.0, "est_miles": 1000.0}],
            boundary)
        assert out["tl_confidence"].iloc[0] == "high"

    def test_ltl_loads_are_graded_not_applicable(self, boundary):
        out = self._grade([{"carrier_cost": 300.0, "est_miles": 500.0}], boundary)
        assert out["tl_confidence"].iloc[0] == "n/a"

    def test_relative_floor_catches_a_partial_on_a_premium_lane(self, boundary):
        """The case an absolute floor cannot see.

        A lane whose own market clears about $4.00/mile is priced by deadhead. A
        load at $1.90/mile there is obviously a partial, and yet it clears the
        $1.60 absolute floor comfortably. Exactly the lanes the benchmark cares
        most about are where the absolute floor stops protecting it.
        """
        rows = [{"carrier_cost": 1600.0, "est_miles": 400.0,
                 "origin_zip": "46803", "dest_zip": "26505"}] * 12
        rows.append({"carrier_cost": 760.0, "est_miles": 400.0,
                     "origin_zip": "46803", "dest_zip": "26505"})
        out = self._grade(rows, boundary)

        partial = out.iloc[-1]
        assert partial["cost_per_mile"] > 1.60          # clears the absolute floor
        assert partial["tl_confidence"] == "low"        # caught by the lane floor
        assert (out.iloc[:-1]["tl_confidence"] == "high").all()

    def test_relative_floor_is_skipped_on_thin_lanes(self, boundary):
        """A three-load lane has no stable median to gate against."""
        rows = [{"carrier_cost": 1600.0, "est_miles": 400.0}] * 2
        rows.append({"carrier_cost": 800.0, "est_miles": 400.0})
        out = self._grade(rows, boundary)
        assert out.iloc[-1]["tl_confidence"] == "high"

    def test_disabling_the_relative_floor_lets_the_partial_through(self, boundary):
        rows = [{"carrier_cost": 1600.0, "est_miles": 400.0,
                 "origin_zip": "46803", "dest_zip": "26505"}] * 12
        rows.append({"carrier_cost": 760.0, "est_miles": 400.0,
                     "origin_zip": "46803", "dest_zip": "26505"})
        out = self._grade(rows, boundary, relative_fraction=0.0)
        assert out.iloc[-1]["tl_confidence"] == "high"


class TestBenchmarkUniverse:
    def test_keeps_only_high_confidence_truckload_inside_the_window(self, classified):
        universe = classify.benchmark_universe(classified, "2025-04-01", "2026-04-01")

        assert (universe["mode_derived"] == "TL").all()
        assert (universe["tl_confidence"] == "high").all()
        assert universe["est_miles"].notna().all()
        assert pd.to_datetime(universe["start_date"]).max() < pd.Timestamp("2026-04-01")

    def test_window_excludes_partial_months_at_the_end(self, classified):
        """The generated book runs past the window; the tail must be dropped."""
        full = pd.to_datetime(classified["start_date"]).max()
        universe = classify.benchmark_universe(classified, "2025-04-01", "2026-04-01")
        assert full > pd.Timestamp("2026-04-01")
        assert len(universe) < len(classified)


class TestScoring:
    def test_perfect_agreement_scores_one(self):
        df = pd.DataFrame({"mode_derived": ["TL", "LTL"], "true_mode": ["TL", "LTL"]})
        score = classify.score_against_truth(df)
        assert score["accuracy"] == 1.0
        assert score["tl_precision"] == 1.0
        assert score["tl_recall"] == 1.0

    def test_counts_land_in_the_right_cells(self):
        df = pd.DataFrame({
            "mode_derived": ["TL", "TL", "LTL", "LTL"],
            "true_mode": ["TL", "LTL", "TL", "LTL"],
        })
        score = classify.score_against_truth(df)
        assert (score["tl_true_positive"], score["tl_false_positive"],
                score["tl_false_negative"], score["tl_true_negative"]) == (1, 1, 1, 1)
        assert score["tl_precision"] == 0.5
        assert score["tl_recall"] == 0.5

    def test_classifier_beats_the_mode_column_it_replaces(self, classified):
        """The claim the whole stage rests on, measured rather than asserted."""
        truth = classified["true_mode"]
        derived_accuracy = (classified["mode_derived"] == truth).mean()
        flag_accuracy = (classified["mode"].where(
            classified["mode"] == "TL", "LTL") == truth).mean()

        assert derived_accuracy > 0.90
        assert derived_accuracy > flag_accuracy + 0.30

    def test_confidence_gate_raises_purity(self, classified):
        truckload = classified[classified["mode_derived"] == "TL"]
        high = truckload[truckload["tl_confidence"] == "high"]

        purity_all = (truckload["true_mode"] == "TL").mean()
        purity_high = (high["true_mode"] == "TL").mean()
        assert purity_high > purity_all

    def test_confidence_gate_holds_out_mostly_ltl(self, classified):
        """If it were discarding real truckload, it would be costing coverage
        for nothing."""
        truckload = classified[classified["mode_derived"] == "TL"]
        held = truckload[truckload["tl_confidence"] == "low"]
        if len(held):
            assert (held["true_mode"] == "LTL").mean() > 0.75
