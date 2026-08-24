"""Distance estimation."""
import numpy as np
import pandas as pd
import pytest

from tlbench import config, geo
from conftest import make_loads


class TestHaversine:
    def test_known_distance(self):
        """Chicago to Saint Louis is about 260 great-circle miles."""
        miles = geo.haversine(41.8858, -87.6181, 38.6346, -90.1913)
        assert 255 < miles < 265

    def test_zero_distance(self):
        assert geo.haversine(41.0, -85.0, 41.0, -85.0) == pytest.approx(0.0)

    def test_symmetric(self):
        there = geo.haversine(41.0, -85.0, 33.0, -84.0)
        back = geo.haversine(33.0, -84.0, 41.0, -85.0)
        assert there == pytest.approx(back)

    def test_vectorised(self):
        result = geo.haversine(np.array([41.0, 33.0]), np.array([-85.0, -84.0]),
                               np.array([38.0, 34.0]), np.array([-90.0, -85.0]))
        assert result.shape == (2,)


class TestTableResolver:
    def test_resolves_bundled_codes(self, resolver):
        out = resolver.resolve(pd.Index(["46803", "60601"]))
        assert out.loc["46803", "lat"] == pytest.approx(41.0695, abs=0.01)
        assert out["lat"].notna().all()

    def test_unknown_code_returns_nan_rather_than_raising(self, resolver):
        out = resolver.resolve(pd.Index(["99999"]))
        assert out["lat"].isna().all()

    def test_normalises_codes_on_load(self, resolver):
        """A degraded zip must still resolve once normalised."""
        out = resolver.resolve(pd.Index(["04769"]))
        assert out["lat"].notna().all()


class TestChainResolver:
    def test_second_resolver_fills_the_first_one_gaps(self):
        primary = geo.TableResolver(pd.DataFrame({
            "postal_code": ["11111"], "latitude": [40.0], "longitude": [-80.0]}))
        secondary = geo.TableResolver(pd.DataFrame({
            "postal_code": ["22222"], "latitude": [35.0], "longitude": [-90.0]}))

        out = geo.ChainResolver(primary, secondary).resolve(pd.Index(["11111", "22222"]))
        assert out.loc["11111", "lat"] == 40.0
        assert out.loc["22222", "lat"] == 35.0

    def test_first_resolver_wins(self):
        primary = geo.TableResolver(pd.DataFrame({
            "postal_code": ["11111"], "latitude": [40.0], "longitude": [-80.0]}))
        secondary = geo.TableResolver(pd.DataFrame({
            "postal_code": ["11111"], "latitude": [99.0], "longitude": [-99.0]}))
        out = geo.ChainResolver(primary, secondary).resolve(pd.Index(["11111"]))
        assert out.loc["11111", "lat"] == 40.0


class TestAttachMileage:
    def test_adds_distance_and_rate_per_mile(self, resolver):
        loads = make_loads([{"origin_zip": "46803", "dest_zip": "60601",
                             "carrier_cost": 1000.0}])
        out = geo.attach_mileage(loads, resolver)

        assert out["est_miles"].iloc[0] > 100
        assert out["cost_per_mile"].iloc[0] == pytest.approx(
            1000.0 / out["est_miles"].iloc[0], abs=0.01)

    def test_circuity_is_applied(self, resolver):
        loads = make_loads([{"origin_zip": "46803", "dest_zip": "60601"}])
        out = geo.attach_mileage(loads, resolver, circuity=1.17)
        assert out["est_miles"].iloc[0] == pytest.approx(
            out["gc_miles"].iloc[0] * 1.17, abs=0.2)

    def test_same_city_move_is_floored(self, resolver):
        """Origin equals destination on a real share of loads.

        Without a floor the rate per mile is infinite and every aggregate that
        touches the lane becomes NaN.
        """
        loads = make_loads([{"origin_zip": "46803", "dest_zip": "46803",
                             "carrier_cost": 400.0}])
        out = geo.attach_mileage(loads, resolver)

        assert out["est_miles"].iloc[0] == config.MIN_MILES
        assert np.isfinite(out["cost_per_mile"].iloc[0])

    def test_unresolvable_postal_yields_nan_miles_not_a_crash(self, resolver):
        loads = make_loads([{"origin_zip": "99999", "dest_zip": "60601"}])
        out = geo.attach_mileage(loads, resolver)
        assert out["est_miles"].isna().all()

    def test_degraded_zip_is_normalised_before_lookup(self, resolver):
        """`4769.0` must resolve, because stage 03 sees exports in that state."""
        loads = make_loads([{"origin_zip": "46803", "dest_zip": "4769.0"}])
        out = geo.attach_mileage(loads, resolver)
        assert out["est_miles"].notna().all()


class TestCalibration:
    def test_positive_error_means_the_estimate_runs_long(self):
        error = geo.calibration_error(pd.Series([110.0]), pd.Series([100.0]))
        assert error.iloc[0] == pytest.approx(10.0)

    def test_negative_error_means_the_estimate_runs_short(self):
        error = geo.calibration_error(pd.Series([95.0]), pd.Series([100.0]))
        assert error.iloc[0] == pytest.approx(-5.0)
