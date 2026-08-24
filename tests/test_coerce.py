"""Coercions for a messy export.

Every case here corresponds to something that silently corrupted a real read.
"""
import numpy as np
import pandas as pd
import pytest

from tlbench import coerce


class TestDateParsing:
    def test_excel_serial_round_trips(self):
        serial = pd.Series([45748.0])  # 2025-04-01
        assert coerce.to_datetime(serial).iloc[0] == pd.Timestamp("2025-04-01")

    def test_epoch_microseconds_parse(self):
        target = pd.Timestamp("2025-06-15")
        micros = pd.Series([target.value / 1000.0])
        assert coerce.to_datetime(micros).iloc[0] == target

    def test_mixed_encodings_in_one_column(self):
        """The failure that motivates the whole function.

        A single export carries both encodings, because some rows were last
        touched by the UI and others by an API sync. Parsing the column with one
        unit destroys whichever group does not match.
        """
        target = pd.Timestamp("2025-06-15")
        mixed = pd.Series([45748.0, target.value / 1000.0, 45900.0])
        parsed = coerce.to_datetime(mixed)

        assert parsed.notna().all()
        assert parsed.iloc[0] == pd.Timestamp("2025-04-01")
        assert parsed.iloc[1] == target
        assert parsed.iloc[2] == pd.Timestamp("2025-08-31")

    @pytest.mark.parametrize("unit,divisor", [
        ("s", 1_000_000_000), ("ms", 1_000_000), ("us", 1_000), ("ns", 1),
    ])
    def test_every_epoch_resolution(self, unit, divisor):
        target = pd.Timestamp("2025-09-09")
        value = pd.Series([target.value / divisor])
        assert coerce.to_datetime(value).iloc[0] == target

    def test_unparseable_becomes_nat_not_an_exception(self):
        """Bad dates must not abort the run; they have to be counted."""
        parsed = coerce.to_datetime(pd.Series(["", None, "not a date"]))
        assert parsed.isna().all()

    def test_real_date_strings_still_parse(self):
        parsed = coerce.to_datetime(pd.Series(["2025-04-01", "2025-12-25"]))
        assert parsed.iloc[0] == pd.Timestamp("2025-04-01")
        assert parsed.iloc[1] == pd.Timestamp("2025-12-25")


class TestPostalNormalisation:
    def test_leading_zero_restored_after_spreadsheet_round_trip(self):
        """`04769.0` must come back as `04769`, not `4769`.

        Left alone this drops every New England load out of the benchmark
        without it ever appearing in a count.
        """
        assert coerce.normalize_postal(pd.Series(["4769.0"])).iloc[0] == "04769"

    def test_short_zips_are_padded(self):
        result = coerce.normalize_postal(pd.Series(["7004", "601"]))
        assert list(result) == ["07004", "00601"]

    def test_canadian_postal_codes_lose_the_space(self):
        result = coerce.normalize_postal(pd.Series(["n1h 4g8", "L5T 2R7"]))
        assert list(result) == ["N1H4G8", "L5T2R7"]

    def test_bare_canadian_fsa_survives(self):
        assert coerce.normalize_postal(pd.Series(["n1h"])).iloc[0] == "N1H"

    def test_postal_key_is_three_characters(self):
        keys = coerce.postal_key(pd.Series(["04769", "N1H 4G8", "60601"]))
        assert list(keys) == ["047", "N1H", "606"]

    def test_whitespace_and_case_normalised(self):
        assert coerce.normalize_postal(pd.Series([" n1h 4g8 "])).iloc[0] == "N1H4G8"


class TestRepeatedHeaders:
    def test_header_rows_inside_the_data_block_are_dropped(self):
        df = pd.DataFrame({"id": ["1", "2", "id", "3"], "carrier_cost": [1, 2, "carrier_cost", 3]})
        cleaned = coerce.drop_repeated_headers(df)

        assert len(cleaned) == 3
        assert cleaned["id"].tolist() == [1, 2, 3]

    def test_id_column_ends_up_integer(self):
        df = pd.DataFrame({"id": ["10", "20"]})
        assert coerce.drop_repeated_headers(df)["id"].dtype == np.int64

    def test_a_repeated_header_would_otherwise_poison_a_sum(self):
        """Why this matters: the banner row survives read_excel as data."""
        df = pd.DataFrame({"id": ["1", "id", "2"],
                           "carrier_cost": ["100", "carrier_cost", "200"]})
        cleaned = coerce.drop_repeated_headers(df)
        assert pd.to_numeric(cleaned["carrier_cost"]).sum() == 300


class TestCompletedFilter:
    def test_only_delivered_loads_with_cost_survive(self):
        df = pd.DataFrame({
            "status_description": ["Completed", "Cancelled", "Completed", "In Transit"],
            "carrier_cost": [1000.0, 0.0, 0.0, 500.0],
        })
        assert len(coerce.completed(df)) == 1

    def test_zero_cost_completed_loads_are_excluded(self):
        """They are usually unbilled, and they drag every average toward zero."""
        df = pd.DataFrame({"status_description": ["Completed"], "carrier_cost": [0.0]})
        assert coerce.completed(df).empty


class TestCoerceExport:
    def test_applies_every_coercion_together(self):
        df = pd.DataFrame({
            "id": ["1", "id", "2"],
            " carrier_cost ": ["1000", "carrier_cost", "2000"],
            "start_date": [45748.0, np.nan, 45749.0],
            "origin_zip": ["4769.0", "origin_zip", "60601"],
            "dest_zip": ["7004", "dest_zip", "N1H 4G8"],
        })
        out = coerce.coerce_export(df)

        assert len(out) == 2
        assert "carrier_cost" in out.columns  # header whitespace stripped
        assert out["start_date"].iloc[0] == pd.Timestamp("2025-04-01")
        assert out["origin_zip"].tolist() == ["04769", "60601"]
        assert out["dest_zip"].tolist() == ["07004", "N1H4G8"]

    def test_missing_optional_columns_are_tolerated(self):
        """A real export does not always carry every column."""
        out = coerce.coerce_export(pd.DataFrame({"id": ["1"], "carrier_cost": ["5"]}))
        assert len(out) == 1
