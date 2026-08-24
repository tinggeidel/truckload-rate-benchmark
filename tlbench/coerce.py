"""Coercions for a messy TMS export.

Every function here exists because a real export broke a naive read. They are
kept separate from the pipeline so each failure mode has a test pinning it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Excel's day-zero. Excel's 1900 leap-year bug means the usable epoch is Dec 30, 1899.
EXCEL_EPOCH = pd.Timestamp("1899-12-30")

# Excel serials in a plausible shipping era: 1954 .. 2119.
_SERIAL_LO, _SERIAL_HI = 20_000, 80_000

# Epoch integers arrive at whichever resolution the exporting layer happened to use.
_EPOCH_SCALES = ((1e18, "ns"), (1e15, "us"), (1e12, "ms"), (1e9, "s"))

DATE_COLUMNS = [
    "start_date", "end_date", "created_at", "updated_at",
    "invoice_date", "actual_pickup_arrival", "actual_delivery_arrival",
]

NUMERIC_COLUMNS = [
    "revenue", "carrier_cost", "gross_profit", "gross_profit_pct",
    "total_weight", "total_pieces", "total_handling_units",
    "customer_id", "carrier_id",
]


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse a date column that mixes Excel serials with epoch integers.

    A single export can carry both: most rows are Excel serials (~45,000) while
    rows touched by an API sync carry epoch microseconds (~1.7e15). Feeding the
    whole column to `pd.to_datetime` with one unit silently destroys one group,
    so each scale is converted on its own terms and the rest is left as NaT.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NaT, index=numeric.index, dtype="datetime64[ns]")

    serial = numeric.between(_SERIAL_LO, _SERIAL_HI)
    if serial.any():
        out[serial] = EXCEL_EPOCH + pd.to_timedelta(numeric[serial], unit="D")

    for scale, unit in _EPOCH_SCALES:
        mask = numeric.ge(scale) & numeric.lt(scale * 1000) & out.isna()
        if mask.any():
            out[mask] = pd.to_datetime(numeric[mask], unit=unit, errors="coerce")

    # Anything that was already a real date string rather than a number.
    remaining = out.isna() & numeric.isna() & series.notna()
    if remaining.any():
        out[remaining] = pd.to_datetime(series[remaining], errors="coerce")
    return out


def normalize_postal(series: pd.Series) -> pd.Series:
    """Normalise a mixed US/Canadian postal column to a comparable string.

    Spreadsheet round-trips turn US zips into floats, so `07004` becomes `7004.0`
    and would geocode to the wrong state or not at all. Canadian codes arrive
    with and without the internal space.
    """
    s = (series.astype("string").str.strip().str.upper()
         .str.replace(r"\.0+$", "", regex=True))

    us = s.str.fullmatch(r"\d{1,5}").fillna(False)
    s = s.mask(us, s.str.zfill(5))

    canadian = s.str.match(r"^[A-Z]\d[A-Z]").fillna(False)
    s = s.mask(canadian, s.str.replace(" ", "", regex=False))
    return s


def postal_key(series: pd.Series) -> pd.Series:
    """The 3-character grouping key: US 3-digit zip, Canadian FSA."""
    return normalize_postal(series).str[:3]


def drop_repeated_headers(df: pd.DataFrame, id_column: str = "id") -> pd.DataFrame:
    """Drop banner or repeated header rows that landed inside the data block.

    Multi-sheet exports concatenated by the TMS repeat their header every N rows.
    Those rows survive `read_excel` as data and poison any numeric aggregate.
    """
    keep = df[id_column].astype("string").str.fullmatch(r"\d+").fillna(False)
    out = df[keep].copy()
    out[id_column] = out[id_column].astype("int64")
    return out


def coerce_export(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every column coercion the raw export needs, in order."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Repeated header rows go first. Left in place, their literal column names
    # reach every parser below as unparseable strings, and the date parser in
    # particular then falls back to a slow per-element path just to reject them.
    if "id" in df.columns:
        df = drop_repeated_headers(df)

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = to_datetime(df[column])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("origin_zip", "dest_zip"):
        if column in df.columns:
            df[column] = normalize_postal(df[column])

    return df


def completed(df: pd.DataFrame) -> pd.DataFrame:
    """The billable universe: delivered loads that actually cost us something.

    Cancelled and in-flight loads carry partial or zero cost and would drag every
    average down if they were left in.
    """
    return df[(df["status_description"] == "Completed") & (df["carrier_cost"] > 0)].copy()
