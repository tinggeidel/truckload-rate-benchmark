"""Distance estimation from postal centroids.

The export carries no mileage column, which is normal — the TMS stores what the
carrier invoiced, not how far they drove. Great-circle distance between postal
centroids times a circuity factor is the standard stand-in, and stage 09
calibrates it against the market provider's authoritative mileage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from . import config
from .coerce import normalize_postal

EARTH_RADIUS_MILES = 3958.7613


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two arrays of coordinates."""
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


class CentroidResolver(Protocol):
    """Maps normalised postal codes to centroid coordinates."""

    def resolve(self, postal_codes: pd.Index) -> pd.DataFrame:
        """Return a frame indexed by postal code with `lat` and `lon` columns."""


class TableResolver:
    """Resolve from a bundled centroid table.

    Used by the demo and the tests so the whole pipeline runs with no network
    call and no reference-data download.
    """

    def __init__(self, table: pd.DataFrame):
        t = table.copy()
        t["postal_code"] = normalize_postal(t["postal_code"])
        self._lookup = t.set_index("postal_code")[["latitude", "longitude"]]

    @classmethod
    def from_csv(cls, path: Path | None = None) -> "TableResolver":
        path = path or config.REFERENCE / "postal_centroids.csv"
        return cls(pd.read_csv(path, dtype={"postal_code": str}))

    def resolve(self, postal_codes: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=postal_codes, columns=["lat", "lon"], dtype="float64")
        hits = postal_codes.intersection(self._lookup.index)
        if len(hits):
            out.loc[hits, "lat"] = self._lookup.loc[hits, "latitude"].to_numpy()
            out.loc[hits, "lon"] = self._lookup.loc[hits, "longitude"].to_numpy()
        return out


class PgeocodeResolver:
    """Resolve US zips and Canadian FSAs via the `pgeocode` reference dataset.

    Canadian points arrive either as a full postal code or a bare forward
    sortation area; both reduce to the three-character FSA, which is the finest
    granularity the free dataset carries.
    """

    def __init__(self):
        import pgeocode  # imported lazily: the demo path never needs it

        self._us = pgeocode.Nominatim("us")
        self._ca = pgeocode.Nominatim("ca")

    def resolve(self, postal_codes: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=postal_codes, columns=["lat", "lon"], dtype="float64")

        is_us = postal_codes.str.fullmatch(r"\d{5}").fillna(False)
        if is_us.any():
            r = self._us.query_postal_code(list(postal_codes[is_us]))
            out.loc[postal_codes[is_us], "lat"] = r.latitude.values
            out.loc[postal_codes[is_us], "lon"] = r.longitude.values

        is_ca = postal_codes.str.match(r"^[A-Z]\d[A-Z]").fillna(False)
        if is_ca.any():
            fsa = postal_codes[is_ca].str.replace(" ", "", regex=False).str[:3]
            r = self._ca.query_postal_code(list(fsa))
            out.loc[postal_codes[is_ca], "lat"] = r.latitude.values
            out.loc[postal_codes[is_ca], "lon"] = r.longitude.values
        return out


class ChainResolver:
    """Try each resolver in turn, filling gaps left by the one before it."""

    def __init__(self, *resolvers: CentroidResolver):
        self._resolvers = resolvers

    def resolve(self, postal_codes: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=postal_codes, columns=["lat", "lon"], dtype="float64")
        for resolver in self._resolvers:
            missing = out.index[out["lat"].isna()]
            if not len(missing):
                break
            filled = resolver.resolve(missing)
            out.loc[missing, ["lat", "lon"]] = filled[["lat", "lon"]].to_numpy()
        return out


def attach_mileage(
    df: pd.DataFrame,
    resolver: CentroidResolver,
    circuity: float = config.CIRCUITY,
    min_miles: float = config.MIN_MILES,
) -> pd.DataFrame:
    """Attach centroid coordinates, estimated road miles, and cost per mile."""
    df = df.copy()
    df["origin_zip"] = normalize_postal(df["origin_zip"])
    df["dest_zip"] = normalize_postal(df["dest_zip"])

    both = pd.concat([df["origin_zip"], df["dest_zip"]])
    unique = pd.Index(both.dropna().unique())
    centroids = resolver.resolve(unique)

    for side, column in (("o", "origin_zip"), ("d", "dest_zip")):
        df[f"{side}_lat"] = df[column].map(centroids["lat"])
        df[f"{side}_lon"] = df[column].map(centroids["lon"])

    df["gc_miles"] = haversine(df.o_lat, df.o_lon, df.d_lat, df.d_lon)
    df["est_miles"] = (df["gc_miles"] * circuity).round(1)
    df.loc[df["est_miles"] < min_miles, "est_miles"] = min_miles
    df["cost_per_mile"] = (df["carrier_cost"] / df["est_miles"]).round(3)
    return df


def calibration_error(estimated: pd.Series, authoritative: pd.Series) -> pd.Series:
    """Percent error of the estimator against a provider's own mileage."""
    return 100 * (estimated - authoritative) / authoritative
