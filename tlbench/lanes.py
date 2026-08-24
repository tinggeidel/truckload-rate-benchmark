"""Lane aggregation and the internal benchmark.

Before any external market data is bought, the book benchmarks against itself.
On a lane where we buy repeatedly from several different carriers, the 25th
percentile is not a theoretical target -- it is a price we have already paid, on
this lane, for this freight, recently. The gap between what we actually paid and
that price is recoverable without renegotiating anything.

This is deliberately the first benchmark rather than the last. It needs no
subscription, it cannot be wrong about the market, and if it finds nothing then
the expensive external pull is unlikely to find much either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .coerce import postal_key


def add_lane_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the lane key and a human-readable lane name.

    Three-digit postal prefixes are the right granularity: five-digit zips
    fragment a real lane into dozens of singletons, and state-to-state buckets
    average across markets that price nothing alike.
    """
    df = df.copy()
    df["o3"] = postal_key(df["origin_zip"])
    df["d3"] = postal_key(df["dest_zip"])
    df["lane"] = df["o3"] + ">" + df["d3"]
    df["lane_name"] = (df["origin_city"].str.title() + ", " + df["origin_state"]
                       + " > " + df["dest_city"].str.title() + ", " + df["dest_state"])
    return df


def _mode_or_blank(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if len(m) else ""


def summarize_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the truckload universe to one row per lane."""
    grouped = df.groupby("lane")
    lanes = grouped.agg(
        loads=("id", "size"),
        spend=("carrier_cost", "sum"),
        avg_cost=("carrier_cost", "mean"),
        p25_cost=("carrier_cost", lambda s: s.quantile(0.25)),
        med_cost=("carrier_cost", "median"),
        avg_cpm=("cost_per_mile", "mean"),
        p25_cpm=("cost_per_mile", lambda s: s.quantile(0.25)),
        miles=("est_miles", "median"),
        carriers=("carrier_name", "nunique"),
        customers=("customer_name", "nunique"),
        top_city=("lane_name", _mode_or_blank),
        o_state=("origin_state", _mode_or_blank),
        d_state=("dest_state", _mode_or_blank),
    ).reset_index()

    lanes = lanes.sort_values("spend", ascending=False)
    lanes["cum_spend_pct"] = 100 * lanes["spend"].cumsum() / lanes["spend"].sum()
    return lanes


def recoverable_vs_own_p25(df: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
    """Dollars above each lane's own 25th percentile, summed load by load.

    Computed per load rather than from lane averages. Averaging first would let a
    lane with a handful of very expensive loads look the same as one that is
    uniformly slightly expensive, and only the first is worth a phone call.
    """
    merged = df.merge(lanes[["lane", "p25_cost"]], on="lane", how="left")
    merged["over_p25"] = (merged["carrier_cost"] - merged["p25_cost"]).clip(lower=0)
    recoverable = merged.groupby("lane")["over_p25"].sum().rename("recoverable")
    return lanes.merge(recoverable, on="lane", how="left")


def benchmarkable(lanes: pd.DataFrame,
                  min_loads: int = config.MIN_LANE_LOADS,
                  min_carriers: int = config.MIN_LANE_CARRIERS) -> pd.DataFrame:
    """Lanes with enough repeat volume and enough carriers to have a real p25.

    Both thresholds matter. Ten loads from one carrier is one negotiated rate
    repeated ten times, and its 25th percentile is not evidence of anything.
    """
    return lanes[(lanes["loads"] >= min_loads)
                 & (lanes["carriers"] >= min_carriers)].copy()


def lane_concentration(lanes: pd.DataFrame, share: float = 80.0) -> int:
    """How many lanes carry the given share of spend.

    Freight books are almost always far more concentrated than they feel from
    the inside, and this number decides how much external data is worth buying.
    """
    return int((lanes["cum_spend_pct"] <= share).sum()) + 1


def carrier_spread(df: pd.DataFrame, lane: str) -> pd.DataFrame:
    """Per-carrier cost on a single lane.

    A lane with seventy carriers is either healthy competition or an absence of
    routing discipline, and the price spread is what tells them apart.
    """
    subset = df[df["lane"] == lane]
    return subset.groupby("carrier_name").agg(
        loads=("id", "size"),
        avg_cost=("carrier_cost", "mean"),
        avg_cpm=("cost_per_mile", "mean"),
        spend=("carrier_cost", "sum"),
    ).sort_values("loads", ascending=False)


def price_percentiles(series: pd.Series) -> dict:
    """The spread summary used when reporting a single lane."""
    return {f"p{int(q * 100)}": float(series.quantile(q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
