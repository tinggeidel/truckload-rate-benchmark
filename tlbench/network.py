"""Carrier-network health on the truckload book.

Everything here answers a sourcing question using only the shipment file -- no
accounting system, no carrier scorecard, no survey. That constraint is the point:
these are the questions a broker can answer today with what the TMS already has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USAGE_BINS = [0, 1, 2, 5, 20, 100, 10 ** 6]
USAGE_LABELS = ["1 load", "2", "3-5", "6-20", "21-100", "100+"]


def monthly_network(df: pd.DataFrame) -> pd.DataFrame:
    """Carrier count, arrivals, and departures by month.

    A network that adds and loses the same number of carriers every month is not
    growing, it is churning, and the two look identical if only the headcount is
    tracked.
    """
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["ym"] = df["start_date"].dt.to_period("M")

    first_seen = df.groupby("carrier_name")["start_date"].min().dt.to_period("M")
    last_seen = df.groupby("carrier_name")["start_date"].max().dt.to_period("M")

    rows = []
    for month, group in df.groupby("ym"):
        carriers = set(group["carrier_name"])
        rows.append({
            "month": str(month),
            "loads": len(group),
            "active_carriers": len(carriers),
            "new": sum(first_seen[c] == month for c in carriers),
            "last_time_seen": sum(last_seen[c] == month for c in carriers),
            "median_cpm": group["cost_per_mile"].median(),
            "spend": group["carrier_cost"].sum(),
        })

    out = pd.DataFrame(rows)
    out["churn_pct"] = 100 * out["last_time_seen"] / out["active_carriers"]
    return out


def usage_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """How spend distributes across one-off carriers versus repeat partners."""
    usage = df.groupby("carrier_name").agg(
        loads=("id", "size"), spend=("carrier_cost", "sum"))
    usage["bucket"] = pd.cut(usage["loads"], bins=USAGE_BINS, labels=USAGE_LABELS)

    out = usage.groupby("bucket", observed=True).agg(
        carriers=("loads", "size"), loads=("loads", "sum"), spend=("spend", "sum"))
    out["pct_carriers"] = 100 * out["carriers"] / out["carriers"].sum()
    out["pct_spend"] = 100 * out["spend"] / out["spend"].sum()
    return out


def first_load_premium(df: pd.DataFrame, min_lane_loads: int = 20) -> pd.DataFrame:
    """Does a carrier's first load with us price above that lane's norm?

    Compared within lane rather than across the book, because a carrier's first
    load is not a random draw -- new carriers get called on the hard freight, and
    an unconditional average would measure lane mix instead of onboarding cost.
    """
    df = df.sort_values("start_date").copy()
    df["seq"] = df.groupby("carrier_name").cumcount() + 1

    lane_median = df.groupby("lane")["cost_per_mile"].transform("median")
    lane_count = df.groupby("lane")["cost_per_mile"].transform("size")
    df["cpm_vs_lane_pct"] = 100 * (df["cost_per_mile"] - lane_median) / lane_median

    dense = df[lane_count >= min_lane_loads]
    seq_bin = pd.cut(dense["seq"], [0, 1, 2, 5, 20, 10 ** 6],
                     labels=["1st load", "2nd", "3-5", "6-20", "21+"])
    return dense.groupby(seq_bin, observed=True).agg(
        loads=("id", "size"),
        median_vs_lane_pct=("cpm_vs_lane_pct", "median"),
        mean_vs_lane_pct=("cpm_vs_lane_pct", "mean"),
    )


def carrier_lifespan(df: pd.DataFrame, active_within_days: int = 60) -> pd.DataFrame:
    """First load, last load, and whether the carrier is still with us."""
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])

    life = df.groupby("carrier_name").agg(
        loads=("id", "size"), spend=("carrier_cost", "sum"),
        first=("start_date", "min"), last=("start_date", "max"))
    life["days_active"] = (life["last"] - life["first"]).dt.days
    cutoff = df["start_date"].max() - pd.Timedelta(days=active_within_days)
    life["still_active"] = life["last"] >= cutoff
    life["bucket"] = pd.cut(life["loads"], bins=USAGE_BINS, labels=USAGE_LABELS)
    return life


def matched_lane_trend(df: pd.DataFrame, split_date: str,
                       min_loads_each_side: int = 10) -> dict:
    """Price trend on a fixed basket of lanes present in both halves.

    A raw average cost per load moves whenever the mix moves, so a book that
    simply hauled further this quarter reads as a rate increase. Holding the lane
    basket fixed removes that, and it is the difference between reporting a rate
    change and reporting a mix change.
    """
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    before = df[df["start_date"] < split_date]
    after = df[df["start_date"] >= split_date]

    a = before.groupby("lane").agg(n_a=("id", "size"), cost_a=("carrier_cost", "mean"))
    b = after.groupby("lane").agg(n_b=("id", "size"), cost_b=("carrier_cost", "mean"))
    matched = a.join(b, how="inner")
    matched = matched[(matched["n_a"] >= min_loads_each_side)
                      & (matched["n_b"] >= min_loads_each_side)]
    if matched.empty:
        return {"lanes": 0, "loads": 0, "before": float("nan"),
                "after": float("nan"), "change_pct": float("nan")}

    weight = matched["n_a"] + matched["n_b"]
    before_avg = float((matched["cost_a"] * weight).sum() / weight.sum())
    after_avg = float((matched["cost_b"] * weight).sum() / weight.sum())
    return {
        "lanes": int(len(matched)),
        "loads": int(weight.sum()),
        "before": before_avg,
        "after": after_avg,
        "change_pct": 100 * (after_avg - before_avg) / before_avg,
    }
