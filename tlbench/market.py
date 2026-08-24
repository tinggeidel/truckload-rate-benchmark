"""Benchmarking the book against market rates -- and the trap that ruins it.

This module exists mostly to encode one expensive lesson.

The first pass of the real engagement benchmarked the book against published
*national* spot averages, because those are free and immediate. It returned a
headline of roughly +21% above market and about $1.5M of annual overpay. That
number was wrong, and it was wrong in the most dangerous way available: it was
large, it was plausible, and it pointed at a team that had done nothing wrong.

The error is a mix problem. A national average is the mean of a lane population
that looks nothing like any one brokerage's book. This book was concentrated
into rural, low-backhaul destinations -- places a carrier reaches loaded and
leaves empty. That deadhead is priced into the rate, legitimately, and it is
enormous: one lane into a rural destination cleared $4.61/mile against a $2.41
national average. Measured nationally that lane reads as 127% overpay. Measured
against its own lane market it was about 5% above, which is noise.

Re-run against lane-level market data, the same book came out roughly 2% *below*
market, and the finding inverted completely: sourcing was fine, and the margin
problem was on the sell side. `national_benchmark` is kept here, and run by the
demo, so that inversion is reproducible rather than merely asserted.

Two further provider behaviours are handled here because both silently corrupt
an aggregate:

  match rate     the provider reports how many of *our* loads it actually saw on
                 a lane. Well under our real load count means the comparison is
                 unmeasured, not merely uncertain.
  market pairs   the provider silently aggregates thin lanes up to a market-to-
                 market average, so two distinct lanes come back with identical
                 figures. Summed without deduping, that double-counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def national_benchmark(lanes: pd.DataFrame, national_rate_per_mile: float,
                       min_miles: float = config.NATIONAL_MIN_MILES) -> pd.DataFrame:
    """Benchmark every lane against one national average rate per mile.

    Kept deliberately: this is the method that produced the wrong answer, and the
    demo runs it to show how far wrong. Published national averages cover hauls
    at or above `min_miles`, so shorter lanes are excluded rather than compared
    against a number that does not describe them.

    Do not use this to make a sourcing decision. See the module docstring.
    """
    out = lanes[lanes["miles"] >= min_miles].copy()
    out["national_rate"] = national_rate_per_mile * out["miles"]
    out["var_per_load"] = out["avg_cost"] - out["national_rate"]
    out["var_pct"] = 100 * out["var_per_load"] / out["national_rate"]
    out["annual_var"] = out["var_per_load"] * out["loads"]
    return out.sort_values("annual_var", ascending=False)


def grade_match_rate(market: pd.DataFrame,
                     min_match_rate: float = config.MIN_MATCH_RATE) -> pd.DataFrame:
    """Grade each lane on how much of our own volume the provider actually saw.

    The provider's "your reported rate" figure is built from whatever subset of
    our loads reached its panel. When that subset is small, the comparison
    describes a handful of loads and gets presented with the same confidence as
    one built from hundreds.

    In the real pull, one lane matched 7 of our 63 loads and read as 12% above
    market. It was not 12% above market; it was unmeasured. Acting on it would
    have meant repricing a lane on the strength of seven observations.
    """
    out = market.copy()
    out["match_rate"] = out["provider_reports_ours"] / out["loads"]
    out["measured"] = out["match_rate"] >= min_match_rate
    out["grade"] = np.where(out["measured"], "measured", "unmeasured")
    return out


def dedupe_market_pairs(market: pd.DataFrame) -> pd.DataFrame:
    """Collapse lanes the provider silently aggregated to the same market pair.

    When a lane is too thin to price on its own, the provider quietly answers
    with the market-to-market average instead, at the same granularity of
    presentation. Two different origin-destination pairs then come back with
    byte-identical rate figures. Summing both double-counts that market's
    contribution to the total.

    Detection is on the returned figures rather than on any granularity flag,
    since the flag is not always present, and identical rate, report count, and
    mileage across two distinct lanes is not a coincidence.
    """
    signature = ["market_rate", "market_rate_90d", "provider_reports_total", "provider_miles"]
    present = [c for c in signature if c in market.columns]
    out = market.copy()
    out["market_pair_dupe"] = out.duplicated(subset=present, keep="first")
    return out


def lane_benchmark(lanes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join lane-level market rates to our own lane costs and size the gap.

    The provider's mileage is authoritative where present and replaces the
    centroid estimate, which matters more than it sounds: a 5% mileage error
    moves every derived rate per mile by 5% in the same direction.
    """
    merged = market.merge(
        lanes[["lane", "loads", "spend", "avg_cost", "med_cost", "miles", "carriers"]],
        on="lane", how="left",
    )
    merged["use_miles"] = merged["provider_miles"].fillna(merged["miles"])
    merged["our_cpm"] = merged["avg_cost"] / merged["use_miles"]
    merged["market_cpm"] = merged["market_rate"] / merged["use_miles"]
    merged["var_per_load"] = merged["avg_cost"] - merged["market_rate"]
    merged["var_pct"] = 100 * merged["var_per_load"] / merged["market_rate"]
    merged["annual_var"] = merged["var_per_load"] * merged["loads"]

    if "market_rate_90d" in merged.columns:
        merged["var_vs_90d"] = merged["avg_cost"] - merged["market_rate_90d"]
        merged["annual_var_90d"] = merged["var_vs_90d"] * merged["loads"]
    return merged.sort_values("spend", ascending=False)


def variance_summary(benchmarked: pd.DataFrame, rate_column: str = "market_rate",
                     cost_column: str = "avg_cost") -> dict:
    """Load-weighted variance of the book against the market.

    Weighted by loads, never a mean of per-lane percentages. An unweighted mean
    lets a 40-load lane and a 4-load lane vote equally, which on a concentrated
    book is how a rounding error on the long tail overturns the headline.
    """
    df = benchmarked.dropna(subset=[rate_column, cost_column, "loads"])
    if df.empty:
        return {"lanes": 0, "loads": 0, "spend": 0.0, "at_market": 0.0,
                "variance": 0.0, "variance_pct": float("nan")}

    actual = float((df[cost_column] * df["loads"]).sum())
    at_market = float((df[rate_column] * df["loads"]).sum())
    return {
        "lanes": int(len(df)),
        "loads": int(df["loads"].sum()),
        "spend": actual,
        "at_market": at_market,
        "variance": actual - at_market,
        "variance_pct": 100 * (actual - at_market) / at_market if at_market else float("nan"),
        "lanes_above": int((df[cost_column] > df[rate_column]).sum()),
        "lanes_below": int((df[cost_column] < df[rate_column]).sum()),
    }


def robustness_cuts(benchmarked: pd.DataFrame) -> pd.DataFrame:
    """Re-run the headline under progressively stricter subsets.

    A finding that survives every cut is a finding. One that only holds on the
    full set is an artifact of whatever the weakest lanes contributed, and the
    real engagement's first headline failed exactly this test.
    """
    cuts = {
        "all verified lanes": benchmarked,
        "market-pair deduped": benchmarked[~benchmarked.get(
            "market_pair_dupe", pd.Series(False, index=benchmarked.index))],
        "measured only": benchmarked[benchmarked.get(
            "measured", pd.Series(True, index=benchmarked.index))],
    }
    rows = []
    for name, subset in cuts.items():
        summary = variance_summary(subset)
        summary["cut"] = name
        rows.append(summary)
    columns = ["cut", "lanes", "loads", "spend", "at_market", "variance", "variance_pct"]
    return pd.DataFrame(rows)[columns]


def sell_side_view(df: pd.DataFrame, benchmarked: pd.DataFrame,
                   recent_months: int = 4, collapse_points: float = 6.0,
                   recent_margin_floor: float = 10.0) -> pd.DataFrame:
    """Put buy-side variance and the *trajectory* of gross margin on one row.

    This is the join that reframed the whole engagement, and the trajectory is
    the part that matters. A committed lane is repriced once and then left, while
    the buy side keeps tracking the market every week. In a rising market the
    margin bleeds away a point at a time, and none of the individual months looks
    alarming enough to escalate.

    Whole-window margin hides this completely -- it averages a healthy first half
    against an underwater second half and reports something unremarkable. So the
    comparison here is the recent window against everything before it, and the
    lanes worth flagging are the ones that buy at or below market and are still
    losing margin: no carrier negotiation can fix those, because sourcing is not
    what is wrong with them.
    """
    df = df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    cutoff = df["start_date"].max() - pd.DateOffset(months=recent_months)

    def _aggregate(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        out = frame.groupby("lane").agg(**{
            f"revenue{suffix}": ("revenue", "sum"),
            f"cost{suffix}": ("carrier_cost", "sum"),
            f"loads_actual{suffix}": ("id", "size"),
        })
        out[f"margin_pct{suffix}"] = 100 * (
            out[f"revenue{suffix}"] - out[f"cost{suffix}"]
        ) / out[f"revenue{suffix}"].replace(0, np.nan)
        return out

    overall = _aggregate(df, "")
    overall["gross_profit"] = overall["revenue"] - overall["cost"]
    earlier = _aggregate(df[df["start_date"] < cutoff], "_earlier")
    recent = _aggregate(df[df["start_date"] >= cutoff], "_recent")

    margin = overall.join(earlier, how="left").join(recent, how="left").reset_index()
    margin["margin_change_pts"] = margin["margin_pct_recent"] - margin["margin_pct_earlier"]

    out = benchmarked.merge(margin, on="lane", how="left")
    out["buys_below_market"] = out["var_per_load"] <= 0
    # Sourced at or below market, and the margin is going the wrong way or has
    # already gone. Either condition alone is common; together they are the
    # signature of a sell rate nobody has revisited.
    out["sell_side_problem"] = out["buys_below_market"] & (
        (out["margin_change_pts"] <= -collapse_points)
        | (out["margin_pct_recent"] < recent_margin_floor)
    )
    return out.sort_values("margin_change_pts")
