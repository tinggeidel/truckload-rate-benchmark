"""Derive a trustworthy truckload / LTL flag.

The TMS `mode` column cannot be used. In the engagement this came from, it called
a carrier running seven-figure truckload volume "LTL" on 96% of its loads, and
the reverse error was just as common: a "TL-Any" bucket that swept up miscoded
partials pricing under a dollar a mile on 2,000-mile hauls. Mode is entered by
whoever built the load, and nothing downstream ever validates it.

So the label is derived instead, from two anchor sets that can be trusted on
their own terms, plus a boundary learned from where those two sets separate:

  anchor LTL   loads tendered to a common carrier that operates a terminal
               network. Those companies do not haul full truckloads for a
               brokerage, so carrier identity alone settles the mode.
  anchor TL    loads on equipment only a truckload carrier provides. Nobody
               moves an LTL shipment on a lowboy or a Conestoga.

Between the anchors sits the ambiguous middle -- dry van on a carrier that runs
both. Those are separated on (cost, miles), which is what physically
distinguishes the modes: LTL is priced off class and weight and saturates with
distance, truckload is priced off distance and capacity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Common carriers operating LTL terminal networks. This is public knowledge about
# the carrier base rather than anything derived from one brokerage's book, so it
# ships with the repo as reference data.
LTL_COMMON_CARRIERS = [
    r"R\+ ?L CARRIERS", r"CENTRAL TRANSPORT", r"DAYTON FREIGHT", r"PITT[- ]OHIO",
    r"ABF FREIGHT", r"WARD TRUCKING", r"ESTES EXPRESS", r"OLD DOMINION",
    r"\bSAIA\b", r"\bXPO\b", r"TFORCE", r"SOUTHEASTERN FREIGHT", r"AAA COOPER",
    r"\bAVERITT\b", r"FEDEX FREIGHT", r"UPS FREIGHT", r"\bYRC\b", r"\bYELLOW\b",
    r"ROADRUNNER", r"A DUIE PYLE", r"NEW PENN", r"STANDARD FORWARDING",
    r"CROSS COUNTRY FREIGHT", r"OAK HARBOR", r"DEPENDABLE HIGHWAY",
    r"PENINSULA TRUCK", r"MIDWEST MOTOR EXPRESS", r"CLEAR LANE", r"MAGNUM LTL",
    r"\bUSF\b", r"\bHOLLAND\b", r"NEW ENGLAND MOTOR", r"DAYLIGHT TRANSPORT",
    r"WILSON TRUCKING", r"\bDOHRN\b", r"SUNBELT FURNITURE",
]

LTL_CARRIER_RE = re.compile("|".join(LTL_COMMON_CARRIERS))

# Equipment no LTL shipment moves on.
TRUCKLOAD_EQUIPMENT = frozenset({
    "Flatbed", "Step deck", "Stepdeck Conestoga", "Conestoga", "Low boy",
    "Double drop (low boy)", "Tanker", "Power only", "Auto transport",
    "Auto carrier", "Hotshot", "Hotshot Flatbed", "Container - standard",
    "Van - sprinter",
})

# Distance bands used to locate the boundary. Truckload economics change shape
# across these; a single global cost cut would misclassify both ends.
DISTANCE_BANDS = [0, 100, 250, 500, 750, 1000, 1500, 3500]

# A load called truckload that prices below these floors is not plausibly a full
# truckload at any point in the last decade's rate cycle. Held out rather than
# forced to a side -- see `assign_confidence`.
TRUCKLOAD_FLOORS = [(0, 250, 2.20), (250, 500, 1.60), (500, np.inf, 1.20)]

# An absolute floor is necessary but not sufficient, and the gap is worth being
# explicit about because it is invisible until the book is concentrated.
#
# The floors above are calibrated to national rate levels. On a lane whose own
# market clears $4.00/mile -- a rural, low-backhaul destination -- a partial
# moving at $2.00/mile is unmistakably not a full truckload, and yet it clears
# the $1.60 floor comfortably. Exactly the lanes the benchmark cares most about
# are the ones where the absolute floor stops protecting it.
#
# So a load must also clear a fraction of its own lane's median rate per mile.
# The median is used rather than the mean because the population being measured
# is the contaminated one; a median tolerates the contamination it is being used
# to detect, and a mean does not.
RELATIVE_FLOOR_FRACTION = 0.62

# Below this many derived-truckload loads, a lane's own median is too unstable
# to gate against and only the absolute floor applies.
MIN_LANE_LOADS_FOR_RELATIVE_FLOOR = 8

# Fitting the boundary needs enough of both anchor sets inside a band for that
# band's percentiles to mean anything.
MIN_ANCHOR_PER_BAND = 30

# When mileage is unresolvable there is no boundary to compare against, so the
# only available signal is absolute cost.
COST_ONLY_CUT = 1200.0


@dataclass(frozen=True)
class Boundary:
    """A distance-dependent cost cut separating LTL from truckload."""

    intercept: float
    slope: float

    def cut(self, miles):
        """The cost above which a load of this distance prices like truckload."""
        return self.intercept + self.slope * miles

    def describe(self) -> str:
        return f"cost_cut = {self.intercept:,.0f} + {self.slope:,.3f} * miles"


def flag_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Mark the two trustworthy anchor sets."""
    df = df.copy()
    names = df["carrier_name"].fillna("").str.upper()
    df["ltl_common_carrier"] = names.str.contains(LTL_CARRIER_RE, regex=True)
    df["tl_equipment"] = df["equipment_type"].isin(TRUCKLOAD_EQUIPMENT)
    return df


def band_separation(df: pd.DataFrame, bands=DISTANCE_BANDS) -> pd.DataFrame:
    """Per distance band, locate the LTL ceiling and the truckload floor.

    The gap between the LTL 90th percentile and the truckload 10th percentile is
    the corridor the fitted boundary has to run through.
    """
    labelled = pd.cut(df["est_miles"], bands)
    ltl = df[df["ltl_common_carrier"]]
    tl = df[df["tl_equipment"]]

    rows = []
    for band in labelled.cat.categories:
        a = ltl[pd.cut(ltl["est_miles"], bands) == band]
        t = tl[pd.cut(tl["est_miles"], bands) == band]
        rows.append({
            "band": str(band),
            "mid_miles": float(np.mean([band.left, band.right])),
            "ltl_n": len(a),
            "ltl_p90_cost": a["carrier_cost"].quantile(0.90) if len(a) else np.nan,
            "tl_n": len(t),
            "tl_p10_cost": t["carrier_cost"].quantile(0.10) if len(t) else np.nan,
        })
    return pd.DataFrame(rows)


def fit_boundary(separation: pd.DataFrame,
                 min_anchor: int = MIN_ANCHOR_PER_BAND) -> Boundary:
    """Fit the cost cut as a line through the per-band anchor corridor.

    The per-band midpoint is taken geometrically rather than arithmetically. Cost
    is roughly log-distributed, so an arithmetic mean of a $600 LTL ceiling and a
    $2,400 truckload floor sits far too close to the truckload side and would
    sweep heavy LTL into the benchmark.
    """
    usable = separation.dropna(subset=["ltl_p90_cost", "tl_p10_cost"])
    usable = usable[(usable["ltl_n"] >= min_anchor) & (usable["tl_n"] >= min_anchor)]
    if len(usable) < 2:
        raise ValueError(
            f"only {len(usable)} distance bands carry >= {min_anchor} loads in both "
            "anchor sets; cannot fit a boundary. Widen the window or lower "
            "MIN_ANCHOR_PER_BAND."
        )

    midpoint = np.sqrt(usable["ltl_p90_cost"] * usable["tl_p10_cost"])
    slope, intercept = np.polyfit(usable["mid_miles"], midpoint, 1)
    return Boundary(intercept=float(intercept), slope=float(slope))


def classify(df: pd.DataFrame, boundary: Boundary,
             cost_only_cut: float = COST_ONLY_CUT) -> pd.DataFrame:
    """Assign `mode_derived` and record which rule decided it.

    Recording the basis matters as much as the label: it is what lets a reader
    see that the benchmark rests mostly on carrier identity and equipment rather
    than on the fitted boundary, and it makes the fitted portion auditable on its
    own.
    """
    df = df.copy()
    cut = boundary.cut(df["est_miles"])

    mode = pd.Series("LTL", index=df.index, dtype="object")
    basis = pd.Series("cost_vs_miles", index=df.index, dtype="object")

    no_miles = df["est_miles"].isna()
    if no_miles.any():
        mode[no_miles] = np.where(
            df.loc[no_miles, "carrier_cost"] >= cost_only_cut, "TL", "LTL")
        basis[no_miles] = "cost_only"

    priced_as_tl = ~no_miles & (df["carrier_cost"] >= cut)
    mode[priced_as_tl] = "TL"

    # Anchors override the fitted boundary: they are evidence, not inference.
    mode[df["tl_equipment"]] = "TL"
    basis[df["tl_equipment"]] = "tl_equipment"
    mode[df["ltl_common_carrier"]] = "LTL"
    basis[df["ltl_common_carrier"]] = "ltl_common_carrier"

    df["cost_cut"] = cut
    df["mode_derived"] = mode
    df["mode_basis"] = basis
    return df


def assign_confidence(df: pd.DataFrame, floors=TRUCKLOAD_FLOORS,
                      relative_fraction: float = RELATIVE_FLOOR_FRACTION,
                      min_lane_loads: int = MIN_LANE_LOADS_FOR_RELATIVE_FLOOR
                      ) -> pd.DataFrame:
    """Grade each derived-truckload load on whether its price is physically possible.

    A load labelled truckload that prices below any plausible truckload floor is
    far more likely a heavy LTL, a partial, or a volume move that the boundary
    swept up. Holding it out costs a little coverage; leaving it in would drag
    every lane average down and manufacture a finding that we buy below market.
    That second failure is much worse than the first, so this fails toward
    exclusion.

    Two floors apply, and a load must clear both. The absolute floor catches
    freight that is cheap by any standard. The relative floor catches freight
    that is only cheap *for its lane*, which on a premium lane is the same thing
    and is otherwise invisible -- see `RELATIVE_FLOOR_FRACTION`.
    """
    df = df.copy()
    absolute_floor = pd.Series(np.nan, index=df.index, dtype="float64")
    for lo, hi, value in floors:
        absolute_floor[(df["est_miles"] > lo) & (df["est_miles"] <= hi)] = value

    is_tl = df["mode_derived"] == "TL"

    # The lane's own median rate per mile, taken over derived-truckload loads
    # only. Lanes too thin to have a stable median get no relative floor rather
    # than an unreliable one.
    lane_key = (df["origin_zip"].astype(str).str[:3] + ">"
                + df["dest_zip"].astype(str).str[:3])
    tl_only = df["cost_per_mile"].where(is_tl)
    lane_median = tl_only.groupby(lane_key).transform("median")
    lane_count = tl_only.groupby(lane_key).transform("count")
    relative_floor = (lane_median * relative_fraction).where(
        lane_count >= min_lane_loads)

    clears_absolute = df["cost_per_mile"] >= absolute_floor
    clears_relative = ~(df["cost_per_mile"] < relative_floor)  # NaN floor passes

    confidence = pd.Series("n/a", index=df.index, dtype="object")
    confidence[is_tl] = np.where(
        (clears_absolute & clears_relative)[is_tl], "high", "low")
    # No mileage means no floor to test against.
    confidence[is_tl & df["est_miles"].isna()] = "low"
    # Equipment is decisive regardless of price.
    confidence[is_tl & df["tl_equipment"]] = "high"

    df["abs_floor"] = absolute_floor
    df["lane_rel_floor"] = relative_floor
    df["tl_confidence"] = confidence
    return df


def derive_mode(df: pd.DataFrame) -> tuple[pd.DataFrame, Boundary]:
    """Run the whole derivation: anchors, boundary fit, labels, confidence."""
    flagged = flag_anchors(df)
    boundary = fit_boundary(band_separation(flagged))
    labelled = classify(flagged, boundary)
    return assign_confidence(labelled), boundary


def benchmark_universe(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """The subset a rate benchmark is allowed to run on.

    High-confidence truckload only, mileage resolved, inside the complete-months
    window. A partial month at either end biases the average toward whatever mix
    happened to move in it.
    """
    out = df[(df["mode_derived"] == "TL")
             & (df["tl_confidence"] == "high")
             & df["est_miles"].notna()].copy()
    dates = pd.to_datetime(out["start_date"])
    return out[(dates >= start) & (dates < end)].copy()


def score_against_truth(df: pd.DataFrame, truth_column: str = "true_mode") -> dict:
    """Score the derived label against a known ground truth.

    Only the synthetic dataset carries a truth column; a real export has none,
    which is the whole reason the derivation exists. Being able to measure the
    classifier at all is what separates this from an untested heuristic.
    """
    truth = df[truth_column]
    derived = df["mode_derived"]
    tp = int(((derived == "TL") & (truth == "TL")).sum())
    fp = int(((derived == "TL") & (truth == "LTL")).sum())
    fn = int(((derived == "LTL") & (truth == "TL")).sum())
    tn = int(((derived == "LTL") & (truth == "LTL")).sum())

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else float("nan"))
    return {
        "n": len(df),
        "tl_true_positive": tp, "tl_false_positive": fp,
        "tl_false_negative": fn, "tl_true_negative": tn,
        "accuracy": (tp + tn) / len(df) if len(df) else float("nan"),
        "tl_precision": precision,
        "tl_recall": recall,
        "tl_f1": f1,
    }
