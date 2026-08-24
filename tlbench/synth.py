"""Generate a synthetic TMS export with known ground truth.

The engagement this pipeline came from ran on a brokerage's shipment history and
a licensed market-rate subscription. Neither can be published, so the repo ships
a generator instead. It is not a fixture dump -- it models the two things that
make the analysis hard, so the pipeline is exercised rather than merely executed:

1. **The export is dirty in specific, real ways.** Dates arrive as Excel serials
   with a minority in epoch microseconds. Header rows repeat inside the data
   block. Zips lose their leading zeros to a spreadsheet round-trip. The `mode`
   column is close to useless. Weight is almost never populated. Some loads
   pick up and deliver in the same city. Every one of these broke a naive read of
   the real file, and each has a test pinning the fix.

2. **The book is concentrated into rural, low-backhaul destinations.** This is
   what makes the national-average benchmark produce a confidently wrong answer,
   and reproducing it is the point of `stages/09_market_benchmark.py`. The
   generator gives those destinations a real, legitimate rate premium, then
   prices the book at a small discount to its own true lane market. A national
   average -- which knows nothing about the premium -- reads that same book as
   badly overpaying.

Ground truth is carried in `true_mode`, which lets the derived classifier be
scored rather than asserted. A real export has no such column; that absence is
the entire reason the derivation in `classify.py` exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Geography. Real places and real postal codes so distances are honest, chosen
# to have nothing to do with any actual brokerage's book.
# ---------------------------------------------------------------------------

# Shipper hubs: (zip, city, state, sampling weight)
ORIGINS = [
    ("46803", "Fort Wayne", "IN", 34),
    ("65803", "Springfield", "MO", 20),
    ("38801", "Tupelo", "MS", 14),
    ("44903", "Mansfield", "OH", 13),
    ("54301", "Green Bay", "WI", 12),
    ("98901", "Yakima", "WA", 7),
]

# Destinations: (zip, city, state, weight, market premium).
#
# The premium is the multiplier a carrier legitimately charges above a
# distance-equivalent lane into a dense market. It is deadhead: the driver
# arrives loaded and leaves empty, and prices the empty half into the rate.
# Major markets sit at 1.0 by definition -- they are what the national average
# is mostly made of.
DESTINATIONS = [
    # Dense markets, no premium.
    ("60601", "Chicago", "IL", 9, 1.00),
    ("75201", "Dallas", "TX", 6, 1.00),
    ("30303", "Atlanta", "GA", 5, 1.02),
    ("43215", "Columbus", "OH", 6, 1.00),
    ("63101", "Saint Louis", "MO", 5, 1.01),
    ("85003", "Phoenix", "AZ", 2, 1.04),
    ("28202", "Charlotte", "NC", 4, 1.03),
    ("19102", "Philadelphia", "PA", 3, 1.06),
    ("84101", "Salt Lake City", "UT", 1, 1.05),
    ("90021", "Los Angeles", "CA", 2, 1.02),
    # Rural, low-backhaul, and mostly a medium-haul run from the hubs above.
    # The book is deliberately concentrated here: that concentration is what
    # makes a national average produce a confidently wrong answer.
    ("26505", "Morgantown", "WV", 14, 1.90),
    ("49829", "Escanaba", "MI", 16, 1.86),
    ("24210", "Abingdon", "VA", 13, 1.84),
    ("56649", "International Falls", "MN", 8, 1.72),
    ("71601", "Pine Bluff", "AR", 7, 1.52),
    ("04769", "Presque Isle", "ME", 3, 1.62),
    ("81101", "Alamosa", "CO", 2, 1.45),
    ("57701", "Rapid City", "SD", 2, 1.40),
    ("59501", "Havre", "MT", 1, 1.55),
    ("89801", "Elko", "NV", 1, 1.30),
    # Cross-border points, excluded from the market pull later.
    ("L5T", "Mississauga", "ON", 2, 1.20),
    ("N1H", "Guelph", "ON", 2, 1.20),
]

# National published spot average, dollars per mile, all-in. Deliberately a
# single number, because that is exactly what makes it wrong for a book shaped
# like this one: it is the mean of a lane population that looks nothing like
# these lanes.
NATIONAL_SPOT_PER_MILE = 2.30

# Carriers that run LTL terminal networks. Public companies, named because the
# classifier keys on carrier identity and the demo has to exercise that path.
LTL_CARRIERS = [
    "ESTES EXPRESS LINES", "OLD DOMINION FREIGHT LINE", "SAIA MOTOR FREIGHT",
    "XPO LOGISTICS FREIGHT", "R+L CARRIERS", "DAYTON FREIGHT LINES",
    "PITT-OHIO EXPRESS", "ABF FREIGHT SYSTEM", "AAA COOPER TRANSPORTATION",
    "SOUTHEASTERN FREIGHT LINES", "TFORCE FREIGHT", "AVERITT EXPRESS",
    "CENTRAL TRANSPORT LLC", "A DUIE PYLE INC", "WARD TRUCKING CORP",
    "MIDWEST MOTOR EXPRESS", "FEDEX FREIGHT EAST",
]

# Invented truckload carriers.
_TL_PREFIX = [
    "Ridgeline", "Copper Creek", "Blackwater", "Northbound", "Silver Birch",
    "Two Rivers", "Harvest Road", "Ironwood", "Pale Horse", "Cedar Gap",
    "Kettle Point", "Long Meadow", "Redstone", "Fair Weather", "Quarry Hill",
    "Windrow", "Stone Arch", "Tallgrass", "Bright Fork", "Cold Spring",
]
_TL_SUFFIX = ["Transport LLC", "Carriers Inc", "Trucking Co", "Freight Lines",
              "Logistics LLC", "Express Inc", "Hauling LLC", "Transit Group"]

TL_EQUIPMENT_POOL = [
    ("Van - standard", 0.72), ("Reefer", 0.09), ("Flatbed", 0.10),
    ("Step deck", 0.04), ("Conestoga", 0.02), ("Power only", 0.02),
    ("Low boy", 0.01),
]

CUSTOMER_PREFIX = [
    "Allanby", "Brightmoor", "Calder", "Dunhaven", "Elmridge", "Fairbank",
    "Glenmara", "Holloway", "Ivanhoe", "Jessup", "Kirkfield", "Lansdowne",
    "Meriden", "Northgate", "Oakhurst", "Pemberton", "Quilliam", "Ravensworth",
    "Stanbridge", "Thornbury",
]
CUSTOMER_SUFFIX = ["Industries", "Manufacturing", "Metals", "Components",
                   "Products Co", "Fabrication", "Materials", "Works"]


def _weighted_choice(rng, items, weights, size):
    p = np.asarray(weights, dtype=float)
    return rng.choice(len(items), size=size, p=p / p.sum())


def _haversine_miles(lat1, lon1, lat2, lon2):
    r1, r2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 3958.7613 * 2 * np.arcsin(np.sqrt(a))


def _truckload_cpm(miles):
    """Base truckload rate per mile into a dense market, before any premium.

    Falls with distance -- the fixed cost of dispatching, loading, and detaining
    a truck spreads over more miles -- but toward a floor, not toward zero. The
    floor is the driver, the fuel, and the equipment, and no length of haul
    makes those cheaper per mile.
    """
    return 1.62 + 235.0 / np.clip(miles, 60, None)


def _month_index(dates, start):
    """A rising market with seasonal shape -- about 16% across the full window.

    The trend is what makes the sell side interesting. A committed rate agreed at
    the start of the window is underwater by the end of it, and the lanes where
    that hurts most are the largest ones.
    """
    months = ((dates.dt.year - start.year) * 12 + dates.dt.month - start.month).to_numpy()
    trend = 1.0 + 0.0105 * months
    seasonal = 1.0 + 0.035 * np.sin((months + 2) / 12 * 2 * np.pi)
    return trend * seasonal


def generate_loads(seed: int = 7, n_loads: int = 20000,
                   start: str = "2025-04-01", end: str = "2026-06-30",
                   centroids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a clean, fully-labelled load table. Messiness is added separately."""
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    if centroids is None:
        from . import config
        centroids = pd.read_csv(config.REFERENCE / "postal_centroids.csv",
                                dtype={"postal_code": str})
    coords = centroids.set_index("postal_code")[["latitude", "longitude"]]

    o_idx = _weighted_choice(rng, ORIGINS, [o[3] for o in ORIGINS], n_loads)
    d_idx = _weighted_choice(rng, DESTINATIONS, [d[3] for d in DESTINATIONS], n_loads)

    origin_zip = np.array([ORIGINS[i][0] for i in o_idx])
    origin_city = np.array([ORIGINS[i][1] for i in o_idx])
    origin_state = np.array([ORIGINS[i][2] for i in o_idx])
    dest_zip = np.array([DESTINATIONS[i][0] for i in d_idx])
    dest_city = np.array([DESTINATIONS[i][1] for i in d_idx])
    dest_state = np.array([DESTINATIONS[i][2] for i in d_idx])
    premium = np.array([DESTINATIONS[i][4] for i in d_idx])

    # About 5% of the book picks up and delivers in the same city -- local
    # cartage and yard moves that the TMS records like any other load.
    local = rng.random(n_loads) < 0.05
    dest_zip[local] = origin_zip[local]
    dest_city[local] = origin_city[local]
    dest_state[local] = origin_state[local]
    premium[local] = 1.0

    o_lat = coords.loc[origin_zip, "latitude"].to_numpy()
    o_lon = coords.loc[origin_zip, "longitude"].to_numpy()
    d_lat = coords.loc[dest_zip, "latitude"].to_numpy()
    d_lon = coords.loc[dest_zip, "longitude"].to_numpy()
    true_miles = _haversine_miles(o_lat, o_lon, d_lat, d_lon) * 1.19
    true_miles = np.clip(true_miles, 12, None)

    span_days = (end_ts - start_ts).days
    offsets = rng.integers(0, span_days, n_loads)
    pickup = pd.to_datetime(start_ts) + pd.to_timedelta(offsets, unit="D")
    market_index = _month_index(pd.Series(pickup), start_ts)

    # Mode. Truckload is the minority of loads but the large majority of spend,
    # which is the usual shape of a brokerage book and the reason a bad mode flag
    # is so costly: it misroutes most of the money.
    is_tl = rng.random(n_loads) < 0.43

    # Partials and volume LTL: genuinely not truckload, but heavy enough to price
    # in the ambiguous middle and brokered to truckload carriers rather than to
    # an LTL network. This is the population that defeats both anchors.
    partial = ~is_tl & (rng.random(n_loads) < 0.12)

    # --- truckload pricing ---------------------------------------------------
    true_market_cpm = _truckload_cpm(true_miles) * premium * market_index
    # The book buys slightly below its own lane market. Lognormal noise, because
    # rates are multiplicative and cannot go negative.
    edge = rng.normal(-0.02, 0.115, n_loads)
    tl_cost = true_market_cpm * true_miles * np.exp(edge)

    # --- LTL pricing ---------------------------------------------------------
    # Priced off class and weight, saturating with distance rather than scaling.
    ltl_weight = rng.lognormal(7.4, 0.75, n_loads).clip(120, 14000)
    ltl_cost = (68 + 0.052 * ltl_weight * (1 + 0.55 * np.log1p(true_miles / 260))
                * market_index * np.exp(rng.normal(0, 0.22, n_loads)))
    ltl_cost = ltl_cost.clip(95, 3200)
    # A partial fills half a trailer and is priced closer to a discounted
    # truckload than to a parcel-style LTL tariff.
    ltl_weight[partial] = rng.lognormal(9.5, 0.3, partial.sum()).clip(6000, 32000)
    ltl_cost[partial] = (true_market_cpm[partial] * true_miles[partial]
                         * rng.uniform(0.34, 0.72, partial.sum()))

    carrier_cost = np.where(is_tl, tl_cost, ltl_cost).round(2)

    # --- carriers ------------------------------------------------------------
    tl_names = np.array([f"{p} {s}" for p in _TL_PREFIX for s in _TL_SUFFIX])
    # A long tail of one-and-done carriers plus a core that repeats: sample a
    # skewed index so a handful of names carry most of the loads.
    tl_pick = np.floor(rng.power(0.45, n_loads) * len(tl_names)).astype(int)
    tl_pick = np.clip(tl_pick, 0, len(tl_names) - 1)
    ltl_pick = rng.integers(0, len(LTL_CARRIERS), n_loads)
    carrier_name = np.where(is_tl, tl_names[tl_pick],
                            np.array(LTL_CARRIERS)[ltl_pick])

    # Partials go to truckload carriers, so the carrier-name anchor cannot see
    # them. Left unchecked they price like cheap truckload and drag every lane
    # average down, which is what the confidence floor in
    # `classify.assign_confidence` exists to prevent.
    carrier_name[partial] = tl_names[tl_pick][partial]

    # --- equipment -----------------------------------------------------------
    eq_names = [e[0] for e in TL_EQUIPMENT_POOL]
    eq_probs = np.array([e[1] for e in TL_EQUIPMENT_POOL])
    eq_pick = rng.choice(len(eq_names), size=n_loads, p=eq_probs / eq_probs.sum())
    equipment = np.where(is_tl, np.array(eq_names)[eq_pick], "Van - standard")

    # --- the unreliable mode column -----------------------------------------
    # This is the headline data-quality failure. The flag is entered at load
    # build and never validated, so it is right about 2% of the time on the
    # freight that matters and confidently wrong the rest.
    mode = np.full(n_loads, "LTL", dtype=object)
    mode[is_tl] = np.where(rng.random(is_tl.sum()) < 0.05, "TL", "LTL")
    swept = ~is_tl & (rng.random(n_loads) < 0.08)
    mode[swept] = "TL-Any"

    customers = np.array([f"{p} {s}" for p in CUSTOMER_PREFIX for s in CUSTOMER_SUFFIX])
    cust_pick = np.floor(rng.power(0.6, n_loads) * len(customers)).astype(int)
    cust_pick = np.clip(cust_pick, 0, len(customers) - 1)
    customer_name = customers[cust_pick]

    # --- sell side -----------------------------------------------------------
    # Spot freight is priced off what it costs today, so margin is roughly
    # stable. Committed lanes are not: the sell rate is agreed once and then
    # held, while the buy side keeps tracking the market. When the market rises
    # across the window, margin on exactly the largest lanes compresses toward
    # zero and then through it.
    #
    # This is modelled because it is the finding the whole engagement turned on.
    # The lanes that looked worst on a sourcing benchmark were sourced fine; what
    # was wrong with them was a sell rate nobody had revisited in eight months.
    committed = is_tl & np.isin(dest_zip, ["26505", "49829", "24210"]) \
        & (origin_zip == "46803")
    # The rate was agreed at the market level prevailing when the window opened,
    # and has not moved since.
    index_at_agreement = float(_month_index(pd.Series([start_ts]), start_ts)[0])
    agreed_cpm = _truckload_cpm(true_miles) * premium * index_at_agreement
    sell = np.where(
        committed,
        agreed_cpm * true_miles * 1.19 * np.exp(rng.normal(0, 0.03, n_loads)),
        carrier_cost * (1 + rng.normal(0.155, 0.085, n_loads)),
    )
    revenue = np.round(sell, 2)

    df = pd.DataFrame({
        "id": np.arange(100000, 100000 + n_loads),
        "custom_id": [f"L{n:07d}" for n in range(1, n_loads + 1)],
        "status_code": 200,
        "status_description": "Completed",
        "phase_key": "delivered",
        "phase_value": "Delivered",
        "customer_id": cust_pick + 5000,
        "customer_name": customer_name,
        "carrier_id": rng.integers(20000, 99999, n_loads),
        "carrier_name": carrier_name,
        "mode": mode,
        "equipment_type": equipment,
        "origin": [f"{c}, {s}" for c, s in zip(origin_city, origin_state)],
        "destination": [f"{c}, {s}" for c, s in zip(dest_city, dest_state)],
        "revenue": revenue,
        "carrier_cost": carrier_cost,
        "gross_profit": (revenue - carrier_cost).round(2),
        "gross_profit_pct": (100 * (revenue - carrier_cost) / revenue).round(2),
        "start_date": pickup,
        "end_date": pickup + pd.to_timedelta(rng.integers(1, 5, n_loads), unit="D"),
        "created_at": pickup - pd.to_timedelta(rng.integers(1, 12, n_loads), unit="D"),
        "updated_at": pickup + pd.to_timedelta(rng.integers(2, 30, n_loads), unit="D"),
        "origin_company_name": [f"{c} Plant" for c in origin_city],
        "origin_street": "1 Industrial Park Dr",
        "origin_city": origin_city,
        "origin_state": origin_state,
        "origin_zip": origin_zip,
        "dest_company_name": [f"{c} DC" for c in dest_city],
        "dest_street": "200 Commerce Way",
        "dest_city": dest_city,
        "dest_state": dest_state,
        "dest_zip": dest_zip,
        "customer_po": [f"PO{n:08d}" for n in rng.integers(1, 9999999, n_loads)],
        "pro_number": [f"{n:09d}" for n in rng.integers(1, 999999999, n_loads)],
        "order_number": [f"ORD{n:06d}" for n in rng.integers(1, 999999, n_loads)],
        "reference_number": "",
        "total_weight": np.where(is_tl, np.nan, ltl_weight.round(0)),
        "total_pieces": np.where(is_tl, np.nan, rng.integers(1, 24, n_loads)),
        "total_handling_units": np.where(is_tl, np.nan, rng.integers(1, 18, n_loads)),
        "invoice_number": [f"INV{n:07d}" for n in range(1, n_loads + 1)],
        "invoice_date": pickup + pd.to_timedelta(rng.integers(3, 40, n_loads), unit="D"),
        "invoice_status": "Invoiced",
        "invoice_is_paid": True,
        "actual_pickup_arrival": pickup,
        "actual_delivery_arrival": pickup + pd.to_timedelta(
            rng.integers(1, 5, n_loads), unit="D"),
        "service_level": "Standard",
        # Ground truth and modelling internals. Stripped before the export is
        # written; kept on the frame so tests and scoring can use them.
        "true_mode": np.where(is_tl, "TL", "LTL"),
        "true_miles": true_miles.round(1),
        "true_market_cpm": true_market_cpm.round(4),
        "dest_premium": premium,
    })

    # Weight is populated on only a small share of truckload loads, so it cannot
    # be used to validate truckload status -- a tempting shortcut that fails.
    tl_rows = df.index[is_tl]
    keep_weight = rng.choice(tl_rows, size=int(0.11 * len(tl_rows)), replace=False)
    df.loc[keep_weight, "total_weight"] = rng.lognormal(
        10.0, 0.28, len(keep_weight)).clip(8000, 45000).round(0)

    # Cancelled and in-flight loads carry no cost and must be filtered out.
    n_open = int(0.045 * n_loads)
    open_rows = rng.choice(df.index, size=n_open, replace=False)
    df.loc[open_rows, "status_description"] = rng.choice(
        ["Cancelled", "In Transit", "Booked"], size=n_open)
    df.loc[open_rows, ["carrier_cost", "revenue", "gross_profit"]] = 0.0

    return df


def _to_excel_serial(dates: pd.Series) -> pd.Series:
    return (dates - pd.Timestamp("1899-12-30")) / pd.Timedelta(days=1)


def dirty_export(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Degrade a clean table into the shape a real TMS export arrives in.

    Each transformation here corresponds to something that broke a first attempt
    at reading the real file.
    """
    rng = np.random.default_rng(seed + 1)
    out = df.drop(columns=["true_mode", "true_miles", "true_market_cpm",
                           "dest_premium"]).copy()

    date_columns = ["start_date", "end_date", "created_at", "updated_at",
                    "invoice_date", "actual_pickup_arrival", "actual_delivery_arrival"]

    # Dates: mostly Excel serials, a minority in epoch microseconds because those
    # rows were last touched by an API sync rather than the UI.
    epoch_rows = rng.random(len(out)) < 0.08
    for column in date_columns:
        values = pd.to_datetime(out[column])
        serial = _to_excel_serial(values)
        # Pin the resolution before taking the integer view. pandas stores
        # datetimes at varying units across versions, so an unqualified
        # `astype("int64")` silently yields milliseconds on some installs and
        # nanoseconds on others -- and the degradation being modelled here is
        # specifically microseconds.
        micros = values.astype("datetime64[ns]").astype("int64") / 1_000.0
        out[column] = np.where(epoch_rows, micros, serial)

    # Zips: a spreadsheet round-trip turned a share of them into numbers, which
    # silently destroys the leading zero on New England postal codes.
    numeric_zip = rng.random(len(out)) < 0.15
    for column in ("origin_zip", "dest_zip"):
        as_number = pd.to_numeric(out[column], errors="coerce")
        degraded = as_number.map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
        out[column] = out[column].astype(object)
        mask = numeric_zip & degraded.ne("")
        out.loc[mask, column] = degraded[mask]

    # Header rows repeated inside the data block, as produced by a paged export.
    header_row = {c: c for c in out.columns}
    banners = pd.DataFrame([header_row] * 6)
    positions = sorted(rng.choice(np.arange(1, len(out)), size=6, replace=False))
    pieces, previous = [], 0
    for banner_index, position in enumerate(positions):
        pieces.append(out.iloc[previous:position])
        pieces.append(banners.iloc[[banner_index]])
        previous = position
    pieces.append(out.iloc[previous:])
    return pd.concat(pieces, ignore_index=True)


def write_export(dirty: pd.DataFrame, path, sheet_name: str = "Sheet1",
                 banner_rows: int = 2) -> None:
    """Write the export with the banner rows a TMS puts above the real header."""
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    sheet.append(["Shipment Detail Export"])
    if banner_rows > 1:
        sheet.append([f"Generated by tlbench synthetic generator -- rows: {len(dirty):,}"])
    sheet.append(list(dirty.columns))
    for row in dirty.itertuples(index=False):
        sheet.append(["" if (isinstance(v, float) and np.isnan(v)) else v for v in row])
    workbook.save(path)


def market_pull(loads: pd.DataFrame, lanes: pd.DataFrame, seed: int = 7,
                thin_match_lanes: int = 2) -> pd.DataFrame:
    """Stand in for a licensed lane-level market-rate pull.

    Rates are derived from the same true market the loads were priced against,
    plus panel noise, so the ground truth stays consistent across the repo. The
    provider's quirks are modelled too, because they change the answer:

    * the provider measures real road miles, which run a few percent above the
      centroid estimate the pipeline computes -- so stage 09 has a genuine
      calibration bias to find and correct;
    * a couple of lanes come back having matched only a fraction of our loads,
      which `market.grade_match_rate` is there to catch;
    * thin lanes are silently answered with a market-to-market average, so two
      distinct lanes return identical figures.
    """
    rng = np.random.default_rng(seed + 2)

    truth = loads[loads["true_mode"] == "TL"].groupby("lane").agg(
        true_cpm=("true_market_cpm", "mean"),
        true_miles=("true_miles", "median"),
        loads_actual=("id", "size"),
    )
    joined = lanes.merge(truth, on="lane", how="inner")
    joined = joined[joined["true_miles"] > 50].copy()

    # US domestic only: the provider does not price cross-border lanes reliably.
    domestic = joined["o_state"].str.fullmatch(r"[A-Z]{2}") & \
        joined["d_state"].str.fullmatch(r"[A-Z]{2}")
    joined = joined[domestic & ~joined["d_state"].isin(["ON", "QC", "BC", "AB"])].copy()

    n = len(joined)
    # The provider reports the road distance the loads actually ran. The
    # pipeline's centroid estimate is derived from a smaller circuity factor and
    # so sits a couple of percent short of this -- which is the bias stage 09
    # measures and then corrects by preferring provider mileage.
    joined["provider_miles"] = (joined["true_miles"] * rng.normal(1.0, 0.012, n)).round(0)
    panel_noise = rng.normal(1.0, 0.028, n)
    joined["market_rate"] = (joined["true_cpm"] * joined["provider_miles"]
                             * panel_noise).round(0)
    # The current market sits above the trailing year: rates rose across the window.
    joined["market_rate_90d"] = (joined["market_rate"]
                                 * rng.normal(1.115, 0.025, n)).round(0)
    joined["provider_reports_total"] = rng.integers(180, 5200, n)
    joined["rate_strength"] = rng.integers(84, 99, n)
    joined["equipment"] = "Van"
    joined["origin"] = joined["top_city"].str.split(" > ").str[0]
    joined["destination"] = joined["top_city"].str.split(" > ").str[-1]

    # Match rate: usually high, deliberately poor on a couple of lanes.
    match = rng.uniform(0.72, 0.97, n)
    if thin_match_lanes and n > thin_match_lanes:
        thin = rng.choice(n, size=thin_match_lanes, replace=False)
        match[thin] = rng.uniform(0.08, 0.34, thin_match_lanes)
    joined["provider_reports_ours"] = (match * joined["loads"]).round(0).astype(int)
    joined["our_reported_rate"] = (joined["avg_cost"]
                                   * rng.normal(1.0, 0.02, n)).round(0)

    # Silent market-pair aggregation. The provider does this only to lanes too
    # thin to price on their own, and substitutes the surrounding market's
    # average -- so the figure that comes back is plausible for the lane, just
    # not specific to it. That is what makes it dangerous: it does not look
    # wrong, it only double-counts when summed alongside its twin.
    if n >= 6:
        order = np.argsort(joined["loads"].to_numpy())
        thin_candidates = order[:max(2, n // 3)]
        miles = joined["provider_miles"].to_numpy()
        shared = ["market_rate", "market_rate_90d", "provider_reports_total",
                  "provider_miles", "rate_strength"]
        for target in rng.permutation(thin_candidates):
            comparable = [i for i in range(n)
                          if i != target and 0.88 <= miles[i] / miles[target] <= 1.14]
            if not comparable:
                continue
            source = int(rng.choice(comparable))
            joined.iloc[target, [joined.columns.get_loc(c) for c in shared]] = \
                joined.iloc[source][shared].to_numpy()
            break

    columns = ["lane", "origin", "destination", "equipment", "provider_miles",
               "market_rate", "market_rate_90d", "provider_reports_total",
               "provider_reports_ours", "our_reported_rate", "rate_strength"]
    return joined[columns].reset_index(drop=True)
