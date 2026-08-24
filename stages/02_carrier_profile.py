"""Stage 02 -- profile every carrier.

Run before any classification. The point is to establish, from evidence rather
than from the mode flag, which carriers behave like LTL networks and which
behave like truckload carriers -- and to find the ones that do both, since those
are where a per-carrier rule would fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import classify, coerce, config, report  # noqa: E402

report.configure_pandas()
report.heading("stage 02 -- carrier profile")

df = pd.read_csv(config.data("loads_raw.csv"), low_memory=False)
completed = coerce.completed(df)

profile = completed.groupby("carrier_name").agg(
    loads=("id", "size"),
    spend=("carrier_cost", "sum"),
    avg_cost=("carrier_cost", "mean"),
    med_cost=("carrier_cost", "median"),
    p10_cost=("carrier_cost", lambda s: s.quantile(0.10)),
    p90_cost=("carrier_cost", lambda s: s.quantile(0.90)),
    pct_mode_says_tl=("mode", lambda s: 100 * (s == "TL").mean()),
    top_equipment=("equipment_type", lambda s: s.mode().iloc[0] if s.notna().any() else ""),
    n_customers=("customer_name", "nunique"),
).reset_index().sort_values("spend", ascending=False)

profile["cum_spend_pct"] = 100 * profile["spend"].cumsum() / profile["spend"].sum()
names = profile["carrier_name"].str.upper()
profile["ltl_network"] = names.str.contains(classify.LTL_CARRIER_RE, regex=True)

print(f"  carriers: {len(profile):,}   spend: {report.money(profile['spend'].sum())}")
n80 = int((profile["cum_spend_pct"] <= 80).sum()) + 1
n95 = int((profile["cum_spend_pct"] <= 95).sum()) + 1
print(f"  top {n80} carriers = 80% of spend;  top {n95} = 95%")

report.section("top 20 carriers by spend")
report.table(profile.head(20)[[
    "carrier_name", "loads", "spend", "avg_cost", "med_cost", "p10_cost",
    "p90_cost", "pct_mode_says_tl", "top_equipment", "ltl_network"]], floats=",.0f")

report.section("carriers running both modes")
repeat = profile[profile["loads"] >= 20].copy()
repeat["p90_over_p10"] = repeat["p90_cost"] / repeat["p10_cost"].clip(lower=1)
mixed = repeat[repeat["p90_over_p10"] > 6]
print(f"  {len(mixed)} carriers (>=20 loads) price across a >6x spread, "
      f"carrying {report.money(mixed['spend'].sum())}")
report.note(
    "A carrier whose 90th percentile is six times its 10th is not running one "
    "kind of freight. Any rule that classifies by carrier alone gets every one "
    "of these loads wrong in one direction or the other, which is why stage 04 "
    "classifies per load and uses carrier identity only as an anchor."
)
if len(mixed):
    report.table(mixed.nlargest(8, "spend")[[
        "carrier_name", "loads", "spend", "p10_cost", "med_cost", "p90_cost",
        "p90_over_p10"]], floats=",.0f")

report.section("what the mode flag claims, by carrier type")
by_type = profile.groupby("ltl_network").agg(
    carriers=("carrier_name", "size"), loads=("loads", "sum"),
    spend=("spend", "sum"), avg_pct_mode_says_tl=("pct_mode_says_tl", "mean"))
report.table(by_type.reset_index(), floats=",.1f")
report.note(
    "If the mode flag were usable, the truckload share it reports would be near "
    "zero on the LTL networks and near total off them. It is neither."
)

out = config.data("carrier_profile.csv")
profile.to_csv(out, index=False)
report.wrote(out)
print()
