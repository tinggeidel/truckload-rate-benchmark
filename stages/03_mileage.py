"""Stage 03 -- attach distance and rate per mile.

The export has no mileage column, so cost per mile -- the only unit in which
truckload rates can be compared at all -- has to be constructed. Stage 09
calibrates the estimate against the market provider's authoritative mileage and
prefers the provider's number wherever it exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import coerce, config, geo, report  # noqa: E402

report.configure_pandas()
report.heading("stage 03 -- mileage and cost per mile")

df = pd.read_csv(config.data("loads_raw.csv"), low_memory=False,
                 dtype={"origin_zip": str, "dest_zip": str})

# The bundled table covers the demo geography with no network access. For a real
# export, pgeocode resolves the full US and Canadian postal ranges; it is tried
# second so the bundled table stays authoritative for the demo.
resolvers = [geo.TableResolver.from_csv()]
try:
    resolvers.append(geo.PgeocodeResolver())
except Exception as exc:  # pragma: no cover - depends on local install
    report.note(f"pgeocode unavailable ({exc.__class__.__name__}); bundled table only.")

df = geo.attach_mileage(df, geo.ChainResolver(*resolvers))
completed = coerce.completed(df)
resolved = completed["est_miles"].notna()

print(f"  completed loads   : {len(completed):,}")
print(f"  mileage resolved  : {int(resolved.sum()):,} ({100 * resolved.mean():.1f}%)")
print(f"  spend covered     : {report.money(completed.loc[resolved, 'carrier_cost'].sum())}"
      f" of {report.money(completed['carrier_cost'].sum())}")

unresolved = completed.loc[~resolved]
if len(unresolved):
    report.section("unresolved postal codes")
    print(f"  {unresolved['origin_zip'].nunique()} origin / "
          f"{unresolved['dest_zip'].nunique()} destination codes")
    report.table(unresolved[["origin_city", "origin_state", "origin_zip",
                             "dest_city", "dest_state", "dest_zip"]].head(8))
    report.note(
        "These carry through as unclassifiable rather than being guessed at. "
        "Cross-border points are the usual cause and they are excluded from the "
        "market pull anyway."
    )

report.section("distance distribution")
quantiles = [5, 10, 25, 50, 75, 90, 95, 99]
print("  " + "  ".join(
    f"p{q}={completed['est_miles'].quantile(q / 100):,.0f}" for q in quantiles))

report.section("same-city moves")
local = completed[completed["origin_zip"] == completed["dest_zip"]]
print(f"  {len(local):,} loads ({100 * len(local) / len(completed):.1f}%), "
      f"{report.money(local['carrier_cost'].sum())}")
report.note(
    f"Origin equals destination on these. Distance is floored at "
    f"{config.MIN_MILES:.0f} miles so cost per mile stays finite, but they are "
    "local cartage rather than truckload lanes and stage 08 excludes them from "
    "the market pull."
)

out = config.data("loads_miles.csv")
df.to_csv(out, index=False)
report.wrote(out)
print()
