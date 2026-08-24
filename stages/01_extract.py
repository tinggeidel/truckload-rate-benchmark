"""Stage 01 -- read the export and cache a clean load table.

The only stage that touches the source workbook. Everything downstream reads the
CSV this writes, so a slow, fragile Excel parse happens once rather than on every
iteration of the analysis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from tlbench import coerce, config, report  # noqa: E402

report.configure_pandas()
report.heading("stage 01 -- extract and clean the source export")

source = config.SOURCE_EXPORT or config.data("synthetic_tms_export.xlsx")
if not Path(source).exists():
    raise SystemExit(
        f"source export not found: {source}\n"
        "Run stage 00 to generate a synthetic one, or set TLBENCH_SOURCE_EXPORT "
        "to point at a real export."
    )

raw = pd.read_excel(source, sheet_name=config.SOURCE_SHEET,
                    header=config.SOURCE_HEADER_ROW, engine="openpyxl")
print(f"  read {source}")
report.note(f"raw: {len(raw):,} rows x {len(raw.columns)} columns")

before = len(raw)
numeric_dates = {c: int(pd.to_numeric(raw[c], errors="coerce").notna().sum())
                 for c in coerce.DATE_COLUMNS if c in raw.columns}

df = coerce.coerce_export(raw)

report.section("row-level cleaning")
print(f"  repeated header / banner rows dropped : {before - len(df):>7,}")
print(f"  rows retained                         : {len(df):>7,}")

report.section("date parsing (numeric values in -> timestamps out)")
for column, count in numeric_dates.items():
    parsed = int(df[column].notna().sum())
    flag = "   <-- LOSS" if parsed < count else ""
    print(f"  {column:<26} {count:>7,} -> {parsed:>7,}{flag}")
report.note(
    "A loss here means a date encoding the coercion does not know about. It is "
    "worth chasing rather than tolerating: the benchmark window is a date filter, "
    "so unparsed dates silently shrink the universe."
)

report.section("postal normalisation")
for column in ("origin_zip", "dest_zip"):
    resolved = int(df[column].notna().sum())
    print(f"  {column:<12} {resolved:>7,} populated, {df[column].nunique():>4} distinct")
report.note(
    "Leading zeros are restored here. A zip that arrived as 4769.0 would "
    "otherwise geocode to nowhere, and the loads on it would drop out of the "
    "benchmark without ever appearing in a count."
)

completed = coerce.completed(df)
report.section("sanity")
print(f"  date range          : {df['start_date'].min():%Y-%m-%d} .. "
      f"{df['start_date'].max():%Y-%m-%d}")
print(f"  completed w/ cost>0 : {len(completed):,} loads")
print(f"  carrier spend       : {report.money(completed['carrier_cost'].sum())}")
print(f"  revenue             : {report.money(completed['revenue'].sum())}")
print(f"  distinct carriers   : {completed['carrier_name'].nunique():,}")
print(f"  distinct customers  : {completed['customer_name'].nunique():,}")

report.section("the mode column, for the record")
print(df["mode"].value_counts(dropna=False).to_string())
report.note(
    "Stage 04 replaces this. It is printed now so the comparison at the end of "
    "stage 04 has a baseline."
)

out = config.data("loads_raw.csv")
df.to_csv(out, index=False)
report.wrote(out)
print()
