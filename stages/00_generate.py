"""Stage 00 -- generate a synthetic TMS export and a synthetic market pull.

Skip this stage when running against a real export: set TLBENCH_SOURCE_EXPORT
and start at stage 01. Everything downstream is identical either way, which is
the point -- the demo exercises the real code path rather than a special case.

See `tlbench/synth.py` for what is being modelled and why.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tlbench import config, report, synth  # noqa: E402

report.configure_pandas()
report.heading("stage 00 -- generate synthetic source data")

loads = synth.generate_loads(seed=7, n_loads=20000)
report.note(
    f"Generated {len(loads):,} loads across {loads['customer_name'].nunique()} "
    f"customers and {loads['carrier_name'].nunique()} carriers."
)

completed = loads[(loads["status_description"] == "Completed") & (loads["carrier_cost"] > 0)]
truckload = completed[completed["true_mode"] == "TL"]
report.note(
    f"Ground truth: {len(truckload):,} of {len(completed):,} completed loads are "
    f"truckload ({100 * len(truckload) / len(completed):.0f}% by count) but carry "
    f"{100 * truckload['carrier_cost'].sum() / completed['carrier_cost'].sum():.0f}% "
    "of carrier spend. That asymmetry is why a bad mode flag is expensive: it "
    "misroutes most of the money."
)
says_truckload = int((completed["mode"] == "TL").sum())
report.note(
    f"The TMS mode column calls only {says_truckload:,} of them truckload. Stage 04 "
    "has to derive the label instead."
)

# The ground-truth columns are kept in a separate file. They never touch the
# export, because a real export has no such thing and the pipeline must not
# quietly come to depend on them.
truth_columns = ["id", "true_mode", "true_miles", "true_market_cpm", "dest_premium"]
truth_path = config.data("truth_labels.csv")
loads[truth_columns].to_csv(truth_path, index=False)
report.wrote(truth_path)

dirty = synth.dirty_export(loads, seed=7)
export_path = config.data("synthetic_tms_export.xlsx")
synth.write_export(dirty, export_path)
report.note(
    f"Wrote a {len(dirty):,}-row export with two banner rows above the header, "
    "six repeated header rows inside the data block, dates split between Excel "
    "serials and epoch microseconds, and a share of postal codes degraded to "
    "floats. Stage 01 has to survive all of it."
)
report.wrote(export_path)
print()
