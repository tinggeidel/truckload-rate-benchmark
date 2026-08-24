"""Run every stage in order.

Each stage is also runnable on its own -- they hand off through CSVs in `data/`
rather than through memory, so an expensive early stage does not have to re-run
while a later one is being iterated on.

    python run_pipeline.py              # everything, synthetic data
    python run_pipeline.py --from 04    # resume from the classifier
    python run_pipeline.py --only 09    # one stage
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES_DIR = Path(__file__).resolve().parent / "stages"


def discover() -> list[Path]:
    return sorted(STAGES_DIR.glob("[0-9][0-9]_*.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", metavar="NN",
                        help="first stage to run, e.g. 04")
    parser.add_argument("--only", metavar="NN", help="run a single stage")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress stage output, report pass/fail only")
    args = parser.parse_args()

    stages = discover()
    if not stages:
        print(f"no stages found in {STAGES_DIR}", file=sys.stderr)
        return 1
    if args.only:
        stages = [s for s in stages if s.name.startswith(args.only)]
    elif args.start:
        stages = [s for s in stages if s.name[:2] >= args.start]
    if not stages:
        print("no stages matched", file=sys.stderr)
        return 1

    for stage in stages:
        if args.quiet:
            print(f"  {stage.name} ... ", end="", flush=True)
        result = subprocess.run(
            [sys.executable, str(stage)],
            capture_output=args.quiet, text=True,
        )
        if result.returncode != 0:
            if args.quiet:
                print("FAILED")
                print(result.stdout or "", file=sys.stderr)
                print(result.stderr or "", file=sys.stderr)
            print(f"\nstage {stage.name} failed with exit code "
                  f"{result.returncode}", file=sys.stderr)
            return result.returncode
        if args.quiet:
            print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
