"""Console formatting for the stage scripts.

Each stage prints a readable narrative rather than dumping frames, because the
output is the working record of the analysis -- it is what gets pasted into a
findings note and what a reviewer reads to decide whether to trust the number.
"""
from __future__ import annotations

import pandas as pd

WIDTH = 78


def configure_pandas() -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)


def heading(text: str) -> None:
    print()
    print("=" * WIDTH)
    print(text.upper())
    print("=" * WIDTH)


def section(text: str) -> None:
    print()
    print(f"-- {text} " + "-" * max(0, WIDTH - len(text) - 4))


def note(text: str) -> None:
    """A line of interpretation, not data. Prefixed so it reads as commentary."""
    for line in _wrap(text, WIDTH - 2):
        print(f"  {line}")


def table(df: pd.DataFrame, floats: str = ",.2f", index: bool = False) -> None:
    print(df.to_string(index=index, float_format=lambda x: format(x, floats)))


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float, places: int = 1) -> str:
    return f"{value:+.{places}f}%"


def wrote(path) -> None:
    print(f"\n  wrote {path}")


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
