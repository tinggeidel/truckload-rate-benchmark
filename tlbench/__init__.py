"""Truckload buy-rate benchmarking against lane-level market data.

The package holds the analysis logic; `stages/` holds thin numbered scripts that
run it in order. Everything here is importable and side-effect free so it can be
tested without a TMS export, a market-data subscription, or a network connection.
"""

__version__ = "1.0.0"
