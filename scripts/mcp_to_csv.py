#!/usr/bin/env python3
"""Convert Robinhood MCP ``get_equity_historicals`` output into CSVs.

The MCP tools are available to an agent session, not to a plain Python
process, so the ingest path is: have the agent save the tool's JSON response
to a file, then run this script to turn it into the CSVs the pipeline reads.

    # 1. In an agent session, call the MCP tool and save the response:
    #      get_equity_historicals(symbols=["HOOD","AAPL"],
    #                             start_time="2019-01-01T00:00:00Z",
    #                             interval="day")
    #    ... writing the JSON to data/mcp/batch1.json
    #
    # 2. Then:
    python scripts/mcp_to_csv.py data/mcp/batch1.json --out data/

Accepts the full tool envelope, a bare ``results`` list, or a directory of
either. Multiple files covering the same symbol are merged, with later bars
winning on overlapping dates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stockforecast.data import _iter_mcp_results, validate_ohlcv  # noqa: E402


def collect(paths: list[Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {}

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"! {path}: not valid JSON ({exc})")
            continue

        for result in _iter_mcp_results(payload):
            symbol = str(result.get("symbol", "")).upper()
            bars = result.get("bars") or []
            if not symbol or not bars:
                continue
            frames.setdefault(symbol, []).append(pd.DataFrame(bars))

    merged: dict[str, pd.DataFrame] = {}
    for symbol, parts in frames.items():
        combined = pd.concat(parts, ignore_index=True)
        try:
            merged[symbol] = validate_ohlcv(combined, symbol)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"! {symbol}: {exc}")
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON files or directories")
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.exists():
            paths.append(path)
        else:
            print(f"! not found: {path}")

    if not paths:
        print("no input files found")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = collect(paths)
    if not frames:
        print("no symbols extracted")
        return 1

    for symbol, frame in sorted(frames.items()):
        destination = out_dir / f"{symbol}.csv"
        frame.to_csv(destination, index_label="date")
        span = f"{frame.index[0].date()} to {frame.index[-1].date()}"
        print(f"  wrote {destination}  ({len(frame)} bars, {span})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
