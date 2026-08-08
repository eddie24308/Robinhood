"""Data access layer.

Price history can come from several places depending on where this package is
running. Each source is wrapped in a :class:`DataProvider` so the rest of the
pipeline never knows the difference:

``csv``        Local CSV files, one per symbol. Always available.
``yfinance``   Yahoo Finance via the ``yfinance`` package. Needs network access
               to Yahoo, which some sandboxes block.
``mcp-json``   Saved JSON responses from the Robinhood MCP
               ``get_equity_historicals`` tool. See ``scripts/mcp_to_csv.py``.
``synthetic``  Simulated bars with volatility clustering and fat tails. Used by
               the test suite so tests are fast, offline and deterministic.
               Never use it to make claims about a real company.

All providers return the same frame: a ``DatetimeIndex`` named ``date`` (tz
naive, sorted, unique) and float columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Column aliases seen in the wild, normalised to our canonical names.
_ALIASES: dict[str, str] = {
    "adj close": "close",
    "adj_close": "close",
    "adjclose": "close",
    "begins_at": "date",
    "close_price": "close",
    "datetime": "date",
    "high_price": "high",
    "low_price": "low",
    "open_price": "open",
    "time": "date",
    "timestamp": "date",
}


class DataError(RuntimeError):
    """Raised when price history cannot be loaded or fails validation."""


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_")
        renamed[column] = _ALIASES.get(key, _ALIASES.get(key.replace("_", " "), key))
    return frame.rename(columns=renamed)


def validate_ohlcv(frame: pd.DataFrame, symbol: str = "?") -> pd.DataFrame:
    """Coerce ``frame`` to the canonical OHLCV contract or raise ``DataError``.

    Validation is deliberately strict. A silently mis-parsed price column
    produces a model that looks fine in backtest and is worthless in use, so
    problems are surfaced at load time rather than absorbed.
    """
    if frame is None or len(frame) == 0:
        raise DataError(f"{symbol}: no rows returned")

    frame = _normalise_columns(frame.copy())

    if "date" in frame.columns:
        frame = frame.set_index("date")

    index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if index.isna().any():
        raise DataError(f"{symbol}: {int(index.isna().sum())} rows have unparseable dates")
    # Drop tz so downstream date maths never mixes aware and naive stamps.
    frame.index = index.tz_convert(None).normalize()
    frame.index.name = "date"

    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(
            f"{symbol}: missing required column(s) {missing}; got {sorted(frame.columns)}"
        )

    frame = frame.loc[:, list(OHLCV_COLUMNS)]
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Bars synthesised to fill gaps (Robinhood marks these ``interpolated``)
    # arrive as zero-volume repeats of the prior close. They carry no
    # information and would otherwise be scored as real predictions.
    zero_volume_flat = (
        (frame["volume"].fillna(0) <= 0)
        & (frame["open"] == frame["close"])
        & (frame["high"] == frame["low"])
    )
    if bool(zero_volume_flat.any()):
        frame = frame.loc[~zero_volume_flat]

    frame = frame.dropna(subset=["close"])
    if frame.empty:
        raise DataError(f"{symbol}: every row had a missing close price")

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if (frame["close"] <= 0).any():
        raise DataError(f"{symbol}: non-positive close prices present")

    # high/low should bracket open/close. Small violations happen in real feeds
    # around auctions; large ones mean the columns are swapped.
    body_high = frame[["open", "close"]].max(axis=1)
    body_low = frame[["open", "close"]].min(axis=1)
    bad_bars = ((frame["high"] < body_high * 0.999) | (frame["low"] > body_low * 1.001)).sum()
    if bad_bars > max(3, 0.01 * len(frame)):
        raise DataError(
            f"{symbol}: {int(bad_bars)} bars have high/low inconsistent with open/close; "
            "columns are probably mismapped"
        )

    return frame


class DataProvider(ABC):
    """Base class for price history sources."""

    name: str = "base"

    @abstractmethod
    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Return validated OHLCV history for one symbol."""

    def fetch_many(
        self,
        symbols: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch several symbols, skipping (and reporting) the ones that fail."""
        out: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        for symbol in symbols:
            try:
                out[symbol.upper()] = self.fetch(symbol, start=start, end=end)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not sink the run
                errors[symbol.upper()] = str(exc)
        if errors and not out:
            detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
            raise DataError(f"no symbols could be loaded ({detail})")
        for symbol, message in errors.items():
            print(f"  ! skipping {symbol}: {message}")
        return out


class CsvProvider(DataProvider):
    """Reads ``<directory>/<SYMBOL>.csv``.

    The CSV needs a date column and open/high/low/close/volume columns; common
    header spellings are normalised automatically.
    """

    name = "csv"

    def __init__(self, directory: str | Path = "data") -> None:
        self.directory = Path(directory)

    def _path_for(self, symbol: str) -> Path:
        candidates = [
            self.directory / f"{symbol.upper()}.csv",
            self.directory / f"{symbol.lower()}.csv",
            self.directory / f"{symbol}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise DataError(
            f"{symbol}: no CSV found in {self.directory}/ (looked for {candidates[0].name})"
        )

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        frame = pd.read_csv(self._path_for(symbol))
        frame = validate_ohlcv(frame, symbol)
        return _slice_dates(frame, start, end)

    def available_symbols(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem.upper() for path in self.directory.glob("*.csv"))


class YFinanceProvider(DataProvider):
    """Yahoo Finance via ``yfinance``. Requires outbound access to Yahoo."""

    name = "yfinance"

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        try:
            import yfinance  # noqa: PLC0415 - optional dependency, imported on use
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DataError(
                "yfinance is not installed; run `pip install yfinance` or use --provider csv"
            ) from exc

        frame = yfinance.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=self.auto_adjust,
            actions=False,
        )
        if frame is None or frame.empty:
            raise DataError(
                f"{symbol}: yfinance returned no rows (bad ticker, or network blocked to Yahoo)"
            )
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.reset_index()
        return validate_ohlcv(frame, symbol)


class RobinhoodMcpJsonProvider(DataProvider):
    """Reads saved Robinhood MCP ``get_equity_historicals`` JSON payloads.

    Accepts either the full tool envelope (``{"data": {"results": [...]}}``) or a
    bare results list. A file may hold several symbols; the matching one is
    selected. Point it at a directory of ``.json`` files or a single file.
    """

    name = "mcp-json"

    def __init__(self, path: str | Path = "data/mcp") -> None:
        self.path = Path(path)

    def _candidate_files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if self.path.is_dir():
            return sorted(self.path.glob("*.json"))
        raise DataError(f"MCP JSON path not found: {self.path}")

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        wanted = symbol.upper()
        for file_path in self._candidate_files():
            payload = json.loads(file_path.read_text())
            for result in _iter_mcp_results(payload):
                if str(result.get("symbol", "")).upper() != wanted:
                    continue
                bars = result.get("bars") or []
                if not bars:
                    continue
                frame = validate_ohlcv(pd.DataFrame(bars), symbol)
                return _slice_dates(frame, start, end)
        raise DataError(f"{symbol}: not present in any JSON under {self.path}")


def _iter_mcp_results(payload: object) -> Iterable[dict]:
    """Yield per-symbol result dicts from any of the MCP payload shapes."""
    if isinstance(payload, dict):
        if "bars" in payload and "symbol" in payload:
            yield payload
            return
        for key in ("data", "result", "results"):
            if key in payload:
                yield from _iter_mcp_results(payload[key])
                return
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_mcp_results(item)


@dataclass
class SyntheticProvider(DataProvider):
    """Simulated price history for tests and offline demos.

    Generates a GARCH-flavoured process: persistent volatility, Student-t
    shocks, and a small drift. There is deliberately **no** predictable signal
    beyond mild volatility persistence, so a correctly-built pipeline should
    report near-zero return predictability on this data. That property is what
    makes it a useful test fixture — it catches lookahead leakage, which shows
    up as impossibly good scores.
    """

    name: str = "synthetic"
    n_bars: int = 1500
    seed: int = 7
    start_price: float = 100.0
    annual_drift: float = 0.08
    base_annual_vol: float = 0.35

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        # Per-symbol seed so different tickers differ but each is reproducible.
        rng = np.random.default_rng(self.seed + (abs(hash(symbol.upper())) % 100_000))

        daily_drift = self.annual_drift / 252.0
        base_var = (self.base_annual_vol / math.sqrt(252.0)) ** 2

        omega, alpha, beta = base_var * 0.05, 0.08, 0.87
        variance = base_var
        returns = np.empty(self.n_bars)
        for i in range(self.n_bars):
            shock = rng.standard_t(df=5) / math.sqrt(5 / 3)  # unit-variance t
            step = math.sqrt(variance) * shock
            returns[i] = daily_drift + step
            variance = omega + alpha * step**2 + beta * variance

        close = self.start_price * np.exp(np.cumsum(returns))
        intraday = np.abs(rng.normal(0.0, 0.004, self.n_bars))
        open_ = close * np.exp(rng.normal(0.0, 0.003, self.n_bars))
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)
        volume = rng.lognormal(mean=15.0, sigma=0.4, size=self.n_bars)

        end_date = pd.Timestamp(end) if end else pd.Timestamp("2026-08-07")
        dates = pd.bdate_range(end=end_date, periods=self.n_bars, name="date")

        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        frame = validate_ohlcv(frame.reset_index(), symbol)
        return _slice_dates(frame, start, end)


def _slice_dates(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        raise DataError(f"no bars left after restricting to [{start}, {end}]")
    return frame


_PROVIDERS: dict[str, type[DataProvider]] = {
    "csv": CsvProvider,
    "yfinance": YFinanceProvider,
    "mcp-json": RobinhoodMcpJsonProvider,
    "synthetic": SyntheticProvider,
}


def get_provider(name: str, **kwargs: object) -> DataProvider:
    """Construct a provider by name (``csv``/``yfinance``/``mcp-json``/``synthetic``)."""
    try:
        provider_class = _PROVIDERS[name]
    except KeyError:
        raise DataError(
            f"unknown provider {name!r}; choose one of {sorted(_PROVIDERS)}"
        ) from None
    return provider_class(**kwargs)  # type: ignore[arg-type]


def load_prices(
    symbols: Sequence[str],
    provider: DataProvider | str = "csv",
    start: str | None = None,
    end: str | None = None,
    **provider_kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Load validated OHLCV history for ``symbols`` from ``provider``."""
    if isinstance(provider, str):
        provider = get_provider(provider, **provider_kwargs)
    return provider.fetch_many(symbols, start=start, end=end)
