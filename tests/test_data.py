"""Data loading and validation tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stockforecast.data import (
    CsvProvider,
    DataError,
    RobinhoodMcpJsonProvider,
    SyntheticProvider,
    get_provider,
    validate_ohlcv,
)


def _frame(**overrides: object) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=10),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000.0,
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


def test_validate_accepts_canonical_frame() -> None:
    result = validate_ohlcv(_frame(), "TEST")
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.name == "date"
    assert result.index.is_monotonic_increasing


def test_validate_normalises_column_aliases() -> None:
    frame = _frame().rename(
        columns={
            "date": "begins_at",
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "close": "close_price",
        }
    )
    result = validate_ohlcv(frame, "TEST")
    assert "close" in result.columns


def test_validate_handles_yahoo_style_headers() -> None:
    frame = _frame().rename(columns={"date": "Date", "close": "Adj Close"})
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low"})
    frame = frame.rename(columns={"volume": "Volume"})
    result = validate_ohlcv(frame, "TEST")
    assert len(result) == 10


def test_validate_rejects_missing_columns() -> None:
    frame = _frame().drop(columns=["volume"])
    with pytest.raises(DataError, match="missing required column"):
        validate_ohlcv(frame, "TEST")


def test_validate_rejects_empty() -> None:
    with pytest.raises(DataError, match="no rows"):
        validate_ohlcv(pd.DataFrame(), "TEST")


def test_validate_rejects_non_positive_prices() -> None:
    frame = _frame()
    frame.loc[3, "close"] = -1.0
    with pytest.raises(DataError, match="non-positive"):
        validate_ohlcv(frame, "TEST")


def test_validate_rejects_swapped_high_low() -> None:
    frame = _frame(high=99.0, low=101.0)
    with pytest.raises(DataError, match="mismapped"):
        validate_ohlcv(frame, "TEST")


def test_interpolated_gap_bars_are_dropped() -> None:
    """Zero-volume flat bars carry no information and must not be scored."""
    frame = _frame()
    frame.loc[5, ["open", "high", "low", "close"]] = 100.0
    frame.loc[5, "volume"] = 0.0

    result = validate_ohlcv(frame, "TEST")
    assert len(result) == 9


def test_duplicate_dates_keep_last() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[0]].assign(close=123.0)], ignore_index=True)
    result = validate_ohlcv(frame, "TEST")
    assert len(result) == 10
    assert result["close"].iloc[0] == 123.0


def test_csv_round_trip(tmp_path) -> None:
    prices = SyntheticProvider(n_bars=300, seed=5).fetch("ABC")
    prices.to_csv(tmp_path / "ABC.csv", index_label="date")

    provider = CsvProvider(tmp_path)
    loaded = provider.fetch("abc")

    assert len(loaded) == len(prices)
    pd.testing.assert_series_equal(loaded["close"], prices["close"], rtol=1e-9)
    assert provider.available_symbols() == ["ABC"]


def test_csv_missing_symbol_is_clear(tmp_path) -> None:
    with pytest.raises(DataError, match="no CSV found"):
        CsvProvider(tmp_path).fetch("NOPE")


def test_mcp_json_provider(tmp_path) -> None:
    payload = {
        "data": {
            "results": [
                {
                    "symbol": "HOOD",
                    "interval": "day",
                    "bars": [
                        {
                            "begins_at": "2026-08-03T00:00:00Z",
                            "open_price": "86.685",
                            "high_price": "92.35",
                            "low_price": "85.30",
                            "close_price": "90.34",
                            "volume": 19740541,
                        },
                        {
                            "begins_at": "2026-08-04T00:00:00Z",
                            "open_price": "92.615",
                            "high_price": "95.60",
                            "low_price": "90.0717",
                            "close_price": "93.51",
                            "volume": 22639539,
                        },
                    ],
                }
            ]
        }
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(payload))

    frame = RobinhoodMcpJsonProvider(path).fetch("HOOD")

    assert len(frame) == 2
    assert frame["close"].iloc[-1] == pytest.approx(93.51)


def test_mcp_json_unknown_symbol(tmp_path) -> None:
    (tmp_path / "empty.json").write_text(json.dumps({"data": {"results": []}}))
    with pytest.raises(DataError, match="not present"):
        RobinhoodMcpJsonProvider(tmp_path).fetch("HOOD")


def test_synthetic_is_deterministic_per_symbol() -> None:
    first = SyntheticProvider(n_bars=200, seed=9).fetch("XYZ")
    second = SyntheticProvider(n_bars=200, seed=9).fetch("XYZ")
    other = SyntheticProvider(n_bars=200, seed=9).fetch("ZZZ")

    pd.testing.assert_frame_equal(first, second)
    assert not first["close"].equals(other["close"])


def test_fetch_many_skips_bad_symbols(tmp_path, capsys) -> None:
    SyntheticProvider(n_bars=200, seed=2).fetch("GOOD").to_csv(
        tmp_path / "GOOD.csv", index_label="date"
    )
    result = CsvProvider(tmp_path).fetch_many(["GOOD", "MISSING"])

    assert set(result) == {"GOOD"}
    assert "MISSING" in capsys.readouterr().out


def test_fetch_many_raises_when_everything_fails(tmp_path) -> None:
    with pytest.raises(DataError, match="no symbols could be loaded"):
        CsvProvider(tmp_path).fetch_many(["A", "B"])


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(DataError, match="unknown provider"):
        get_provider("bloomberg")


def test_real_hood_csv_loads() -> None:
    """The committed HOOD sample must stay loadable."""
    frame = CsvProvider("data").fetch("HOOD")
    assert len(frame) > 100
    assert frame.index.is_monotonic_increasing
    assert frame["close"].between(50, 200).all()
