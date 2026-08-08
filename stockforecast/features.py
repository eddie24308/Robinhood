"""Strictly causal feature engineering.

Every column produced here is computed from bars at or before the timestamp it
is indexed on. The convention throughout the package is:

* Features on row ``t`` are known once bar ``t`` has closed.
* The target on row ``t`` is the return **after** bar ``t``, over the next
  ``horizon`` bars.

That split is the only thing standing between a real backtest and an
accidental time machine, so ``tests/test_features.py`` verifies it mechanically
by truncating the input and checking that already-computed rows do not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class FeatureConfig:
    """Which features to build. Defaults are a reasonable general-purpose set."""

    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 21)
    momentum_windows: tuple[int, ...] = (5, 10, 21, 63, 126)
    volatility_windows: tuple[int, ...] = (5, 21, 63)
    sma_windows: tuple[int, ...] = (10, 20, 50, 200)
    rsi_period: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    atr_period: int = 14
    volume_windows: tuple[int, ...] = (5, 21)
    range_windows: tuple[int, ...] = (21, 63)
    include_calendar: bool = True
    winsorize_quantile: float | None = 0.001
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def warmup_bars(self) -> int:
        """Bars consumed before the first complete feature row exists.

        The longest lookback window drives this. It is worth surfacing because
        a config whose warm-up exceeds the available history silently produces
        an empty dataset, which is a confusing way to fail.
        """
        return max(
            (
                max(self.return_lags),
                max(self.momentum_windows),
                max(self.volatility_windows),
                max(self.sma_windows),
                max(self.volume_windows),
                max(self.range_windows),
                self.rsi_period,
                max(self.macd),
                self.atr_period,
            )
        )

    @classmethod
    def short_history(cls) -> FeatureConfig:
        """A lighter feature set for tickers with under ~1 year of bars.

        Drops the long windows so the warm-up fits. Fewer features on less data
        is also the right bias/variance trade-off, but be clear-eyed: a model
        fitted on a few hundred bars is a weak estimate whatever you feed it.
        """
        return cls(
            return_lags=(1, 2, 3, 5),
            momentum_windows=(5, 10, 21),
            volatility_windows=(5, 10, 21),
            sma_windows=(5, 10, 20),
            rsi_period=14,
            macd=(6, 13, 5),
            atr_period=10,
            volume_windows=(5, 10),
            range_windows=(10, 21),
            include_calendar=True,
            winsorize_quantile=None,
        )


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI. Uses an EWMA of gains/losses over closed bars only."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows give an undefined ratio; RSI is 100 there by definition.
    return rsi.where(avg_loss > 0, 100.0).where(avg_gain > 0, rsi.fillna(50.0))


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def build_features(
    prices: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build the causal feature matrix for one symbol's OHLCV history.

    Returns a frame indexed like ``prices`` (minus the warm-up rows that cannot
    be computed). Feature scales are mostly unit-free — returns, ratios and
    z-scores — so the same model configuration transfers across tickers with
    very different share prices.
    """
    config = config or FeatureConfig()

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted by date before building features")

    close = prices["close"].astype(float)
    high = prices["high"].astype(float)
    low = prices["low"].astype(float)
    volume = prices["volume"].astype(float)

    log_return = _log_returns(close)
    out: dict[str, pd.Series] = {}

    # --- momentum / mean-reversion -------------------------------------------
    for lag in config.return_lags:
        # ret_lag_1 is today's return, which is known at today's close.
        out[f"ret_lag_{lag}"] = log_return.shift(lag - 1)

    for window in config.momentum_windows:
        out[f"mom_{window}"] = log_return.rolling(window, min_periods=window).sum()

    # --- volatility ----------------------------------------------------------
    for window in config.volatility_windows:
        vol = log_return.rolling(window, min_periods=window).std(ddof=0)
        out[f"vol_{window}"] = vol * np.sqrt(TRADING_DAYS)

    short_vol_window = min(config.volatility_windows)
    long_vol_window = max(config.volatility_windows)
    if short_vol_window != long_vol_window:
        short_vol = log_return.rolling(short_vol_window, min_periods=short_vol_window).std(ddof=0)
        long_vol = log_return.rolling(long_vol_window, min_periods=long_vol_window).std(ddof=0)
        out["vol_ratio"] = short_vol / long_vol.replace(0.0, np.nan)

    # --- trend location ------------------------------------------------------
    for window in config.sma_windows:
        sma = close.rolling(window, min_periods=window).mean()
        out[f"close_over_sma_{window}"] = close / sma - 1.0

    if len(config.sma_windows) >= 2:
        fast, slow = sorted(config.sma_windows)[:2]
        fast_sma = close.rolling(fast, min_periods=fast).mean()
        slow_sma = close.rolling(slow, min_periods=slow).mean()
        out[f"sma_{fast}_over_{slow}"] = fast_sma / slow_sma - 1.0

    # Position within the trailing range: 0 = at the low, 1 = at the high.
    for window in config.range_windows:
        window_high = high.rolling(window, min_periods=window).max()
        window_low = low.rolling(window, min_periods=window).min()
        span = (window_high - window_low).replace(0.0, np.nan)
        out[f"range_pos_{window}"] = (close - window_low) / span

    # --- oscillators ---------------------------------------------------------
    rsi = _rsi(close, config.rsi_period)
    out[f"rsi_{config.rsi_period}"] = rsi / 100.0 - 0.5

    fast_span, slow_span, signal_span = config.macd
    macd_line = (
        close.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
        - close.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()
    )
    signal_line = macd_line.ewm(span=signal_span, adjust=False, min_periods=signal_span).mean()
    # Normalised by price so the scale is comparable across tickers.
    out["macd"] = macd_line / close
    out["macd_hist"] = (macd_line - signal_line) / close

    # --- range / liquidity ---------------------------------------------------
    true_range = _true_range(prices)
    atr = true_range.ewm(
        alpha=1.0 / config.atr_period, adjust=False, min_periods=config.atr_period
    ).mean()
    out[f"atr_{config.atr_period}"] = atr / close

    log_volume = np.log(volume.replace(0.0, np.nan))
    for window in config.volume_windows:
        mean = log_volume.rolling(window, min_periods=window).mean()
        std = log_volume.rolling(window, min_periods=window).std(ddof=0)
        out[f"volume_z_{window}"] = (log_volume - mean) / std.replace(0.0, np.nan)

    out["overnight_gap"] = np.log(prices["open"].astype(float) / close.shift(1))
    out["intraday_return"] = np.log(close / prices["open"].astype(float))

    # --- calendar ------------------------------------------------------------
    if config.include_calendar:
        index = prices.index
        out["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 5.0)
        out["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 5.0)
        out["month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * (index.month - 1) / 12.0)

    features = pd.DataFrame(out, index=prices.index)
    features = features.replace([np.inf, -np.inf], np.nan)

    if config.winsorize_quantile:
        features = _winsorize_expanding(features, config.winsorize_quantile)

    return features


def _winsorize_expanding(features: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Clip extreme feature values using only past observations.

    A conventional winsorisation computes quantiles over the whole sample,
    which leaks future information into the training window. Expanding
    quantiles keep the operation causal at the cost of being noisier early on.
    """
    lower = features.expanding(min_periods=60).quantile(quantile)
    upper = features.expanding(min_periods=60).quantile(1.0 - quantile)
    clipped = features.clip(lower=lower, upper=upper, axis=1)
    # Before min_periods is reached the bounds are NaN; keep the raw values.
    return clipped.where(~lower.isna(), features)


def make_target(
    prices: pd.DataFrame,
    horizon: int = 5,
    kind: str = "log_return",
) -> pd.Series:
    """Forward return over the next ``horizon`` bars, indexed at decision time.

    Row ``t`` holds the return realised from bar ``t``'s close to bar
    ``t + horizon``'s close. The last ``horizon`` rows are NaN because that
    return has not happened yet — those rows are exactly where live forecasts
    are made.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = prices["close"].astype(float)
    forward = close.shift(-horizon)

    if kind == "log_return":
        target = np.log(forward / close)
    elif kind == "simple_return":
        target = forward / close - 1.0
    elif kind == "direction":
        target = (forward > close).astype(float).where(forward.notna())
    else:
        raise ValueError(f"unknown target kind {kind!r}")

    return pd.Series(target, index=prices.index, name=f"target_{kind}_{horizon}")


def assemble_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    config: FeatureConfig | None = None,
    target_kind: str = "log_return",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build ``(X, y, X_live)`` for one symbol.

    ``X``/``y`` are the rows where both features and a realised target exist —
    the supervised training set. ``X_live`` holds the most recent rows whose
    features are complete but whose target has not resolved yet; these are the
    rows a live forecast is made from.
    """
    features = build_features(prices, config)
    target = make_target(prices, horizon=horizon, kind=target_kind)

    complete_features = features.dropna(how="any")
    aligned_target = target.reindex(complete_features.index)

    trainable = aligned_target.notna()
    X = complete_features.loc[trainable]
    y = aligned_target.loc[trainable]
    X_live = complete_features.loc[~trainable]

    return X, y, X_live
