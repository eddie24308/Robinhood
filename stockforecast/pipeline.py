"""Orchestration: evaluate, forecast, and screen.

The intended order of operations is always the same:

1. :func:`evaluate_symbol` measures out-of-sample skill by walk-forward
   validation, and calibrates prediction intervals on the same held-out
   predictions.
2. :func:`forecast_symbol` refits on all available history and produces a live
   forecast — but only ever bundled with the evaluation from step 1.

There is no path through this module that yields a point forecast without its
measured skill. That is intentional.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from stockforecast.data import DataError, DataProvider, load_prices
from stockforecast.features import FeatureConfig, assemble_dataset
from stockforecast.intervals import ConformalCalibrator, PredictionInterval, empirical_coverage
from stockforecast.metrics import ForecastMetrics, evaluate_predictions
from stockforecast.models import build_model, feature_importances
from stockforecast.validation import WalkForwardSplitter


@dataclass
class EvaluationResult:
    """Walk-forward results for one symbol."""

    symbol: str
    model_name: str
    horizon: int
    n_bars: int
    n_folds: int
    metrics: ForecastMetrics
    baseline_metrics: dict[str, ForecastMetrics]
    oos_predictions: pd.DataFrame
    interval_half_width: float
    interval_alpha: float
    realised_coverage: float
    top_features: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_skill(self) -> bool:
        return self.metrics.beats_baseline


@dataclass
class Forecast:
    """A live forecast, inseparable from the evidence about its reliability."""

    symbol: str
    as_of: pd.Timestamp
    last_close: float
    horizon: int
    model_name: str
    interval: PredictionInterval
    evaluation: EvaluationResult

    @property
    def expected_return(self) -> float:
        """Point forecast as a simple return over the horizon."""
        return float(np.exp(self.interval.point) - 1.0)

    @property
    def price_range(self) -> tuple[float, float, float]:
        return self.interval.as_price_range(self.last_close)

    @property
    def has_skill(self) -> bool:
        return self.evaluation.has_skill

    @property
    def trustworthiness(self) -> str:
        if not self.has_skill:
            return "NO MEASURED SKILL - treat the point estimate as noise"
        if self.evaluation.metrics.skill_rmse < 0.01:
            return "marginal skill - statistically detectable but economically small"
        return "measurable skill vs baseline"


def evaluate_symbol(
    symbol: str,
    prices: pd.DataFrame,
    model_name: str = "ridge",
    horizon: int = 5,
    n_splits: int = 5,
    min_train_size: int = 250,
    embargo: int = 5,
    alpha: float = 0.2,
    feature_config: FeatureConfig | None = None,
    max_train_size: int | None = None,
    baselines: Sequence[str] = ("zero", "mean", "momentum"),
    model_kwargs: dict | None = None,
) -> EvaluationResult:
    """Walk-forward evaluate one symbol and calibrate its prediction intervals."""
    config = feature_config or FeatureConfig()
    X, y, _ = assemble_dataset(prices, horizon=horizon, config=config)

    if X.empty:
        raise DataError(
            f"{symbol}: no usable rows. The feature set needs {config.warmup_bars} bars of "
            f"warm-up plus {horizon} for the target, but only {len(prices)} bars were loaded. "
            "Load more history, or use the short-history feature preset "
            "(--preset short / FeatureConfig.short_history())."
        )

    splitter = WalkForwardSplitter(
        n_splits=n_splits,
        horizon=horizon,
        embargo=embargo,
        min_train_size=min_train_size,
        max_train_size=max_train_size,
    )
    folds = list(splitter.split(len(X)))
    if not folds:
        raise DataError(f"{symbol}: {splitter.describe_requirements(len(X))}")

    collected_warnings: list[str] = []

    model_predictions: list[np.ndarray] = []
    baseline_predictions: dict[str, list[np.ndarray]] = {name: [] for name in baselines}
    truths: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    last_fitted = None

    for fold in folds:
        X_train, y_train = X.iloc[fold.train], y.iloc[fold.train]
        X_test, y_test = X.iloc[fold.test], y.iloc[fold.test]

        model = build_model(model_name, **(model_kwargs or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            model_predictions.append(np.asarray(model.predict(X_test), dtype=float))

            for name in baselines:
                baseline = build_model(name)
                baseline.fit(X_train, y_train)
                baseline_predictions[name].append(
                    np.asarray(baseline.predict(X_test), dtype=float)
                )

        truths.append(np.asarray(y_test, dtype=float))
        timestamps.append(X_test.index)
        last_fitted = model

    y_true = np.concatenate(truths)
    y_pred = np.concatenate(model_predictions)
    index = pd.DatetimeIndex(np.concatenate([ts.values for ts in timestamps]), name="date")

    stacked_baselines = {
        name: np.concatenate(values) for name, values in baseline_predictions.items()
    }
    # "zero" is the reference the headline metrics are computed against.
    reference = stacked_baselines.get("zero", np.zeros_like(y_true))

    metrics = evaluate_predictions(y_true, y_pred, reference, horizon=horizon)
    baseline_metrics = {
        name: evaluate_predictions(y_true, values, reference, horizon=horizon)
        for name, values in stacked_baselines.items()
    }

    # Calibrate intervals on held-out residuals only.
    try:
        calibrator = ConformalCalibrator(alpha=alpha).fit(y_true, y_pred)
        half_width = calibrator.quantile()
        coverage = empirical_coverage(y_true, y_pred, half_width)
    except ValueError as exc:
        collected_warnings.append(f"conformal calibration unavailable: {exc}")
        half_width = float("nan")
        coverage = float("nan")

    if np.isfinite(coverage) and coverage < (1 - alpha) - 0.07:
        collected_warnings.append(
            f"interval coverage {coverage:.0%} is below the nominal {1 - alpha:.0%}; "
            "intervals are optimistically narrow (volatility clustering)"
        )
    if len(X) < 500:
        collected_warnings.append(
            f"only {len(X)} usable training rows; results are noisy - prefer 3+ years of history"
        )
    if metrics.n < 100:
        collected_warnings.append(
            f"only {metrics.n} out-of-sample predictions; skill estimates are very imprecise"
        )

    oos = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, **stacked_baselines},
        index=index,
    ).sort_index()

    top = feature_importances(last_fitted, list(X.columns)) if last_fitted is not None else {}

    return EvaluationResult(
        symbol=symbol.upper(),
        model_name=model_name,
        horizon=horizon,
        n_bars=len(prices),
        n_folds=len(folds),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        oos_predictions=oos,
        interval_half_width=half_width,
        interval_alpha=alpha,
        realised_coverage=coverage,
        top_features=dict(list(top.items())[:10]),
        warnings=collected_warnings,
    )


def forecast_symbol(
    symbol: str,
    prices: pd.DataFrame,
    model_name: str = "ridge",
    horizon: int = 5,
    alpha: float = 0.2,
    feature_config: FeatureConfig | None = None,
    evaluation: EvaluationResult | None = None,
    **evaluate_kwargs: object,
) -> Forecast:
    """Produce a live forecast for ``symbol``, bundled with its evaluation.

    If ``evaluation`` is not supplied it is computed first — a forecast is
    never returned without one.
    """
    if evaluation is None:
        evaluation = evaluate_symbol(
            symbol,
            prices,
            model_name=model_name,
            horizon=horizon,
            alpha=alpha,
            feature_config=feature_config,
            **evaluate_kwargs,  # type: ignore[arg-type]
        )

    X, y, X_live = assemble_dataset(prices, horizon=horizon, config=feature_config)
    if X_live.empty:
        raise DataError(
            f"{symbol}: no live rows to forecast from; the history ends before a full "
            "feature window could be formed"
        )

    model = build_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
        point = float(np.asarray(model.predict(X_live.iloc[[-1]]), dtype=float)[0])

    half_width = evaluation.interval_half_width
    if not np.isfinite(half_width):
        # Fall back to a random-walk interval, clearly weaker but better than none.
        from stockforecast.intervals import volatility_scaled_interval  # noqa: PLC0415

        daily_volatility = float(np.log(prices["close"]).diff().tail(63).std(ddof=0))
        interval = volatility_scaled_interval(point, daily_volatility, horizon, alpha=alpha)
    else:
        interval = PredictionInterval(
            point=point,
            lower=point - half_width,
            upper=point + half_width,
            alpha=alpha,
            calibration_n=evaluation.metrics.n,
            empirical_coverage=evaluation.realised_coverage,
        )

    return Forecast(
        symbol=symbol.upper(),
        as_of=X_live.index[-1],
        last_close=float(prices["close"].iloc[-1]),
        horizon=horizon,
        model_name=model_name,
        interval=interval,
        evaluation=evaluation,
    )


def screen_symbols(
    symbols: Sequence[str],
    provider: DataProvider | str = "csv",
    model_name: str = "ridge",
    horizon: int = 5,
    start: str | None = None,
    end: str | None = None,
    alpha: float = 0.2,
    feature_config: FeatureConfig | None = None,
    verbose: bool = True,
    **evaluate_kwargs: object,
) -> tuple[list[Forecast], pd.DataFrame]:
    """Evaluate and forecast a whole universe.

    Returns ``(forecasts, summary_frame)``. The summary carries a
    Benjamini-Hochberg adjusted q-value per symbol: screening many tickers and
    keeping the best-looking one is the single easiest way to fool yourself,
    and the raw p-values do not account for it.
    """
    price_history = load_prices(symbols, provider=provider, start=start, end=end)

    forecasts: list[Forecast] = []
    rows: list[dict[str, object]] = []

    for symbol, prices in price_history.items():
        try:
            forecast = forecast_symbol(
                symbol,
                prices,
                model_name=model_name,
                horizon=horizon,
                alpha=alpha,
                feature_config=feature_config,
                **evaluate_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue the screen
            if verbose:
                print(f"  ! {symbol}: {exc}")
            continue

        forecasts.append(forecast)
        metrics = forecast.evaluation.metrics
        low, high = forecast.price_range[1], forecast.price_range[2]
        rows.append(
            {
                "symbol": symbol,
                "last_close": forecast.last_close,
                "expected_return": forecast.expected_return,
                "low": low,
                "high": high,
                "skill_rmse": metrics.skill_rmse,
                "r2_oos": metrics.r2_oos,
                "dir_acc": metrics.directional_accuracy,
                "ic": metrics.information_coefficient,
                "dm_p": metrics.dm_p_value,
                "n_oos": metrics.n,
                "has_skill": forecast.has_skill,
            }
        )
        if verbose:
            print(f"  {symbol:6s} {metrics.summary_line()}")

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_value"] = _benjamini_hochberg(summary["dm_p"].to_numpy())
        summary["skill_after_fdr"] = (summary["q_value"] < 0.05) & (summary["skill_rmse"] > 0)
        summary = summary.sort_values("expected_return", ascending=False).reset_index(drop=True)

    return forecasts, summary


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment; NaNs pass through untouched."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)

    finite = np.isfinite(p_values)
    if not finite.any():
        return q_values

    values = p_values[finite]
    n = values.size
    order = np.argsort(values)
    ranked = values[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p-value downwards.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty(n)
    restored[order] = adjusted
    q_values[finite] = restored
    return q_values
