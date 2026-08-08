"""Split-conformal prediction intervals.

A point forecast of "+1.2% over 5 days" is close to meaningless without a
sense of spread. Conformal prediction supplies that with almost no
assumptions: given residuals from data the model did not train on, the
interval

    prediction +/- quantile(|residual|, 1 - alpha)

covers the truth at least ``1 - alpha`` of the time, provided residuals are
exchangeable. Financial residuals are not perfectly exchangeable — volatility
clusters — so the realised coverage is checked empirically and reported rather
than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionInterval:
    """A point forecast with a conformal interval attached."""

    point: float
    lower: float
    upper: float
    alpha: float
    calibration_n: int
    empirical_coverage: float | None = None

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_price_range(self, last_price: float) -> tuple[float, float, float]:
        """Convert log-return bounds to prices ``(point, low, high)``."""
        return (
            last_price * float(np.exp(self.point)),
            last_price * float(np.exp(self.lower)),
            last_price * float(np.exp(self.upper)),
        )


class ConformalCalibrator:
    """Calibrates absolute-residual quantiles from held-out predictions."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.residuals_: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> ConformalCalibrator:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        residuals = np.abs(y_true[mask] - y_pred[mask])
        if residuals.size < 20:
            raise ValueError(
                f"need >=20 calibration residuals for a usable interval, got {residuals.size}"
            )
        self.residuals_ = np.sort(residuals)
        return self

    def quantile(self) -> float:
        """Finite-sample-corrected absolute-residual quantile."""
        if self.residuals_ is None:
            raise RuntimeError("call fit() before quantile()")
        n = self.residuals_.size
        # The (n+1) correction is what makes the coverage guarantee finite-sample.
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        return float(np.quantile(self.residuals_, level, method="higher"))

    def interval(self, point: float) -> PredictionInterval:
        half_width = self.quantile()
        return PredictionInterval(
            point=float(point),
            lower=float(point - half_width),
            upper=float(point + half_width),
            alpha=self.alpha,
            calibration_n=int(self.residuals_.size) if self.residuals_ is not None else 0,
        )


def empirical_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    half_width: float,
) -> float:
    """Fraction of outcomes that actually fell inside +/- ``half_width``.

    Compare this to the nominal confidence level. Materially lower realised
    coverage means the intervals are too narrow to trust.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) <= half_width))


def volatility_scaled_interval(
    point: float,
    recent_volatility: float,
    horizon: int,
    alpha: float = 0.2,
) -> PredictionInterval:
    """Fallback interval from a random-walk volatility estimate.

    Used when there are too few calibration residuals for conformal intervals.
    Assumes returns scale with the square root of time and are roughly normal —
    both of which understate tail risk, so this is the weaker option and is
    labelled as such wherever it is surfaced.
    """
    from scipy import stats  # noqa: PLC0415 - keeps import cost off the hot path

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    half_width = z * recent_volatility * np.sqrt(horizon)
    return PredictionInterval(
        point=float(point),
        lower=float(point - half_width),
        upper=float(point + half_width),
        alpha=alpha,
        calibration_n=0,
    )
