"""Out-of-sample scoring, with emphasis on *skill relative to doing nothing*.

An RMSE of 0.04 on 5-day returns sounds precise until you notice that
predicting zero every day scores 0.0398. The metrics here always express model
quality relative to a baseline, and attach uncertainty to the headline numbers,
because a directional accuracy of 54% over 200 samples is indistinguishable
from a coin flip.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ForecastMetrics:
    """Out-of-sample scores for one model against one baseline."""

    n: int
    rmse: float
    mae: float
    baseline_rmse: float
    baseline_mae: float
    r2_oos: float
    skill_rmse: float
    directional_accuracy: float
    directional_ci_low: float
    directional_ci_high: float
    directional_p_value: float
    dm_statistic: float
    dm_p_value: float
    information_coefficient: float
    ic_p_value: float
    hit_rate_baseline: float = 0.5
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def beats_baseline(self) -> bool:
        """True only when the error reduction is statistically credible.

        Deliberately conservative: a model must both reduce error and clear a
        two-sided Diebold-Mariano test at the 5% level. Point-estimate
        improvements that fail this test are noise more often than not.
        """
        return self.skill_rmse > 0 and self.dm_p_value < 0.05

    @property
    def directional_edge_is_significant(self) -> bool:
        return self.directional_p_value < 0.05

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_line(self) -> str:
        def pct(value: float) -> str:
            return f"{value:+.2%}" if np.isfinite(value) else "n/a"

        def acc(value: float) -> str:
            return f"{value:.1%}" if np.isfinite(value) else "n/a"

        def num(value: float) -> str:
            return f"{value:+.3f}" if np.isfinite(value) else "n/a"

        verdict = "BEATS baseline" if self.beats_baseline else "no significant edge"
        return (
            f"n={self.n}  RMSE={self.rmse:.5f} vs baseline {self.baseline_rmse:.5f}  "
            f"skill={pct(self.skill_rmse)}  R2_oos={num(self.r2_oos)}  "
            f"dir={acc(self.directional_accuracy)} "
            f"[{acc(self.directional_ci_low)},{acc(self.directional_ci_high)}]  "
            f"IC={num(self.information_coefficient)}  -> {verdict}"
        )


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fraction of correct sign calls, with a Wilson CI and a vs-coin p-value.

    Two exclusions keep the number meaningful. Samples where the realised move
    was exactly zero are dropped, so flat bars cannot inflate the score. So are
    samples where the forecast is exactly zero: a zero forecast expresses no
    directional view, and scoring it as a miss would make the zero baseline
    look like a 0%-accuracy predictor rather than an abstention. When every
    forecast is zero the result is NaN, which is the truthful answer.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & (y_true != 0.0)
        & (y_pred != 0.0)
    )
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    correct = np.sign(y_pred[mask]) == np.sign(y_true[mask])
    n = int(mask.sum())
    k = int(correct.sum())
    accuracy = k / n

    # Wilson score interval — better than normal approximation for small n.
    z = 1.959963984540054
    denominator = 1.0 + z**2 / n
    centre = (accuracy + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(accuracy * (1 - accuracy) / n + z**2 / (4 * n**2))
    low, high = centre - margin, centre + margin

    p_value = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return accuracy, max(0.0, low), min(1.0, high), p_value


def diebold_mariano(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    horizon: int = 1,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal squared-error accuracy.

    Returns ``(statistic, two_sided_p_value)``. A positive statistic means the
    model has lower loss than the baseline. Uses a Newey-West variance with
    ``horizon - 1`` lags because overlapping forward returns make the loss
    differential autocorrelated; ignoring that inflates significance badly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_baseline = np.asarray(y_baseline, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_baseline)
    y_true, y_pred, y_baseline = y_true[mask], y_pred[mask], y_baseline[mask]

    n = y_true.size
    if n < 10:
        return float("nan"), float("nan")

    loss_differential = (y_baseline - y_true) ** 2 - (y_pred - y_true) ** 2
    mean_d = float(np.mean(loss_differential))

    centred = loss_differential - mean_d
    variance = float(np.mean(centred**2))
    for lag in range(1, max(1, horizon)):
        if lag >= n:
            break
        autocovariance = float(np.mean(centred[lag:] * centred[:-lag]))
        variance += 2.0 * (1.0 - lag / horizon) * autocovariance

    if variance <= 0 or not np.isfinite(variance):
        return float("nan"), float("nan")

    statistic = mean_d / math.sqrt(variance / n)
    # Harvey-Leybourne-Newbold small-sample correction.
    correction = math.sqrt(max(1e-12, (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n))
    statistic *= correction

    p_value = float(2 * (1 - stats.t.cdf(abs(statistic), df=n - 1)))
    return statistic, p_value


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray | None = None,
    horizon: int = 1,
) -> ForecastMetrics:
    """Score predictions against the truth and a baseline.

    When ``y_baseline`` is omitted, the zero-return forecast is used — the
    honest null hypothesis for short-horizon equity returns.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_baseline is None:
        y_baseline = np.zeros_like(y_true)
    y_baseline = np.asarray(y_baseline, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_baseline)
    y_true, y_pred, y_baseline = y_true[mask], y_pred[mask], y_baseline[mask]
    n = int(y_true.size)

    if n == 0:
        raise ValueError("no finite prediction/target pairs to score")

    errors = y_pred - y_true
    baseline_errors = y_baseline - y_true

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors**2)))
    baseline_mae = float(np.mean(np.abs(baseline_errors)))

    # Out-of-sample R^2 against the baseline. Negative means worse than doing
    # nothing, which is the common outcome and must not be hidden.
    sum_squared_error = float(np.sum(errors**2))
    sum_squared_baseline = float(np.sum(baseline_errors**2))
    r2_oos = 1.0 - sum_squared_error / sum_squared_baseline if sum_squared_baseline > 0 else float("nan")
    skill_rmse = 1.0 - rmse / baseline_rmse if baseline_rmse > 0 else float("nan")

    accuracy, ci_low, ci_high, dir_p = directional_accuracy(y_true, y_pred)
    dm_statistic, dm_p = diebold_mariano(y_true, y_pred, y_baseline, horizon=horizon)

    # Information coefficient: rank correlation between forecast and outcome.
    if n >= 3 and np.std(y_pred) > 0:
        ic, ic_p = stats.spearmanr(y_pred, y_true)
        ic, ic_p = float(ic), float(ic_p)
    else:
        ic, ic_p = float("nan"), float("nan")

    return ForecastMetrics(
        n=n,
        rmse=rmse,
        mae=mae,
        baseline_rmse=baseline_rmse,
        baseline_mae=baseline_mae,
        r2_oos=r2_oos,
        skill_rmse=skill_rmse,
        directional_accuracy=accuracy,
        directional_ci_low=ci_low,
        directional_ci_high=ci_high,
        directional_p_value=dir_p,
        dm_statistic=dm_statistic,
        dm_p_value=dm_p,
        information_coefficient=ic,
        ic_p_value=ic_p,
        hit_rate_baseline=float(np.mean(y_true > 0)) if n else float("nan"),
    )


def annualised_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe ratio of a return stream, zero risk-free rate."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2 or np.std(returns, ddof=1) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Largest peak-to-trough fractional decline of an equity curve."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    if equity_curve.size == 0:
        return float("nan")
    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_peak - 1.0
    return float(np.min(drawdowns))
