"""Metric tests, with emphasis on the honesty properties."""

from __future__ import annotations

import numpy as np
import pytest

from stockforecast.intervals import ConformalCalibrator, empirical_coverage
from stockforecast.metrics import (
    diebold_mariano,
    directional_accuracy,
    evaluate_predictions,
    max_drawdown,
)
from stockforecast.pipeline import _benjamini_hochberg


def test_perfect_forecast_scores_perfectly() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.02, 500)

    metrics = evaluate_predictions(y_true, y_true.copy())

    assert metrics.rmse == pytest.approx(0.0, abs=1e-12)
    assert metrics.r2_oos == pytest.approx(1.0)
    assert metrics.directional_accuracy == pytest.approx(1.0)


def test_zero_forecast_has_exactly_zero_skill() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.normal(0, 0.02, 500)
    y_pred = np.zeros_like(y_true)

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.skill_rmse == pytest.approx(0.0)
    assert metrics.r2_oos == pytest.approx(0.0)
    assert not metrics.beats_baseline


def test_pure_noise_forecast_is_reported_as_no_skill() -> None:
    """The headline property: noise must not be sold as signal."""
    rng = np.random.default_rng(2)
    y_true = rng.normal(0, 0.02, 800)
    y_pred = rng.normal(0, 0.02, 800)  # independent of y_true

    metrics = evaluate_predictions(y_true, y_pred)

    assert not metrics.beats_baseline
    assert metrics.r2_oos < 0.05
    assert metrics.directional_ci_low < 0.5 < metrics.directional_ci_high


def test_beats_baseline_requires_significance_not_just_improvement() -> None:
    """A real but tiny edge on few samples must not be reported as skill.

    ``y_pred`` here is the true conditional mean of ``y_true``, so it is
    genuinely informative - but the signal is swamped by noise and n is small.
    Point-estimate RMSE improves; that improvement is not distinguishable from
    luck, and ``beats_baseline`` must say so.
    """
    rng = np.random.default_rng(5)
    signal = rng.normal(0, 0.02, 60)
    y_true = 0.05 * signal + rng.normal(0, 0.02, 60)
    y_pred = 0.05 * signal

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.skill_rmse > 0, "expected an apparent improvement"
    assert metrics.dm_p_value > 0.05
    assert not metrics.beats_baseline


def test_genuine_signal_is_detected() -> None:
    """The flip side: a real edge must actually be found."""
    rng = np.random.default_rng(4)
    signal = rng.normal(0, 0.02, 1500)
    noise = rng.normal(0, 0.01, 1500)
    y_true = signal + noise

    metrics = evaluate_predictions(y_true, signal)

    assert metrics.beats_baseline
    assert metrics.r2_oos > 0.5
    assert metrics.directional_accuracy > 0.7


def test_zero_forecast_has_no_directional_opinion() -> None:
    """An abstention must not be scored as 0% accuracy."""
    rng = np.random.default_rng(12)
    y_true = rng.normal(0, 0.02, 200)

    accuracy, _, _, _ = directional_accuracy(y_true, np.zeros_like(y_true))
    assert np.isnan(accuracy)

    metrics = evaluate_predictions(y_true, np.zeros_like(y_true))
    assert np.isnan(metrics.directional_accuracy)


def test_directional_accuracy_ignores_flat_outcomes() -> None:
    y_true = np.array([0.01, -0.01, 0.0, 0.0, 0.02])
    y_pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    accuracy, low, high, _ = directional_accuracy(y_true, y_pred)

    assert accuracy == pytest.approx(1.0)  # the two zero rows are excluded
    assert 0.0 <= low <= high <= 1.0


def test_diebold_mariano_sign_and_symmetry() -> None:
    rng = np.random.default_rng(5)
    y_true = rng.normal(0, 0.02, 600)
    good = y_true + rng.normal(0, 0.005, 600)
    bad = rng.normal(0, 0.02, 600)

    statistic, p_value = diebold_mariano(y_true, good, bad, horizon=1)
    assert statistic > 0 and p_value < 0.01

    reversed_statistic, reversed_p = diebold_mariano(y_true, bad, good, horizon=1)
    assert reversed_statistic < 0
    assert reversed_p == pytest.approx(p_value, rel=1e-6)


def test_diebold_mariano_widens_for_overlapping_horizons() -> None:
    """Overlapping targets must reduce, not inflate, confidence."""
    rng = np.random.default_rng(6)
    y_true = rng.normal(0, 0.02, 500)
    y_pred = y_true * 0.15 + rng.normal(0, 0.02, 500)

    _, p_h1 = diebold_mariano(y_true, y_pred, np.zeros_like(y_true), horizon=1)
    _, p_h10 = diebold_mariano(y_true, y_pred, np.zeros_like(y_true), horizon=10)

    assert p_h10 > p_h1


def test_conformal_interval_achieves_nominal_coverage() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(0, 0.02, 2000)
    y_pred = np.zeros_like(y_true)

    calibrator = ConformalCalibrator(alpha=0.2).fit(y_true[:1000], y_pred[:1000])
    half_width = calibrator.quantile()

    coverage = empirical_coverage(y_true[1000:], y_pred[1000:], half_width)
    assert coverage == pytest.approx(0.8, abs=0.05)


def test_conformal_requires_enough_calibration_data() -> None:
    with pytest.raises(ValueError, match="need >=20"):
        ConformalCalibrator().fit(np.zeros(5), np.zeros(5))


def test_benjamini_hochberg_controls_false_discoveries() -> None:
    # 100 null p-values, uniform: at q<0.05 we expect essentially no survivors.
    rng = np.random.default_rng(8)
    p_values = rng.uniform(0, 1, 100)

    q_values = _benjamini_hochberg(p_values)

    assert np.all(q_values >= p_values - 1e-12)
    assert (q_values < 0.05).sum() <= 1
    # Monotonic in the same order as the raw p-values.
    order = np.argsort(p_values)
    assert np.all(np.diff(q_values[order]) >= -1e-12)


def test_benjamini_hochberg_keeps_strong_signals() -> None:
    p_values = np.array([1e-8, 1e-7, 0.4, 0.6, 0.9])
    q_values = _benjamini_hochberg(p_values)

    assert q_values[0] < 0.05
    assert q_values[1] < 0.05
    assert q_values[2] > 0.05


def test_max_drawdown() -> None:
    equity = np.array([1.0, 1.2, 0.9, 1.1, 0.6])
    assert max_drawdown(equity) == pytest.approx(0.6 / 1.2 - 1.0)
