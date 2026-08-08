"""End-to-end pipeline tests, including a deliberate-leak canary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockforecast.data import SyntheticProvider
from stockforecast.features import make_target
from stockforecast.pipeline import evaluate_symbol, forecast_symbol, screen_symbols


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return SyntheticProvider(n_bars=1400, seed=21).fetch("SYN")


def test_no_skill_found_on_signal_free_data(prices: pd.DataFrame) -> None:
    """The central correctness check.

    Synthetic prices carry no predictable drift. A pipeline free of lookahead
    must therefore report approximately no skill. A strong positive result here
    means something is leaking.
    """
    result = evaluate_symbol(prices=prices, symbol="SYN", model_name="ridge", horizon=5)

    assert result.metrics.n > 100
    assert result.metrics.r2_oos < 0.10, (
        f"R2_oos={result.metrics.r2_oos:.3f} on data with no signal - "
        "this indicates lookahead leakage"
    )
    assert result.metrics.directional_accuracy < 0.62


@pytest.mark.parametrize("model_name", ["ridge", "elasticnet", "gbm", "rf"])
def test_all_models_stay_honest_on_noise(prices: pd.DataFrame, model_name: str) -> None:
    result = evaluate_symbol(
        prices=prices, symbol="SYN", model_name=model_name, horizon=5, n_splits=4
    )
    assert result.metrics.r2_oos < 0.15, f"{model_name} scored impossibly well on noise"


def test_leak_canary_is_caught(prices: pd.DataFrame) -> None:
    """Prove the harness *would* detect a leak if one existed.

    A test that only ever checks "no skill found" passes trivially if the
    pipeline is broken and predicts nothing. So we inject the answer directly
    into the price history and confirm the metrics light up. If this test ever
    fails, the no-skill assertions above have stopped being meaningful.
    """
    leaked = prices.copy()
    future_return = make_target(prices, horizon=5).shift(0)
    # Contaminate volume with the future return: a strictly-causal feature
    # builder cannot exploit this, but our feature set reads volume at t, and
    # here volume at t encodes the t->t+5 move.
    leaked["volume"] = np.exp(12.0 + 30.0 * future_return.fillna(0.0))

    result = evaluate_symbol(prices=leaked, symbol="LEAK", model_name="gbm", horizon=5)

    assert result.metrics.r2_oos > 0.30, (
        "injected leakage was not detected; the no-skill assertions in this "
        "module cannot be trusted"
    )
    assert result.has_skill


def test_forecast_is_bundled_with_evaluation(prices: pd.DataFrame) -> None:
    forecast = forecast_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)

    assert forecast.evaluation is not None
    assert forecast.interval.lower < forecast.interval.point < forecast.interval.upper

    point, low, high = forecast.price_range
    assert low < point < high
    assert low > 0

    # The assessment string must state the skill situation either way.
    assert forecast.trustworthiness
    if not forecast.has_skill:
        assert "NO MEASURED SKILL" in forecast.trustworthiness


def test_forecast_interval_is_not_absurdly_narrow(prices: pd.DataFrame) -> None:
    """A 5-day interval tighter than ~1% would be claiming impossible precision."""
    forecast = forecast_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)
    assert forecast.interval.width > 0.02


def test_evaluation_warns_on_short_history() -> None:
    short_prices = SyntheticProvider(n_bars=430, seed=3).fetch("SHORT")
    result = evaluate_symbol(
        prices=short_prices,
        symbol="SHORT",
        model_name="ridge",
        horizon=5,
        n_splits=3,
        min_train_size=200,
    )
    assert any("usable training rows" in message for message in result.warnings)


def test_too_little_data_raises_clear_error() -> None:
    tiny = SyntheticProvider(n_bars=180, seed=4).fetch("TINY")
    with pytest.raises(Exception, match="usable rows"):
        evaluate_symbol(prices=tiny, symbol="TINY", model_name="ridge", horizon=5)


def test_screen_applies_multiple_testing_correction() -> None:
    provider = SyntheticProvider(n_bars=1200, seed=31)
    symbols = [f"SYN{i}" for i in range(6)]

    _, summary = screen_symbols(
        symbols, provider=provider, model_name="ridge", horizon=5, verbose=False
    )

    assert len(summary) == len(symbols)
    assert {"q_value", "skill_after_fdr", "expected_return"} <= set(summary.columns)
    # q-values are at least as large as the raw p-values.
    assert (summary["q_value"] >= summary["dm_p"] - 1e-12).all()
    # On pure noise, essentially nothing should survive FDR control.
    assert int(summary["skill_after_fdr"].sum()) <= 1


def test_screen_sorted_by_expected_return() -> None:
    provider = SyntheticProvider(n_bars=1200, seed=41)
    _, summary = screen_symbols(
        [f"S{i}" for i in range(4)], provider=provider, horizon=5, verbose=False
    )
    assert summary["expected_return"].is_monotonic_decreasing
