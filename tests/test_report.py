"""Reporting tests.

These assert the package's user-facing promise: a forecast is never shown
without its skill evidence, and a model with no skill says so loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockforecast.data import SyntheticProvider
from stockforecast.pipeline import evaluate_symbol, forecast_symbol, screen_symbols
from stockforecast.report import _pct, format_evaluation, format_forecast, format_screen


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return SyntheticProvider(n_bars=1200, seed=77).fetch("SYN")


def test_nan_renders_as_not_available() -> None:
    assert _pct(float("nan")) == "n/a"
    assert _pct(np.nan, 2, signed=True) == "n/a"
    assert _pct(0.1234) == "12.3%"
    assert _pct(0.1234, 2, signed=True) == "+12.34%"


def test_evaluation_report_states_a_verdict(prices: pd.DataFrame) -> None:
    result = evaluate_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)
    text = format_evaluation(result)

    assert "WALK-FORWARD EVALUATION" in text
    assert "VERDICT" in text
    # Exactly one verdict must be stated.
    verdicts = [
        "NO SIGNIFICANT SKILL" in text,
        "SIGNIFICANTLY WORSE" in text,
        "Model beats the zero baseline" in text,
    ]
    assert sum(verdicts) == 1

    assert "nan%" not in text, "NaN leaked into user-facing output"


def test_evaluation_report_shows_baselines(prices: pd.DataFrame) -> None:
    result = evaluate_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)
    text = format_evaluation(result)

    for baseline in ("zero", "mean", "momentum"):
        assert baseline in text
    # The zero baseline abstains from direction; it must be labelled, not scored 0%.
    assert "no directional view" in text


def test_forecast_report_never_shows_a_bare_number(prices: pd.DataFrame) -> None:
    forecast = forecast_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)
    text = format_forecast(forecast)

    assert "expected return" in text
    # The point estimate is always accompanied by an interval and evidence.
    assert "interval" in text
    assert "RMSE skill vs baseline" in text
    assert "directional accuracy" in text
    assert "ASSESSMENT" in text
    assert "nan%" not in text


def test_forecast_report_warns_loudly_when_skill_is_absent(prices: pd.DataFrame) -> None:
    forecast = forecast_symbol(symbol="SYN", prices=prices, model_name="ridge", horizon=5)

    if not forecast.has_skill:
        text = format_forecast(forecast)
        assert "NO MEASURED OUT-OF-SAMPLE SKILL" in text
        assert "Do not trade on them." in text
    else:  # pragma: no cover - synthetic data should not produce skill
        pytest.skip("synthetic data unexpectedly showed skill")


def test_screen_report_foregrounds_multiple_testing() -> None:
    provider = SyntheticProvider(n_bars=1100, seed=88)
    _, summary = screen_symbols(
        [f"T{i}" for i in range(5)], provider=provider, horizon=5, verbose=False
    )
    text = format_screen(summary)

    assert "Benjamini-Hochberg" in text
    assert "false positives" in text
    assert "q" in text


def test_screen_report_handles_empty_frame() -> None:
    assert "No symbols" in format_screen(pd.DataFrame())
