"""Feature causality tests.

The most important test in the package is ``test_features_are_causal``. If
features could see the future, every downstream metric would be inflated and
the whole thing would be a very convincing lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockforecast.data import SyntheticProvider
from stockforecast.features import (
    FeatureConfig,
    assemble_dataset,
    build_features,
    make_target,
)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return SyntheticProvider(n_bars=600, seed=11).fetch("TEST")


def test_features_are_causal(prices: pd.DataFrame) -> None:
    """Truncating the future must not change already-computed feature rows.

    This is the mechanical definition of a causal transform: what the model
    could have known on day t cannot depend on day t+1 existing.
    """
    full = build_features(prices)

    for cutoff in (300, 420, 555):
        truncated = build_features(prices.iloc[:cutoff])
        overlap = truncated.index.intersection(full.index)
        assert len(overlap) > 50, "truncation left too few rows to compare"

        left = full.loc[overlap]
        right = truncated.loc[overlap]

        pd.testing.assert_frame_equal(
            left,
            right,
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
            obj=f"features differ after truncating at {cutoff} - LOOKAHEAD LEAK",
        )


def test_no_feature_correlates_perfectly_with_future_return(prices: pd.DataFrame) -> None:
    """A near-perfect correlation with the future is the signature of a leak."""
    features = build_features(prices)
    target = make_target(prices, horizon=5)

    aligned = features.join(target.rename("target")).dropna()
    correlations = aligned.corr()["target"].drop("target").abs()

    worst = correlations.max()
    assert worst < 0.5, (
        f"feature {correlations.idxmax()!r} correlates {worst:.3f} with the future "
        "return on data with no real signal - almost certainly a leak"
    )


def test_target_is_forward_return(prices: pd.DataFrame) -> None:
    horizon = 5
    target = make_target(prices, horizon=horizon)
    close = prices["close"]

    for position in (10, 100, 250):
        expected = np.log(close.iloc[position + horizon] / close.iloc[position])
        assert target.iloc[position] == pytest.approx(expected)


def test_target_tail_is_unresolved(prices: pd.DataFrame) -> None:
    """The last `horizon` targets have not happened yet and must be NaN."""
    horizon = 7
    target = make_target(prices, horizon=horizon)

    assert target.iloc[-horizon:].isna().all()
    assert target.iloc[: -horizon].notna().all()


def test_direction_target_matches_sign(prices: pd.DataFrame) -> None:
    log_target = make_target(prices, horizon=3, kind="log_return").dropna()
    direction = make_target(prices, horizon=3, kind="direction").dropna()

    assert ((log_target > 0).astype(float) == direction).all()


def test_assemble_dataset_separates_live_rows(prices: pd.DataFrame) -> None:
    horizon = 5
    X, y, X_live = assemble_dataset(prices, horizon=horizon)

    assert not X.empty and not y.isna().any()
    # Live rows are exactly the unresolved tail.
    assert len(X_live) == horizon
    assert X_live.index.max() == prices.index.max()
    # Train and live rows must not overlap.
    assert X.index.intersection(X_live.index).empty
    # Every train row precedes every live row.
    assert X.index.max() < X_live.index.min()


def test_features_have_no_infinities(prices: pd.DataFrame) -> None:
    features = build_features(prices)
    assert not np.isinf(features.to_numpy(dtype=float)).any()


def test_config_controls_columns(prices: pd.DataFrame) -> None:
    config = FeatureConfig(
        return_lags=(1, 2),
        momentum_windows=(5,),
        volatility_windows=(10,),
        sma_windows=(20,),
        volume_windows=(5,),
        include_calendar=False,
    )
    features = build_features(prices, config)

    assert "ret_lag_1" in features.columns
    assert "ret_lag_10" not in features.columns
    assert not any(column.startswith("dow_") for column in features.columns)


def test_unsorted_input_is_rejected(prices: pd.DataFrame) -> None:
    shuffled = prices.iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        build_features(shuffled)
