"""Walk-forward splitter tests: ordering, purging and embargo."""

from __future__ import annotations

import numpy as np
import pytest

from stockforecast.validation import WalkForwardSplitter


def test_train_always_precedes_test() -> None:
    splitter = WalkForwardSplitter(n_splits=5, horizon=5, embargo=5, min_train_size=200)
    folds = list(splitter.split(1000))

    assert folds, "expected folds for a 1000-row dataset"
    for fold in folds:
        assert fold.train.max() < fold.test.min(), "training data leaks past the test start"


def test_purge_and_embargo_gap_is_respected() -> None:
    horizon, embargo = 10, 3
    splitter = WalkForwardSplitter(
        n_splits=4, horizon=horizon, embargo=embargo, min_train_size=200
    )

    expected_gap = (horizon - 1) + embargo
    for fold in splitter.split(1200):
        gap = fold.test.min() - fold.train.max() - 1
        assert gap >= expected_gap, (
            f"gap of {gap} rows is smaller than the required {expected_gap}; "
            "overlapping targets would leak into the test fold"
        )


def test_test_folds_are_contiguous_and_ordered() -> None:
    splitter = WalkForwardSplitter(n_splits=5, horizon=5, embargo=2, min_train_size=150)
    folds = list(splitter.split(900))

    for fold in folds:
        assert np.array_equal(fold.test, np.arange(fold.test.min(), fold.test.max() + 1))

    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test.max() < later.test.min(), "test folds must move forward in time"


def test_folds_cover_the_end_of_the_sample() -> None:
    """The most recent data must always be tested — it is the regime that matters."""
    n_samples = 800
    splitter = WalkForwardSplitter(n_splits=4, horizon=5, embargo=5, min_train_size=200)
    folds = list(splitter.split(n_samples))

    assert folds[-1].test.max() == n_samples - 1


def test_expanding_window_grows() -> None:
    splitter = WalkForwardSplitter(n_splits=4, horizon=5, embargo=5, min_train_size=200)
    folds = list(splitter.split(1000))

    sizes = [fold.n_train for fold in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_rolling_window_is_capped() -> None:
    cap = 300
    splitter = WalkForwardSplitter(
        n_splits=4, horizon=5, embargo=5, min_train_size=200, max_train_size=cap
    )
    for fold in splitter.split(1500):
        assert fold.n_train <= cap


def test_too_small_dataset_yields_no_folds() -> None:
    splitter = WalkForwardSplitter(n_splits=5, horizon=5, embargo=5, min_train_size=250)
    assert list(splitter.split(100)) == []
    assert splitter.n_usable_splits(100) == 0

    message = splitter.describe_requirements(100)
    assert "100 usable rows" in message
    assert "need at least" in message


@pytest.mark.parametrize("n_samples", [400, 750, 1500, 4000])
def test_no_index_is_reused_across_train_and_test(n_samples: int) -> None:
    splitter = WalkForwardSplitter(n_splits=5, horizon=5, embargo=5, min_train_size=200)
    for fold in splitter.split(n_samples):
        assert not set(fold.train) & set(fold.test)
        assert fold.test.max() < n_samples
