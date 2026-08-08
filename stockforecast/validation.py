"""Walk-forward cross-validation with purging and embargo.

Standard k-fold cross-validation is invalid for time series: it trains on the
future to predict the past. Even a plain time-ordered split is subtly wrong
when the target is a multi-bar forward return, because the last few training
targets *overlap* the first test bars. The model then sees part of the answer.

:class:`WalkForwardSplitter` handles both problems:

* folds move forward in time only;
* the final ``horizon - 1`` training rows before each test fold are dropped
  (purging), because their target windows reach into the test period;
* an additional ``embargo`` of rows is dropped to blunt short-range
  autocorrelation between adjacent samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Fold:
    """One walk-forward fold, as positional indices into the dataset."""

    index: int
    train: np.ndarray
    test: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)


@dataclass
class WalkForwardSplitter:
    """Expanding- or rolling-window splits over a time-ordered dataset.

    Parameters
    ----------
    n_splits:
        Number of test folds.
    horizon:
        Target horizon in bars. Drives how many training rows are purged.
    embargo:
        Extra rows dropped from the end of training, on top of purging.
    min_train_size:
        Minimum training rows required before the first fold is emitted.
    max_train_size:
        If set, training uses a rolling window of at most this many rows
        instead of an expanding one.
    """

    n_splits: int = 5
    horizon: int = 5
    embargo: int = 5
    min_train_size: int = 250
    max_train_size: int | None = None

    def split(self, n_samples: int) -> Iterator[Fold]:
        """Yield folds for a dataset of ``n_samples`` time-ordered rows."""
        for fold in self._compute_folds(n_samples):
            yield fold

    def _compute_folds(self, n_samples: int) -> list[Fold]:
        if n_samples <= 0:
            return []

        purge = max(0, self.horizon - 1) + max(0, self.embargo)
        # Every fold needs training rows, purged rows, and its own test rows.
        usable = n_samples - self.min_train_size - purge
        if usable < self.n_splits:
            return []

        test_size = usable // self.n_splits
        if test_size < 1:
            return []

        folds: list[Fold] = []
        # Work backwards from the end so the most recent data is always tested.
        first_test_start = n_samples - test_size * self.n_splits

        for i in range(self.n_splits):
            test_start = first_test_start + i * test_size
            test_stop = test_start + test_size if i < self.n_splits - 1 else n_samples

            train_stop = test_start - purge
            if train_stop < self.min_train_size:
                continue

            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_stop - self.max_train_size)

            folds.append(
                Fold(
                    index=len(folds),
                    train=np.arange(train_start, train_stop),
                    test=np.arange(test_start, test_stop),
                )
            )

        return folds

    def n_usable_splits(self, n_samples: int) -> int:
        """How many folds this dataset actually supports (may be < n_splits)."""
        return len(self._compute_folds(n_samples))

    def describe_requirements(self, n_samples: int) -> str:
        """Explain why a dataset is too small, for user-facing errors."""
        purge = max(0, self.horizon - 1) + max(0, self.embargo)
        needed = self.min_train_size + purge + self.n_splits
        return (
            f"{n_samples} usable rows; need at least {needed} for {self.n_splits} folds "
            f"(min_train_size={self.min_train_size}, purge+embargo={purge}). "
            "Load more history, or lower --min-train / --splits."
        )
