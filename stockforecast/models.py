"""Model zoo: naive baselines plus a few well-regularised learners.

Model choice matters far less than validation discipline for this problem. The
signal-to-noise ratio in daily equity returns is low enough that flexible
models mostly fit noise, so the defaults here are heavily regularised and the
baselines are first-class citizens rather than an afterthought.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ZeroForecast(BaseEstimator, RegressorMixin):
    """Predicts zero return always.

    The honest null for short-horizon returns, and a surprisingly hard baseline
    to beat on RMSE.
    """

    def fit(self, X, y=None):  # noqa: N803, ANN001
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):  # noqa: N803, ANN001
        return np.zeros(np.asarray(X).shape[0])


class MeanForecast(BaseEstimator, RegressorMixin):
    """Predicts the training-set mean return (a drift-only model)."""

    def fit(self, X, y):  # noqa: N803, ANN001
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):  # noqa: N803, ANN001
        return np.full(np.asarray(X).shape[0], getattr(self, "mean_", 0.0))


class MomentumForecast(BaseEstimator, RegressorMixin):
    """Scaled recent momentum, a classic weak-signal benchmark.

    Included so that learned models are compared against a plausible rule of
    thumb, not just against zero.
    """

    def __init__(self, feature: str = "mom_21", scale: float = 0.05) -> None:
        self.feature = feature
        self.scale = scale

    def fit(self, X, y):  # noqa: N803, ANN001
        columns = list(getattr(X, "columns", []))
        self.column_index_ = columns.index(self.feature) if self.feature in columns else None
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):  # noqa: N803, ANN001
        values = np.asarray(X, dtype=float)
        if getattr(self, "column_index_", None) is None:
            return np.zeros(values.shape[0])
        return self.scale * values[:, self.column_index_]


def _ridge(**kwargs: object) -> Pipeline:
    params = {"alpha": 10.0, "random_state": None}
    params.update(kwargs)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(params["alpha"]))),
        ]
    )


def _elastic_net(**kwargs: object) -> Pipeline:
    params: dict[str, object] = {"alpha": 0.001, "l1_ratio": 0.5}
    params.update(kwargs)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=float(params["alpha"]),
                    l1_ratio=float(params["l1_ratio"]),
                    max_iter=20_000,
                    random_state=0,
                ),
            ),
        ]
    )


def _gradient_boosting(**kwargs: object) -> GradientBoostingRegressor:
    params: dict[str, object] = {
        "n_estimators": 200,
        "learning_rate": 0.02,
        "max_depth": 2,
        "subsample": 0.7,
        "min_samples_leaf": 20,
    }
    params.update(kwargs)
    return GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        random_state=0,
    )


def _random_forest(**kwargs: object) -> RandomForestRegressor:
    params: dict[str, object] = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 25,
        "max_features": 0.5,
    }
    params.update(kwargs)
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        random_state=0,
        n_jobs=-1,
    )


BASELINES: dict[str, Callable[..., BaseEstimator]] = {
    "zero": lambda **_: ZeroForecast(),
    "mean": lambda **_: MeanForecast(),
    "momentum": lambda **kwargs: MomentumForecast(**kwargs),  # type: ignore[arg-type]
}

MODELS: dict[str, Callable[..., BaseEstimator]] = {
    **BASELINES,
    "ridge": _ridge,
    "elasticnet": _elastic_net,
    "gbm": _gradient_boosting,
    "rf": _random_forest,
}


def build_model(name: str, **kwargs: object) -> BaseEstimator:
    """Instantiate a model by name."""
    try:
        factory = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose one of {sorted(MODELS)}") from None
    return factory(**kwargs)


def feature_importances(model: BaseEstimator, feature_names: list[str]) -> dict[str, float]:
    """Best-effort importance extraction, normalised to sum to 1.

    Returns an empty dict for models that expose neither coefficients nor
    tree importances, rather than inventing numbers.
    """
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        return {}

    if values.size != len(feature_names) or not np.isfinite(values).any():
        return {}

    total = float(values.sum())
    if total <= 0:
        return {}

    ranked = sorted(zip(feature_names, values / total), key=lambda kv: kv[1], reverse=True)
    return {name: float(weight) for name, weight in ranked}
