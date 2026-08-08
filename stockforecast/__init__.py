"""stockforecast — an honest short-horizon return forecasting toolkit for equities.

The design goal of this package is *not* to produce impressive-looking price
predictions. It is to produce forecasts whose measured out-of-sample skill is
reported alongside them, so a user can tell whether the number is worth
anything.

Three rules are enforced throughout:

1. Every feature is strictly causal. A feature at bar ``t`` uses only data from
   bars ``<= t``. This is unit-tested, not just asserted.
2. Every model is scored against naive baselines on walk-forward splits with
   purging and embargo. A model that cannot beat "tomorrow's return is zero"
   is reported as having no skill.
3. Every point forecast is accompanied by a conformal prediction interval and
   the model's measured skill. The reporting layer refuses to emit a bare
   point estimate.
"""

__version__ = "0.1.0"

from stockforecast.data import (
    OHLCV_COLUMNS,
    CsvProvider,
    DataProvider,
    RobinhoodMcpJsonProvider,
    SyntheticProvider,
    YFinanceProvider,
    get_provider,
    load_prices,
    validate_ohlcv,
)
from stockforecast.features import FeatureConfig, build_features, make_target
from stockforecast.metrics import (
    ForecastMetrics,
    diebold_mariano,
    directional_accuracy,
    evaluate_predictions,
)
from stockforecast.models import BASELINES, MODELS, build_model
from stockforecast.pipeline import (
    EvaluationResult,
    Forecast,
    evaluate_symbol,
    forecast_symbol,
    screen_symbols,
)
from stockforecast.validation import WalkForwardSplitter

__all__ = [
    "OHLCV_COLUMNS",
    "BASELINES",
    "MODELS",
    "CsvProvider",
    "DataProvider",
    "EvaluationResult",
    "FeatureConfig",
    "Forecast",
    "ForecastMetrics",
    "RobinhoodMcpJsonProvider",
    "SyntheticProvider",
    "WalkForwardSplitter",
    "YFinanceProvider",
    "build_features",
    "build_model",
    "diebold_mariano",
    "directional_accuracy",
    "evaluate_predictions",
    "evaluate_symbol",
    "forecast_symbol",
    "get_provider",
    "load_prices",
    "make_target",
    "screen_symbols",
    "validate_ohlcv",
]
