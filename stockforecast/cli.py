"""Command-line interface.

    python -m stockforecast evaluate HOOD --provider csv --data-dir data
    python -m stockforecast forecast HOOD AAPL --horizon 5
    python -m stockforecast screen --symbols-file universe.txt --model gbm
    python -m stockforecast demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from stockforecast.data import DataError, get_provider, load_prices
from stockforecast.features import FeatureConfig
from stockforecast.models import MODELS
from stockforecast.pipeline import evaluate_symbol, forecast_symbol, screen_symbols
from stockforecast.report import format_evaluation, format_forecast, format_screen


def _provider_kwargs(args: argparse.Namespace) -> dict[str, object]:
    if args.provider == "csv":
        return {"directory": args.data_dir}
    if args.provider == "mcp-json":
        return {"path": args.data_dir}
    return {}


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    symbols = [symbol.upper() for symbol in getattr(args, "symbols", [])]

    if getattr(args, "symbols_file", None):
        path = Path(args.symbols_file)
        if not path.exists():
            raise SystemExit(f"symbols file not found: {path}")
        for line in path.read_text().splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                symbols.append(token.upper())

    if not symbols and args.provider == "csv":
        provider = get_provider("csv", directory=args.data_dir)
        symbols = provider.available_symbols()  # type: ignore[attr-defined]
        if symbols:
            print(f"No symbols given; using all {len(symbols)} CSVs in {args.data_dir}/")

    # De-duplicate, preserving order.
    return list(dict.fromkeys(symbols))


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("symbols", nargs="*", help="ticker symbols, e.g. HOOD AAPL MSFT")
    parser.add_argument("--symbols-file", help="file with one ticker per line")
    parser.add_argument(
        "--provider",
        default="csv",
        choices=["csv", "yfinance", "mcp-json", "synthetic"],
        help="where price history comes from (default: csv)",
    )
    parser.add_argument("--data-dir", default="data", help="directory for csv/mcp-json providers")
    parser.add_argument("--start", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", help="last date, YYYY-MM-DD")
    parser.add_argument("--model", default="ridge", choices=sorted(MODELS), help="model to fit")
    parser.add_argument("--horizon", type=int, default=5, help="forecast horizon in trading days")
    parser.add_argument("--splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--min-train", type=int, default=250, help="minimum training rows")
    parser.add_argument("--embargo", type=int, default=5, help="embargo rows between train/test")
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="rolling training window size (default: expanding)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="1 - confidence for prediction intervals (default 0.2 => 80%%)",
    )
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "short"],
        help=(
            "feature set. 'default' needs ~200 bars of warm-up; 'short' uses lighter "
            "windows for tickers with under a year of history"
        ),
    )
    parser.add_argument("--json-out", help="also write machine-readable results here")


def _feature_config(args: argparse.Namespace) -> FeatureConfig:
    if getattr(args, "preset", "default") == "short":
        return FeatureConfig.short_history()
    return FeatureConfig()


def _evaluate_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "n_splits": args.splits,
        "min_train_size": args.min_train,
        "embargo": args.embargo,
        "max_train_size": args.max_train,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("no symbols to evaluate (pass tickers, --symbols-file, or populate data/)")

    history = load_prices(
        symbols,
        provider=args.provider,
        start=args.start,
        end=args.end,
        **_provider_kwargs(args),
    )

    payload: list[dict[str, object]] = []
    failures = 0
    for symbol, prices in history.items():
        try:
            result = evaluate_symbol(
                symbol,
                prices,
                model_name=args.model,
                horizon=args.horizon,
                alpha=args.alpha,
                feature_config=_feature_config(args),
                **_evaluate_kwargs(args),
            )
        except DataError as exc:
            print(f"! {symbol}: {exc}")
            failures += 1
            continue

        print(format_evaluation(result))
        print()
        payload.append(
            {
                "symbol": result.symbol,
                "model": result.model_name,
                "horizon": result.horizon,
                "n_folds": result.n_folds,
                "has_skill": result.has_skill,
                "metrics": result.metrics.to_dict(),
                "warnings": result.warnings,
            }
        )

    _maybe_write_json(args, payload)
    return 1 if failures and not payload else 0


def command_forecast(args: argparse.Namespace) -> int:
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("no symbols to forecast")

    history = load_prices(
        symbols,
        provider=args.provider,
        start=args.start,
        end=args.end,
        **_provider_kwargs(args),
    )

    payload: list[dict[str, object]] = []
    for symbol, prices in history.items():
        try:
            forecast = forecast_symbol(
                symbol,
                prices,
                model_name=args.model,
                horizon=args.horizon,
                alpha=args.alpha,
                feature_config=_feature_config(args),
                **_evaluate_kwargs(args),
            )
        except DataError as exc:
            print(f"! {symbol}: {exc}")
            continue

        print(format_forecast(forecast))
        print()
        point, low, high = forecast.price_range
        payload.append(
            {
                "symbol": forecast.symbol,
                "as_of": str(forecast.as_of.date()),
                "last_close": forecast.last_close,
                "horizon": forecast.horizon,
                "expected_return": forecast.expected_return,
                "interval_low_return": forecast.interval.lower,
                "interval_high_return": forecast.interval.upper,
                "implied_price": point,
                "implied_price_low": low,
                "implied_price_high": high,
                "confidence": forecast.interval.confidence,
                "has_measured_skill": forecast.has_skill,
                "assessment": forecast.trustworthiness,
                "metrics": forecast.evaluation.metrics.to_dict(),
                "warnings": forecast.evaluation.warnings,
            }
        )

    _maybe_write_json(args, payload)
    return 0 if payload else 1


def command_screen(args: argparse.Namespace) -> int:
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("no symbols to screen")

    print(f"Screening {len(symbols)} symbols with model={args.model}, horizon={args.horizon}d ...")
    _, summary = screen_symbols(
        symbols,
        provider=args.provider,
        model_name=args.model,
        horizon=args.horizon,
        start=args.start,
        end=args.end,
        alpha=args.alpha,
        feature_config=_feature_config(args),
        **_provider_kwargs(args) if args.provider in {"csv", "mcp-json"} else {},
        **_evaluate_kwargs(args),
    )
    print()
    print(format_screen(summary))

    if args.json_out and not summary.empty:
        Path(args.json_out).write_text(summary.to_json(orient="records", indent=2))
        print(f"\nWrote {args.json_out}")
    return 0 if not summary.empty else 1


def command_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on synthetic data with no network access needed."""
    print(
        "Running on SYNTHETIC data with no real predictable signal.\n"
        "A correct pipeline should report roughly no skill here. If it reports\n"
        "strong skill, there is lookahead leakage somewhere.\n"
    )
    provider = get_provider("synthetic", n_bars=1500)
    symbols = ["SYN1", "SYN2", "SYN3"]
    history = {symbol: provider.fetch(symbol) for symbol in symbols}

    for symbol, prices in history.items():
        result = evaluate_symbol(
            symbol,
            prices,
            model_name=args.model,
            horizon=args.horizon,
            **_evaluate_kwargs(args),
        )
        print(format_evaluation(result))
        print()
    return 0


def _maybe_write_json(args: argparse.Namespace, payload: list[dict[str, object]]) -> None:
    if args.json_out and payload:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"Wrote {args.json_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockforecast",
        description=(
            "Short-horizon equity return forecasting with walk-forward validation. "
            "Reports measured out-of-sample skill alongside every forecast."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="measure out-of-sample skill (run this before trusting anything)"
    )
    _shared_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=command_evaluate)

    forecast_parser = subparsers.add_parser(
        "forecast", help="live forecast plus the evidence about its reliability"
    )
    _shared_arguments(forecast_parser)
    forecast_parser.set_defaults(func=command_forecast)

    screen_parser = subparsers.add_parser(
        "screen", help="run a whole universe, with multiple-testing correction"
    )
    _shared_arguments(screen_parser)
    screen_parser.set_defaults(func=command_screen)

    demo_parser = subparsers.add_parser("demo", help="offline sanity run on synthetic data")
    _shared_arguments(demo_parser)
    demo_parser.set_defaults(func=command_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pd.set_option("display.width", 120)
    try:
        return int(args.func(args))
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
