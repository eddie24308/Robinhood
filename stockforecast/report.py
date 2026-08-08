"""Rendering forecasts and evaluations as text.

The formatting rules here are the user-facing half of the package's premise:

* a point forecast is never printed without its interval and its measured
  skill;
* when a model shows no significant skill, the report says so before it says
  anything else;
* warnings from the evaluation are always shown, never collapsed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockforecast.pipeline import EvaluationResult, Forecast

RULE = "=" * 78
THIN = "-" * 78


def _pct(value: float, digits: int = 1, signed: bool = False) -> str:
    """Percent, rendering NaN as ``n/a`` rather than the misleading ``nan%``."""
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}%}" if signed else f"{value:.{digits}%}"


def _num(value: float, digits: int = 3, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def format_evaluation(result: EvaluationResult) -> str:
    """Render walk-forward evaluation results."""
    metrics = result.metrics
    lines: list[str] = [
        RULE,
        f"WALK-FORWARD EVALUATION  {result.symbol}  "
        f"model={result.model_name}  horizon={result.horizon}d",
        RULE,
        f"bars={result.n_bars}  folds={result.n_folds}  "
        f"out-of-sample predictions={metrics.n}",
        "",
        "Error vs the zero-return baseline",
        THIN,
        f"  model RMSE          {metrics.rmse:.5f}",
        f"  baseline RMSE       {metrics.baseline_rmse:.5f}",
        f"  RMSE skill          {_pct(metrics.skill_rmse, 2, signed=True)}  "
        f"(positive = better than predicting zero)",
        f"  out-of-sample R2    {_num(metrics.r2_oos, 4, signed=True)}  "
        "(negative = worse than baseline)",
        f"  Diebold-Mariano     stat={_num(metrics.dm_statistic, 2, signed=True)}  "
        f"p={_num(metrics.dm_p_value)}",
        "",
        "Direction and ranking",
        THIN,
        f"  directional acc.    {_pct(metrics.directional_accuracy)}  "
        f"95% CI [{_pct(metrics.directional_ci_low)}, {_pct(metrics.directional_ci_high)}]  "
        f"p={_num(metrics.directional_p_value)}",
        f"  base rate (up)      {_pct(metrics.hit_rate_baseline)}",
        f"  information coef.   {_num(metrics.information_coefficient, 3, signed=True)}  "
        f"p={_num(metrics.ic_p_value)}",
        "",
        "Other baselines (same folds)",
        THIN,
    ]

    for name, baseline in result.baseline_metrics.items():
        note = "  (no directional view)" if name == "zero" else ""
        lines.append(
            f"  {name:10s} RMSE={baseline.rmse:.5f}  "
            f"skill={_pct(baseline.skill_rmse, 2, signed=True)}  "
            f"dir={_pct(baseline.directional_accuracy)}{note}"
        )

    lines += [
        "",
        "Prediction intervals",
        THIN,
        f"  nominal confidence  {1 - result.interval_alpha:.0%}",
        f"  half-width          {_num(result.interval_half_width, 4)} log-return",
        f"  realised coverage   {_pct(result.realised_coverage)}",
    ]

    if result.top_features:
        lines += ["", "Top features (last fold)", THIN]
        for name, weight in list(result.top_features.items())[:8]:
            bar = "#" * max(1, int(round(weight * 60)))
            lines.append(f"  {name:22s} {weight:6.2%}  {bar}")

    lines += ["", "VERDICT", THIN]
    if result.has_skill:
        lines.append(
            f"  Model beats the zero baseline (RMSE skill "
            f"{_pct(metrics.skill_rmse, 2, signed=True)}, DM p={_num(metrics.dm_p_value)})."
        )
        lines.append(
            "  Statistical skill is not the same as profit: transaction costs, "
            "slippage and"
        )
        lines.append("  capacity are not modelled here.")
    elif metrics.skill_rmse < 0 and metrics.dm_p_value < 0.05:
        lines.append(
            f"  SIGNIFICANTLY WORSE than the zero baseline "
            f"({_pct(metrics.skill_rmse, 2, signed=True)} skill, "
            f"DM p={_num(metrics.dm_p_value)})."
        )
        lines.append(
            "  The model is not merely uninformative, it is actively harmful at this"
        )
        lines.append(
            "  configuration - usually too many features for too little data, or a"
        )
        lines.append(
            "  horizon/regime mismatch. Shorten the feature set, add history, or stop."
        )
    else:
        lines.append("  NO SIGNIFICANT SKILL over the zero-return baseline.")
        lines.append(
            "  Point forecasts from this configuration should be treated as noise. "
            "This is"
        )
        lines.append(
            "  the expected result for most tickers at most horizons - it is the "
            "honest answer,"
        )
        lines.append("  not a bug.")

    if result.warnings:
        lines += ["", "WARNINGS", THIN]
        lines += [f"  ! {message}" for message in result.warnings]

    lines.append(RULE)
    return "\n".join(lines)


def format_forecast(forecast: Forecast) -> str:
    """Render a live forecast. Never emits a bare point estimate."""
    metrics = forecast.evaluation.metrics
    point, low, high = forecast.price_range
    confidence = forecast.interval.confidence

    lines = [
        RULE,
        f"FORECAST  {forecast.symbol}  next {forecast.horizon} trading days",
        RULE,
    ]

    if not forecast.has_skill:
        lines += [
            "",
            "  >>> THIS MODEL HAS NO MEASURED OUT-OF-SAMPLE SKILL FOR THIS TICKER. <<<",
            "  The numbers below are shown for completeness. On the walk-forward test "
            "they were",
            "  not better than assuming a zero return. Do not trade on them.",
            "",
        ]

    lines += [
        f"  as of              {forecast.as_of.date()}  (last close {forecast.last_close:,.2f})",
        f"  model              {forecast.model_name}",
        "",
        f"  expected return    {_pct(forecast.expected_return, 2, signed=True)} "
        f"over {forecast.horizon} days",
        f"  {confidence:.0%} interval      "
        f"[{forecast.interval.lower * 100:+.2f}%, {forecast.interval.upper * 100:+.2f}%]",
        f"  implied price      {point:,.2f}   range {low:,.2f} to {high:,.2f}",
        "",
        "  Evidence for taking the number seriously",
        THIN,
        f"    RMSE skill vs baseline   {_pct(metrics.skill_rmse, 2, signed=True)}  "
        f"(DM p={_num(metrics.dm_p_value)})",
        f"    directional accuracy     {_pct(metrics.directional_accuracy)}  "
        f"95% CI [{_pct(metrics.directional_ci_low)}, {_pct(metrics.directional_ci_high)}]",
        f"    out-of-sample sample     {metrics.n} predictions "
        f"over {forecast.evaluation.n_folds} folds",
        f"    interval coverage        {_pct(forecast.evaluation.realised_coverage)} "
        f"realised vs {confidence:.0%} nominal",
        "",
        f"  ASSESSMENT: {forecast.trustworthiness}",
    ]

    if forecast.evaluation.warnings:
        lines += ["", "  WARNINGS", THIN]
        lines += [f"    ! {message}" for message in forecast.evaluation.warnings]

    lines += [
        "",
        "  The interval is wide because 5-day equity returns are mostly noise. A "
        "narrow",
        "  interval here would be a lie, not a better model.",
        RULE,
    ]
    return "\n".join(lines)


def format_screen(summary: pd.DataFrame, top_n: int = 20) -> str:
    """Render a multi-ticker screen, foregrounding the multiple-testing problem."""
    if summary.empty:
        return "No symbols produced a usable forecast."

    lines = [
        RULE,
        f"SCREEN  {len(summary)} symbols  (sorted by expected return)",
        RULE,
        f"{'sym':<7}{'close':>10}{'exp.ret':>9}{'low':>10}{'high':>10}"
        f"{'skill':>11}{'dir':>8}{'IC':>7}{'DM p':>8}{'q':>8}  verdict",
        THIN,
    ]

    for _, row in summary.head(top_n).iterrows():
        verdict = "skill (FDR ok)" if row.get("skill_after_fdr") else (
            "nominal only" if row["has_skill"] else "no skill"
        )
        lines.append(
            f"{row['symbol']:<7}{row['last_close']:>10,.2f}{row['expected_return']:>9.2%}"
            f"{row['low']:>10,.2f}{row['high']:>10,.2f}"
            f"{_pct(row['skill_rmse'], 2, signed=True):>11}"
            f"{_pct(row['dir_acc']):>8}{_num(row['ic'], 2, signed=True):>7}"
            f"{_num(row['dm_p']):>8}{_num(row.get('q_value', float('nan'))):>8}  {verdict}"
        )

    survivors = int(summary.get("skill_after_fdr", pd.Series(dtype=bool)).sum())
    nominal = int(summary["has_skill"].sum())

    lines += [
        THIN,
        f"  {nominal}/{len(summary)} symbols beat baseline at nominal p<0.05.",
        f"  {survivors}/{len(summary)} survive Benjamini-Hochberg FDR control at q<0.05.",
        "",
        "  Screening many tickers guarantees some look good by chance: at p<0.05 you "
        "expect",
        f"  about {0.05 * len(summary):.1f} false positives from {len(summary)} symbols "
        "with no signal at all.",
        "  The q-value column is the one to read. Ranking by expected return alone "
        "mostly",
        "  ranks by noise.",
        RULE,
    ]
    return "\n".join(lines)
