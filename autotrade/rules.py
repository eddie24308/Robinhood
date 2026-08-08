"""User-specified buying rules.

Each rule answers one question: given the current quote and recent history,
should an order be *proposed* right now? Rules never size orders beyond the
``amount_usd`` the user wrote down, and never place anything — a fired rule
produces a proposal that still has to clear every risk guard and then a human.

Adding a condition means adding a :class:`RuleType` to ``RULE_TYPES``. Each one
owns both its validation and its evaluation so a half-specified rule fails at
config load rather than at 09:31 on a Tuesday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Callable, Protocol

import pandas as pd

from autotrade.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autotrade.config import RuleConfig

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
}


@dataclass(frozen=True)
class MarketContext:
    """Everything a rule is allowed to look at."""

    symbol: str
    price: float
    quote_time: datetime
    today: date
    history: pd.Series | None = None
    last_fired: date | None = None

    def trailing_high(self, days: int) -> float | None:
        if self.history is None or self.history.empty:
            return None
        window = self.history.tail(days)
        return float(window.max()) if len(window) else None

    def trailing_low(self, days: int) -> float | None:
        if self.history is None or self.history.empty:
            return None
        window = self.history.tail(days)
        return float(window.min()) if len(window) else None


@dataclass(frozen=True)
class RuleEvaluation:
    """Outcome of evaluating one rule."""

    rule_id: str
    fired: bool
    reason: str
    detail: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            object.__setattr__(self, "detail", {})


class RuleType(Protocol):
    """A condition's validation and evaluation logic."""

    description: str

    def validate(self, rule: RuleConfig) -> None: ...

    def evaluate(self, rule: RuleConfig, context: MarketContext) -> RuleEvaluation: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class _PriceThreshold:
    """Fires when price crosses a fixed level."""

    below: bool
    description: str

    def validate(self, rule: RuleConfig) -> None:
        _require(
            rule.threshold is not None and rule.threshold > 0,
            f"rule {rule.id!r}: condition {rule.condition!r} needs a positive `threshold`",
        )

    def evaluate(self, rule: RuleConfig, context: MarketContext) -> RuleEvaluation:
        threshold = float(rule.threshold)  # type: ignore[arg-type]
        fired = context.price < threshold if self.below else context.price > threshold
        comparator = "<" if self.below else ">"
        return RuleEvaluation(
            rule_id=rule.id,
            fired=fired,
            reason=(
                f"{context.symbol} {context.price:.2f} {comparator} {threshold:.2f}"
                if fired
                else f"{context.symbol} {context.price:.2f} did not cross {threshold:.2f}"
            ),
            detail={"price": context.price, "threshold": threshold},
        )


@dataclass(frozen=True)
class _MoveFromExtreme:
    """Fires on a percentage move away from a trailing high or low."""

    from_high: bool
    description: str

    def validate(self, rule: RuleConfig) -> None:
        _require(
            rule.threshold is not None and 0 < rule.threshold < 100,
            f"rule {rule.id!r}: `threshold` must be a percentage between 0 and 100",
        )
        _require(
            rule.lookback_days is not None and rule.lookback_days > 1,
            f"rule {rule.id!r}: `lookback_days` must be > 1",
        )

    def evaluate(self, rule: RuleConfig, context: MarketContext) -> RuleEvaluation:
        lookback = int(rule.lookback_days)  # type: ignore[arg-type]
        threshold = float(rule.threshold)  # type: ignore[arg-type]

        extreme = (
            context.trailing_high(lookback) if self.from_high else context.trailing_low(lookback)
        )
        if extreme is None or extreme <= 0:
            return RuleEvaluation(
                rule_id=rule.id,
                fired=False,
                reason=f"no price history for {context.symbol}; cannot evaluate",
            )

        change_pct = (context.price / extreme - 1.0) * 100.0
        fired = change_pct <= -threshold if self.from_high else change_pct >= threshold
        label = "high" if self.from_high else "low"

        return RuleEvaluation(
            rule_id=rule.id,
            fired=fired,
            reason=(
                f"{context.symbol} {change_pct:+.1f}% vs {lookback}-day {label} "
                f"{extreme:.2f} (trigger {'-' if self.from_high else '+'}{threshold:.1f}%)"
            ),
            detail={"price": context.price, "extreme": extreme, "change_pct": change_pct},
        )


@dataclass(frozen=True)
class _WeeklySchedule:
    """Fires on a given weekday — plain dollar-cost averaging."""

    description: str = "Fire on a specific weekday (scheduled DCA)"

    def validate(self, rule: RuleConfig) -> None:
        _require(
            rule.weekday is not None and rule.weekday.lower() in WEEKDAYS,
            f"rule {rule.id!r}: `weekday` must be one of {sorted(WEEKDAYS)}",
        )

    def evaluate(self, rule: RuleConfig, context: MarketContext) -> RuleEvaluation:
        target = WEEKDAYS[str(rule.weekday).lower()]
        fired = context.today.weekday() == target
        return RuleEvaluation(
            rule_id=rule.id,
            fired=fired,
            reason=(
                f"today is {context.today:%A}"
                + ("" if fired else f", not {str(rule.weekday).title()}")
            ),
            detail={"price": context.price},
        )


@dataclass(frozen=True)
class _EveryRun:
    """Always fires; cadence is controlled entirely by ``cooldown_days``."""

    description: str = "Fire on every run, throttled by cooldown_days"

    def validate(self, rule: RuleConfig) -> None:
        _require(
            rule.cooldown_days >= 1,
            f"rule {rule.id!r}: condition 'every_run' needs cooldown_days >= 1, "
            "otherwise it proposes an order on every single invocation",
        )

    def evaluate(self, rule: RuleConfig, context: MarketContext) -> RuleEvaluation:
        return RuleEvaluation(
            rule_id=rule.id,
            fired=True,
            reason=f"unconditional (every {rule.cooldown_days}d)",
            detail={"price": context.price},
        )


RULE_TYPES: dict[str, RuleType] = {
    "price_below": _PriceThreshold(below=True, description="Fire when price falls below a level"),
    "price_above": _PriceThreshold(below=False, description="Fire when price rises above a level"),
    "drawdown_from_high": _MoveFromExtreme(
        from_high=True, description="Fire after an N% fall from the trailing high"
    ),
    "rally_from_low": _MoveFromExtreme(
        from_high=False, description="Fire after an N% rise from the trailing low"
    ),
    "weekly_schedule": _WeeklySchedule(),
    "every_run": _EveryRun(),
}


def evaluate_rule(rule: RuleConfig, context: MarketContext) -> RuleEvaluation:
    """Evaluate one rule, applying its cooldown before its condition.

    Cooldown is checked first because it is the cheaper and more important
    guard: a rule that fired yesterday must not fire again today regardless of
    what the price is doing.
    """
    if not rule.enabled:
        return RuleEvaluation(rule_id=rule.id, fired=False, reason="rule disabled in config")

    if context.last_fired is not None and rule.cooldown_days > 0:
        days_since = (context.today - context.last_fired).days
        if days_since < rule.cooldown_days:
            remaining = rule.cooldown_days - days_since
            return RuleEvaluation(
                rule_id=rule.id,
                fired=False,
                reason=(
                    f"cooling down: fired {days_since}d ago, "
                    f"{remaining}d remaining of {rule.cooldown_days}d"
                ),
            )

    return RULE_TYPES[rule.condition].evaluate(rule, context)


def describe_rule_types() -> str:
    """Human-readable catalogue of available conditions."""
    lines = []
    for name, rule_type in sorted(RULE_TYPES.items()):
        lines.append(f"  {name:22s} {rule_type.description}")
    return "\n".join(lines)
