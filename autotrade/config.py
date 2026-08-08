"""Configuration loading and validation.

Config is TOML, read with the standard library's ``tomllib`` — no dependency,
and human-editable, which matters because this file is the thing standing
between a typo and your money.

Validation is aggressive and fail-closed. Every limit has a conservative
default, an omitted limit is never treated as "unlimited", and anything
unrecognised raises rather than being silently ignored (a misspelled
``max_notional_per_ordr`` must not read as "no cap").
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autotrade.errors import ConfigError

# Ceilings that apply no matter what the config says. A config cannot raise
# these; it can only set something lower. They exist so that a bad edit has a
# bounded blast radius.
ABSOLUTE_MAX_NOTIONAL_PER_ORDER = 5_000.0
ABSOLUTE_MAX_NOTIONAL_PER_DAY = 10_000.0
ABSOLUTE_MAX_ORDERS_PER_DAY = 20

VALID_MODES = ("paper", "live")


@dataclass(frozen=True)
class AccountConfig:
    """Which Robinhood account to act on, and in what mode."""

    number: str
    mode: str = "paper"

    def __post_init__(self) -> None:
        if not self.number or not self.number.isdigit():
            raise ConfigError(f"account.number must be a digit string, got {self.number!r}")
        if self.mode not in VALID_MODES:
            raise ConfigError(f"account.mode must be one of {VALID_MODES}, got {self.mode!r}")

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def masked(self) -> str:
        return f"****{self.number[-4:]}"


@dataclass(frozen=True)
class RiskLimits:
    """Hard caps applied to every intent.

    Defaults are deliberately small. The intent is that turning this system on
    with a half-written config risks lunch money, not rent.
    """

    max_notional_per_order: float = 100.0
    max_notional_per_day: float = 300.0
    max_position_notional_per_symbol: float = 1_000.0
    max_orders_per_day: int = 3
    daily_loss_limit: float = 200.0
    allowlist: tuple[str, ...] = ()
    max_quote_age_seconds: int = 300
    limit_offset_bps: float = 10.0
    require_confirmation: bool = True
    # Buying less than one share forces a market order (the broker allows
    # fractional only on type=market). Off by default: giving up price
    # protection should be an explicit choice, not a silent fallback.
    allow_fractional: bool = False
    # Only applies to market orders, which have no price cap of their own.
    max_spread_bps: float = 25.0

    def __post_init__(self) -> None:
        positive_fields = {
            "max_notional_per_order": self.max_notional_per_order,
            "max_notional_per_day": self.max_notional_per_day,
            "max_position_notional_per_symbol": self.max_position_notional_per_symbol,
            "daily_loss_limit": self.daily_loss_limit,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ConfigError(f"limits.{name} must be > 0, got {value}")

        if self.max_orders_per_day <= 0:
            raise ConfigError(f"limits.max_orders_per_day must be > 0, got {self.max_orders_per_day}")
        if self.max_quote_age_seconds <= 0:
            raise ConfigError("limits.max_quote_age_seconds must be > 0")
        if not 0 <= self.limit_offset_bps <= 500:
            raise ConfigError(
                f"limits.limit_offset_bps must be between 0 and 500, got {self.limit_offset_bps}"
            )

        if self.max_notional_per_order > ABSOLUTE_MAX_NOTIONAL_PER_ORDER:
            raise ConfigError(
                f"limits.max_notional_per_order ({self.max_notional_per_order}) exceeds the "
                f"hard ceiling of {ABSOLUTE_MAX_NOTIONAL_PER_ORDER}. Raise the ceiling in "
                "autotrade/config.py deliberately if you really mean it."
            )
        if self.max_notional_per_day > ABSOLUTE_MAX_NOTIONAL_PER_DAY:
            raise ConfigError(
                f"limits.max_notional_per_day ({self.max_notional_per_day}) exceeds the hard "
                f"ceiling of {ABSOLUTE_MAX_NOTIONAL_PER_DAY}."
            )
        if self.max_orders_per_day > ABSOLUTE_MAX_ORDERS_PER_DAY:
            raise ConfigError(
                f"limits.max_orders_per_day ({self.max_orders_per_day}) exceeds the hard "
                f"ceiling of {ABSOLUTE_MAX_ORDERS_PER_DAY}."
            )
        if not 0 < self.max_spread_bps <= 1000:
            raise ConfigError(
                f"limits.max_spread_bps must be between 0 and 1000, got {self.max_spread_bps}"
            )
        if self.max_notional_per_order > self.max_notional_per_day:
            raise ConfigError(
                "limits.max_notional_per_order cannot exceed limits.max_notional_per_day"
            )
        if not self.allowlist:
            raise ConfigError(
                "limits.allowlist must list at least one symbol. An empty allowlist is "
                "treated as 'nothing is tradable', never as 'everything is'."
            )


@dataclass(frozen=True)
class RuleConfig:
    """One user-specified buying rule."""

    id: str
    symbol: str
    condition: str
    action: str = "buy"
    amount_usd: float = 0.0
    threshold: float | None = None
    lookback_days: int | None = None
    weekday: str | None = None
    cooldown_days: int = 7
    note: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        from autotrade.rules import RULE_TYPES  # noqa: PLC0415 - avoids a cycle

        if not self.id:
            raise ConfigError("every rule needs a unique id")
        if self.condition not in RULE_TYPES:
            raise ConfigError(
                f"rule {self.id!r}: unknown condition {self.condition!r}; "
                f"valid conditions are {sorted(RULE_TYPES)}"
            )
        if self.action != "buy":
            raise ConfigError(
                f"rule {self.id!r}: only action='buy' is supported. Selling is intentionally "
                "not automated - exits deserve a human looking at them."
            )
        if self.amount_usd <= 0:
            raise ConfigError(f"rule {self.id!r}: amount_usd must be > 0")
        if self.cooldown_days < 0:
            raise ConfigError(f"rule {self.id!r}: cooldown_days must be >= 0")

        RULE_TYPES[self.condition].validate(self)


@dataclass(frozen=True)
class AutotradeConfig:
    """The whole validated configuration."""

    account: AccountConfig
    limits: RiskLimits
    rules: tuple[RuleConfig, ...]
    state_dir: Path = field(default=Path(".autotrade"))

    def __post_init__(self) -> None:
        if not self.rules:
            raise ConfigError("no rules defined; there is nothing to evaluate")

        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ConfigError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)

        allowed = {symbol.upper() for symbol in self.limits.allowlist}
        for rule in self.rules:
            if rule.symbol.upper() not in allowed:
                raise ConfigError(
                    f"rule {rule.id!r} targets {rule.symbol} which is not in "
                    f"limits.allowlist {sorted(allowed)}"
                )

    @property
    def enabled_rules(self) -> tuple[RuleConfig, ...]:
        return tuple(rule for rule in self.rules if rule.enabled)

    @property
    def kill_switch_path(self) -> Path:
        return self.state_dir / "HALT"


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _reject_unknown(data: dict[str, Any], known: set[str], where: str) -> None:
    """Fail on unrecognised keys.

    A silently ignored key in a risk config is how a cap you believed was set
    turns out never to have applied.
    """
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {where}: {sorted(unknown)}. "
            f"Valid keys are {sorted(known)}."
        )


def load_config(path: str | Path) -> AutotradeConfig:
    """Load and validate an autotrade TOML config."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"config not found: {path}. Copy autotrade.example.toml and edit it."
        )

    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML - {exc}") from exc

    _reject_unknown(raw, {"account", "limits", "rules", "state_dir"}, "the top level")

    if "account" not in raw:
        raise ConfigError("config is missing the [account] table")
    account_raw = _require_mapping(raw["account"], "account")
    _reject_unknown(account_raw, {"number", "mode"}, "[account]")
    account = AccountConfig(**account_raw)

    limits_raw = _require_mapping(raw.get("limits", {}), "limits")
    _reject_unknown(
        limits_raw,
        {
            "max_notional_per_order",
            "max_notional_per_day",
            "max_position_notional_per_symbol",
            "max_orders_per_day",
            "daily_loss_limit",
            "allowlist",
            "max_quote_age_seconds",
            "limit_offset_bps",
            "require_confirmation",
            "allow_fractional",
            "max_spread_bps",
        },
        "[limits]",
    )
    if "allowlist" in limits_raw:
        limits_raw["allowlist"] = tuple(
            str(symbol).upper() for symbol in limits_raw["allowlist"]
        )
    limits = RiskLimits(**limits_raw)

    rules_raw = raw.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ConfigError("[[rules]] must be an array of tables")

    known_rule_keys = {
        "id",
        "symbol",
        "condition",
        "action",
        "amount_usd",
        "threshold",
        "lookback_days",
        "weekday",
        "cooldown_days",
        "note",
        "enabled",
    }
    rules = []
    for entry in rules_raw:
        entry = _require_mapping(entry, "rules")
        _reject_unknown(entry, known_rule_keys, f"rule {entry.get('id', '?')!r}")
        entry = {**entry, "symbol": str(entry.get("symbol", "")).upper()}
        rules.append(RuleConfig(**entry))

    state_dir = Path(raw.get("state_dir", ".autotrade"))

    config = AutotradeConfig(
        account=account,
        limits=limits,
        rules=tuple(rules),
        state_dir=state_dir,
    )

    if config.account.is_live and not config.limits.require_confirmation:
        raise ConfigError(
            "refusing to load a config with mode='live' and require_confirmation=false. "
            "Unattended live order placement is not a supported configuration."
        )

    return config
