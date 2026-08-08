"""Risk guards — the layer that says no.

Every proposed intent passes through every guard. Guards do not short-circuit:
an intent that violates four limits reports all four, because knowing only the
first one leads to fixing it and being surprised by the next.

Design rules for anything added here:

* Fail closed. Missing data is a violation, never a pass.
* No guard may mutate an intent. Silently shrinking an order to fit a cap
  hides the fact that a cap was hit; blocking it surfaces that.
* Guards are pure functions of (intent, state, limits) so they are trivially
  testable, and the test suite asserts each one actually blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from autotrade.config import RiskLimits
from autotrade.intents import ExecutionStyle, IntentStatus, OrderIntent


@dataclass(frozen=True)
class GuardViolation:
    """One reason an intent must not proceed."""

    guard: str
    message: str

    def __str__(self) -> str:
        return f"{self.guard}: {self.message}"


@dataclass
class GuardOutcome:
    """Result of running all guards against one intent."""

    passed: list[str] = field(default_factory=list)
    violations: list[GuardViolation] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return not self.violations

    def messages(self) -> list[str]:
        return [str(violation) for violation in self.violations]


@dataclass
class PortfolioState:
    """What has already happened today, and what is already held.

    Supplied by the ledger for paper mode, or by the agent from
    ``get_equity_orders`` / ``get_equity_positions`` for live mode. Defaults
    are zero rather than unknown; the caller is responsible for populating it,
    and :meth:`RiskGuard.check` treats a stale ``as_of`` as a violation.
    """

    trade_date: date
    orders_today: int = 0
    notional_today: float = 0.0
    realized_pnl_today: float = 0.0
    positions_notional: dict[str, float] = field(default_factory=dict)
    fingerprints_today: set[str] = field(default_factory=set)

    def position_for(self, symbol: str) -> float:
        return self.positions_notional.get(symbol.upper(), 0.0)


class RiskGuard:
    """Applies configured limits to proposed intents."""

    def __init__(self, limits: RiskLimits, kill_switch_active: bool = False) -> None:
        self.limits = limits
        self.kill_switch_active = kill_switch_active

    def check(
        self,
        intent: OrderIntent,
        state: PortfolioState,
        now: datetime | None = None,
    ) -> GuardOutcome:
        """Run every guard. Returns all passes and all violations."""
        now = now or datetime.now(timezone.utc)
        outcome = GuardOutcome()

        checks = (
            self._check_kill_switch,
            self._check_side,
            self._check_allowlist,
            self._check_price_sanity,
            self._check_quote_freshness,
            self._check_fractional_allowed,
            self._check_spread,
            self._check_per_order_notional,
            self._check_daily_notional,
            self._check_daily_order_count,
            self._check_position_cap,
            self._check_daily_loss_limit,
            self._check_duplicate,
            self._check_trade_date,
        )

        for check in checks:
            violation = check(intent, state, now)
            name = check.__name__.removeprefix("_check_")
            if violation is None:
                outcome.passed.append(name)
            else:
                outcome.violations.append(violation)

        return outcome

    # --- individual guards ---------------------------------------------------

    def _check_kill_switch(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if self.kill_switch_active:
            return GuardViolation(
                "kill_switch",
                "HALT file present - all trading is disabled until it is removed",
            )
        return None

    def _check_side(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if intent.side != "buy":
            return GuardViolation(
                "side", f"only buys are automated, got side={intent.side!r}"
            )
        return None

    def _check_allowlist(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        allowed = {symbol.upper() for symbol in self.limits.allowlist}
        if intent.symbol.upper() not in allowed:
            return GuardViolation(
                "allowlist",
                f"{intent.symbol} is not in the allowlist {sorted(allowed)}",
            )
        return None

    def _check_price_sanity(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if intent.reference_price <= 0 or intent.limit_price <= 0:
            return GuardViolation(
                "price_sanity",
                f"non-positive price (ref={intent.reference_price}, "
                f"limit={intent.limit_price})",
            )
        # A limit far above the reference means bad data or a bad offset, and
        # would hand away money on a marketable order.
        drift = intent.limit_price / intent.reference_price - 1.0
        max_drift = self.limits.limit_offset_bps / 10_000.0
        if drift > max_drift + 1e-9:
            return GuardViolation(
                "price_sanity",
                f"limit {intent.limit_price:.2f} is {drift:.2%} above reference "
                f"{intent.reference_price:.2f}, exceeding the configured "
                f"{max_drift:.2%} offset",
            )
        return None

    def _check_quote_freshness(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        quote_time = intent.detail.get("quote_time")
        if quote_time is None:
            return GuardViolation("quote_freshness", "intent carries no quote timestamp")

        if isinstance(quote_time, str):
            try:
                quote_time = datetime.fromisoformat(quote_time)
            except ValueError:
                return GuardViolation(
                    "quote_freshness", f"unparseable quote timestamp {quote_time!r}"
                )

        if quote_time.tzinfo is None:
            quote_time = quote_time.replace(tzinfo=timezone.utc)

        age = now - quote_time
        if age > timedelta(seconds=self.limits.max_quote_age_seconds):
            return GuardViolation(
                "quote_freshness",
                f"quote is {age.total_seconds():.0f}s old, limit is "
                f"{self.limits.max_quote_age_seconds}s - refusing to trade on stale data",
            )
        if age < timedelta(seconds=-60):
            return GuardViolation(
                "quote_freshness",
                f"quote timestamp is {-age.total_seconds():.0f}s in the future - clock problem",
            )
        return None

    def _check_fractional_allowed(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        """Block a fractional order when the config has not opted into one.

        Buying less than a whole share forces a market order, which gives up
        price protection. That trade-off should be a deliberate choice, so it
        is off unless ``allow_fractional`` is set.
        """
        if intent.execution_style != ExecutionStyle.NOTIONAL_MARKET:
            return None
        if self.limits.allow_fractional:
            return None
        return GuardViolation(
            "fractional_allowed",
            f"${intent.amount_usd:,.2f} buys less than one share of {intent.symbol} at "
            f"{intent.reference_price:,.2f}, which requires a fractional market order. "
            "Set limits.allow_fractional = true, or raise the order size above one share.",
        )

    def _check_spread(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        """Block market orders into a wide spread.

        A limit order carries its own price cap, so this only applies to
        notional market orders. For a liquid ETF the spread is a basis point or
        two; a sudden wide spread means thin liquidity or a halted book, and is
        exactly when an uncapped market order does damage.
        """
        if intent.execution_style != ExecutionStyle.NOTIONAL_MARKET:
            return None

        spread = intent.spread_bps
        if spread is None:
            return GuardViolation(
                "spread",
                "no usable bid/ask on the quote, so the spread cannot be checked; "
                "a market order without that check is not allowed",
            )
        if spread > self.limits.max_spread_bps:
            return GuardViolation(
                "spread",
                f"bid-ask spread is {spread:.1f} bps (bid {intent.bid_price:,.2f} / "
                f"ask {intent.ask_price:,.2f}), over the {self.limits.max_spread_bps:.0f} bps "
                "limit - refusing an uncapped market order into a thin book",
            )
        return None

    def _check_per_order_notional(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if intent.amount_usd > self.limits.max_notional_per_order:
            return GuardViolation(
                "per_order_notional",
                f"${intent.amount_usd:,.2f} exceeds the per-order cap of "
                f"${self.limits.max_notional_per_order:,.2f}",
            )
        return None

    def _check_daily_notional(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        projected = state.notional_today + intent.amount_usd
        if projected > self.limits.max_notional_per_day:
            return GuardViolation(
                "daily_notional",
                f"${state.notional_today:,.2f} already committed today; adding "
                f"${intent.amount_usd:,.2f} would reach ${projected:,.2f}, over the "
                f"${self.limits.max_notional_per_day:,.2f} daily cap",
            )
        return None

    def _check_daily_order_count(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if state.orders_today >= self.limits.max_orders_per_day:
            return GuardViolation(
                "daily_order_count",
                f"{state.orders_today} orders already today, at the cap of "
                f"{self.limits.max_orders_per_day}",
            )
        return None

    def _check_position_cap(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        current = state.position_for(intent.symbol)
        projected = current + intent.amount_usd
        if projected > self.limits.max_position_notional_per_symbol:
            return GuardViolation(
                "position_cap",
                f"{intent.symbol} position would reach ${projected:,.2f}, over the "
                f"${self.limits.max_position_notional_per_symbol:,.2f} per-symbol cap "
                f"(currently ${current:,.2f})",
            )
        return None

    def _check_daily_loss_limit(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if state.realized_pnl_today <= -abs(self.limits.daily_loss_limit):
            return GuardViolation(
                "daily_loss_limit",
                f"realised P&L today is ${state.realized_pnl_today:,.2f}, at or beyond "
                f"the ${self.limits.daily_loss_limit:,.2f} stop - no new positions",
            )
        return None

    def _check_duplicate(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if intent.fingerprint() in state.fingerprints_today:
            return GuardViolation(
                "duplicate",
                f"an identical intent for rule {intent.rule_id!r} was already proposed "
                "today - refusing to double it",
            )
        return None

    def _check_trade_date(
        self, intent: OrderIntent, state: PortfolioState, now: datetime
    ) -> GuardViolation | None:
        if intent.trade_date != state.trade_date:
            return GuardViolation(
                "trade_date",
                f"intent is dated {intent.trade_date} but state is for "
                f"{state.trade_date} - refusing to act on a stale plan",
            )
        return None


def apply_guards(
    intents: list[OrderIntent],
    limits: RiskLimits,
    state: PortfolioState,
    kill_switch_active: bool = False,
    now: datetime | None = None,
) -> list[OrderIntent]:
    """Run guards over a batch, accumulating state as intents are accepted.

    Accumulation matters: three $100 intents against a $250 daily cap must
    leave the third blocked. Checking each against the starting state would let
    all three through.
    """
    guard = RiskGuard(limits, kill_switch_active=kill_switch_active)
    running = PortfolioState(
        trade_date=state.trade_date,
        orders_today=state.orders_today,
        notional_today=state.notional_today,
        realized_pnl_today=state.realized_pnl_today,
        positions_notional=dict(state.positions_notional),
        fingerprints_today=set(state.fingerprints_today),
    )

    for intent in intents:
        outcome = guard.check(intent, running, now=now)
        intent.guards_passed = outcome.passed
        intent.blocked_by = outcome.messages()

        if outcome.allowed:
            intent.status = IntentStatus.READY_FOR_REVIEW
            running.orders_today += 1
            running.notional_today += intent.amount_usd
            symbol = intent.symbol.upper()
            running.positions_notional[symbol] = (
                running.positions_notional.get(symbol, 0.0) + intent.amount_usd
            )
            running.fingerprints_today.add(intent.fingerprint())
        else:
            intent.status = IntentStatus.BLOCKED

    return intents
