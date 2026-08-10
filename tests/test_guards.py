"""Guard tests.

Each guard gets a test that proves it actually blocks. A risk limit that is
configured but not enforced is worse than no limit, because it produces
confidence without protection.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from autotrade.config import RiskLimits
from autotrade.guards import PortfolioState, RiskGuard, apply_guards
from autotrade.intents import ExecutionStyle, IntentStatus, OrderIntent

NOW = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
TODAY = date(2026, 8, 10)


def make_limits(**overrides) -> RiskLimits:
    base = {
        "max_notional_per_order": 100.0,
        "max_notional_per_day": 300.0,
        "max_position_notional_per_symbol": 1000.0,
        "max_orders_per_day": 3,
        "daily_loss_limit": 200.0,
        "allowlist": ("HOOD", "VOO"),
        "max_quote_age_seconds": 300,
        "limit_offset_bps": 10.0,
    }
    base.update(overrides)
    return RiskLimits(**base)


def make_intent(**overrides) -> OrderIntent:
    base = dict(
        intent_id="i1",
        created_at=NOW,
        trade_date=TODAY,
        account_number="771654944",
        mode="paper",
        symbol="HOOD",
        side="buy",
        amount_usd=100.0,
        reference_price=93.28,
        order_type="limit",
        limit_price=93.37,
        time_in_force="gfd",
        rule_id="r1",
        rule_reason="test",
        status=IntentStatus.READY_FOR_REVIEW,
        # $100 affords a whole share of HOOD at ~93, so the engine would pick
        # the limit style here. Fractional cases are covered explicitly below.
        execution_style=ExecutionStyle.WHOLE_SHARE_LIMIT,
        bid_price=93.20,
        ask_price=93.35,
        detail={"quote_time": NOW.isoformat()},
    )
    base.update(overrides)
    return OrderIntent(**base)


def make_state(**overrides) -> PortfolioState:
    base = {"trade_date": TODAY}
    base.update(overrides)
    return PortfolioState(**base)


def check(intent, state=None, limits=None, kill=False):
    guard = RiskGuard(limits or make_limits(), kill_switch_active=kill)
    return guard.check(intent, state or make_state(), now=NOW)


def test_clean_intent_passes() -> None:
    outcome = check(make_intent())
    assert outcome.allowed, outcome.messages()
    assert "allowlist" in outcome.passed


def test_kill_switch_blocks_everything() -> None:
    outcome = check(make_intent(), kill=True)
    assert not outcome.allowed
    assert any("kill_switch" in message for message in outcome.messages())


def test_symbol_not_in_allowlist_is_blocked() -> None:
    outcome = check(make_intent(symbol="TSLA"))
    assert not outcome.allowed
    assert any("allowlist" in message for message in outcome.messages())


def test_sell_side_is_blocked() -> None:
    """Selling is never automated."""
    outcome = check(make_intent(side="sell"))
    assert not outcome.allowed
    assert any("side" in message for message in outcome.messages())


def test_per_order_notional_cap() -> None:
    outcome = check(make_intent(amount_usd=250.0))
    assert not outcome.allowed
    assert any("per_order_notional" in message for message in outcome.messages())


def test_daily_notional_cap() -> None:
    outcome = check(make_intent(amount_usd=100.0), state=make_state(notional_today=250.0))
    assert not outcome.allowed
    assert any("daily_notional" in message for message in outcome.messages())


def test_daily_order_count_cap() -> None:
    outcome = check(make_intent(), state=make_state(orders_today=3))
    assert not outcome.allowed
    assert any("daily_order_count" in message for message in outcome.messages())


def test_position_cap_per_symbol() -> None:
    outcome = check(
        make_intent(amount_usd=100.0),
        state=make_state(positions_notional={"HOOD": 950.0}),
    )
    assert not outcome.allowed
    assert any("position_cap" in message for message in outcome.messages())


def test_daily_loss_limit_stops_new_positions() -> None:
    outcome = check(make_intent(), state=make_state(realized_pnl_today=-200.0))
    assert not outcome.allowed
    assert any("daily_loss_limit" in message for message in outcome.messages())


def test_stale_quote_is_blocked() -> None:
    stale = (NOW - timedelta(seconds=600)).isoformat()
    outcome = check(make_intent(detail={"quote_time": stale}))
    assert not outcome.allowed
    assert any("quote_freshness" in message for message in outcome.messages())


def test_missing_quote_timestamp_is_blocked() -> None:
    """Fail closed: absent data is a violation, not a pass."""
    outcome = check(make_intent(detail={}))
    assert not outcome.allowed
    assert any("quote_freshness" in message for message in outcome.messages())


def test_future_quote_timestamp_is_blocked() -> None:
    future = (NOW + timedelta(seconds=300)).isoformat()
    outcome = check(make_intent(detail={"quote_time": future}))
    assert not outcome.allowed
    assert any("clock problem" in message for message in outcome.messages())


def test_limit_far_above_reference_is_blocked() -> None:
    outcome = check(make_intent(reference_price=93.28, limit_price=120.0))
    assert not outcome.allowed
    assert any("price_sanity" in message for message in outcome.messages())


def fractional_intent(**overrides) -> OrderIntent:
    """A $100 buy of a $710 ETF — necessarily fractional, hence market."""
    base = dict(
        symbol="VOO",
        amount_usd=100.0,
        reference_price=710.57,
        order_type="market",
        limit_price=711.28,
        execution_style=ExecutionStyle.NOTIONAL_MARKET,
        bid_price=710.52,
        ask_price=711.00,
    )
    base.update(overrides)
    return make_intent(**base)


def test_cent_rounding_does_not_trip_price_sanity() -> None:
    """Regression: VXUS at 87.21 with a 10 bps offset rounds to 87.30.

    That is 0.103% above reference, a hair over the 0.10% offset purely from
    quoting in whole cents. The guard must not read a rounding artifact as bad
    data, or every cheap security gets blocked.
    """
    outcome = check(make_intent(reference_price=87.21, limit_price=87.30))
    assert outcome.allowed, outcome.messages()


def test_genuinely_bad_limit_still_blocked_on_cheap_shares() -> None:
    """The cent of slack must not become a licence for a real overshoot."""
    outcome = check(make_intent(reference_price=87.21, limit_price=88.50))
    assert not outcome.allowed
    assert any("price_sanity" in message for message in outcome.messages())


def test_price_drift_not_checked_for_market_orders() -> None:
    """A market order carries no limit price, so the field is meaningless there."""
    outcome = check(
        fractional_intent(reference_price=87.21, limit_price=999.0),
        limits=make_limits(allow_fractional=True),
    )
    assert outcome.allowed, outcome.messages()


def test_non_positive_price_is_blocked() -> None:
    outcome = check(make_intent(reference_price=0.0, limit_price=0.0))
    assert not outcome.allowed
    assert any("price_sanity" in message for message in outcome.messages())


def test_duplicate_fingerprint_is_blocked() -> None:
    intent = make_intent()
    outcome = check(intent, state=make_state(fingerprints_today={intent.fingerprint()}))
    assert not outcome.allowed
    assert any("duplicate" in message for message in outcome.messages())


def test_stale_plan_date_is_blocked() -> None:
    outcome = check(make_intent(trade_date=date(2026, 8, 3)))
    assert not outcome.allowed
    assert any("trade_date" in message for message in outcome.messages())


def test_fractional_blocked_unless_explicitly_allowed() -> None:
    """Giving up price protection must be opted into, not defaulted into."""
    outcome = check(fractional_intent(), limits=make_limits(allow_fractional=False))
    assert not outcome.allowed
    assert any("fractional_allowed" in message for message in outcome.messages())


def test_fractional_passes_when_allowed() -> None:
    outcome = check(fractional_intent(), limits=make_limits(allow_fractional=True))
    assert outcome.allowed, outcome.messages()


def test_wide_spread_blocks_market_order() -> None:
    outcome = check(
        fractional_intent(bid_price=700.00, ask_price=720.00),
        limits=make_limits(allow_fractional=True, max_spread_bps=25.0),
    )
    assert not outcome.allowed
    assert any("spread" in message for message in outcome.messages())


def test_missing_bid_ask_blocks_market_order() -> None:
    """Fail closed: an unpriceable book is not a tradable one."""
    outcome = check(
        fractional_intent(bid_price=None, ask_price=None),
        limits=make_limits(allow_fractional=True),
    )
    assert not outcome.allowed
    assert any("spread" in message for message in outcome.messages())


def test_spread_guard_does_not_apply_to_limit_orders() -> None:
    """A limit order carries its own cap, so a wide spread is not fatal."""
    outcome = check(make_intent(bid_price=90.0, ask_price=97.0))
    assert outcome.allowed, outcome.messages()


def test_all_violations_are_reported_not_just_the_first() -> None:
    """Fixing one blocker should not reveal a surprise second one."""
    outcome = check(
        make_intent(symbol="TSLA", amount_usd=500.0, detail={}),
        state=make_state(orders_today=5),
    )
    guards_hit = {message.split(":")[0] for message in outcome.messages()}
    assert {"allowlist", "per_order_notional", "quote_freshness", "daily_order_count"} <= guards_hit


def test_batch_accumulates_against_daily_cap() -> None:
    """Three $100 intents against a $250 cap must leave the third blocked."""
    limits = make_limits(max_notional_per_day=250.0, max_orders_per_day=10)
    intents = [
        make_intent(intent_id=f"i{i}", rule_id=f"r{i}", amount_usd=100.0) for i in range(3)
    ]

    result = apply_guards(intents, limits, make_state(), now=NOW)

    assert result[0].status == IntentStatus.READY_FOR_REVIEW
    assert result[1].status == IntentStatus.READY_FOR_REVIEW
    assert result[2].status == IntentStatus.BLOCKED
    assert any("daily_notional" in message for message in result[2].blocked_by)


def test_batch_accumulates_against_order_count() -> None:
    limits = make_limits(max_orders_per_day=2, max_notional_per_day=10_000.0)
    intents = [
        make_intent(intent_id=f"i{i}", rule_id=f"r{i}", amount_usd=50.0) for i in range(4)
    ]

    result = apply_guards(intents, limits, make_state(), now=NOW)

    assert sum(i.status == IntentStatus.READY_FOR_REVIEW for i in result) == 2
    assert sum(i.status == IntentStatus.BLOCKED for i in result) == 2


def test_batch_accumulates_against_position_cap() -> None:
    limits = make_limits(
        max_position_notional_per_symbol=150.0, max_orders_per_day=10, max_notional_per_day=10_000.0
    )
    intents = [
        make_intent(intent_id=f"i{i}", rule_id=f"r{i}", amount_usd=100.0) for i in range(2)
    ]

    result = apply_guards(intents, limits, make_state(), now=NOW)

    assert result[0].status == IntentStatus.READY_FOR_REVIEW
    assert result[1].status == IntentStatus.BLOCKED
    assert any("position_cap" in message for message in result[1].blocked_by)


def test_guards_never_mutate_order_size() -> None:
    """Blocked, not shrunk. Silently resizing hides that a cap was hit."""
    intent = make_intent(amount_usd=500.0)
    apply_guards([intent], make_limits(), make_state(), now=NOW)
    assert intent.amount_usd == 500.0
    assert intent.status == IntentStatus.BLOCKED


def test_blocked_intent_refuses_to_produce_review_arguments() -> None:
    intent = make_intent(symbol="TSLA")
    apply_guards([intent], make_limits(), make_state(), now=NOW)

    with pytest.raises(ValueError, match="not ready for review"):
        intent.review_call_arguments()


# --- buying power / partial funding -------------------------------------------


def test_buying_power_not_supplied_is_skipped() -> None:
    """Paper runs have no broker; the check is a pre-filter, not the gate."""
    outcome = check(make_intent(), state=make_state(available_buying_power=None))
    assert outcome.allowed, outcome.messages()


def test_unfundable_intent_is_blocked() -> None:
    outcome = check(
        make_intent(amount_usd=15.0), state=make_state(available_buying_power=0.0)
    )
    assert not outcome.allowed
    assert any("buying_power" in message for message in outcome.messages())


def test_exactly_affordable_intent_passes() -> None:
    outcome = check(
        make_intent(amount_usd=15.0), state=make_state(available_buying_power=15.0)
    )
    assert outcome.allowed, outcome.messages()


def test_partial_funding_buys_what_it_can_afford() -> None:
    """The point of the guard: $18 available funds the $15, not the $10 after it."""
    limits = make_limits(max_orders_per_day=10, max_notional_per_day=1000.0)
    intents = [
        make_intent(intent_id="a", rule_id="vti",  amount_usd=15.0),
        make_intent(intent_id="b", rule_id="vxus", amount_usd=10.0),
    ]

    result = apply_guards(
        intents, limits, make_state(available_buying_power=18.0), now=NOW
    )

    assert result[0].status == IntentStatus.READY_FOR_REVIEW
    assert result[1].status == IntentStatus.BLOCKED
    assert any("buying_power" in m for m in result[1].blocked_by)


def test_full_funding_buys_everything() -> None:
    limits = make_limits(max_orders_per_day=10, max_notional_per_day=1000.0)
    intents = [
        make_intent(intent_id="a", rule_id="vti",  amount_usd=15.0),
        make_intent(intent_id="b", rule_id="vxus", amount_usd=10.0),
    ]

    result = apply_guards(
        intents, limits, make_state(available_buying_power=25.0), now=NOW
    )
    assert all(i.status == IntentStatus.READY_FOR_REVIEW for i in result)


def test_zero_buying_power_blocks_the_whole_batch() -> None:
    """Today's actual situation: $80 of cash, none of it spendable."""
    limits = make_limits(max_orders_per_day=10, max_notional_per_day=1000.0)
    intents = [
        make_intent(intent_id="a", rule_id="vti",  amount_usd=15.0),
        make_intent(intent_id="b", rule_id="vxus", amount_usd=10.0),
    ]

    result = apply_guards(
        intents, limits, make_state(available_buying_power=0.0), now=NOW
    )
    assert all(i.status == IntentStatus.BLOCKED for i in result)
