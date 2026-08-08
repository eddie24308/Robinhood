"""Config, rules, ledger and engine tests — including the mode-isolation checks."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from autotrade.config import ConfigError, RiskLimits, load_config
from autotrade.engine import Engine, EngineError, Quote, load_quotes
from autotrade.intents import ExecutionStyle, IntentStatus, OrderIntent
from autotrade.ledger import Ledger, PaperBroker
from autotrade.rules import MarketContext, evaluate_rule

NOW = datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc)
TODAY = date(2026, 8, 10)

BASE_CONFIG = """
[account]
number = "771654944"
mode = "paper"

[limits]
max_notional_per_order = 100.0
max_notional_per_day = 300.0
allowlist = ["HOOD"]

[[rules]]
id = "hood-dip"
symbol = "HOOD"
condition = "price_below"
threshold = 95.0
amount_usd = 100.0
cooldown_days = 7
"""


def write_config(tmp_path: Path, text: str = BASE_CONFIG) -> Path:
    path = tmp_path / "autotrade.toml"
    # state_dir must precede every table header: a bare key written after
    # [[rules]] would land inside that rule, not at the top level.
    state_dir = str(tmp_path / ".autotrade").replace("\\", "\\\\")
    path.write_text(f'state_dir = "{state_dir}"\n' + text)
    return path


# --- config ------------------------------------------------------------------


def test_valid_config_loads(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    assert config.account.number == "771654944"
    assert config.account.masked() == "****4944"
    assert not config.account.is_live
    assert len(config.enabled_rules) == 1


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo'd cap must not read as 'no cap'."""
    text = BASE_CONFIG.replace(
        "max_notional_per_order = 100.0", "max_notional_per_ordr = 100.0"
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config(tmp_path, text))


def test_empty_allowlist_is_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace('allowlist = ["HOOD"]', "allowlist = []")
    with pytest.raises(ConfigError, match="allowlist"):
        load_config(write_config(tmp_path, text))


def test_rule_targeting_non_allowlisted_symbol_is_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace('allowlist = ["HOOD"]', 'allowlist = ["VOO"]')
    with pytest.raises(ConfigError, match="not in"):
        load_config(write_config(tmp_path, text))


def test_live_mode_without_confirmation_is_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace('mode = "paper"', 'mode = "live"').replace(
        'allowlist = ["HOOD"]', 'allowlist = ["HOOD"]\nrequire_confirmation = false'
    )
    with pytest.raises(ConfigError, match="require_confirmation"):
        load_config(write_config(tmp_path, text))


def test_absolute_ceiling_cannot_be_exceeded() -> None:
    with pytest.raises(ConfigError, match="hard ceiling"):
        RiskLimits(max_notional_per_order=50_000.0, allowlist=("HOOD",))


def test_sell_action_is_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG + '\naction = "sell"\n'
    with pytest.raises(ConfigError, match="only action='buy'"):
        load_config(write_config(tmp_path, text))


def test_duplicate_rule_ids_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG + """
[[rules]]
id = "hood-dip"
symbol = "HOOD"
condition = "price_below"
threshold = 80.0
amount_usd = 50.0
"""
    with pytest.raises(ConfigError, match="duplicate rule id"):
        load_config(write_config(tmp_path, text))


def test_unknown_condition_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace('condition = "price_below"', 'condition = "vibes"')
    with pytest.raises(ConfigError, match="unknown condition"):
        load_config(write_config(tmp_path, text))


def test_missing_threshold_rejected(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace("threshold = 95.0\n", "")
    with pytest.raises(ConfigError, match="threshold"):
        load_config(write_config(tmp_path, text))


def test_example_config_is_valid() -> None:
    """The shipped example must always load."""
    config = load_config("autotrade.example.toml")
    assert config.account.mode == "paper"
    assert config.limits.require_confirmation


# --- rules -------------------------------------------------------------------


def context(price: float, **overrides) -> MarketContext:
    base = dict(symbol="HOOD", price=price, quote_time=NOW, today=TODAY, history=None)
    base.update(overrides)
    return MarketContext(**base)


def rule(**overrides):
    from autotrade.config import RuleConfig

    base = dict(
        id="r1", symbol="HOOD", condition="price_below", threshold=95.0, amount_usd=100.0
    )
    base.update(overrides)
    return RuleConfig(**base)


def test_price_below_fires_and_holds() -> None:
    assert evaluate_rule(rule(), context(93.0)).fired
    assert not evaluate_rule(rule(), context(97.0)).fired


def test_price_above_fires() -> None:
    r = rule(condition="price_above", threshold=95.0)
    assert evaluate_rule(r, context(97.0)).fired
    assert not evaluate_rule(r, context(93.0)).fired


def test_drawdown_from_high() -> None:
    history = pd.Series([100.0, 105.0, 110.0, 100.0, 95.0])
    r = rule(condition="drawdown_from_high", threshold=10.0, lookback_days=5)

    assert evaluate_rule(r, context(95.0, history=history)).fired  # 13.6% off 110
    assert not evaluate_rule(r, context(105.0, history=history)).fired


def test_drawdown_without_history_does_not_fire() -> None:
    r = rule(condition="drawdown_from_high", threshold=10.0, lookback_days=30)
    evaluation = evaluate_rule(r, context(50.0, history=None))
    assert not evaluation.fired
    assert "no price history" in evaluation.reason


def test_near_period_low_fires_only_near_the_low() -> None:
    # Trailing low of 358; 2% tolerance puts the trigger at 365.16.
    history = pd.Series([380.0, 370.0, 358.0, 365.0, 381.0])
    r = rule(condition="near_period_low", threshold=2.0, lookback_days=5)

    assert evaluate_rule(r, context(362.0, history=history)).fired
    assert evaluate_rule(r, context(358.0, history=history)).fired
    assert not evaluate_rule(r, context(370.0, history=history)).fired


def test_near_period_low_is_silent_at_a_high() -> None:
    """The realistic case: VTI at a new high must not trigger a dip buy."""
    history = pd.Series([358.04, 365.0, 372.0, 379.07])
    r = rule(condition="near_period_low", threshold=2.0, lookback_days=60)

    evaluation = evaluate_rule(r, context(381.74, history=history))
    assert not evaluation.fired
    assert "above the" in evaluation.reason


def test_near_period_low_zero_tolerance_needs_the_actual_low() -> None:
    history = pd.Series([380.0, 360.0, 370.0])
    r = rule(condition="near_period_low", threshold=0.0, lookback_days=5)

    assert evaluate_rule(r, context(360.0, history=history)).fired
    assert not evaluate_rule(r, context(360.01, history=history)).fired


def test_near_period_low_without_history_does_not_fire() -> None:
    r = rule(condition="near_period_low", threshold=2.0, lookback_days=60)
    evaluation = evaluate_rule(r, context(100.0, history=None))
    assert not evaluation.fired
    assert "no price history" in evaluation.reason


def test_near_period_low_rejects_bad_threshold() -> None:
    from autotrade.config import RuleConfig

    with pytest.raises(ConfigError, match="tolerance above the low"):
        RuleConfig(
            id="bad",
            symbol="VTI",
            condition="near_period_low",
            threshold=150.0,
            lookback_days=60,
            amount_usd=25.0,
        )


def test_weekly_schedule_fires_on_its_day() -> None:
    r = rule(condition="weekly_schedule", weekday="monday", threshold=None)
    assert evaluate_rule(r, context(93.0, today=date(2026, 8, 10))).fired  # Monday
    assert not evaluate_rule(r, context(93.0, today=date(2026, 8, 11))).fired  # Tuesday


def test_cooldown_blocks_before_condition_is_even_checked() -> None:
    evaluation = evaluate_rule(
        rule(cooldown_days=7), context(50.0, last_fired=date(2026, 8, 8))
    )
    assert not evaluation.fired
    assert "cooling down" in evaluation.reason


def test_cooldown_expires() -> None:
    assert evaluate_rule(
        rule(cooldown_days=7), context(50.0, last_fired=date(2026, 8, 1))
    ).fired


def test_disabled_rule_never_fires() -> None:
    assert not evaluate_rule(rule(enabled=False), context(50.0)).fired


# --- ledger ------------------------------------------------------------------


def make_intent(tmp_path=None, **overrides) -> OrderIntent:
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


def test_ledger_round_trip(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    intent = make_intent()
    ledger.record_intent(intent)

    recovered = ledger.intents()
    assert len(recovered) == 1
    assert recovered[0].symbol == "HOOD"
    assert recovered[0].fingerprint() == intent.fingerprint()


def test_ledger_records_blocked_intents(tmp_path: Path) -> None:
    """The log of what did NOT happen is the half you will actually need."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    blocked = make_intent(status=IntentStatus.BLOCKED, blocked_by=["allowlist: nope"])
    ledger.record_intent(blocked)

    assert ledger.intents()[0].status == IntentStatus.BLOCKED
    assert ledger.intents()[0].blocked_by == ["allowlist: nope"]


def test_blocked_intent_does_not_consume_daily_budget(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_intent(make_intent(status=IntentStatus.BLOCKED))

    state = ledger.state_for(TODAY)
    assert state.notional_today == 0.0
    assert state.orders_today == 0


def test_paper_fill_updates_state(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    PaperBroker(ledger).fill(make_intent())

    state = ledger.state_for(TODAY)
    assert state.orders_today == 1
    assert state.notional_today == pytest.approx(100.0)
    assert state.position_for("HOOD") == pytest.approx(100.0)


def test_paper_fill_is_pessimistic(tmp_path: Path) -> None:
    """Fills at the limit plus slippage, never better."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    fill = PaperBroker(ledger, slippage_bps=5.0).fill(make_intent(limit_price=100.0))
    assert fill.price > 100.0


def test_paper_broker_refuses_live_intents(tmp_path: Path) -> None:
    """Structural guarantee: paper mode cannot reach a real order."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="PaperBroker refuses"):
        PaperBroker(ledger).fill(make_intent(mode="live"))


def test_last_fired_tracks_committed_only(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_intent(make_intent(intent_id="a", status=IntentStatus.BLOCKED))
    assert ledger.last_fired() == {}

    ledger.record_intent(make_intent(intent_id="b", status=IntentStatus.PAPER_FILLED))
    assert ledger.last_fired()["r1"] == TODAY


def test_truncated_ledger_line_is_survivable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record_intent(make_intent())
    with open(path, "a") as handle:
        handle.write('{"event": "intent", "inten')  # simulate a crash mid-write

    assert len(ledger.intents()) == 1


# --- engine ------------------------------------------------------------------


def test_engine_plans_and_paper_fills(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    engine = Engine(config)

    quotes = {"HOOD": Quote("HOOD", 93.0, NOW)}
    result = engine.plan(quotes, today=TODAY, now=NOW)

    assert len(result.actionable) == 1
    intent = result.actionable[0]
    assert intent.symbol == "HOOD"
    assert intent.order_type == "limit"
    assert intent.limit_price > intent.reference_price  # marketable

    fills = engine.execute_paper(result)
    assert len(fills) == 1


def test_engine_respects_kill_switch(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    config.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
    config.kill_switch_path.write_text("halted")

    engine = Engine(config)
    result = engine.plan({"HOOD": Quote("HOOD", 93.0, NOW)}, today=TODAY, now=NOW)

    assert result.actionable == []
    assert any("KILL SWITCH" in note for note in result.notes)


def test_engine_blocks_stale_quotes(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    engine = Engine(config)

    stale_quote = Quote("HOOD", 93.0, NOW - timedelta(hours=2))
    result = engine.plan({"HOOD": stale_quote}, today=TODAY, now=NOW)

    assert result.actionable == []
    assert any("quote_freshness" in b for i in result.blocked for b in i.blocked_by)


def test_engine_does_not_fire_without_a_quote(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    result = Engine(config).plan({}, today=TODAY, now=NOW)

    assert result.intents == []
    assert any("no quote" in e.reason for e in result.evaluations)


def test_cooldown_prevents_same_day_double_buy(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    engine = Engine(config)
    quotes = {"HOOD": Quote("HOOD", 93.0, NOW)}

    first = engine.plan(quotes, today=TODAY, now=NOW)
    engine.execute_paper(first)

    second = engine.plan(quotes, today=TODAY, now=NOW)
    assert second.actionable == []


def test_execute_paper_refuses_in_live_mode(tmp_path: Path) -> None:
    """The most important negative test in the package."""
    text = BASE_CONFIG.replace('mode = "paper"', 'mode = "live"')
    config = load_config(write_config(tmp_path, text))
    engine = Engine(config)

    result = engine.plan({"HOOD": Quote("HOOD", 93.0, NOW)}, today=TODAY, now=NOW)
    with pytest.raises(EngineError, match="never through the paper broker"):
        engine.execute_paper(result)


def test_live_mode_emits_intents_without_placing(tmp_path: Path) -> None:
    text = BASE_CONFIG.replace('mode = "paper"', 'mode = "live"')
    config = load_config(write_config(tmp_path, text))
    engine = Engine(config)

    result = engine.plan({"HOOD": Quote("HOOD", 93.0, NOW)}, today=TODAY, now=NOW)
    out = engine.emit_for_review(result, tmp_path / "intents.json")

    payload = json.loads(out.read_text())
    assert out.exists()
    assert len(payload["intents"]) == 1
    assert payload["mode"] == "live"
    assert payload["requires_confirmation"] is True

    # The emitted arguments must be a valid review_equity_order call.
    arguments = result.actionable[0].review_call_arguments()
    assert set(arguments) >= {"account_number", "symbol", "side", "type", "limit_price"}
    assert arguments["side"] == "buy"
    assert arguments["type"] == "limit"


def test_only_actionable_intents_are_emitted(tmp_path: Path) -> None:
    """A blocked intent must never reach the execution layer in any form."""
    text = BASE_CONFIG.replace('mode = "paper"', 'mode = "live"').replace(
        "max_notional_per_order = 100.0", "max_notional_per_order = 10.0"
    )
    config = load_config(write_config(tmp_path, text))
    engine = Engine(config)

    result = engine.plan({"HOOD": Quote("HOOD", 93.0, NOW)}, today=TODAY, now=NOW)
    assert result.blocked and not result.actionable

    out = engine.emit_for_review(result, tmp_path / "intents.json")
    assert '"intents": []' in out.read_text()


ETF_CONFIG = """
[account]
number = "771654944"
mode = "paper"

[limits]
max_notional_per_order = 100.0
max_notional_per_day = 300.0
allowlist = ["VOO"]
allow_fractional = true

[[rules]]
id = "voo-weekly"
symbol = "VOO"
condition = "every_run"
amount_usd = 100.0
cooldown_days = 7
"""


def test_sub_share_order_uses_fractional_market(tmp_path: Path) -> None:
    """$100 of a $710 ETF cannot be a limit order — the broker rejects those."""
    config = load_config(write_config(tmp_path, ETF_CONFIG))
    engine = Engine(config)

    quote = Quote("VOO", 710.57, NOW, bid=710.52, ask=711.00)
    result = engine.plan({"VOO": quote}, today=TODAY, now=NOW)

    assert len(result.actionable) == 1
    intent = result.actionable[0]
    assert intent.execution_style is ExecutionStyle.NOTIONAL_MARKET

    arguments = intent.review_call_arguments()
    assert arguments["type"] == "market"
    assert arguments["dollar_amount"] == "100.00"
    assert arguments["market_hours"] == "regular_hours"
    # A fractional quantity on a limit order is exactly what gets rejected.
    assert "quantity" not in arguments
    assert "limit_price" not in arguments


def test_affordable_order_uses_whole_share_limit(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))
    engine = Engine(config)

    quote = Quote("HOOD", 93.28, NOW, bid=93.20, ask=93.35)
    result = engine.plan({"HOOD": quote}, today=TODAY, now=NOW)

    intent = result.actionable[0]
    assert intent.execution_style is ExecutionStyle.WHOLE_SHARE_LIMIT

    arguments = intent.review_call_arguments()
    assert arguments["type"] == "limit"
    assert arguments["quantity"] == "1"
    assert float(arguments["limit_price"]) > 93.28
    assert "dollar_amount" not in arguments


def test_whole_share_quantity_is_never_fractional(tmp_path: Path) -> None:
    """Whatever the budget, a limit order's quantity must be an integer."""
    config = load_config(write_config(tmp_path))
    engine = Engine(config)

    result = engine.plan(
        {"HOOD": Quote("HOOD", 93.28, NOW, bid=93.20, ask=93.35)}, today=TODAY, now=NOW
    )
    quantity = result.actionable[0].review_call_arguments()["quantity"]
    assert float(quantity).is_integer()


def test_fractional_blocked_when_not_opted_in(tmp_path: Path) -> None:
    text = ETF_CONFIG.replace("allow_fractional = true", "allow_fractional = false")
    config = load_config(write_config(tmp_path, text))
    engine = Engine(config)

    quote = Quote("VOO", 710.57, NOW, bid=710.52, ask=711.00)
    result = engine.plan({"VOO": quote}, today=TODAY, now=NOW)

    assert not result.actionable
    assert any("fractional_allowed" in b for i in result.blocked for b in i.blocked_by)


def test_intent_round_trips_execution_style(tmp_path: Path) -> None:
    intent = make_intent(
        execution_style=ExecutionStyle.NOTIONAL_MARKET, bid_price=710.52, ask_price=711.0
    )
    restored = OrderIntent.from_dict(intent.to_dict())
    assert restored.execution_style is ExecutionStyle.NOTIONAL_MARKET
    assert restored.bid_price == pytest.approx(710.52)


def test_quote_must_carry_a_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "quotes.json"
    path.write_text('{"quotes": {"HOOD": 93.28}}')
    with pytest.raises(EngineError, match="bare number"):
        load_quotes(path)


def test_autotrade_cannot_import_broker_tools() -> None:
    """The package must have no path to order placement.

    If someone later adds a network client here, this test is the tripwire.
    """
    import autotrade

    source_root = Path(autotrade.__file__).parent
    banned = ("place_equity_order", "requests", "httpx", "urllib.request", "socket")

    for module in source_root.glob("*.py"):
        text = module.read_text()
        for token in banned:
            # place_equity_order may appear in docstrings describing the handoff,
            # never in an import or call position.
            assert f"import {token}" not in text, f"{module.name} imports {token}"
            assert f"{token}(" not in text, f"{module.name} calls {token}"
