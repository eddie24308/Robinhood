"""The decision engine.

Ties together config, rules, guards and the ledger:

    quotes + history  ->  rules fire  ->  intents built  ->  guards applied
                                                                 |
                                        paper mode: simulated fill
                                        live mode:  written out for agent review

The engine has no network access and no way to reach a broker. In live mode
its output is a JSON file. Something else has to pick that file up, show it to
a human, and act — which is the point.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from autotrade.config import AutotradeConfig
from autotrade.errors import EngineError
from autotrade.guards import PortfolioState, apply_guards
from autotrade.intents import ExecutionStyle, IntentStatus, OrderIntent
from autotrade.ledger import Ledger, PaperBroker
from autotrade.rules import MarketContext, RuleEvaluation, evaluate_rule


@dataclass
class Quote:
    """A price observation with the time it was taken."""

    symbol: str
    price: float
    time: datetime
    bid: float | None = None
    ask: float | None = None

    @classmethod
    def from_dict(cls, symbol: str, data: dict | float) -> Quote:
        if isinstance(data, (int, float)):
            raise EngineError(
                f"quote for {symbol} is a bare number. It must be "
                '{"price": ..., "time": "<ISO timestamp>"} so staleness can be checked.'
            )
        if "price" not in data or "time" not in data:
            raise EngineError(f"quote for {symbol} needs both 'price' and 'time'")

        time = data["time"]
        if isinstance(time, str):
            time = datetime.fromisoformat(time.replace("Z", "+00:00"))
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)

        def optional(key: str) -> float | None:
            value = data.get(key)
            return float(value) if value not in (None, "", 0) else None

        return cls(
            symbol=symbol.upper(),
            price=float(data["price"]),
            time=time,
            bid=optional("bid"),
            ask=optional("ask"),
        )


@dataclass
class PlanResult:
    """Everything the engine decided on one run."""

    trade_date: date
    mode: str
    evaluations: list[RuleEvaluation] = field(default_factory=list)
    intents: list[OrderIntent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> list[OrderIntent]:
        return [intent for intent in self.intents if intent.is_actionable]

    @property
    def blocked(self) -> list[OrderIntent]:
        return [intent for intent in self.intents if intent.status == IntentStatus.BLOCKED]

    @property
    def total_notional(self) -> float:
        return sum(intent.amount_usd for intent in self.actionable)


def load_quotes(path: str | Path) -> dict[str, Quote]:
    """Load a quotes JSON file written by the agent from ``get_equity_quotes``."""
    path = Path(path)
    if not path.exists():
        raise EngineError(
            f"quotes file not found: {path}. Have the agent fetch quotes and write them, "
            "or use --from-csv to price off stored history."
        )

    payload = json.loads(path.read_text())
    raw = payload.get("quotes", payload)
    if not isinstance(raw, dict) or not raw:
        raise EngineError(f"{path} contains no quotes")

    return {symbol.upper(): Quote.from_dict(symbol, data) for symbol, data in raw.items()}


def quotes_from_history(
    history: dict[str, pd.DataFrame], as_of: datetime | None = None
) -> dict[str, Quote]:
    """Derive quotes from the last close of stored history.

    For paper runs only. The timestamps are synthetic, so a live config that
    tried to use this would be caught by the quote-freshness guard.
    """
    as_of = as_of or datetime.now(timezone.utc)
    return {
        symbol.upper(): Quote(symbol.upper(), float(frame["close"].iloc[-1]), as_of)
        for symbol, frame in history.items()
    }


class Engine:
    """Evaluates rules and produces guarded intents."""

    def __init__(
        self,
        config: AutotradeConfig,
        ledger: Ledger | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger or Ledger(config.state_dir / "ledger.jsonl")

    @property
    def kill_switch_active(self) -> bool:
        return self.config.kill_switch_path.exists()

    def plan(
        self,
        quotes: dict[str, Quote],
        history: dict[str, pd.DataFrame] | None = None,
        today: date | None = None,
        now: datetime | None = None,
    ) -> PlanResult:
        """Evaluate every enabled rule and return guarded intents."""
        now = now or datetime.now(timezone.utc)
        today = today or now.date()
        history = history or {}

        result = PlanResult(trade_date=today, mode=self.config.account.mode)

        if self.kill_switch_active:
            result.notes.append(
                f"KILL SWITCH ACTIVE ({self.config.kill_switch_path}) - everything blocked"
            )

        state = self.ledger.state_for(today)
        last_fired = self.ledger.last_fired()

        proposals: list[OrderIntent] = []
        for rule in self.config.enabled_rules:
            quote = quotes.get(rule.symbol.upper())
            if quote is None:
                result.evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.id,
                        fired=False,
                        reason=f"no quote available for {rule.symbol}",
                    )
                )
                continue

            context = MarketContext(
                symbol=rule.symbol.upper(),
                price=quote.price,
                quote_time=quote.time,
                today=today,
                history=(
                    history[rule.symbol.upper()]["close"]
                    if rule.symbol.upper() in history
                    else None
                ),
                last_fired=last_fired.get(rule.id),
            )

            evaluation = evaluate_rule(rule, context)
            result.evaluations.append(evaluation)
            if not evaluation.fired:
                continue

            proposals.append(self._build_intent(rule, quote, today, now, evaluation))

        result.intents = apply_guards(
            proposals,
            self.config.limits,
            state,
            kill_switch_active=self.kill_switch_active,
            now=now,
        )

        # Blocked intents are logged too — the record of what did *not* happen
        # is the more useful half when something looks wrong later.
        for intent in result.intents:
            if intent.status == IntentStatus.BLOCKED:
                self.ledger.record_intent(intent)

        return result

    def _build_intent(
        self,
        rule,  # RuleConfig
        quote: Quote,
        today: date,
        now: datetime,
        evaluation: RuleEvaluation,
    ) -> OrderIntent:
        # Marketable limit: priced slightly through the reference so it fills
        # promptly, but never at an unbounded market price.
        offset = self.config.limits.limit_offset_bps / 10_000.0
        # Floor to the cent rather than rounding. On a buy, rounding up would
        # push the limit past the offset the user configured; flooring keeps it
        # at or under, costing at most a cent of fill probability.
        limit_price = math.floor(quote.price * (1.0 + offset) * 100.0) / 100.0

        # A limit order needs at least one whole share; anything smaller can
        # only be expressed as a fractional market order.
        affords_whole_share = rule.amount_usd >= limit_price
        style = (
            ExecutionStyle.WHOLE_SHARE_LIMIT
            if affords_whole_share
            else ExecutionStyle.NOTIONAL_MARKET
        )

        return OrderIntent(
            intent_id=str(uuid.uuid4()),
            created_at=now,
            trade_date=today,
            account_number=self.config.account.number,
            mode=self.config.account.mode,
            symbol=rule.symbol.upper(),
            side="buy",
            amount_usd=float(rule.amount_usd),
            reference_price=quote.price,
            order_type="limit" if style == ExecutionStyle.WHOLE_SHARE_LIMIT else "market",
            limit_price=limit_price,
            time_in_force="gfd",
            execution_style=style,
            bid_price=quote.bid,
            ask_price=quote.ask,
            rule_id=rule.id,
            rule_reason=evaluation.reason,
            status=IntentStatus.READY_FOR_REVIEW,
            detail={
                "quote_time": quote.time.isoformat(),
                "condition": rule.condition,
                "note": rule.note,
                **evaluation.detail,
            },
        )

    def execute_paper(self, result: PlanResult) -> list:
        """Fill actionable intents against the simulated book.

        Refuses outright in live mode. The check is here as well as in
        :class:`PaperBroker` because a single guard on a money path is not
        enough.
        """
        if self.config.account.is_live:
            raise EngineError(
                "execute_paper called with mode='live'. Live intents must go through "
                "agent review and human approval, never through the paper broker."
            )

        broker = PaperBroker(self.ledger)
        return [broker.fill(intent) for intent in result.actionable]

    def emit_for_review(self, result: PlanResult, path: str | Path) -> Path:
        """Write actionable intents for the agent to review with the user.

        Only actionable intents are written. A blocked intent is never handed
        to the execution layer in any form.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "account_number": self.config.account.number,
            "mode": self.config.account.mode,
            "requires_confirmation": self.config.limits.require_confirmation,
            "auto_execute": self.config.limits.auto_execute,
            "workflow": [
                "1. For each intent, call review_equity_order with review_call_arguments.",
                "2. Show the user the reviewed cost and every alert returned.",
                "3. Place with place_equity_order ONLY after explicit user approval.",
                "4. Record the outcome with `python -m autotrade record`.",
            ],
            "intents": [intent.to_dict() for intent in result.actionable],
            "review_calls": [
                {"intent_id": intent.intent_id, "arguments": intent.review_call_arguments()}
                for intent in result.actionable
            ],
        }
        path.write_text(json.dumps(payload, indent=2))

        for intent in result.actionable:
            self.ledger.record_intent(intent)

        return path
