"""Append-only audit log and derived portfolio state.

Every decision is recorded, including the ones that were blocked — a log of
only the orders that happened cannot answer "why didn't it buy the dip?", which
is the question you will actually have.

The log is JSONL: one JSON object per line, appended, never rewritten. That
makes it durable against a crash mid-write (you lose at most the last line) and
trivially greppable. Portfolio state is *derived* by replaying the log rather
than stored separately, so the state and its explanation cannot disagree.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autotrade.guards import PortfolioState
from autotrade.intents import IntentStatus, OrderIntent


@dataclass(frozen=True)
class Fill:
    """A simulated or recorded fill."""

    symbol: str
    quantity: float
    price: float
    notional: float
    trade_date: date
    intent_id: str
    mode: str


class Ledger:
    """Append-only JSONL event log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- writing -------------------------------------------------------------

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one event, flushed and fsynced before returning.

        The fsync is deliberate. This log is the only record that an order was
        proposed; losing it to a page cache on a crash would leave real orders
        with no provenance.
        """
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_intent(self, intent: OrderIntent) -> None:
        self.append("intent", {"intent": intent.to_dict()})

    def record_fill(self, fill: Fill) -> None:
        self.append(
            "fill",
            {
                "symbol": fill.symbol,
                "quantity": fill.quantity,
                "price": fill.price,
                "notional": fill.notional,
                "trade_date": fill.trade_date.isoformat(),
                "intent_id": fill.intent_id,
                "mode": fill.mode,
            },
        )

    def record_decision(self, intent_id: str, decision: str, note: str = "") -> None:
        """Record a human approve/reject, or an agent-side placement result."""
        self.append(
            "decision", {"intent_id": intent_id, "decision": decision, "note": note}
        )

    # --- reading -------------------------------------------------------------

    def events(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line is expected after a crash; skip it
                    # rather than losing the whole history.
                    print(f"  ! {self.path}:{line_number} is not valid JSON, skipping")

    def fills(self) -> list[Fill]:
        out = []
        for event in self.events():
            if event.get("event") != "fill":
                continue
            out.append(
                Fill(
                    symbol=event["symbol"],
                    quantity=float(event["quantity"]),
                    price=float(event["price"]),
                    notional=float(event["notional"]),
                    trade_date=date.fromisoformat(event["trade_date"]),
                    intent_id=event["intent_id"],
                    mode=event.get("mode", "paper"),
                )
            )
        return out

    def intents(self) -> list[OrderIntent]:
        out = []
        for event in self.events():
            if event.get("event") == "intent":
                out.append(OrderIntent.from_dict(event["intent"]))
        return out

    # --- derived state -------------------------------------------------------

    def state_for(self, trade_date: date) -> PortfolioState:
        """Replay the log to derive today's limits state and open positions.

        Positions accumulate across all history; the daily counters reset on
        each new trade date. Only intents that reached a terminal *committed*
        status count toward the daily caps — a blocked proposal must not
        consume the day's budget.
        """
        positions: dict[str, float] = {}
        orders_today = 0
        notional_today = 0.0
        fingerprints_today: set[str] = set()

        committed = {
            IntentStatus.PAPER_FILLED.value,
            IntentStatus.APPROVED.value,
            IntentStatus.PLACED.value,
        }

        for event in self.events():
            if event.get("event") == "fill":
                symbol = str(event["symbol"]).upper()
                positions[symbol] = positions.get(symbol, 0.0) + float(event["notional"])
                if date.fromisoformat(event["trade_date"]) == trade_date:
                    orders_today += 1
                    notional_today += float(event["notional"])

            elif event.get("event") == "intent":
                payload = event["intent"]
                if date.fromisoformat(payload["trade_date"]) != trade_date:
                    continue
                if payload.get("status") in committed:
                    fingerprints_today.add(payload.get("fingerprint", ""))

        return PortfolioState(
            trade_date=trade_date,
            orders_today=orders_today,
            notional_today=notional_today,
            realized_pnl_today=0.0,  # realised P&L requires sells; buys only for now
            positions_notional=positions,
            fingerprints_today=fingerprints_today,
        )

    def last_fired(self) -> dict[str, date]:
        """Most recent committed fire date per rule, for cooldown checks."""
        out: dict[str, date] = {}
        committed = {
            IntentStatus.PAPER_FILLED.value,
            IntentStatus.APPROVED.value,
            IntentStatus.PLACED.value,
        }
        for event in self.events():
            if event.get("event") != "intent":
                continue
            payload = event["intent"]
            if payload.get("status") not in committed:
                continue
            rule_id = payload["rule_id"]
            fired_on = date.fromisoformat(payload["trade_date"])
            if rule_id not in out or fired_on > out[rule_id]:
                out[rule_id] = fired_on
        return out


class PaperBroker:
    """Simulates fills so the whole system can run without touching money.

    Fills are pessimistic on purpose: a buy fills at the limit price plus a
    configurable slippage, never better. A paper broker that fills at the
    midpoint flatters every strategy tested through it.
    """

    def __init__(self, ledger: Ledger, slippage_bps: float = 5.0) -> None:
        self.ledger = ledger
        self.slippage_bps = slippage_bps

    def fill(self, intent: OrderIntent) -> Fill:
        if intent.status != IntentStatus.READY_FOR_REVIEW:
            raise ValueError(
                f"cannot fill intent {intent.intent_id} with status {intent.status.value}"
            )
        if intent.mode != "paper":
            raise ValueError(
                f"PaperBroker refuses intent in mode {intent.mode!r}. This is the "
                "structural guarantee that paper mode cannot place a real order."
            )

        fill_price = intent.limit_price * (1.0 + self.slippage_bps / 10_000.0)
        quantity = intent.amount_usd / fill_price

        fill = Fill(
            symbol=intent.symbol.upper(),
            quantity=quantity,
            price=fill_price,
            notional=intent.amount_usd,
            trade_date=intent.trade_date,
            intent_id=intent.intent_id,
            mode="paper",
        )

        intent.status = IntentStatus.PAPER_FILLED
        self.ledger.record_intent(intent)
        self.ledger.record_fill(fill)
        return fill
