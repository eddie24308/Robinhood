"""Order intents — the only thing this package produces.

An :class:`OrderIntent` is a *proposal*. It is inert data: a description of an
order someone might place, plus the full record of why it was proposed and
which guards it passed. Turning one into a real order requires an agent to run
``review_equity_order``, a human to approve the reviewed cost, and only then
``place_equity_order``.

Intents are serialisable so the handoff between the decision layer, the agent,
and the audit log is a file on disk rather than a function call that could be
made by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class IntentStatus(str, Enum):
    """Where an intent stands.

    Note there is no status an intent can reach on its own that means
    "placed". Only the agent, after human approval, records ``PLACED``.
    """

    READY_FOR_REVIEW = "ready_for_review"
    BLOCKED = "blocked"
    PAPER_FILLED = "paper_filled"
    APPROVED = "approved"
    PLACED = "placed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderIntent:
    """A proposed buy, with its full provenance."""

    intent_id: str
    created_at: datetime
    trade_date: date
    account_number: str
    mode: str
    symbol: str
    side: str
    amount_usd: float
    reference_price: float
    order_type: str
    limit_price: float
    time_in_force: str
    rule_id: str
    rule_reason: str
    status: IntentStatus
    guards_passed: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_shares(self) -> float:
        if self.limit_price <= 0:
            return 0.0
        return self.amount_usd / self.limit_price

    @property
    def estimated_cost(self) -> float:
        """Worst-case cost if the limit fills at its limit price."""
        return self.estimated_shares * self.limit_price

    @property
    def is_actionable(self) -> bool:
        return self.status == IntentStatus.READY_FOR_REVIEW

    def fingerprint(self) -> str:
        """Stable hash of the economically meaningful fields.

        Used to detect a duplicate proposal for the same rule, symbol and day,
        so a re-run of ``plan`` cannot quietly double an order.
        """
        payload = f"{self.trade_date}|{self.rule_id}|{self.symbol}|{self.amount_usd:.2f}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["trade_date"] = self.trade_date.isoformat()
        data["status"] = self.status.value
        data["estimated_shares"] = round(self.estimated_shares, 6)
        data["estimated_cost"] = round(self.estimated_cost, 2)
        data["fingerprint"] = self.fingerprint()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderIntent:
        data = dict(data)
        for derived in ("estimated_shares", "estimated_cost", "fingerprint"):
            data.pop(derived, None)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["trade_date"] = date.fromisoformat(data["trade_date"])
        data["status"] = IntentStatus(data["status"])
        return cls(**data)

    def review_call_arguments(self) -> dict[str, Any]:
        """Exact arguments for ``review_equity_order``.

        Emitted rather than executed. The agent copies these verbatim, which
        keeps the decision layer authoritative about order parameters while
        leaving execution entirely outside this process.
        """
        if not self.is_actionable:
            raise ValueError(
                f"intent {self.intent_id} is {self.status.value}, not ready for review"
            )
        return {
            "account_number": self.account_number,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
            "quantity": f"{self.estimated_shares:.6f}",
            "limit_price": f"{self.limit_price:.2f}",
            "time_in_force": self.time_in_force,
            "market_hours": "regular_hours",
        }

    def summary_line(self) -> str:
        marker = {
            IntentStatus.READY_FOR_REVIEW: "READY",
            IntentStatus.BLOCKED: "BLOCKED",
            IntentStatus.PAPER_FILLED: "PAPER",
        }.get(self.status, self.status.value.upper())

        base = (
            f"[{marker:7s}] {self.symbol:6s} buy ${self.amount_usd:,.2f} "
            f"~{self.estimated_shares:.4f}sh @ limit {self.limit_price:.2f}  "
            f"(rule: {self.rule_id})"
        )
        if self.blocked_by:
            base += f"\n            blocked by: {'; '.join(self.blocked_by)}"
        return base


def write_intents(path: str, intents: list[OrderIntent]) -> None:
    """Write intents to JSON for the agent to pick up."""
    payload = [intent.to_dict() for intent in intents]
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def read_intents(path: str) -> list[OrderIntent]:
    """Read intents back from JSON."""
    with open(path) as handle:
        payload = json.load(handle)
    return [OrderIntent.from_dict(entry) for entry in payload]
