"""autotrade — rule-driven order preparation with mandatory human approval.

This package decides *what* to buy and enforces *how much*. It deliberately
cannot place an order. The separation is structural, not a matter of
discipline:

    autotrade (Python)        decides + enforces limits, emits OrderIntents
    agent session             runs review_equity_order, shows you the cost
    you                       approve or reject, per order
    agent session             runs place_equity_order only after approval

Nothing in this package imports or can reach the Robinhood tools. The worst a
bug here can do is emit an intent that a human then declines. That property is
what makes it safe to iterate on the rules.

Two modes:

``paper``  Intents are filled against a simulated book. No real order exists.
``live``   Intents are written out for the agent to review with you. Reaching
           this mode requires an explicit config change plus a per-order
           confirmation, and every guard still applies.
"""

__version__ = "0.1.0"

from autotrade.config import (
    AccountConfig,
    AutotradeConfig,
    RiskLimits,
    RuleConfig,
    load_config,
)
from autotrade.engine import Engine, PlanResult
from autotrade.errors import AutotradeError, ConfigError, EngineError
from autotrade.guards import GuardOutcome, GuardViolation, RiskGuard
from autotrade.intents import ExecutionStyle, IntentStatus, OrderIntent
from autotrade.ledger import Ledger
from autotrade.rules import RULE_TYPES, RuleEvaluation, evaluate_rule

__all__ = [
    "RULE_TYPES",
    "AutotradeError",
    "ConfigError",
    "EngineError",
    "AccountConfig",
    "AutotradeConfig",
    "Engine",
    "GuardOutcome",
    "GuardViolation",
    "ExecutionStyle",
    "IntentStatus",
    "Ledger",
    "OrderIntent",
    "PlanResult",
    "RiskGuard",
    "RiskLimits",
    "RuleConfig",
    "RuleEvaluation",
    "evaluate_rule",
    "load_config",
]
