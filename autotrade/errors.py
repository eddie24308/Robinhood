"""Shared exception types.

These live in their own module so ``config`` and ``rules`` can both raise the
same class without importing each other. A rule-validation failure and a
config-validation failure must be the *same* exception type, or the CLI's
``except ConfigError`` silently misses half of them and a bad rule surfaces as
an unhandled traceback instead of a readable message.
"""

from __future__ import annotations


class AutotradeError(Exception):
    """Base class for every error this package raises."""


class ConfigError(AutotradeError, ValueError):
    """Config or rule definition is missing, malformed, or unsafe."""


class EngineError(AutotradeError, RuntimeError):
    """The engine cannot safely produce a plan."""
