"""Command line for autotrade.

    python -m autotrade rules                      # what conditions exist
    python -m autotrade check                      # validate config, print limits
    python -m autotrade plan --from-csv            # evaluate rules, show decisions
    python -m autotrade run --from-csv             # plan + paper-fill (paper mode only)
    python -m autotrade status                     # positions and today's usage
    python -m autotrade log --limit 20             # recent audit events
    python -m autotrade record --intent-id X --decision placed
    python -m autotrade halt / resume              # kill switch
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from autotrade.config import ConfigError, load_config
from autotrade.engine import Engine, EngineError, load_quotes, quotes_from_history
from autotrade.intents import IntentStatus
from autotrade.ledger import Ledger
from autotrade.rules import describe_rule_types

RULE = "=" * 78
THIN = "-" * 78


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    ledger = Ledger(config.state_dir / "ledger.jsonl")
    return config, ledger, Engine(config, ledger)


def _load_history(config, symbols: list[str]) -> dict:
    """Load stored CSV history for drawdown-style rules."""
    from stockforecast.data import CsvProvider, DataError  # noqa: PLC0415

    provider = CsvProvider(config.state_dir.parent / "data")
    history = {}
    for symbol in symbols:
        try:
            history[symbol.upper()] = provider.fetch(symbol)
        except DataError:
            pass
    return history


def command_rules(args: argparse.Namespace) -> int:
    print("Available rule conditions:\n")
    print(describe_rule_types())
    print(
        "\nEach rule also takes: id, symbol, amount_usd, cooldown_days, note, enabled.\n"
        "Only action='buy' is supported - exits are deliberately not automated."
    )
    return 0


def command_check(args: argparse.Namespace) -> int:
    config, ledger, engine = _load(args)
    limits = config.limits

    print(RULE)
    print(f"CONFIG OK  {args.config}")
    print(RULE)
    print(f"  account            {config.account.masked()}  mode={config.account.mode.upper()}")
    if config.account.is_live:
        print("                     *** LIVE MODE - intents will be real orders ***")
    print(f"  kill switch        {'ACTIVE' if engine.kill_switch_active else 'inactive'}"
          f"  ({config.kill_switch_path})")
    print()
    print("  Limits")
    print(THIN)
    print(f"    per order        ${limits.max_notional_per_order:,.2f}")
    print(f"    per day          ${limits.max_notional_per_day:,.2f}")
    print(f"    per symbol       ${limits.max_position_notional_per_symbol:,.2f}")
    print(f"    orders/day       {limits.max_orders_per_day}")
    print(f"    daily loss stop  ${limits.daily_loss_limit:,.2f}")
    print(f"    max quote age    {limits.max_quote_age_seconds}s")
    print(f"    limit offset     {limits.limit_offset_bps:.0f} bps above reference")
    print(f"    confirmation     {'REQUIRED' if limits.require_confirmation else 'not required'}")
    print(f"    allowlist        {', '.join(limits.allowlist)}")
    print()
    print(f"  Rules ({len(config.enabled_rules)} enabled of {len(config.rules)})")
    print(THIN)
    for rule in config.rules:
        flag = "x" if rule.enabled else " "
        detail = []
        if rule.threshold is not None:
            detail.append(f"threshold={rule.threshold}")
        if rule.lookback_days is not None:
            detail.append(f"lookback={rule.lookback_days}d")
        if rule.weekday:
            detail.append(f"weekday={rule.weekday}")
        detail.append(f"cooldown={rule.cooldown_days}d")
        print(
            f"   [{flag}] {rule.id:20s} {rule.symbol:6s} {rule.condition:20s} "
            f"${rule.amount_usd:>8,.2f}  {' '.join(detail)}"
        )
    print(RULE)
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config, ledger, engine = _load(args)
    symbols = sorted({rule.symbol for rule in config.enabled_rules})
    history = _load_history(config, symbols)

    if args.from_csv:
        if config.account.is_live:
            raise SystemExit(
                "refusing --from-csv in live mode: stored closes are not live quotes. "
                "Have the agent write a quotes file and pass --quotes."
            )
        quotes = quotes_from_history(history)
        if not quotes:
            raise SystemExit("no CSV history found; populate data/<SYMBOL>.csv first")
    else:
        quotes = load_quotes(args.quotes)

    today = date.fromisoformat(args.date) if args.date else None
    result = engine.plan(quotes, history=history, today=today)
    _print_plan(result, config)

    if result.actionable and config.account.is_live:
        path = engine.emit_for_review(result, args.out or config.state_dir / "intents.json")
        print(f"\nWrote {len(result.actionable)} intent(s) for review to {path}")
        print("Nothing has been ordered. The agent must review each one with you first.")

    return 0


def command_run(args: argparse.Namespace) -> int:
    config, ledger, engine = _load(args)

    if config.account.is_live:
        raise SystemExit(
            "`run` executes fills directly and is paper-only. In live mode use `plan`, "
            "which emits intents for agent review and human approval."
        )

    symbols = sorted({rule.symbol for rule in config.enabled_rules})
    history = _load_history(config, symbols)
    quotes = quotes_from_history(history) if args.from_csv else load_quotes(args.quotes)

    today = date.fromisoformat(args.date) if args.date else None
    result = engine.plan(quotes, history=history, today=today)
    _print_plan(result, config)

    fills = engine.execute_paper(result)
    if fills:
        print(f"\nPaper filled {len(fills)} order(s):")
        for fill in fills:
            print(
                f"  {fill.symbol:6s} {fill.quantity:.4f}sh @ {fill.price:.2f} "
                f"= ${fill.notional:,.2f}"
            )
    else:
        print("\nNothing to fill.")
    return 0


def _print_plan(result, config) -> None:
    print(RULE)
    print(f"PLAN  {result.trade_date}  mode={result.mode.upper()}  "
          f"account={config.account.masked()}")
    print(RULE)

    for note in result.notes:
        print(f"  !! {note}")
    if result.notes:
        print()

    print("  Rule evaluations")
    print(THIN)
    for evaluation in result.evaluations:
        marker = "FIRED" if evaluation.fired else "  -  "
        print(f"   [{marker}] {evaluation.rule_id:20s} {evaluation.reason}")

    print()
    if not result.intents:
        print("  No intents proposed.")
        print(RULE)
        return

    print("  Intents")
    print(THIN)
    for intent in result.intents:
        print(f"   {intent.summary_line()}")

    print()
    print(f"  {len(result.actionable)} actionable, {len(result.blocked)} blocked, "
          f"${result.total_notional:,.2f} total notional")
    if result.mode == "paper":
        print("  PAPER MODE - no real order can result from this plan.")
    print(RULE)


def command_status(args: argparse.Namespace) -> int:
    config, ledger, engine = _load(args)
    today = date.fromisoformat(args.date) if args.date else datetime.now(timezone.utc).date()
    state = ledger.state_for(today)

    print(RULE)
    print(f"STATUS  {today}  account={config.account.masked()}  mode={config.account.mode}")
    print(RULE)
    print(f"  kill switch        {'ACTIVE' if engine.kill_switch_active else 'inactive'}")
    print()
    print("  Today")
    print(THIN)
    print(f"    orders           {state.orders_today} / {config.limits.max_orders_per_day}")
    print(f"    notional         ${state.notional_today:,.2f} / "
          f"${config.limits.max_notional_per_day:,.2f}")
    print()
    print("  Positions (cumulative cost basis from the ledger)")
    print(THIN)
    if not state.positions_notional:
        print("    none")
    else:
        for symbol, notional in sorted(state.positions_notional.items()):
            cap = config.limits.max_position_notional_per_symbol
            print(f"    {symbol:6s} ${notional:>10,.2f} / ${cap:,.2f}")

    fired = ledger.last_fired()
    if fired:
        print()
        print("  Last fired")
        print(THIN)
        for rule_id, fired_on in sorted(fired.items()):
            print(f"    {rule_id:20s} {fired_on}")
    print(RULE)
    return 0


def command_log(args: argparse.Namespace) -> int:
    _, ledger, _ = _load(args)
    events = list(ledger.events())
    if not events:
        print("no events logged yet")
        return 0

    for event in events[-args.limit :]:
        kind = event.get("event")
        timestamp = event.get("ts", "")[:19]
        if kind == "intent":
            intent = event["intent"]
            print(
                f"{timestamp}  INTENT   {intent['status']:16s} {intent['symbol']:6s} "
                f"${intent['amount_usd']:,.2f}  rule={intent['rule_id']}"
            )
            for reason in intent.get("blocked_by", []):
                print(f"{'':21s}   blocked: {reason}")
        elif kind == "fill":
            print(
                f"{timestamp}  FILL     {event['symbol']:6s} {float(event['quantity']):.4f}sh "
                f"@ {float(event['price']):.2f} = ${float(event['notional']):,.2f} "
                f"[{event.get('mode')}]"
            )
        elif kind == "decision":
            print(
                f"{timestamp}  DECISION {event['decision']:16s} intent={event['intent_id'][:8]} "
                f"{event.get('note', '')}"
            )
        else:
            print(f"{timestamp}  {kind}")
    return 0


def command_record(args: argparse.Namespace) -> int:
    """Record the outcome of an agent-executed order."""
    _, ledger, _ = _load(args)
    valid = {status.value for status in IntentStatus}
    if args.decision not in valid:
        raise SystemExit(f"--decision must be one of {sorted(valid)}")

    ledger.record_decision(args.intent_id, args.decision, args.note)
    print(f"Recorded {args.decision} for intent {args.intent_id}")
    return 0


def command_halt(args: argparse.Namespace) -> int:
    config, ledger, _ = _load(args)
    path = config.kill_switch_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Halted at {datetime.now(timezone.utc).isoformat()}\n"
        f"Reason: {args.reason or 'not given'}\n"
    )
    ledger.append("halt", {"reason": args.reason or ""})
    print(f"KILL SWITCH ACTIVE - created {path}")
    print("Every intent will now be blocked. Remove it with `python -m autotrade resume`.")
    return 0


def command_resume(args: argparse.Namespace) -> int:
    config, ledger, _ = _load(args)
    path = config.kill_switch_path
    if not path.exists():
        print("kill switch was not active; nothing to do")
        return 0
    path.unlink()
    ledger.append("resume", {})
    print(f"Removed {path}. Trading rules will be evaluated again on the next run.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotrade",
        description=(
            "Rule-driven order preparation with mandatory human approval. "
            "This tool cannot place an order."
        ),
    )
    parser.add_argument(
        "--config", default="autotrade.toml", help="path to the TOML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rules_parser = subparsers.add_parser("rules", help="list available rule conditions")
    rules_parser.set_defaults(func=command_rules)

    check_parser = subparsers.add_parser("check", help="validate config and show limits")
    check_parser.set_defaults(func=command_check)

    for name, handler, help_text in (
        ("plan", command_plan, "evaluate rules and show what would be ordered"),
        ("run", command_run, "plan and paper-fill (paper mode only)"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--quotes", help="quotes JSON written by the agent")
        sub.add_argument(
            "--from-csv",
            action="store_true",
            help="price off stored data/<SYMBOL>.csv closes (paper only)",
        )
        sub.add_argument("--date", help="override the trade date, YYYY-MM-DD")
        sub.add_argument("--out", help="where to write intents.json (live mode)")
        sub.set_defaults(func=handler)

    status_parser = subparsers.add_parser("status", help="positions and today's limit usage")
    status_parser.add_argument("--date", help="YYYY-MM-DD")
    status_parser.set_defaults(func=command_status)

    log_parser = subparsers.add_parser("log", help="recent audit events")
    log_parser.add_argument("--limit", type=int, default=20)
    log_parser.set_defaults(func=command_log)

    record_parser = subparsers.add_parser("record", help="record an order outcome")
    record_parser.add_argument("--intent-id", required=True)
    record_parser.add_argument("--decision", required=True)
    record_parser.add_argument("--note", default="")
    record_parser.set_defaults(func=command_record)

    halt_parser = subparsers.add_parser("halt", help="activate the kill switch")
    halt_parser.add_argument("--reason", default="")
    halt_parser.set_defaults(func=command_halt)

    resume_parser = subparsers.add_parser("resume", help="clear the kill switch")
    resume_parser.set_defaults(func=command_resume)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
