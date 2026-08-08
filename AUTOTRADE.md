# autotrade — runbook

Rule-driven order preparation for the Robinhood "Agentic" account (••••4944),
with mandatory human approval on every order.

## The one thing to understand

**This package cannot place an order.** Not "is configured not to" — it has no
network client, no broker credentials, and no import path to the Robinhood
tools. A test (`test_autotrade_cannot_import_broker_tools`) enforces that.

Execution is split across three parties that each hold a veto:

| Stage | Who | Can it spend money? |
|---|---|---|
| Decide + enforce limits | `autotrade` (Python) | No — emits a JSON file |
| Review with you | agent session | No — `review_equity_order` only simulates |
| Approve | **you** | This is the gate |
| Place | agent session | Only after your explicit approval |

A bug in the rules produces a proposal you decline. That is the worst case, by
construction.

## Setup

```bash
cp autotrade.example.toml autotrade.toml
$EDITOR autotrade.toml
python -m autotrade check          # validates and prints every limit
```

`autotrade.toml` is gitignored — it carries your account number and your real
limits. Only the example is committed.

## Daily use (paper)

```bash
python -m autotrade run --from-csv     # evaluate rules, simulate fills
python -m autotrade status             # positions, today's limit usage
python -m autotrade log --limit 20     # every decision, including blocked ones
```

`--from-csv` prices off stored `data/<SYMBOL>.csv` closes. It is refused in
live mode — stale closes are not quotes.

## Going live

Two changes are required, and they are deliberately separate:

1. Set `mode = "live"` in `autotrade.toml`.
2. Ask the agent to run the live cycle.

Then the loop is:

```
agent: get_equity_quotes  ->  writes quotes.json
you:   python -m autotrade plan --quotes quotes.json
       -> writes .autotrade/intents.json, places nothing
agent: review_equity_order for each intent, shows you cost + alerts
you:   approve or decline, per order
agent: place_equity_order  (only for the ones you approved)
       python -m autotrade record --intent-id <id> --decision placed
```

`run` is paper-only and refuses in live mode; live must go through `plan`, so
the intent file and the review step cannot be skipped by habit.

## Kill switch

```bash
python -m autotrade halt --reason "stepping away"
python -m autotrade resume
```

`halt` writes `.autotrade/HALT`. While it exists, every intent is blocked
before any other guard runs. Deleting the file by hand works too — that is the
point of using a file rather than a config flag.

## Guards

All of these are enforced in `autotrade/guards.py`, and each has a test that
proves it blocks. Guards never resize an order to fit — an order over a cap is
blocked, so you find out the cap was hit.

| Guard | Blocks when |
|---|---|
| `kill_switch` | `.autotrade/HALT` exists |
| `side` | anything other than a buy |
| `allowlist` | symbol not explicitly listed |
| `price_sanity` | non-positive price, or limit above the configured offset |
| `quote_freshness` | quote older than `max_quote_age_seconds`, missing, or in the future |
| `per_order_notional` | order above `max_notional_per_order` |
| `daily_notional` | today's committed total would exceed the daily cap |
| `daily_order_count` | already at `max_orders_per_day` |
| `position_cap` | symbol position would exceed its cap |
| `daily_loss_limit` | realised losses today at or past the stop |
| `duplicate` | identical intent already proposed today |
| `trade_date` | plan is dated for a different day |

Violations accumulate across a batch: three $100 intents against a $250 daily
cap leave the third blocked, not all three approved.

Two ceilings in `autotrade/config.py` cannot be raised from the config at all —
$5,000/order and $10,000/day. Changing those requires editing the source, which
is a deliberate speed bump.

## Rules

```bash
python -m autotrade rules      # list conditions
```

| Condition | Fires when |
|---|---|
| `price_below` / `price_above` | price crosses `threshold` |
| `drawdown_from_high` | price is `threshold`% below the `lookback_days` high |
| `rally_from_low` | price is `threshold`% above the `lookback_days` low |
| `weekly_schedule` | it is `weekday` (plain DCA) |
| `every_run` | always, throttled by `cooldown_days` |

`cooldown_days` is checked *before* the condition, so a rule that fired
yesterday cannot fire again today no matter what the price does.

**Only buys are automated.** `action = "sell"` is rejected at config load.
Exits deserve a human looking at them, and an automated stop-loss that fires
during a data glitch is a well-known way to sell the bottom.

## What is deliberately not here

- **No model-driven trading.** The `stockforecast` model measured *negative*
  out-of-sample skill on HOOD. Wiring it to orders would pay spreads to trade
  noise. If a signal ever clears its own walk-forward test, revisit this.
- **No selling, no options, no margin.**
- **No unattended live placement.** `mode = "live"` with
  `require_confirmation = false` is rejected at config load.

## Audit log

`.autotrade/ledger.jsonl` — append-only, fsynced, one JSON object per line.
Blocked intents are recorded alongside filled ones, because "why didn't it buy
the dip?" is the question you will actually have. Positions and daily counters
are *derived* by replaying the log, so state and its explanation cannot
disagree.
