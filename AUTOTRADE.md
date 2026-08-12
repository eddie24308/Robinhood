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

## Weekly checklist

Nominally Monday, but the rules are `every_run` with a 6-day cooldown, so any
day works — if Monday is blocked for funding, running this on Tuesday buys the
week rather than skipping it.

`autotrade.toml` is gitignored and the working container is ephemeral, so a new
session starts from the repo with no live config. Recreating it is two lines:

```bash
cp autotrade.example.toml autotrade.toml
sed -i 's/^mode = "paper".*/mode = "live"/' autotrade.toml
python -m autotrade check          # confirm LIVE, $15 VTI + $10 VXUS, caps
```

Then ask the agent to run the weekly buy. It will:

1. `get_equity_quotes` for VTI and VXUS, and write them to a quotes file.
2. `python -m autotrade plan --quotes ...` — rules evaluate, guards run,
   intents are written. Nothing is ordered.
3. `review_equity_order` per intent, and show you cost plus any broker alerts.
4. Place **only** what you approve, then `python -m autotrade record`.

Current state as of 2026-08-08: buying power $30, no positions. The plan spends
$25, so week one fits and week two does not without a deposit.

## Unattended execution

`auto_execute` lets the agent place a guarded intent without asking first. It is
off by default and turning it on takes two keys, not one:

```toml
require_confirmation = false
auto_execute = true
```

Setting only the first is rejected at config load — the human gate can never come
off by omission. A dedicated **$500/day ceiling** applies whenever `auto_execute`
is on, independent of `max_notional_per_day`, and raising it needs a source edit.

Every machine guard still runs: allowlist, per-order and daily caps, order count,
position cap, spread, quote staleness, duplicate, kill switch. What is gone is the
person. A bug, a bad quote, or a misread rule moves money before anyone looks.

### Partial funding

Pass `--buying-power <n>` and intents are funded in rule order: what fits is
proposed, what does not is blocked with a `buying_power` violation. A day with $18
against a $15 + $10 plan buys the $15 and blocks the $10, rather than failing both.

Use the broker's `buying_power`, **never the `cash` field**. They differ: on
2026-08-10 the account showed $80 cash and $0.00 buying power, because a $50
deposit was still pending. Cash you cannot spend is not funding.

### Scheduling it

A Routine (`create_trigger`) wakes this session on a cron. Creating one from
inside a session returns a warning that it stores no MCP connectors — but when the
Routine binds to a **persistent session** (the default), the fired turn runs inside
that session and reaches its connectors normally. Verified live on 2026-08-10: the
weekly Routine fired and had full `mcp__Robinhood__*` access.

The warning would matter for a Routine using `create_new_session_on_fire`, where
each firing starts cold. Keep the weekly buy bound to a persistent session.

The Routine's first instruction is still to check for broker tools and report
honestly rather than pretend, because a connector can drop at any time.

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
| `fractional_allowed` | order is sub-share (forcing a market order) without `allow_fractional` |
| `spread` | market order when bid-ask exceeds `max_spread_bps`, or bid/ask missing |
| `per_order_notional` | order above `max_notional_per_order` |
| `daily_notional` | today's committed total would exceed the daily cap |
| `daily_order_count` | already at `max_orders_per_day` |
| `buying_power` | order exceeds spendable buying power (skipped when not supplied) |
| `position_cap` | symbol position would exceed its cap |
| `daily_loss_limit` | realised losses today at or past the stop |
| `duplicate` | identical intent already proposed today |
| `trade_date` | plan is dated for a different day |

Violations accumulate across a batch: three $100 intents against a $250 daily
cap leave the third blocked, not all three approved.

Two ceilings in `autotrade/config.py` cannot be raised from the config at all —
$5,000/order and $10,000/day. Changing those requires editing the source, which
is a deliberate speed bump.

## Order types, and why it is not always a limit order

Robinhood permits fractional shares **only** on `type=market` with a
`dollar_amount`. A limit order must carry an integer quantity. So the order
size decides the order type, and `autotrade` picks it rather than defaulting:

| Situation | Style | What gets sent |
|---|---|---|
| Order affords ≥ 1 whole share | `whole_share_limit` | `type=limit`, integer `quantity`, `limit_price` |
| Order is smaller than 1 share | `notional_market` | `type=market`, `dollar_amount` |

A $25 buy of a $380 ETF is always the second row. That trades away price
protection, which is why two guards exist to compensate:

- `allow_fractional` must be explicitly set — the trade-off is opted into,
  never defaulted into.
- `max_spread_bps` blocks a market order into a wide book. This is not
  theoretical: on its first contact with a live quote it blocked a VTI order
  whose after-hours bid was $309.83 against a $381.74 last trade — a 2,096 bps
  spread. During regular hours the same book is 1-3 bps.

## Rules

```bash
python -m autotrade rules      # list conditions
```

| Condition | Fires when |
|---|---|
| `price_below` / `price_above` | price crosses `threshold` |
| `drawdown_from_high` | price is `threshold`% below the `lookback_days` high |
| `rally_from_low` | price is `threshold`% above the `lookback_days` low |
| `near_period_low` | price is within `threshold`% of the `lookback_days` low |
| `weekly_schedule` | it is `weekday` — fires only on that day, never catches up |
| `every_run` | always, throttled by `cooldown_days` (what the DCA rules use) |

`cooldown_days` is checked *before* the condition, so a rule that fired
yesterday cannot fire again today no matter what the price does.

### Why the weekly buys use `every_run`, not `weekly_schedule`

`weekly_schedule` looks the obvious fit for "buy every Monday", and it is the
wrong one, because it has no catch-up. It compares `today.weekday()` to the
target and nothing else. If Monday's buy is blocked — cash still unsettled, a
wide spread, a run that never happened — the rule is simply silent until the
following Monday, and the money sits idle for a week.

`every_run` fires on the first day the order can actually clear, with
`cooldown_days = 6` doing the weekly throttling. The cooldown reads from
`Ledger.last_fired()`, which counts only *committed* intents, so a blocked
Monday leaves the week open and Tuesday buys instead.
`test_unfunded_day_does_not_skip_the_week` pins that behaviour.

The trade-off is that the buy day drifts: a week delayed once stays delayed.
That is the right trade — a DCA plan cares that the money goes in, not which
weekday it goes in on.

### On "buy at the lowest"

`near_period_low` is the implementable version of that idea. You cannot know a
price is *the* low until well after the fact — the bottom is only visible in
hindsight. What is knowable in real time is that price sits at or near the
lowest point of a defined window.

Know what it does when it fires: a fresh N-day low usually means a downtrend,
so it buys into falling prices and will often be underwater shortly after.
That is the strategy, not a bug. But it is why the shipped config runs it
*alongside* the weekly buy rather than instead of it — a dip-only rule spends
most of its time not buying, and time out of a rising market has historically
cost more than the discount earned by waiting.

As of 2026-08-07, VTI at $381.74 sat 6.6% above its 60-day low of $358.04, so
the dip rule was silent. That is its normal state.

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
