# Robinhood

Two packages that are deliberately **not** wired together:

- **`stockforecast`** — short-horizon return forecasting with honest skill
  measurement. Documented below.
- **`autotrade`** — rule-driven order preparation with mandatory human
  approval. See [AUTOTRADE.md](AUTOTRADE.md).

They stay separate on purpose. `stockforecast` measured *negative*
out-of-sample skill on HOOD, so `autotrade` acts on rules you write down, not
on model output. If a signal ever clears its own walk-forward test, that is the
moment to reconsider — not before.

---

# stockforecast

Short-horizon equity return forecasting for any ticker, built so that the
measured reliability of a forecast travels with the forecast itself.

The hard part of stock prediction is not producing a number. It is knowing
whether the number means anything. Most of this codebase is dedicated to that
second question.

## What it actually does

For any set of tickers it will:

1. Load OHLCV history from CSV, Yahoo Finance, or saved Robinhood MCP responses.
2. Build ~40 strictly causal features (momentum, volatility, trend location,
   RSI/MACD, ATR, volume z-scores, calendar terms).
3. Walk forward through time, refitting on an expanding window, with purging
   and embargo so overlapping forward returns cannot leak into the test folds.
4. Score the model against three baselines — zero return, training mean, and
   scaled momentum — using RMSE skill, out-of-sample R², directional accuracy
   with a Wilson confidence interval, Spearman information coefficient, and a
   Diebold–Mariano test with Newey–West correction for overlapping horizons.
5. Calibrate split-conformal prediction intervals on held-out residuals and
   report the *realised* coverage against nominal.
6. Emit a forecast that always carries the above with it.

## What it will usually tell you

That there is no edge. Run it on almost any liquid large-cap at a 5-day
horizon and it will report no significant skill over predicting zero. That is
the correct answer, not a failure of the code — daily equity returns are close
to unpredictable from price history alone, and a tool that told you otherwise
would be lying.

The value here is a harness that can tell the difference, so that when you add
a genuinely informative feature you will know.

## Install

```bash
pip install -r requirements.txt
```

Python 3.11+. Core dependencies are numpy, pandas, scikit-learn and scipy.
`yfinance` is optional and only needed for the Yahoo provider.

## Quick start

```bash
# Offline sanity check on synthetic data — no network needed.
# A correct pipeline reports ~no skill here.
python -m stockforecast demo

# Evaluate before you trust anything.
python -m stockforecast evaluate AAPL MSFT NVDA --provider yfinance --start 2015-01-01

# Forecast, bundled with the evaluation.
python -m stockforecast forecast AAPL --provider yfinance --start 2015-01-01 --horizon 5

# Screen a universe, with multiple-testing correction.
python -m stockforecast screen --symbols-file universe.txt --provider yfinance \
    --start 2015-01-01 --model gbm --json-out screen.json
```

Options that matter: `--model {zero,mean,momentum,ridge,elasticnet,gbm,rf}`,
`--horizon N`, `--splits N`, `--min-train N`, `--alpha` (0.2 → 80% intervals),
and `--preset {default,short}`. The default feature set needs ~200 bars of
warm-up; `--preset short` trades features for a shorter warm-up when a ticker
has under a year of history.

## Data sources

| Provider | Use | Notes |
|---|---|---|
| `csv` | default | `data/<SYMBOL>.csv`, common header spellings normalised |
| `yfinance` | live | needs outbound access to Yahoo |
| `mcp-json` | Robinhood | saved `get_equity_historicals` payloads |
| `synthetic` | tests | simulated, never for real claims |

To use Robinhood MCP data, have an agent session save the tool response and
convert it:

```bash
python scripts/mcp_to_csv.py data/mcp/*.json --out data/
```

`data/HOOD.csv` is a real sample (125 sessions, Feb–Aug 2026) pulled from the
Robinhood MCP tools.

## Worked example: HOOD

```bash
python -m stockforecast forecast HOOD --preset short --splits 3 --min-train 55
```

```
  >>> THIS MODEL HAS NO MEASURED OUT-OF-SAMPLE SKILL FOR THIS TICKER. <<<

  expected return    -6.12% over 5 days
  80% interval      [-33.10%, +20.47%]
  implied price      87.57   range 66.99 to 114.47

    RMSE skill vs baseline   -98.86%  (DM p=0.011)
    directional accuracy     33.3%  95% CI [19.8%, 50.4%]
    out-of-sample sample     33 predictions over 3 folds
```

Read that as: the model is *worse* than assuming zero, on 33 out-of-sample
predictions, from 99 training rows. The −6.12% is noise. This is what the tool
is supposed to do with insufficient data — the sample is committed precisely
because it is a realistic failure case, not a flattering one.

## Design decisions worth knowing

**Causality is tested, not asserted.** `tests/test_features.py` truncates the
price history and verifies that already-computed feature rows do not change.
Any feature that peeks at the future fails this immediately.

**The leak canary.** A test that only checks "no skill found" passes trivially
if the pipeline is broken and predicts nothing. So `test_leak_canary_is_caught`
deliberately injects the answer into the input and asserts that the metrics
light up. If that test ever fails, the no-skill assertions have stopped meaning
anything.

**Purge and embargo.** With a 5-day target, the last 4 training rows overlap
the test window. They are dropped, plus an embargo, before every fold.

**`beats_baseline` requires significance.** A lower RMSE is not enough; the
Diebold–Mariano test must also clear p < 0.05. Small improvements on small
samples are noise more often than not.

**Screening applies FDR control.** Testing 100 tickers at p < 0.05 yields ~5
false positives from pure noise. The screen reports Benjamini–Hochberg
q-values, and the summary says how many survive.

**Intervals are wide on purpose.** An 80% interval on a 5-day return is
roughly ±10% for a volatile name. Narrowing it would misrepresent the
uncertainty.

## Limitations

- **Price history only.** No fundamentals, earnings dates, options flow,
  short interest, analyst revisions, or news. Real edges usually live there.
- **No transaction costs.** Statistical skill is not profit. Spreads,
  slippage, borrow, taxes and capacity are not modelled.
- **Survivorship bias.** Screening today's tickers ignores companies that
  delisted.
- **Conformal coverage assumes exchangeable residuals.** Volatility clusters,
  so realised coverage drops in turbulent regimes. The evaluation reports
  realised coverage and warns when it falls materially short of nominal.
- **Not investment advice.** This is a measurement instrument. It is at its
  most useful when it tells you a signal you liked does not survive contact
  with a clean backtest.

## Layout

```
stockforecast/
  data.py         providers, OHLCV validation
  features.py     causal feature engineering
  validation.py   walk-forward splits with purge/embargo
  metrics.py      skill, significance, calibration
  models.py       baselines + regularised learners
  intervals.py    split-conformal prediction intervals
  pipeline.py     evaluate / forecast / screen
  report.py       text rendering
  cli.py          command line
scripts/mcp_to_csv.py

autotrade/        rule-driven order prep (see AUTOTRADE.md)
  config.py       TOML config, fail-closed validation
  rules.py        user-specified buy conditions
  guards.py       risk limits that block, never resize
  intents.py      inert order proposals
  ledger.py       append-only audit log + paper broker
  engine.py       plan / paper-fill / emit for review
  cli.py          command line

tests/            130 tests
```

## Tests

```bash
python -m pytest tests/ -q
```
