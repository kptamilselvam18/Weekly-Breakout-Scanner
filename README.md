# ⚡ India Breakout Scanner (NSE)

An 8-filter momentum screener for Indian equities (NSE), built on **weekly
candles** aggregated from daily OHLCV data via [yfinance](https://pypi.org/project/yfinance/).

Originally prototyped in Google Colab; this repo is the same screening
logic packaged as a standalone command-line script you can run locally or
in CI (e.g. a scheduled GitHub Action).

> ⚠️ For educational and research purposes only. **Not financial advice.**

## What it checks

A stock must pass **all 8** checks to appear in the results:

| # | Check | Rule |
|---|-------|------|
| 1 | Consolidation | Price range over the last 10 weeks ≤ 12% |
| 2 | Breakout | Weekly close ≥ 2% above the consolidation high |
| 3 | Body size | Breakout candle body ≥ 5% of open |
| 4 | Relative volume | Weekly volume ≥ 1.5× the 10-week average |
| 5 | Liquidity | 20-day avg daily volume ≥ 500,000 shares |
| 6 | Market cap | ≥ ₹2,000 Crore |
| 7 | Near high | Within 10% of the 20d or 50d high |
| 8 | Trend | Close above both the 20d and 50d SMA |

## Setup

```bash
git clone <this-repo-url>
cd india-breakout-scanner
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `X | None` type hints).

## Usage

```bash
# Scan the entire NSE equity list (slow — can take ~15-20 min due to rate limits)
python scanner.py

# Quick test: only scan the first 200 tickers
python scanner.py --limit 200

# Scan specific tickers only
python scanner.py --tickers RELIANCE INFY TCS HDFCBANK

# Deep-dive a single ticker and print every check's result
python scanner.py --deep-dive RELIANCE.NS

# Skip chart generation for a faster / CI-friendly run
python scanner.py --no-charts

# Cap the number of charts generated (default: 10, sorted by relative volume)
python scanner.py --chart-limit 5
```

## Output

Results are written to `./output/`:

- `india_breakouts_<timestamp>.csv` — full results table with all metrics
- `charts/<TICKER>_breakout.png` — one weekly candlestick chart per breakout (unless `--no-charts`)

## Screening parameters

All thresholds live as constants at the top of `scanner.py` — edit them
directly to tune the screen:

```python
CONSOLIDATION_LOOKBACK = 10
MAX_RANGE_PCT          = 12.0
MIN_BREAKOUT_ABOVE_PCT = 2.0
MIN_BODY_SIZE_PCT      = 5.0
MIN_REL_VOL            = 1.5
MIN_DAILY_AVG_VOL      = 500_000
MIN_MARKET_CAP_CR      = 2000
NEAR_HIGH_PCT          = 10.0
```

## Notes on the GitHub port

This is a direct port of the original Colab notebook, with only the
execution environment changed:

- Uses the `Agg` (non-interactive) matplotlib backend instead of inline
  notebook rendering, so it runs headless in a terminal or CI.
- `display(df)` / notebook-only calls replaced with `print(df.to_string())`.
- Wrapped in an `argparse` CLI (`--tickers`, `--limit`, `--deep-dive`,
  `--no-charts`) instead of hardcoded notebook cell variables.
- Charts and CSVs are written to `./output/` instead of the Colab
  ephemeral filesystem.

The screening logic, thresholds, and objective are unchanged from the
original notebook.

## Disclaimer

This tool is provided for educational and research purposes only. It
does not constitute financial advice. Always do your own due diligence
before making investment decisions.
