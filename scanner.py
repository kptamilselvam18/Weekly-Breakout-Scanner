#!/usr/bin/env python3
"""
India Breakout Scanner (NSE) — Weekly Timeframe
=================================================

8-filter momentum screener for Indian equities (NSE), built from daily
OHLCV data aggregated into weekly candles.

Checks applied (a ticker must pass ALL of them):
  1. Consolidation   -> price range over lookback window <= MAX_RANGE_PCT
  2. Breakout         -> close >= consolidation high * (1 + MIN_BREAKOUT_ABOVE_PCT)
  3. Candle body size -> |close - open| / open >= MIN_BODY_SIZE_PCT
  4. Relative volume  -> current week volume >= MIN_REL_VOL x lookback avg
  5. Liquidity        -> 20d avg daily volume >= MIN_DAILY_AVG_VOL
  6. Market cap       -> >= MIN_MARKET_CAP_CR (in INR Crore)
  7. Near high        -> within NEAR_HIGH_PCT of the 20d or 50d high
  8. Trend            -> close above both the 20d and 50d SMA

This is a command-line port of the original Google Colab notebook. Same
screening logic and objective, restructured so it runs standalone from a
terminal / CI job instead of a notebook cell.

Usage:
    python scanner.py                          # scan all NSE stocks
    python scanner.py --limit 200               # scan only first 200 tickers (quick test)
    python scanner.py --tickers RELIANCE INFY   # scan specific tickers only
    python scanner.py --deep-dive RELIANCE.NS   # single-ticker deep dive + chart
    python scanner.py --no-charts               # skip PNG chart generation (faster)

Output:
    ./output/india_breakouts_<timestamp>.csv    (results table)
    ./output/charts/<TICKER>_breakout.png        (one chart per breakout, if enabled)

⚠️ For educational and research purposes only. Not financial advice.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless CI/servers
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
import requests

warnings.filterwarnings("ignore")

# ── Screening parameters ───────────────────────────────────────────────────
CONSOLIDATION_LOOKBACK = 10
MAX_RANGE_PCT = 12.0
MIN_BREAKOUT_ABOVE_PCT = 2.0
MIN_BODY_SIZE_PCT = 5.0
MIN_REL_VOL = 1.5
MIN_DAILY_AVG_VOL = 500_000
MIN_MARKET_CAP_CR = 2000
MIN_MARKET_CAP_INR = MIN_MARKET_CAP_CR * 1e7
NEAR_HIGH_PCT = 10.0

NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

def send_telegram(results):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram secrets not found.")
        return

    if results:
        message = "🚨 Weekly Breakout Scanner\n\n"
        message += f"✅ {len(results)} breakout(s) found\n\n"

        for i, r in enumerate(results, 1):
            message += (
                f"{i}. {r['Ticker']}\n"
                f"💰 Price: ₹{r['Price']}\n"
                f"📈 Breakout: {r['Breakout%']}%\n"
                f"📊 RelVol: {r['RelVol']}x\n\n"
            )
    else:
        message = (
            "🔇 Weekly Breakout Scanner\n\n"
            "No breakout stocks found.\n\n"
            f"Scan Time: {datetime.now().strftime('%d-%b-%Y %I:%M %p')}"
        )

    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={
            "chat_id": chat_id,
            "text": message
        },
        timeout=20
    )


# ── Ticker list ──────────────────────────────────────────────────────────
def load_nse_tickers() -> list[str]:
    """Fetch the full list of NSE-listed equity tickers as Yahoo Finance symbols."""
    try:
        df = pd.read_csv(NSE_URL)
        symbols = df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
        return [f"{s}.NS" for s in symbols]
    except Exception as e:
        print(f"⚠️  Error loading NSE stock list: {e}", file=sys.stderr)
        return []


# ── Data helpers ────────────────────────────────────────────────────────
def build_weekly_candles(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV into weekly (Mon-Fri) candles."""
    if len(daily_df) < 10:
        return pd.DataFrame()
    daily_df = daily_df.copy()
    daily_df.index = pd.to_datetime(daily_df.index)
    weekly = daily_df.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()
    return weekly


def sma(series: pd.Series, period: int) -> float | None:
    """Simple moving average of the last `period` values."""
    if len(series) < period:
        return None
    return series.iloc[-period:].mean()


def fetch_ticker_data(ticker: str):
    """Fetch 1yr daily data + info for a ticker."""
    try:
        obj = yf.Ticker(ticker)
        hist = obj.history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 60:
            return None, None
        info = obj.info or {}
        return hist, info
    except Exception:
        return None, None


# ── 8-Check screening function ─────────────────────────────────────────
def screen_ticker(ticker: str) -> dict | None:
    """Run all 8 checks on a ticker. Returns a result dict if it passes, else None."""
    hist, info = fetch_ticker_data(ticker)
    if hist is None:
        return None

    weekly = build_weekly_candles(hist)
    if len(weekly) < CONSOLIDATION_LOOKBACK + 2:
        return None

    current = weekly.iloc[-1]
    lookback = weekly.iloc[-(CONSOLIDATION_LOOKBACK + 1):-1]

    daily_close = hist["Close"].dropna()
    daily_vol = hist["Volume"].dropna()

    # 1. Consolidation
    cons_high = max(max(r.Open, r.Close) for _, r in lookback.iterrows())
    cons_low = min(min(r.Open, r.Close) for _, r in lookback.iterrows())
    range_pct = (cons_high - cons_low) / cons_low * 100
    if range_pct > MAX_RANGE_PCT:
        return None

    # 2. Breakout
    if current.Close < cons_high * (1 + MIN_BREAKOUT_ABOVE_PCT / 100):
        return None

    # 3. Candle body size
    body_size = abs(current.Close - current.Open) / current.Open * 100
    if body_size < MIN_BODY_SIZE_PCT:
        return None

    # 4. Relative volume
    avg_vol = lookback["Volume"].mean()
    rel_vol = current.Volume / avg_vol if avg_vol > 0 else 0
    if rel_vol < MIN_REL_VOL:
        return None

    # 5. Liquidity
    avg_daily_vol = daily_vol.iloc[-20:].mean() if len(daily_vol) >= 20 else daily_vol.mean()
    if avg_daily_vol < MIN_DAILY_AVG_VOL:
        return None

    # 6. Market cap
    market_cap = info.get("marketCap", 0) or 0
    if market_cap < MIN_MARKET_CAP_INR:
        return None

    # 7. Near high
    high20 = daily_close.iloc[-20:].max() if len(daily_close) >= 20 else daily_close.max()
    high50 = daily_close.iloc[-50:].max() if len(daily_close) >= 50 else daily_close.max()
    pct_from_20 = (high20 - current.Close) / high20 * 100
    pct_from_50 = (high50 - current.Close) / high50 * 100
    if pct_from_20 > NEAR_HIGH_PCT and pct_from_50 > NEAR_HIGH_PCT:
        return None

    # 8. Trend
    sma20 = sma(daily_close, 20)
    sma50 = sma(daily_close, 50)
    if sma20 is None or sma50 is None:
        return None
    if current.Close <= sma20 or current.Close <= sma50:
        return None

    breakout_pct = (current.Close - cons_high) / cons_high * 100
    cap_cr = market_cap / 1e7

    return {
        "Ticker": ticker,
        "Price": round(current.Close, 2),
        "Breakout%": round(breakout_pct, 2),
        "BodySize%": round(body_size, 2),
        "RelVol": round(rel_vol, 2),
        "vs20dHigh%": round(pct_from_20, 2),
        "vs50dHigh%": round(pct_from_50, 2),
        "MktCap_Cr": round(cap_cr, 0),
        "Range%": round(range_pct, 2),
        "ConsHigh": round(cons_high, 2),
        "ConsLow": round(cons_low, 2),
        "SMA20": round(sma20, 2),
        "SMA50": round(sma50, 2),
        "_weekly": weekly,  # for charting only, stripped before export
    }


# ── Charting ────────────────────────────────────────────────────────────
def plot_breakout(result: dict, lookback_weeks: int = 20, out_dir: Path = CHARTS_DIR) -> Path:
    """Plot weekly candlestick with consolidation range for a breakout, save as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker = result["Ticker"]
    weekly = result["_weekly"].tail(lookback_weeks)
    cons_h = result["ConsHigh"]
    cons_l = result["ConsLow"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, facecolor="#0f172a"
    )
    fig.suptitle(f"{ticker} — Weekly Breakout Chart", color="#e2e8f0", fontsize=14, fontweight="bold", y=0.98)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[["top", "right", "left", "bottom"]].set_color("#334155")
        ax.yaxis.label.set_color("#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")

    xs = range(len(weekly))
    dates = weekly.index

    for i, (idx, row) in enumerate(weekly.iterrows()):
        is_last = i == len(weekly) - 1
        is_up = row.Close >= row.Open
        col = "#10b981" if is_last else ("#10b981" if is_up else "#ef4444")
        body_bot = min(row.Open, row.Close)
        body_top = max(row.Open, row.Close)
        lw = 3 if is_last else 1

        ax1.plot([i, i], [row.Low, row.High], color=col, linewidth=0.8)
        ax1.add_patch(
            plt.Rectangle((i - 0.3, body_bot), 0.6, body_top - body_bot, color=col, linewidth=lw, zorder=3)
        )

    ax1.axhline(cons_h, color="#f43f5e", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Cons. High: {cons_h:.2f}")
    ax1.axhline(cons_l, color="#f43f5e", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Cons. Low: {cons_l:.2f}")
    ax1.axhspan(cons_l, cons_h, alpha=0.08, color="#f43f5e")

    close_vals = weekly["Close"].values
    if len(close_vals) >= 10:
        sma10 = pd.Series(close_vals).rolling(10).mean().values
        ax1.plot(xs, sma10, color="#f59e0b", linewidth=1, label="SMA10(weekly)", alpha=0.7)

    ax1.set_ylabel("Price (₹)", color="#94a3b8")
    ax1.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=8, loc="upper left")
    ax1.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.5)

    vols = weekly["Volume"].values
    avg_v = vols[:-1].mean() if len(vols) > 1 else vols.mean()
    for i, (idx, row) in enumerate(weekly.iterrows()):
        is_last = i == len(weekly) - 1
        col = "#10b981" if is_last else ("#475569" if row.Close >= row.Open else "#7f1d1d")
        ax2.bar(i, row.Volume, color=col, width=0.6, alpha=0.9)
    ax2.axhline(avg_v, color="#f59e0b", linestyle="--", linewidth=1, alpha=0.7, label=f"Avg Vol: {avg_v:,.0f}")
    ax2.set_ylabel("Volume", color="#94a3b8")
    ax2.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=8)
    ax2.grid(axis="y", color="#334155", linewidth=0.5, alpha=0.5)

    step = max(1, len(weekly) // 8)
    tick_pos = list(range(0, len(weekly), step))
    tick_labels = [dates[i].strftime("%d %b") for i in tick_pos]
    for ax in (ax1, ax2):
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8, color="#94a3b8")

    stats = (
        f"Price: ₹{result['Price']:.2f}\n"
        f"Breakout: +{result['Breakout%']:.2f}%\n"
        f"Body Size: {result['BodySize%']:.2f}%\n"
        f"Rel Vol: {result['RelVol']:.2f}×\n"
        f"Mkt Cap: ₹{result['MktCap_Cr']:,.0f} Cr"
    )
    ax1.text(
        0.01,
        0.97,
        stats,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="#e2e8f0",
        bbox=dict(boxstyle="round", facecolor="#0f172a", alpha=0.8, edgecolor="#334155"),
    )

    plt.tight_layout()
    out_path = out_dir / f"{ticker.replace('.NS', '')}_breakout.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return out_path


# ── Scan orchestration ──────────────────────────────────────────────────
def run_scan(tickers: list[str], make_charts: bool = True, chart_limit: int = 10) -> list[dict]:
    results, failed = [], []

    print(f"🔍 Scanning {len(tickers)} Indian tickers (Weekly | ₹{MIN_MARKET_CAP_CR}Cr+)...")
    for i, ticker in enumerate(tickers):
        try:
            result = screen_ticker(ticker)
            if result:
                results.append(result)
                print(f"  ✅ {ticker:<20} → Breakout {result['Breakout%']:+.1f}%  RelVol {result['RelVol']:.1f}×")
            else:
                print(f"  ○  {ticker}")
        except Exception as e:
            failed.append(ticker)
            print(f"  ✗  {ticker} — {e}")

        if (i + 1) % 5 == 0:
            time.sleep(0.5)  # be gentle on the API

    results.sort(key=lambda x: x["RelVol"], reverse=True)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ Scan complete: {len(results)} breakout(s) found")
    if failed:
        print(f"⚠️  {len(failed)} tickers failed to fetch")

    if results:
        display_cols = [
            "Ticker", "Price", "Breakout%", "BodySize%", "RelVol",
            "vs20dHigh%", "vs50dHigh%", "MktCap_Cr", "Range%",
        ]
        df = pd.DataFrame(results)[display_cols]
        print("\n⚡ India Breakout Scanner — Weekly Timeframe\n")
        print(df.to_string(index=False))

        if make_charts:
            CHARTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\n📊 Plotting up to {chart_limit} breakout chart(s)...")
            for r in results[:chart_limit]:
                path = plot_breakout(r)
                print(f"  💾 Chart saved: {path}")
    else:
        print("🔇 No breakouts found. Try again in a different market session.")

    return results


def export_results(results: list[dict], out_dir: Path = OUTPUT_DIR) -> Path | None:
    if not results:
        print("No results to export.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "Ticker", "Price", "Breakout%", "BodySize%", "RelVol",
        "vs20dHigh%", "vs50dHigh%", "MktCap_Cr", "Range%",
        "ConsHigh", "ConsLow", "SMA20", "SMA50",
    ]
    df_export = pd.DataFrame(results)[export_cols]
    filename = out_dir / f"india_breakouts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df_export.to_csv(filename, index=False)
    print(f"✅ Results exported to {filename}")
    return filename


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="India Breakout Scanner (NSE) — Weekly 8-filter momentum screener."
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific NSE symbols to scan (without .NS suffix), e.g. --tickers RELIANCE INFY TCS",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only scan the first N tickers from the full NSE list (useful for a quick test run).",
    )
    p.add_argument(
        "--deep-dive", type=str, default=None,
        help="Run a single-ticker deep dive (e.g. --deep-dive RELIANCE.NS) and print all check results.",
    )
    p.add_argument(
        "--no-charts", action="store_true",
        help="Skip PNG chart generation (faster, useful for CI runs).",
    )
    p.add_argument(
        "--chart-limit", type=int, default=10,
        help="Max number of charts to generate, sorted by relative volume (default: 10).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Single-ticker deep dive mode
    if args.deep_dive:
        ticker = args.deep_dive if args.deep_dive.upper().endswith(".NS") else f"{args.deep_dive.upper()}.NS"
        print(f"🔬 Deep dive: {ticker}")
        r = screen_ticker(ticker)
        if r:
            print("✅ PASSES all 8 checks!")
            for k, v in r.items():
                if not k.startswith("_"):
                    print(f"  {k:<20}: {v}")
            if not args.no_charts:
                path = plot_breakout(r)
                print(f"💾 Chart saved: {path}")
        else:
            print(f"✗ {ticker} did not pass all 8 screening checks.")
            print("Tip: Adjust parameters in scanner.py or try during a breakout session.")
        return

    # Build ticker universe
    if args.tickers:
        tickers = [t if t.upper().endswith(".NS") else f"{t.upper()}.NS" for t in args.tickers]
    else:
        tickers = load_nse_tickers()
        print(f"📋 {len(tickers)} tickers loaded")
        print(f"📊 Market cap filter: ≥ ₹{MIN_MARKET_CAP_CR} Cr")
        if args.limit:
            tickers = tickers[: args.limit]

    if not tickers:
        print("No tickers to scan. Check your network connection or --tickers argument.", file=sys.stderr)
        sys.exit(1)

        results = run_scan(
        tickers,
        make_charts=not args.no_charts,
        chart_limit=args.chart_limit,
    )

    export_results(results)

    send_telegram(results)


if __name__ == "__main__":
    main() 


if __name__ == "__main__":
    main()
