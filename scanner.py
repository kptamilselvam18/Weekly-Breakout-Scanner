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
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless CI/servers
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

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

# Networking behavior tuned for shared CI IPs (GitHub Actions), where Yahoo
# Finance is more prone to blocking/500s than on Colab's Google IPs.
BATCH_SIZE = 150               # tickers per yf.download() batch call
REQUEST_RETRIES = 3            # retries for both batch downloads and fast_info lookups
RETRY_SLEEP_SECONDS = 2.0      # backoff between retries
BATCH_PAUSE_SECONDS = 1.0      # pause between batches to be gentle on the API

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
IST = ZoneInfo("Asia/Kolkata")


# ── Watchlist / market hours helpers ───────────────────────────────────
def load_watchlist(path: str) -> list[str]:
    """Load tickers from a plain-text file, one symbol per line (# comments allowed)."""
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbol = line.split()[0].upper()
            tickers.append(symbol if symbol.endswith(".NS") else f"{symbol}.NS")
    return tickers


def is_market_open(now: datetime | None = None) -> bool:
    """True if within NSE cash market hours (Mon-Fri, 09:15-15:30 IST)."""
    now = now.astimezone(IST) if now else datetime.now(IST)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)


def _telegram_send(text: str) -> bool:
    """Low-level Telegram sender. Returns True on success, False otherwise (never raises)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping Telegram notification.")
        return False

    try:
        import requests
    except ImportError:
        print("⚠️  'requests' not installed — skipping Telegram notification (pip install requests).")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"⚠️  Telegram notify failed: {resp.status_code} {resp.text}")
            return False
        print("✅ Telegram notification sent.")
        return True
    except Exception as e:
        print(f"⚠️  Telegram notify error: {e}")
        return False


def send_telegram(results: list[dict], failed: list[str] | None = None, total: int | None = None) -> None:
    """Post a scan summary to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set.

    Produces a different message depending on outcome:
      - Total failure (every ticker failed to fetch) -> failure alert
      - No breakouts found (scan ran fine, just nothing matched) -> "no breakouts"
      - Breakouts found -> list of results

    Silently no-ops if either env var is missing, so this is safe to call
    unconditionally (e.g. from CI where secrets may not be configured).
    """
    failed = failed or []
    now_ist = datetime.now(IST).strftime("%d-%b-%Y %I:%M %p IST")

    if total is not None and total > 0 and len(failed) == total:
        text = (
            f"🔴 *India Breakout Scanner* — scan FAILED\n\n"
            f"All {total} ticker(s) failed to fetch data (Yahoo Finance "
            f"errors/rate-limit likely). No screening was performed.\n"
            f"{now_ist}"
        )
    elif not results:
        text = f"⚡ India Breakout Scanner: no breakouts found this run.\n{now_ist}"
        if failed:
            text += f"\n⚠️ {len(failed)} ticker(s) failed to fetch."
    else:
        lines = [f"⚡ *India Breakout Scanner* — {len(results)} breakout(s) found\n"]
        for r in results[:20]:  # keep messages short
            lines.append(
                f"• {r['Ticker'].replace('.NS', '')}: ₹{r['Price']:.2f} "
                f"(+{r['Breakout%']:.1f}%, RelVol {r['RelVol']:.1f}x)"
            )
        if len(results) > 20:
            lines.append(f"... and {len(results) - 20} more (see CSV in Actions artifacts).")
        if failed:
            lines.append(f"\n⚠️ {len(failed)} ticker(s) failed to fetch.")
        lines.append(f"\n{now_ist}")
        text = "\n".join(lines)

    _telegram_send(text)


def send_telegram_skip(reason: str) -> None:
    """Notify that a scheduled run was skipped without scanning (e.g. market closed)."""
    now_ist = datetime.now(IST).strftime("%d-%b-%Y %I:%M %p IST")
    text = f"🕒 India Breakout Scanner — run skipped\n{reason}\n{now_ist}"
    _telegram_send(text)


def send_test_ping() -> bool:
    """Send a simple 'scanner is alive' message, regardless of scan results.

    Use `--notify-test` to trigger this without waiting for a real breakout —
    useful for confirming TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are wired up
    correctly end-to-end (locally or in GitHub Actions).
    """
    now_ist = datetime.now(IST).strftime("%d-%b-%Y %I:%M %p IST")
    text = f"🟢 India Breakout Scanner — test ping\nScanner pipeline is alive.\n{now_ist}"
    return _telegram_send(text)


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


# ── Data fetching (batched, with retries) ──────────────────────────────
#
# GitHub Actions runs from shared IPs that Yahoo Finance rate-limits/blocks
# far more aggressively than Colab's Google IPs. Two changes address this:
#   1. `obj.info` (a slow, heavy, easily-blocked endpoint) is replaced with
#      `obj.fast_info`, which is lighter and much less likely to 500.
#   2. Daily history is fetched in batches via `yf.download(...)` instead of
#      one `yf.Ticker(t).history()` call per stock — this turns ~2400
#      requests into ~16 batch requests.
def download_batch_history(tickers_batch: list[str], retries: int = REQUEST_RETRIES) -> pd.DataFrame | None:
    """Download 1y daily OHLCV for a batch of tickers in one call, with retries."""
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=tickers_batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                return data
        except Exception as e:
            print(f"  ⚠️  Batch download attempt {attempt}/{retries} failed: {e}")
        time.sleep(RETRY_SLEEP_SECONDS)
    return None


def extract_ticker_history(batch_data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker's OHLCV frame out of a multi-ticker batch download result."""
    try:
        if isinstance(batch_data.columns, pd.MultiIndex):
            hist = batch_data[ticker].dropna(how="all")
        else:
            # Only happens if the batch itself contained a single ticker
            hist = batch_data.dropna(how="all")
        if hist is None or hist.empty or len(hist) < 60:
            return None
        return hist
    except (KeyError, Exception):
        return None


def fetch_market_cap(ticker: str, retries: int = REQUEST_RETRIES) -> float:
    """Fetch market cap via the lightweight fast_info endpoint, with retries."""
    for attempt in range(1, retries + 1):
        try:
            fi = yf.Ticker(ticker).fast_info
            for key in ("market_cap", "marketCap"):
                try:
                    val = fi[key]
                    if val:
                        return float(val)
                except Exception:
                    continue
            val = getattr(fi, "market_cap", None)
            if val:
                return float(val)
            return 0.0
        except Exception:
            time.sleep(RETRY_SLEEP_SECONDS)
    return 0.0


def fetch_ticker_data(ticker: str, retries: int = REQUEST_RETRIES):
    """Single-ticker fetch (history + market cap) with retries. Used for --deep-dive only —
    the main scan path uses the batched functions above for efficiency."""
    hist = None
    for attempt in range(1, retries + 1):
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
            if hist is not None and len(hist) >= 60:
                break
            hist = None
        except Exception as e:
            print(f"  ⚠️  History fetch attempt {attempt}/{retries} for {ticker} failed: {e}")
        time.sleep(RETRY_SLEEP_SECONDS)

    if hist is None:
        return None, None

    market_cap = fetch_market_cap(ticker, retries=retries)
    return hist, {"marketCap": market_cap}


# ── 8-Check screening function ─────────────────────────────────────────
def screen_ticker_core(ticker: str, hist: pd.DataFrame, market_cap: float) -> dict | None:
    """Run all 8 checks given already-fetched history and market cap.

    Pure computation, no network calls — this is what both the batched
    scan path and the deep-dive path funnel into.
    """
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


def screen_ticker(ticker: str) -> dict | None:
    """Fetch + screen a single ticker end-to-end. Used by --deep-dive.

    The main multi-ticker scan does NOT use this — it uses the batched
    download path in run_scan() for efficiency, then calls
    screen_ticker_core() directly per ticker.
    """
    hist, info = fetch_ticker_data(ticker)
    if hist is None:
        return None
    market_cap = (info or {}).get("marketCap", 0) or 0
    return screen_ticker_core(ticker, hist, market_cap)


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
def run_scan(tickers: list[str], make_charts: bool = True, chart_limit: int = 10) -> tuple[list[dict], list[str]]:
    results, failed = [], []

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    print(
        f"🔍 Scanning {len(tickers)} Indian tickers (Weekly | ₹{MIN_MARKET_CAP_CR}Cr+) "
        f"in {len(batches)} batch(es) of up to {BATCH_SIZE}..."
    )

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n📦 Batch {batch_num}/{len(batches)} ({len(batch)} tickers)")
        batch_data = download_batch_history(batch)

        if batch_data is None:
            print(f"  ✗  Batch download failed after retries — skipping {len(batch)} tickers")
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                hist = extract_ticker_history(batch_data, ticker)
                if hist is None:
                    print(f"  ○  {ticker} (no usable data)")
                    continue

                market_cap = fetch_market_cap(ticker)
                result = screen_ticker_core(ticker, hist, market_cap)
                if result:
                    results.append(result)
                    print(f"  ✅ {ticker:<20} → Breakout {result['Breakout%']:+.1f}%  RelVol {result['RelVol']:.1f}×")
                else:
                    print(f"  ○  {ticker}")
            except Exception as e:
                failed.append(ticker)
                print(f"  ✗  {ticker} — {e}")

        if batch_num < len(batches):
            time.sleep(BATCH_PAUSE_SECONDS)

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

    return results, failed


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
        "--watchlist", type=str, default=None,
        help="Path to a text file of NSE symbols (one per line, # comments allowed). "
             "Recommended for frequent (e.g. every-15-min) runs since a full NSE scan is slow.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only scan the first N tickers from the full NSE list (useful for a quick test run).",
    )
    p.add_argument(
        "--market-hours-only", action="store_true",
        help="Exit immediately (before any network calls) if NSE cash market is currently closed "
             "(Mon-Fri 09:15-15:30 IST). Useful for scheduled jobs that also trigger off-hours.",
    )
    p.add_argument(
        "--notify", action="store_true",
        help="Send a summary to Telegram after the scan. Requires TELEGRAM_BOT_TOKEN and "
             "TELEGRAM_CHAT_ID environment variables (e.g. GitHub Actions secrets).",
    )
    p.add_argument(
        "--notify-test", action="store_true",
        help="Send a simple 'scanner is alive' ping to Telegram and exit immediately — "
             "no scanning, no network calls to Yahoo Finance. Use this to verify "
             "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are configured correctly.",
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

    if args.notify_test:
        ok = send_test_ping()
        sys.exit(0 if ok else 1)

    if args.market_hours_only and not is_market_open():
        msg = "NSE market is currently closed (Mon-Fri 09:15-15:30 IST)."
        print(f"🕒 {msg} Skipping this run.")
        if args.notify:
            send_telegram_skip(msg)
        return

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

    # Build ticker universe (priority: --tickers > --watchlist > full NSE list)
    if args.tickers:
        tickers = [t if t.upper().endswith(".NS") else f"{t.upper()}.NS" for t in args.tickers]
    elif args.watchlist:
        tickers = load_watchlist(args.watchlist)
        print(f"📋 {len(tickers)} tickers loaded from watchlist: {args.watchlist}")
    else:
        tickers = load_nse_tickers()
        print(f"📋 {len(tickers)} tickers loaded")
        print(f"📊 Market cap filter: ≥ ₹{MIN_MARKET_CAP_CR} Cr")
        if args.limit:
            tickers = tickers[: args.limit]

    if not tickers:
        msg = "No tickers to scan. Check your network connection, --tickers, or --watchlist argument."
        print(msg, file=sys.stderr)
        if args.notify:
            send_telegram_skip(msg)
        sys.exit(1)

    try:
        results, failed = run_scan(tickers, make_charts=not args.no_charts, chart_limit=args.chart_limit)
    except Exception as e:
        print(f"✗ Scan crashed: {e}", file=sys.stderr)
        if args.notify:
            send_telegram(results=[], failed=tickers, total=len(tickers))
        raise

    export_results(results)

    if args.notify:
        send_telegram(results, failed=failed, total=len(tickers))


if __name__ == "__main__":
    main()
