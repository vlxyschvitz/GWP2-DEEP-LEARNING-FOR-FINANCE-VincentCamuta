#!/usr/bin/env python3
"""
GWP2 · Deep Learning for Finance
Single-asset price fetcher for the interactive dashboard.
Fetches up to 2000 observations of daily close prices for each ticker.
Auto-updated hourly by GitHub Actions.
"""

import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Installing dependencies…")
    os.system("pip install yfinance pandas --quiet")
    import yfinance as yf
    import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "GLD", "TLT", "TSLA"]
START   = "2017-01-01"   # far enough back for crypto
END     = datetime.utcnow().strftime("%Y-%m-%d")
MAX_OBS = 2000

OUT_DIR  = Path(__file__).parent.parent / "data"
OUT_CSV  = OUT_DIR / "latest_prices.csv"
OUT_META = OUT_DIR / "latest_prices_meta.json"

# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_all():
    OUT_DIR.mkdir(exist_ok=True)
    frames = []

    for ticker in TICKERS:
        try:
            print(f"  Downloading {ticker}…", end=" ")
            df = yf.download(ticker, start=START, end=END,
                             auto_adjust=True, progress=False)
            if df.empty:
                print("EMPTY — skip")
                continue

            # If multi-level columns (yfinance ≥ 0.2)
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Close"]
                df.name = ticker
            else:
                df = df["Close"]
                df.name = ticker

            df = df.tail(MAX_OBS)
            frames.append(df)
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not frames:
        print("No data fetched — abort")
        sys.exit(1)

    # Align on common date index
    combined = pd.concat(frames, axis=1)
    combined.index.name = "Date"
    combined.index = combined.index.strftime("%Y-%m-%d")

    # Forward-fill tiny gaps (weekends already excluded by yfinance)
    combined.ffill(inplace=True)
    combined.dropna(how="all", inplace=True)

    combined.to_csv(OUT_CSV)
    print(f"\n✓ Saved {OUT_CSV}  ({len(combined)} rows × {len(combined.columns)} tickers)")

    # Meta
    meta = {
        "last_updated": datetime.utcnow().strftime("%b %d %Y, %H:%M UTC"),
        "tickers": list(combined.columns),
        "rows": len(combined),
        "start": str(combined.index[0]),
        "end":   str(combined.index[-1]),
        "defaults": {
            "start_date": "2019-01-01",
            "end_date":   END,
            "max_obs":    2000,
            "label_horizon": 5,
            "step1_split": 80,
            "step2a_train": 500, "step2a_test": 500,
            "step2b_train": 500, "step2b_test": 100,
            "step3_embargo": 10,
        }
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved {OUT_META}")

if __name__ == "__main__":
    print(f"GWP2 Price Fetcher — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    fetch_all()
