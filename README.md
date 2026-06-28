# GWP2 · Deep Learning for Finance — Validation Design & Leakage

**MScFE 642 · Group Work Project 2**

Interactive single-asset validation design dashboard exploring information leakage,
walk-forward backtesting, and purge+embargo mitigation.
Runs entirely in the browser — no backend required.
Live price data auto-updated hourly via GitHub Actions.

---

## Live Demo

After setup your dashboard is live at:
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

---

## What This Does

```
GitHub Actions (every hour)
  └─ scripts/fetch_prices.py
        └─ yfinance: downloads BTC-USD ETH-USD SPY AAPL GLD TLT TSLA
        └─ saves  data/latest_prices.csv       ← up to 2000 obs per ticker
        └─ saves  data/latest_prices_meta.json ← timestamp + defaults
              ↓
index.html (browser — no server)
  └─ fetches CSV + meta from raw.githubusercontent.com
  └─ shows ✅ Live · Jun 28 2026, 10:00 UTC
  └─ if fetch fails → synthetic fallback data (seeded per ticker)
```

---

## Setup (5 minutes)

### 1 · Fork / clone
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2 · Fill in your repo details in `index.html`
Open `index.html` and find near the top of the `<script>` block:
```js
const GITHUB_USER = 'YOUR_GITHUB_USERNAME';  // ← your GitHub username
const GITHUB_REPO = 'YOUR_REPO_NAME';         // ← your repo name
```

### 3 · Seed data locally (one-time)
```bash
pip install yfinance pandas
python scripts/fetch_prices.py
git add data/
git commit -m "data: initial prices"
git push
```

### 4 · Enable GitHub Pages
Repo → **Settings → Pages → Source: Deploy from branch → main / root → Save**

### 5 · Trigger first GitHub Actions run
Repo → **Actions → Fetch Prices (GWP2) → Run workflow**

---

## Repo Structure

```
your-repo/
├── index.html                    ← Full dashboard (HTML + CSS + JS)
├── requirements.txt              ← Python deps for GitHub Actions
├── .gitignore
├── README.md
├── scripts/
│   └── fetch_prices.py          ← Multi-ticker price downloader
├── .github/
│   └── workflows/
│       └── fetch_prices.yml     ← Hourly update schedule
└── data/
    ├── latest_prices.csv        ← Auto-updated hourly
    └── latest_prices_meta.json  ← Metadata + defaults
```

---

## Interactive Features

| Control | Default | Description |
|---------|---------|-------------|
| Ticker  | BTC-USD | Any of 7 assets |
| Date range | 2019–2024 | Editable start/end |
| Max obs | 2000 | Slider 500–2000 |
| Label horizon k | 5 | Forward-return days |
| Step 1 train split | 80% | Train/test ratio |
| Step 2a train/test | 500/500 | Walk-forward windows |
| Step 2b train/test | 500/100 | Walk-forward windows |
| Step 3 embargo | 10 | Purge gap length |
| Step 3b train/test | 500/500 | Purged WF windows |
| Step 3c train/test | 500/100 | Purged WF windows |

All charts and discussion questions auto-update when any control changes.

---

## Tickers Available

| Symbol | Asset | Notes |
|--------|-------|-------|
| BTC-USD | Bitcoin | High vol crypto |
| ETH-USD | Ethereum | High vol crypto |
| SPY | S&P 500 ETF | Low vol equity |
| AAPL | Apple Inc | Mid vol equity |
| GLD | Gold ETF | Low vol commodity |
| TLT | 20+ Year Treasuries | Low vol bonds |
| TSLA | Tesla Inc | High vol equity |

---

## Data Banner States

| Banner | Meaning |
|--------|---------|
| ✅ Live · timestamp | GitHub CSV loaded successfully |
| ⚠ Fallback data | GitHub unreachable — synthetic data used |
| ❌ Fetch failed | Config error |

Synthetic fallback is seeded per ticker and per date range so results are
consistent and meaningful even without a GitHub connection.

---

## Notes

- Works offline with synthetic fallback data
- No API keys required for data (yfinance uses Yahoo Finance)
- Free hosting via GitHub Pages
- All computation is in the browser (JavaScript simulation)
- AI Discussion tab requires an Anthropic API key (paste in the field)
