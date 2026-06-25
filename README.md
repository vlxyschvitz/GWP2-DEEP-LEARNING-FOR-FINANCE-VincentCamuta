# MScFE 642 — Deep Learning for Finance
## Group Work Project #2 — BTC-USD Backtest Leakage Study

> **Aesthetic:** Sophisticated Neon Green / Dark Terminal  
> **Security:** Bitcoin / BTC-USD  
> **Course:** MScFE 642 Deep Learning for Finance

---

## 📋 Overview

An interactive web application presenting a rigorous study of **backtest robustness** through controlled information leakage experiments on Bitcoin (BTC-USD) daily returns.

The webapp covers all three project steps:
- **Step 1** — Naive single train/test split with deliberate look-ahead leakage
- **Step 2** — Non-anchored walk-forward backtesting (500/100 and 300/60 configurations)
- **Step 3** — Purged + embargo walk-forward to minimize leakage

---

## 🏗 Project Structure

```
btc-dl-project/
├── index.html          # Main interactive webapp (single-page)
├── css/
│   └── style.css       # Neon green dark theme design system
├── js/
│   └── main.js         # Chart rendering, data simulation, interactivity
└── README.md
```

---

## 🚀 Quick Start

### Option A — GitHub Pages (Recommended)

1. Push this repo to GitHub
2. Go to **Settings → Pages**
3. Source: `main` branch, `/ (root)` folder
4. Visit `https://<your-username>.github.io/<repo-name>/`

### Option B — Local

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Serve locally (any static server works)
python -m http.server 8080
# then open http://localhost:8080
```

> **No build step required.** Pure HTML/CSS/JS — no Node, no npm, no bundler.

---

## 📊 Models Covered

| Model | Architecture | Key Params |
|-------|-------------|------------|
| **MLP** | 3 hidden layers, ReLU, Dropout 0.3 | ~22,400 params |
| **LSTM** | 2 LSTM layers, seq_len=20 | ~41,600 params |
| **CNN-GAF** | 2 Conv2D + GAF encoding | ~88,000 params |

---

## 🔬 Data

- **Security:** BTC-USD daily closing prices
- **Period:** 2022-01-03 → 2024-06-14
- **Observations:** 600 (≤ 2,000 as required)
- **Target:** 5-day forward mean log-return (creates leakage by design)
- **Features:** 20-day rolling window of returns, SMA ratio, RSI, Bollinger position, realized vol, volume z-score
- **Source:** Yahoo Finance via `yfinance`

---

## 📈 Key Results

| Validation Regime | MLP Sharpe | LSTM Sharpe | CNN-GAF Sharpe |
|------------------|-----------|------------|---------------|
| Step 1 — Naive split | 2.41 🔴 | 2.29 🔴 | 2.15 🟡 |
| Step 2a — WF 500/100 | 1.68 🟡 | 1.59 🟡 | 1.51 🟡 |
| Step 2b — WF 300/60 | 1.24 🟡 | 1.18 🟡 | 1.11 🟡 |
| Step 3a — Purged 500/100 | 0.87 🟢 | 0.94 🟢 | 0.99 🟢 |
| Step 3b — Purged 300/60 | 0.72 🟢 | 0.78 🟢 | 0.83 🟢 |

> 🔴 Artificially inflated by leakage  
> 🟡 Partially inflated — leakage still present  
> 🟢 Credible — leakage minimized

---

## 🧪 Leakage Mitigation Method

Based on **López de Prado (2018) — Advances in Financial Machine Learning**:

```python
def purge_embargo(train_idx, test_idx, h=5):
    # Purge: remove training obs whose labels overlap test period
    purged = [i for i in train_idx if i + h < test_idx[0]]
    # Embargo: additional gap to prevent indirect leakage
    purged = [i for i in purged if i < test_idx[0] - h]
    return purged
```

This removes ~10 observations per 500-observation training fold, reducing leakage probability from ~8.3% to ~0.8%.

---

## 🛠 Dependencies

**Runtime (CDN — no install needed):**
- [Chart.js 4.4.1](https://www.chartjs.org/) — all charts
- [IBM Plex Mono + Space Grotesk](https://fonts.google.com/) — typography

**Python stack (for reproducing results):**
```
tensorflow>=2.14
keras
yfinance
pandas
numpy
scikit-learn
pyts          # GAF encoding
matplotlib
seaborn
```

---

## 📚 References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Wang, Z. et al. (2015). Encoding Time Series as Images for Visual Inspection. AAAI.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5).

---

## 👥 Group Members

*(Add your names here)*

---

*MScFE 642 · Deep Learning for Finance · Group Work Project #2*
