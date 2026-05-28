# Market Mood Dashboard

An interactive **Streamlit** dashboard for cross-asset market analysis. It computes a single,
easy-to-read **Market Mood** index (0–100) from a panel of market features and statistical
regimes, then visualizes regimes, volatility clusters, asset-class heatmaps, and cross-asset
correlations.

> Market Mood blends a feature-based score with a regime-based score:
> `Market Mood = 0.70 × Feature Mood + 0.30 × Regime Mood`

---

## Table of contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Project structure](#project-structure)
- [Data inputs](#data-inputs)
- [Installation](#installation)
- [Usage](#usage)
- [How the Market Mood is computed](#how-the-market-mood-is-computed)
- [Dashboard tabs explained](#dashboard-tabs-explained)
- [Configuration & tuning](#configuration--tuning)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)

---

## Features

- **Market Mood index (0–100)** — a single headline sentiment score with intuitive bands
  (bullish / neutral / cautious / bearish).
- **Two regime models** — Gaussian Mixture Model (GMM) and K-Means, viewable side by side
  against BTCUSD price.
- **Asset-class heatmaps** — average daily return or 30-day volatility per asset class over time.
- **Volatility clusters** — per-asset 30-day volatility heatmap to spot market-wide stress.
- **Cross-asset correlation matrix** — full-sample pairwise return correlations, plus a rolling
  30-day average pairwise correlation.
- **In-app explanations** — every chart includes plain-language guidance on how to read it.
- **Batch export** — a standalone script regenerates all enriched datasets and a static chart.

---

## Screenshots

> _Add screenshots here once the app is running, e.g._
>
> ```markdown
> ![Market Mood](docs/market_mood.png)
> ![Regimes](docs/regimes.png)
> ```

---

## Project structure

```
New folder/                       # data root (parent of the app folder)
├── MARKET_1.CSV                   # daily market-level features + regime labels/probabilities
├── metric_engine_long.csv         # long-format per-asset daily metrics
├── REGIME_1.CSV                   # (auxiliary regime data)
├── EURUSD_daily_yahoo.csv         # (auxiliary raw price data)
├── USDINR_daily_yahoo.csv
├── USDJPY_daily_yahoo.csv
│
└── market_dashboard/             # the application
    ├── dashboard.py               # Streamlit app (entry point)
    ├── compute_market_mood.py     # mood/regime computation + batch export
    ├── requirements.txt           # Python dependencies
    ├── README.md                  # this file
    └── outputs/                   # generated CSVs and charts
        ├── market_regimes_with_mood.csv
        ├── market_mood_by_regime.csv
        ├── market_mood_latest.csv
        ├── correlation_matrix.csv
        ├── asset_class_returns_heatmap.csv
        └── asset_class_volatility_heatmap.csv
```

> **Note:** the app reads its input CSVs from the **parent** directory of `market_dashboard/`
> (i.e. `New folder/`). Keep `MARKET_1.CSV` and `metric_engine_long.csv` one level above the app.

---

## Data inputs

### `MARKET_1.CSV` — daily market-level panel

| Column | Description |
| --- | --- |
| `date_utc` | Date (UTC) |
| `mean_return` | Cross-asset mean daily return |
| `return_dispersion` | Spread of returns across assets |
| `avg_vol_30d` | Average 30-day volatility across assets |
| `vol_dispersion` | Spread of volatility across assets |
| `avg_trend` | Average trend signal |
| `pct_uptrend` / `pct_downtrend` | Share of assets trending up / down |
| `crypto_return` / `fx_return` | Asset-class average returns |
| `btc_vol_30d` | BTC 30-day volatility |
| `avg_pairwise_corr_30d` | Rolling 30-day average pairwise correlation |
| `cluster_kmeans`, `regime_kmeans` | K-Means cluster id and mapped regime label |
| `cluster_gmm`, `regime_gmm` | GMM cluster id and mapped regime label |
| `gmm_prob_risk_off`, `gmm_prob_risk_on`, `gmm_prob_high_volatility`, `gmm_prob_calm` | GMM regime probabilities |

### `metric_engine_long.csv` — long-format per-asset metrics

| Column | Description |
| --- | --- |
| `date_utc` | Date (UTC) |
| `asset` | Asset symbol (e.g. `BTCUSD`, `ETHUSD`, `EURUSD`) |
| `close` | Closing price |
| `daily_return` | Daily return |
| `ma_20`, `ma_50` | 20/50-day moving averages |
| `vol_30d` | 30-day volatility |
| `trend_signal` | Trend indicator |
| `asset_class` | `crypto`, `fx`, or `stablecoin` |
| `vol_30d_rank`, `vol_30d_pct` | Volatility rank and percentile |

---

## Installation

Requires **Python 3.10+**.

```bash
cd market_dashboard
python -m pip install -r requirements.txt
```

<details>
<summary>Recommended: use a virtual environment</summary>

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
</details>

---

## Usage

### Run the dashboard

```bash
cd market_dashboard
streamlit run dashboard.py
```

Then open the URL Streamlit prints (default: <http://localhost:8501>).

If `streamlit` is not on your PATH:

```bash
python -m streamlit run dashboard.py
```

### Regenerate the exported datasets (optional)

```bash
python compute_market_mood.py
```

This writes the enriched CSVs and a static mood time-series chart into `outputs/`.

---

## How the Market Mood is computed

Defined in `compute_market_mood.py`.

**1. Feature Mood** — features are z-scored, then combined as
`positive features − negative features`, rescaled to 0–100 and clipped:

- **Positive:** `mean_return`, `avg_trend`, `pct_uptrend`, `crypto_return`, `fx_return`
- **Negative:** `return_dispersion`, `avg_vol_30d`, `vol_dispersion`, `pct_downtrend`,
  `btc_vol_30d`, `avg_pairwise_corr_30d`

```python
raw = z[POSITIVE].mean(axis=1) - z[NEGATIVE].mean(axis=1)
feature_mood = (50 + 15.0 * raw).clip(0, 100)
```

**2. Regime Mood** — a probability-weighted score of the GMM regimes:

| Regime | Score |
| --- | --- |
| `risk_on` | 100 |
| `calm` | 50 |
| `high_volatility` | 25 |
| `risk_off` | 0 |

```python
regime_mood = Σ gmm_prob_<regime> × score(<regime>)
```

> Because `risk_off` scores **0**, the Regime Mood will read **0** on days when the model is
> ~100% confident the market is in the bearish `risk_off` regime. This is expected behavior.

**3. Market Mood** — the blended headline score:

```python
market_mood = 0.70 × feature_mood + 0.30 × regime_mood
```

**Mood labels:**

| Score | Label |
| --- | --- |
| ≥ 65 | bullish |
| 45–65 | neutral |
| 30–45 | cautious |
| < 30 | bearish |

---

## Dashboard tabs explained

- **Market Mood** — Market / Feature / Regime mood over time with threshold guides, plus a table
  of average mood by K-Means regime.
- **Regimes** — Detected regimes shaded behind the BTCUSD price (log scale), switchable between
  GMM and K-Means, plus a days-per-regime bar chart.
- **Asset-Class Heatmaps** — Average daily return or 30-day volatility per asset class over time.
- **Volatility Clusters** — Per-asset 30-day volatility heatmap; market-wide spikes appear as
  vertical dark stripes. Includes a latest-volatility ranking table.
- **Correlations** — Full-sample cross-asset correlation matrix (+1 red / 0 white / −1 blue) plus
  the rolling 30-day average pairwise correlation line.

---

## Configuration & tuning

Key knobs live at the top of `compute_market_mood.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `POSITIVE_FEATURES` / `NEGATIVE_FEATURES` | see above | Which features push mood up vs down |
| `REGIME_SCORES` | risk_on 100 … risk_off 0 | Mood value assigned to each regime |
| `FEATURE_WEIGHT` | `0.70` | Weight of Feature Mood in the blend |
| `REGIME_WEIGHT` | `0.30` | Weight of Regime Mood in the blend |
| `MOOD_SCALE` | `15.0` | Sensitivity of Feature Mood to z-scored features |

For example, to keep Regime Mood off an absolute floor, give `risk_off` a small positive score
(e.g. `10`) in `REGIME_SCORES`.

The dashboard's analysis start date is `2025-02-03` (passed to the heatmap/correlation builders in
`dashboard.py`); adjust it there if you want a different window.

---

## Outputs

Running `python compute_market_mood.py` produces, in `outputs/`:

| File | Contents |
| --- | --- |
| `market_regimes_with_mood.csv` | Full enriched panel with mood columns |
| `market_mood_by_regime.csv` | Mood aggregates (mean/min/max) per K-Means regime |
| `market_mood_latest.csv` | Latest-day mood snapshot |
| `correlation_matrix.csv` | Cross-asset correlation matrix |
| `asset_class_returns_heatmap.csv` | Asset-class daily-return heatmap data |
| `asset_class_volatility_heatmap.csv` | Asset-class volatility heatmap data |
| `market_mood_timeseries.png` | Static mood time-series chart |

---

## Troubleshooting

- **`streamlit` not found** — run `python -m streamlit run dashboard.py`, or reinstall
  requirements.
- **`FileNotFoundError` for `MARKET_1.CSV` / `metric_engine_long.csv`** — make sure these files
  sit in the **parent** folder of `market_dashboard/`, and launch the app from inside
  `market_dashboard/`.
- **A chart looks empty** — refresh hasn't picked up new code; fully stop Streamlit (Ctrl+C) and
  re-run. Heatmaps and the rolling-correlation line are rendered with Matplotlib for reliable
  display inside Streamlit.
- **Port already in use** — `streamlit run dashboard.py --server.port 8502`.
- **Stale data after editing CSVs** — clear the cache from the Streamlit menu (≡ → *Clear cache*)
  or press `C`.

---

## Tech stack

- [Streamlit](https://streamlit.io/) — web UI
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data wrangling
- [Plotly](https://plotly.com/python/) — gauge, line, and bar charts
- [Matplotlib](https://matplotlib.org/) — heatmaps and rolling-correlation chart
- [openpyxl](https://openpyxl.readthedocs.io/) — Excel I/O support
