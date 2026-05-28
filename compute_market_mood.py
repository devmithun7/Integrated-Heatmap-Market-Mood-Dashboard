"""
Compute Market Mood scalar index and export enriched datasets + charts.

Reads MARKET_1.CSV and metric_engine_long.csv from the parent data folder,
writes outputs to ./outputs/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_FEATURES = [
    "mean_return",
    "avg_trend",
    "pct_uptrend",
    "crypto_return",
    "fx_return",
]
NEGATIVE_FEATURES = [
    "return_dispersion",
    "avg_vol_30d",
    "vol_dispersion",
    "pct_downtrend",
    "btc_vol_30d",
    "avg_pairwise_corr_30d",
]
REGIME_NAMES = ["calm", "risk_on", "risk_off", "high_volatility"]
REGIME_COLORS = {
    "calm": "#5BA0E0",
    "risk_on": "#3FB36C",
    "risk_off": "#E07A3F",
    "high_volatility": "#C8362F",
}
REGIME_SCORES = {
    "risk_on": 100,
    "calm": 50,
    "high_volatility": 25,
    "risk_off": 0,
}
FEATURE_WEIGHT = 0.70
REGIME_WEIGHT = 0.30
MOOD_SCALE = 15.0


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


def mood_label(score: float) -> str:
    if score >= 65:
        return "bullish"
    if score >= 45:
        return "neutral"
    if score >= 30:
        return "cautious"
    return "bearish"


def compute_market_mood(labelled: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with feature_mood, regime_mood, and market_mood columns."""
    out = labelled.copy()
    features = POSITIVE_FEATURES + NEGATIVE_FEATURES
    z = out[features].apply(_zscore)
    raw = z[POSITIVE_FEATURES].mean(axis=1) - z[NEGATIVE_FEATURES].mean(axis=1)
    out["feature_mood"] = (50 + MOOD_SCALE * raw).clip(0, 100)

    regime_mood = np.zeros(len(out))
    for regime, score in REGIME_SCORES.items():
        col = f"gmm_prob_{regime}"
        if col in out.columns:
            regime_mood += out[col].values * score
    out["regime_mood"] = regime_mood.clip(0, 100)
    out["market_mood"] = (
        FEATURE_WEIGHT * out["feature_mood"] + REGIME_WEIGHT * out["regime_mood"]
    ).clip(0, 100)
    out["mood_label"] = out["market_mood"].map(mood_label)
    return out


def plot_mood_timeseries(df: pd.DataFrame, outpath: Path) -> None:
    dates = pd.to_datetime(df["date_utc"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(dates, df["market_mood"], color="#1f4e79", lw=1.8, label="Market Mood")
    ax.plot(dates, df["feature_mood"], color="#7aa6d8", lw=1.0, alpha=0.7, label="Feature mood")
    ax.plot(dates, df["regime_mood"], color="#b0b0b0", lw=1.0, alpha=0.7, label="Regime mood")
    ax.axhline(65, color="#3FB36C", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(45, color="#888888", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(30, color="#C8362F", ls="--", lw=0.8, alpha=0.6)
    ax.fill_between(dates, 65, 100, color="#3FB36C", alpha=0.05)
    ax.fill_between(dates, 45, 65, color="#888888", alpha=0.05)
    ax.fill_between(dates, 30, 45, color="#E07A3F", alpha=0.05)
    ax.fill_between(dates, 0, 30, color="#C8362F", alpha=0.05)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Market Mood Index over time")
    ax.legend(loc="upper left", ncol=3, framealpha=0.9)

    ax2 = axes[1]
    colors = [REGIME_COLORS.get(r, "#cccccc") for r in df["regime_gmm"]]
    ax2.bar(dates, np.ones(len(df)), color=colors, width=1.0, align="center")
    ax2.set_yticks([])
    ax2.set_ylabel("GMM regime")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def build_correlation_matrix(metrics: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    m = metrics.copy()
    m["date_utc"] = pd.to_datetime(m["date_utc"])
    if start_date:
        m = m[m["date_utc"] >= pd.Timestamp(start_date)]
    wide = m.pivot_table(index="date_utc", columns="asset", values="daily_return")
    return wide.corr()


def build_asset_class_heatmap(metrics: pd.DataFrame, value_col: str, start_date: str | None = None) -> pd.DataFrame:
    m = metrics.copy()
    m["date_utc"] = pd.to_datetime(m["date_utc"])
    if start_date:
        m = m[m["date_utc"] >= pd.Timestamp(start_date)]
    return (
        m.groupby(["date_utc", "asset_class"], as_index=False)[value_col]
        .mean()
        .pivot(index="date_utc", columns="asset_class", values=value_col)
        .sort_index()
    )


def main() -> None:
    market_path = DATA_DIR / "MARKET_1.CSV"
    metrics_path = DATA_DIR / "metric_engine_long.csv"
    if not market_path.exists():
        raise FileNotFoundError(f"Missing {market_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    labelled = pd.read_csv(market_path, parse_dates=["date_utc"])
    metrics = pd.read_csv(metrics_path, parse_dates=["date_utc"])

    enriched = compute_market_mood(labelled)
    enriched.to_csv(OUT_DIR / "market_regimes_with_mood.csv", index=False)

    summary = (
        enriched.groupby("regime_kmeans")[["feature_mood", "regime_mood", "market_mood"]]
        .agg(["mean", "min", "max"])
        .round(2)
    )
    summary.to_csv(OUT_DIR / "market_mood_by_regime.csv")

    latest = enriched.iloc[-1]
    latest_row = pd.DataFrame(
        [
            {
                "date_utc": latest["date_utc"],
                "market_mood": round(latest["market_mood"], 2),
                "feature_mood": round(latest["feature_mood"], 2),
                "regime_mood": round(latest["regime_mood"], 2),
                "mood_label": latest["mood_label"],
                "regime_kmeans": latest["regime_kmeans"],
                "regime_gmm": latest["regime_gmm"],
            }
        ]
    )
    latest_row.to_csv(OUT_DIR / "market_mood_latest.csv", index=False)

    plot_mood_timeseries(enriched, OUT_DIR / "market_mood_timeseries.png")

    corr = build_correlation_matrix(metrics, start_date="2025-02-03")
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    for col, fname in [("daily_return", "asset_class_returns_heatmap.csv"), ("vol_30d", "asset_class_volatility_heatmap.csv")]:
        heat = build_asset_class_heatmap(metrics, col, start_date="2025-02-03")
        heat.to_csv(OUT_DIR / fname)

    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Latest mood: {latest['market_mood']:.1f} ({latest['mood_label']}) on {latest['date_utc'].date()}")


if __name__ == "__main__":
    main()
