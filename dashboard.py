"""
Interactive market dashboard: mood index, regimes, heatmaps, volatility, correlations.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from compute_market_mood import (
    REGIME_COLORS,
    REGIME_NAMES,
    build_asset_class_heatmap,
    build_correlation_matrix,
    compute_market_mood,
)

DATA_DIR = HERE.parent
OUT_DIR = HERE / "outputs"


@st.cache_data
def load_data():
    market = pd.read_csv(DATA_DIR / "MARKET_1.CSV", parse_dates=["date_utc"])
    metrics = pd.read_csv(DATA_DIR / "metric_engine_long.csv", parse_dates=["date_utc"])
    mood = compute_market_mood(market)
    btc = metrics[metrics["asset"] == "BTCUSD"][["date_utc", "close"]].drop_duplicates()
    return mood, metrics, btc


def mood_gauge(score: float, label: str) -> go.Figure:
    colors = {"bullish": "#3FB36C", "neutral": "#5BA0E0", "cautious": "#E07A3F", "bearish": "#C8362F"}
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": f"  ({label})"},
            title={"text": "Market Mood"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colors.get(label, "#1f4e79")},
                "steps": [
                    {"range": [0, 30], "color": "#fde8e8"},
                    {"range": [30, 45], "color": "#fff0e6"},
                    {"range": [45, 65], "color": "#eef4fb"},
                    {"range": [65, 100], "color": "#e8f6ee"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def show_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str,
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annot: bool = False,
    figsize: tuple[float, float] = (14, 3.5),
) -> None:
    """Render a pivot table (rows=y, cols=x) with matplotlib (reliable in Streamlit)."""
    data = pivot.astype(float).values
    if center is not None and vmin is None and vmax is None:
        limit = float(np.nanmax(np.abs(data)))
        vmin, vmax = -limit, limit

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])

    col_labels = [
        pd.Timestamp(c).strftime("%Y-%m-%d") if isinstance(c, (pd.Timestamp, np.datetime64)) else str(c)
        for c in pivot.columns
    ]
    ncol = len(col_labels)
    step = max(1, ncol // 12)
    tick_idx = list(range(0, ncol, step))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([col_labels[i] for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def regime_timeline(mood: pd.DataFrame, btc: pd.DataFrame, regime_col: str) -> go.Figure:
    df = mood.merge(btc, on="date_utc", how="left")
    fig = go.Figure()
    df = df.sort_values("date_utc")
    df["run"] = (df[regime_col] != df[regime_col].shift()).cumsum()
    for _, grp in df.groupby("run"):
        regime = grp[regime_col].iloc[0]
        fig.add_vrect(
            x0=grp["date_utc"].iloc[0],
            x1=grp["date_utc"].iloc[-1],
            fillcolor=REGIME_COLORS.get(regime, "#cccccc"),
            opacity=0.25,
            layer="below",
            line_width=0,
        )
    fig.add_trace(
        go.Scatter(
            x=df["date_utc"],
            y=df["close"],
            mode="lines",
            name="BTCUSD",
            line=dict(color="black", width=1.5),
        )
    )
    fig.update_layout(
        title=f"Market regimes ({regime_col.replace('regime_', '').upper()}) vs BTCUSD",
        yaxis_type="log",
        yaxis_title="BTCUSD close (log)",
        xaxis_title="Date",
        height=420,
        showlegend=False,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Market Mood Dashboard", layout="wide")
    st.title("Market Mood Dashboard")
    st.caption("Cross-asset regimes, volatility, correlations, and mood index")

    mood, metrics, btc = load_data()
    latest = mood.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market Mood", f"{latest['market_mood']:.1f}", latest["mood_label"])
    col2.metric("Feature Mood", f"{latest['feature_mood']:.1f}")
    col3.metric("Regime Mood", f"{latest['regime_mood']:.1f}")
    col4.metric("GMM Regime", str(latest["regime_gmm"]))

    with st.expander("How to read these numbers", expanded=False):
        st.markdown(
            """
- **Market Mood (0–100)** — the headline risk-sentiment score. It blends the feature mood and
  regime mood as `0.70 × Feature + 0.30 × Regime`. Higher = more bullish/risk-on.
  Bands: **≥65 bullish · 45–65 neutral · 30–45 cautious · <30 bearish**.
- **Feature Mood (0–100)** — built only from market features (returns, trends, volatility,
  dispersion, correlation). It captures the *current* tone of the tape.
- **Regime Mood (0–100)** — derived from the probability of each statistical regime, scored
  `risk_on=100, calm=50, high_volatility=25, risk_off=0`. It can sit at **0** when the model is
  fully confident the market is in the bearish *risk_off* regime.
- **GMM Regime** — the most likely regime label from a Gaussian Mixture Model.
            """
        )

    st.plotly_chart(mood_gauge(latest["market_mood"], latest["mood_label"]), use_container_width=True)
    st.caption("Gauge of the latest Market Mood. The colored bands mark the bearish → bullish zones.")

    tab_mood, tab_regime, tab_heat, tab_vol, tab_corr = st.tabs(
        ["Market Mood", "Regimes", "Asset-Class Heatmaps", "Volatility Clusters", "Correlations"]
    )

    with tab_mood:
        st.markdown(
            "**Mood over time.** Tracks the three mood scores day by day. The dark line is the "
            "blended **Market Mood**; the lighter lines are its **Feature** and **Regime** "
            "components. Dashed guides mark the bullish / neutral / cautious thresholds, so you "
            "can see when sentiment crossed between regimes."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mood["date_utc"], y=mood["market_mood"], name="Market Mood", line=dict(color="#1f4e79", width=2)))
        fig.add_trace(go.Scatter(x=mood["date_utc"], y=mood["feature_mood"], name="Feature mood", line=dict(color="#7aa6d8", width=1)))
        fig.add_trace(go.Scatter(x=mood["date_utc"], y=mood["regime_mood"], name="Regime mood", line=dict(color="#b0b0b0", width=1)))
        for y, label in [(65, "Bullish"), (45, "Neutral"), (30, "Cautious")]:
            fig.add_hline(y=y, line_dash="dash", line_color="#aaaaaa", annotation_text=label)
        fig.update_layout(title="Market Mood over time", yaxis_range=[0, 100], height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mood by regime (K-Means)")
        st.caption(
            "Average mood scores within each K-Means regime. A sanity check: risk-on style "
            "regimes should show higher mood than risk-off ones."
        )
        summary = mood.groupby("regime_kmeans")[["market_mood", "feature_mood", "regime_mood"]].mean().round(1)
        st.dataframe(summary, use_container_width=True)

    with tab_regime:
        st.markdown(
            "**Market regimes vs price.** The black line is BTCUSD (log scale). The shaded "
            "background is the detected market regime on each date, so you can see how price "
            "action lines up with calm, risk-on, risk-off, and high-volatility periods. "
            "Switch between the **GMM** and **K-Means** models below."
        )
        model = st.radio("Regime model", ["GMM", "K-Means"], horizontal=True)
        regime_col = "regime_gmm" if model == "GMM" else "regime_kmeans"
        st.plotly_chart(regime_timeline(mood, btc, regime_col), use_container_width=True)

        st.caption(
            "Below: how many days the market spent in each regime over the sample — a quick view "
            "of which regime has dominated."
        )

        counts = mood[regime_col].value_counts().reindex(REGIME_NAMES, fill_value=0)
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=REGIME_COLORS,
            labels={"x": "Regime", "y": "Days"},
            title=f"Days per regime — {model}",
        )
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with tab_heat:
        st.markdown(
            "**Asset-class heatmap.** Each row is an asset class (crypto, fx, stablecoin) and each "
            "column is a date. Color shows the average metric for that class on that day — pick "
            "**Daily return** (blue = down, red = up) or **30d volatility** (darker = more "
            "volatile). Use it to spot when a whole asset class moved together."
        )
        metric = st.selectbox("Metric", ["daily_return", "vol_30d"], format_func=lambda x: "Daily return" if x == "daily_return" else "30d volatility")
        heat = build_asset_class_heatmap(metrics, metric, start_date="2025-02-03")
        show_heatmap(
            heat.T,
            title=f"Asset-class {metric.replace('_', ' ')} heatmap",
            cmap="RdBu_r" if metric == "daily_return" else "YlOrRd",
            center=0 if metric == "daily_return" else None,
        )

    with tab_vol:
        st.markdown(
            "**Volatility clusters.** Per-asset 30-day volatility over time (darker = higher "
            "volatility). Vertical dark stripes that span many rows are market-wide volatility "
            "spikes — moments when most assets got turbulent at once."
        )
        recent = metrics[metrics["date_utc"] >= "2025-02-03"].dropna(subset=["vol_30d"])
        vol_pivot = recent.pivot_table(index="date_utc", columns="asset", values="vol_30d")
        show_heatmap(
            vol_pivot.T,
            title="Volatility clusters by asset",
            cmap="YlOrRd",
            figsize=(14, 4.5),
        )

        st.subheader("Latest volatility ranking")
        st.caption(
            "Each asset's most recent 30d volatility, sorted high to low. The top rows are "
            "currently the most volatile assets."
        )
        rank = (
            recent.sort_values("date_utc")
            .groupby("asset", as_index=False)
            .last()[["asset", "asset_class", "vol_30d", "vol_30d_rank"]]
            .sort_values("vol_30d", ascending=False)
        )
        st.dataframe(rank, use_container_width=True, hide_index=True)

    with tab_corr:
        st.markdown(
            "**Cross-asset correlations.** The matrix shows how strongly each pair of assets' "
            "daily returns move together over the full sample: **+1 (red)** = move together, "
            "**0 (white)** = unrelated, **−1 (blue)** = move opposite. High correlations across "
            "the board mean less diversification."
        )
        corr = build_correlation_matrix(metrics, start_date="2025-02-03")
        show_heatmap(
            corr,
            title="Cross-asset correlation matrix (full sample)",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            annot=True,
            figsize=(8, 6),
        )

        st.caption(
            "Below: the average pairwise correlation across all assets on a rolling 30-day "
            "window. Rising values mean assets are increasingly moving as one (often a stress "
            "signal); falling values mean more independent, diversified moves."
        )
        corr_series = mood.dropna(subset=["avg_pairwise_corr_30d"])
        fig2, ax2 = plt.subplots(figsize=(14, 3.2))
        ax2.plot(
            corr_series["date_utc"],
            corr_series["avg_pairwise_corr_30d"],
            color="#1f4e79",
            lw=1.5,
        )
        ax2.set_ylim(0, 1)
        ax2.set_title("Rolling average pairwise correlation")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Avg pairwise corr (30d)")
        ax2.grid(True, alpha=0.3)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


if __name__ == "__main__":
    main()
