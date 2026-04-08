import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

from momentum_backtest import (
    fetch_fred_rate,
    fetch_ticker_data,
    MomentumBacktest,
    MonteCarloValidator,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momentum Backtest Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font import */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0a0f;
        color: #e2e8f0;
    }

    /* Main area background */
    .main .block-container {
        background-color: #0a0a0f;
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }
    section[data-testid="stSidebar"] * {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f0f1a 0%, #13131f 100%);
        border: 1px solid #1e1e2e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 600;
        color: #e2e8f0 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #0f0f1a;
        border-bottom: 1px solid #1e1e2e;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        padding: 0.6rem 1.4rem;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: transparent !important;
    }

    /* Divider */
    hr { border-color: #1e1e2e; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #38bdf8 !important; }

    /* Alert / info boxes */
    .stAlert {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        border-radius: 6px;
    }

    /* Page title override */
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        color: #f1f5f9 !important;
    }
    h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
CHART_LAYOUT: dict = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0f0f1a",
    font=dict(family="IBM Plex Mono", size=11, color="#94a3b8"),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zeroline=False),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e", borderwidth=1),
    hovermode="x unified",
)

STRATEGY_COLORS = {
    "baseline":    "#38bdf8",   # sky blue
    "majority":    "#a78bfa",   # violet
    "ma200":       "#34d399",   # emerald
    "long_windows": "#fb923c",  # orange
}

STRATEGIES = ["baseline", "majority", "ma200", "long_windows"]
STRATEGY_LABELS = {
    "baseline":     "Baseline (all windows)",
    "majority":     "Majority vote",
    "ma200":        "MA200 filter",
    "long_windows": "Long windows (30/60/90)",
}

# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_market_data(ticker, period):
    return fetch_ticker_data(ticker, period)

@st.cache_data(show_spinner=False)
def load_risk_free_rate():
    try:
        return fetch_fred_rate(verbose=False)
    except requests.exceptions.RequestException:
        return 0.04

@st.cache_data(show_spinner=False)
def run_all_backtests(ticker, period, tx_cost_bps):
    data = load_market_data(ticker, period)
    rfr  = load_risk_free_rate()
    results = {}
    for s in STRATEGIES:
        bt = MomentumBacktest(data.copy(), rfr, transaction_cost_bps=tx_cost_bps)
        bt.run(strategy=s)
        results[s] = bt
    return results, rfr

@st.cache_data(show_spinner=False)
def run_all_monte_carlos(ticker, period, tx_cost_bps, n_sims):
    backtests, _ = run_all_backtests(ticker, period, tx_cost_bps)
    mc_results = {}
    for s, bt in backtests.items():
        mc = MonteCarloValidator(bt, strategy=s, n_simulations=n_sims)
        mc.run()
        mc_results[s] = mc
    return mc_results

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PARAMETERS")
    st.markdown("---")

    ticker   = st.selectbox("Ticker", ["QQQ", "SPY", "IWM"], index=0)
    period   = st.selectbox("History", ["5y", "10y", "15y"], index=1)
    strategy = st.selectbox(
        "Strategy",
        STRATEGIES,
        format_func=lambda s: STRATEGY_LABELS[s],
        index=0,
    )
    st.markdown("---")
    tx_cost  = st.slider("Transaction cost (bps)", 0, 20, 5)
    capital  = st.number_input("Starting capital ($)", 1_000, 1_000_000, 10_000, step=1_000)
    st.markdown("---")

    st.markdown("### MONTE CARLO")
    n_sims = st.select_slider(
        "Simulations",
        options=[500, 1_000, 2_000, 5_000, 10_000],
        value=1_000,
    )
    run_mc = st.button("▶ Run Monte Carlo", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("MOMENTUM BACKTEST / QQQ")
st.markdown(f"**Strategy:** {STRATEGY_LABELS[strategy]}&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"**Ticker:** {ticker}&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"**Period:** {period}&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"**Tx cost:** {tx_cost} bps")
st.markdown("---")

# ── Load data & run selected backtest ─────────────────────────────────────────
with st.spinner("Loading market data..."):
    all_backtests, rfr = run_all_backtests(ticker, period, tx_cost)

bt     = all_backtests[strategy]
data   = bt.data
res    = bt.results
color  = STRATEGY_COLORS[strategy]

# Scale equity curves to starting capital
eq_strategy = data["cumulative_strategy"] * capital
eq_buyhold  = data["cumulative_buyhold"]  * capital

# ── KPI metrics ───────────────────────────────────────────────────────────────
bh_return    = data["cumulative_buyhold"].iloc[-1].item() - 1
bh_ann       = np.exp(np.log(data["cumulative_buyhold"].iloc[-1].item()) / (len(data) / 252)) - 1
sharpe_delta = f"{res['sharpe_ratio'] - (bh_ann - rfr) / (data['log_return'].std() * np.sqrt(252)):+.2f} vs B&H"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Return",      f"{res['total_return']:.1%}",
          f"{res['total_return'] - bh_return:+.1%} vs B&H")
c2.metric("Ann. Return",       f"{res['annualized_return']:.1%}")
c3.metric("Volatility",        f"{res['volatility']:.1%}")
c4.metric("Sharpe Ratio",      f"{res['sharpe_ratio']:.2f}", sharpe_delta)
c5.metric("Max Drawdown",      f"{res['max_drawdown']:.1%}", delta_color="off")
c6.metric("Win Rate",          f"{res['win_rate']:.1%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "EQUITY CURVE", "DRAWDOWN", "MONTE CARLO", "COMPARISON"
])

# ── Tab 1: Equity Curve ───────────────────────────────────────────────────────
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=eq_strategy,
        name=STRATEGY_LABELS[strategy],
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.05)",
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=eq_buyhold,
        name="Buy & Hold",
        line=dict(color="#475569", width=1.5, dash="dot"),
    ))
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        title="Equity Curve — Growth of Capital",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Position calendar heatmap
    pos = data["position"].copy()
    pos.index = pd.to_datetime(pos.index)

    # Build week x weekday grid
    pos_df = pos.to_frame(name="position")
    pos_df["week"] = pos_df.index.isocalendar().week.astype(int)
    pos_df["year"] = pos_df.index.year
    pos_df["weekday"] = pos_df.index.weekday  # 0=Mon, 4=Fri
    pos_df["week_label"] = pos_df.index.strftime("%Y-W%V")

    # Pivot: rows = weekday, columns = week_label
    pivot = pos_df.pivot_table(
        index="weekday", columns="week_label",
        values="position", aggfunc="first"
    )
    pivot = pivot.sort_index()

    # Only keep Mon-Fri (0-4)
    pivot = pivot.loc[pivot.index.isin([0, 1, 2, 3, 4])]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]

    fig_cal = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=day_labels,
        colorscale=[
            [0.0, "#1a1a2e"],   # cash — dark
            [1.0, color],       # long — strategy colour
        ],
        showscale=False,
        xgap=2,
        ygap=2,
        hovertemplate="Week: %{x}<br>Day: %{y}<br>Position: %{z}<extra></extra>",
    ))

    fig_cal.update_layout(**CHART_LAYOUT)
    fig_cal.update_layout(
        title="Position Calendar (green = long, dark = cash)",
        height=160,
        margin=dict(l=20, r=20, t=40, b=10),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
        ),
        yaxis=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=day_labels,
            showgrid=False,
            autorange="reversed",
        ),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ── Tab 2: Drawdown ───────────────────────────────────────────────────────────
with tab2:
    drawdown_pct = data["drawdown"] * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=data.index, y=drawdown_pct,
        name="Drawdown",
        line=dict(color="#f87171", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248,113,113,0.12)",
    ))
    # Max drawdown line
    fig_dd.add_hline(
        y=res["max_drawdown"] * 100,
        line=dict(color="#ef4444", width=1, dash="dash"),
        annotation_text=f"Max DD {res['max_drawdown']:.1%}",
        annotation_font=dict(color="#ef4444", size=11),
    )
    fig_dd.update_layout(**CHART_LAYOUT)
    fig_dd.update_layout(
        title="Drawdown Over Time",
        yaxis_title="Drawdown (%)",
        yaxis_ticksuffix="%",
        height=460,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Stats row
    underwater = (data["drawdown"] < 0).sum()
    total_days = len(data)
    d1, d2, d3 = st.columns(3)
    d1.metric("Max Drawdown",       f"{res['max_drawdown']:.2%}")
    d2.metric("Days Underwater",    f"{underwater:,}  / {total_days:,}")
    d3.metric("Time in Market",     f"{data['position'].mean():.1%}")

# ── Tab 3: Monte Carlo ────────────────────────────────────────────────────────
with tab3:
    if run_mc:
        with st.spinner(f"Running {n_sims:,} simulations across all 4 strategies..."):
            mc_all = run_all_monte_carlos(ticker, period, tx_cost, n_sims)

        fig_mc = go.Figure()

        for s, mc in mc_all.items():
            assert mc.backtest.results is not None
            actual = mc.backtest.results["sharpe_ratio"]
            p_val  = (mc.simulated_sharpes >= actual).mean()
            c      = STRATEGY_COLORS[s]

            fig_mc.add_trace(go.Histogram(
                x=mc.simulated_sharpes,
                name=f"{STRATEGY_LABELS[s]} (shuffled)",
                marker_color=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.35)",
                nbinsx=80,
                showlegend=True,
            ))
            fig_mc.add_vline(
                x=actual,
                line=dict(color=c, width=2, dash="solid"),
                annotation_text=f"{s}  Sharpe={actual:.2f}  p={p_val:.3f}",
                annotation_font=dict(color=c, size=10),
                annotation_position="top right",
            )

        fig_mc.update_layout(**CHART_LAYOUT)
        fig_mc.update_layout(
            title=f"Monte Carlo Validation — {n_sims:,} Simulations / All Strategies",
            xaxis_title="Sharpe Ratio",
            yaxis_title="Frequency",
            barmode="overlay",
            height=500,
            hovermode="x",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # p-value summary table
        st.markdown("### SIGNIFICANCE SUMMARY")
        rows = []
        for s, mc in mc_all.items():
            assert mc.backtest.results is not None
            actual = mc.backtest.results["sharpe_ratio"]
            p_val  = (mc.simulated_sharpes >= actual).mean()
            pct    = (mc.simulated_sharpes < actual).mean() * 100
            rows.append({
                "Strategy":       STRATEGY_LABELS[s],
                "Actual Sharpe":  f"{actual:.3f}",
                "Sim Mean":       f"{mc.simulated_sharpes.mean():.3f}",
                "Percentile":     f"{pct:.1f}th",
                "p-value":        f"{p_val:.4f}",
                "Significant":    "✅ Yes" if p_val < 0.05 else "❌ No",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.info("Set the number of simulations in the sidebar, then click **▶ Run Monte Carlo**.")
        st.markdown("""
        **What this does:**  
        Shuffles the daily return sequence 10,000 times, breaking any time-series structure  
        (momentum, trends), and re-runs the backtest each time. If your actual Sharpe sits  
        above 95% of random results → p < 0.05 → the edge is statistically significant.
        """)

# ── Tab 4: Strategy Comparison ────────────────────────────────────────────────
with tab4:
    st.markdown("### ALL STRATEGIES vs BUY & HOLD")

    # Equity curves — all strategies on one chart
    fig_cmp = go.Figure()
    for s, bt_s in all_backtests.items():
        eq = bt_s.data["cumulative_strategy"] * capital
        fig_cmp.add_trace(go.Scatter(
            x=bt_s.data.index, y=eq,
            name=STRATEGY_LABELS[s],
            line=dict(color=STRATEGY_COLORS[s], width=2),
        ))
    bh_ref = next(iter(all_backtests.values()))
    fig_cmp.add_trace(go.Scatter(
        x=bh_ref.data.index,
        y=bh_ref.data["cumulative_buyhold"] * capital,
        name="Buy & Hold",
        line=dict(color="#475569", width=1.5, dash="dot"),
    ))
    fig_cmp.update_layout(**CHART_LAYOUT)
    fig_cmp.update_layout(
        title="Equity Curves — All Strategies",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        height=420,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Comparison metrics table
    st.markdown("### METRICS TABLE")
    bh_data    = next(iter(all_backtests.values())).data
    bh_total   = bh_data["cumulative_buyhold"].iloc[-1].item() - 1
    bh_years   = len(bh_data) / 252
    bh_ann     = np.exp(np.log1p(bh_total) / bh_years) - 1

    rows = []
    for s, bt_s in all_backtests.items():
        r = bt_s.results
        rows.append({
            "Strategy":       STRATEGY_LABELS[s],
            "Total Return":   f"{r['total_return']:.2%}",
            "Ann. Return":    f"{r['annualized_return']:.2%}",
            "Volatility":     f"{r['volatility']:.2%}",
            "Sharpe":         f"{r['sharpe_ratio']:.2f}",
            "Max Drawdown":   f"{r['max_drawdown']:.2%}",
            "Win Rate":       f"{r['win_rate']:.2%}",
            "Trades":         int(r["position_changes"]),
        })
    rows.append({
        "Strategy":     "Buy & Hold",
        "Total Return": f"{bh_total:.2%}",
        "Ann. Return":  f"{bh_ann:.2%}",
        "Volatility":   "—",
        "Sharpe":       "—",
        "Max Drawdown": "—",
        "Win Rate":     "—",
        "Trades":       0,
    })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)