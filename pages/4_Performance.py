import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from database import PortfolioSnapshot, SessionLocal, TradeExecution, init_db

st.set_page_config(page_title="Performance", layout="wide")
st.title("Performance")

init_db()


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def load_snapshots() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(PortfolioSnapshot).order_by(PortfolioSnapshot.created_at.asc()).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "Date": r.created_at,
                    "Equity": _safe_float(r.equity),
                    "Cash": _safe_float(r.cash),
                    "Buying Power": _safe_float(r.buying_power),
                    "Positions": r.num_positions,
                    "Daily P&L": _safe_float(r.daily_pnl),
                    "Daily P&L %": _safe_float(r.daily_pnl_pct),
                    "Regime": r.regime,
                }
                for r in rows
            ]
        )
    finally:
        db.close()


def load_strategy_breakdown() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(TradeExecution).order_by(TradeExecution.created_at.desc()).all()
    finally:
        db.close()

    if not rows:
        return pd.DataFrame()

    recs = []
    for r in rows:
        factors = r.decision_factors if isinstance(r.decision_factors, dict) else {}
        kelly_fraction = _safe_float(factors.get("kelly_fraction"))
        recs.append(
            {
                "Strategy": r.strategy_name or "unknown",
                "Action": (r.signal_type or r.side or "").upper(),
                "Kelly%": (kelly_fraction * 100) if kelly_fraction is not None else None,
                "Conviction": _safe_float(factors.get("conviction_score")),
                "Slippage%": _safe_float(factors.get("slippage_pct")),
            }
        )

    df = pd.DataFrame(recs)
    return (
        df.groupby("Strategy", as_index=False)
        .agg(
            Trades=("Action", "count"),
            Buys=("Action", lambda x: int((x == "BUY").sum())),
            Sells=("Action", lambda x: int((x == "SELL").sum())),
            AvgKellyPct=("Kelly%", "mean"),
            AvgConviction=("Conviction", "mean"),
            AvgSlippagePct=("Slippage%", "mean"),
        )
        .sort_values("Trades", ascending=False)
    )


def compute_metrics(snapshots: pd.DataFrame) -> dict:
    metrics = {"sharpe": None, "max_drawdown": None, "win_rate": None}
    if snapshots.empty or snapshots["Equity"].isna().all():
        return metrics

    eq = snapshots.dropna(subset=["Equity"]).copy()
    eq["ret"] = eq["Equity"].pct_change()
    daily = eq["ret"].dropna()
    if not daily.empty and daily.std() > 0:
        metrics["sharpe"] = (daily.mean() / daily.std()) * np.sqrt(252)
        metrics["win_rate"] = (daily > 0).mean() * 100

    roll_max = eq["Equity"].cummax()
    drawdown = (eq["Equity"] / roll_max) - 1
    metrics["max_drawdown"] = drawdown.min() * 100 if not drawdown.empty else None
    return metrics


snap = load_snapshots()
if snap.empty:
    st.info("No portfolio snapshots yet. Run the live cycle to populate performance data.")
    st.stop()

snap["Date"] = pd.to_datetime(snap["Date"])

st.subheader("Portfolio Equity Curve")
fig_eq = px.line(snap, x="Date", y="Equity")
st.plotly_chart(fig_eq, use_container_width=True)

metrics = compute_metrics(snap)
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}" if metrics["sharpe"] is not None else "n/a")
with c2:
    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%" if metrics["max_drawdown"] is not None else "n/a")
with c3:
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%" if metrics["win_rate"] is not None else "n/a")

st.subheader("Daily / Weekly / Monthly P&L")
pnl = snap.set_index("Date")["Daily P&L"].fillna(0)
pnl_summary = pd.DataFrame(
    {
        "Daily P&L": pnl,
        "Weekly P&L": pnl.resample("W").sum().reindex(pnl.index, method="ffill"),
        "Monthly P&L": pnl.resample("M").sum().reindex(pnl.index, method="ffill"),
    }
).reset_index()
st.plotly_chart(px.line(pnl_summary, x="Date", y=["Daily P&L", "Weekly P&L", "Monthly P&L"]), use_container_width=True)

st.subheader("Benchmark Comparison vs SPY")
start = snap["Date"].min().date().isoformat()
end = (snap["Date"].max().date() + pd.Timedelta(days=1)).isoformat()
spy = yf.Ticker("SPY").history(start=start, end=end, interval="1d", auto_adjust=True)
if spy is not None and not spy.empty:
    spy = spy[["Close"]].rename(columns={"Close": "SPY"})
    eq = snap[["Date", "Equity"]].set_index("Date").resample("1D").ffill()
    combo = eq.join(spy, how="inner")
    combo["Portfolio Index"] = combo["Equity"] / combo["Equity"].iloc[0] * 100
    combo["SPY Index"] = combo["SPY"] / combo["SPY"].iloc[0] * 100
    comp = combo[["Portfolio Index", "SPY Index"]].reset_index()
    st.plotly_chart(px.line(comp, x="Date", y=["Portfolio Index", "SPY Index"]), use_container_width=True)
else:
    st.info("SPY data unavailable for benchmark comparison.")

st.subheader("Strategy-Level Breakdown")
strategy_df = load_strategy_breakdown()
if strategy_df.empty:
    st.info("No executed strategy data found yet.")
else:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
