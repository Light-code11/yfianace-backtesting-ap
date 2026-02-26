from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from alpaca_client import AlpacaClient
from database import SessionLocal, LivePosition, TradeJustification, init_db

st.set_page_config(page_title="Position Monitor", layout="wide")
st.title("Position Monitor")

init_db()


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


@st.cache_data(ttl=3600)
def _get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"


def _latest_justifications() -> dict:
    db = SessionLocal()
    try:
        rows = db.query(TradeJustification).filter(TradeJustification.action != "REJECT").order_by(TradeJustification.created_at.desc()).all()
        lookup = {}
        for r in rows:
            if r.ticker not in lookup:
                lookup[r.ticker] = r.justification
        return lookup
    finally:
        db.close()


client = None
try:
    client = AlpacaClient()
except Exception as exc:
    st.error(f"Alpaca client unavailable: {exc}")

if client is None:
    st.stop()

positions_resp = client.get_positions()
if not positions_resp.get("success"):
    st.error(f"Failed to fetch positions: {positions_resp.get('error')}")
    st.stop()

positions = positions_resp.get("positions", [])
if not positions:
    st.info("No open positions.")
    st.stop()

db = SessionLocal()
try:
    db_positions = db.query(LivePosition).filter(LivePosition.is_open == True).all()
    db_lookup = {p.ticker: p for p in db_positions}
finally:
    db.close()

just_lookup = _latest_justifications()

rows = []
for p in positions:
    ticker = p.get("symbol")
    entry = _safe_float(p.get("avg_entry_price"))
    current = _safe_float(p.get("current_price"))
    qty = _safe_float(p.get("qty"))
    pl = _safe_float(p.get("unrealized_pl"))
    plpc = _safe_float(p.get("unrealized_plpc")) * 100
    db_pos = db_lookup.get(ticker)
    stop_loss = getattr(db_pos, "stop_loss_price", None)
    created_at = getattr(db_pos, "created_at", None)
    days_held = None
    if created_at:
        days_held = max((datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)).days, 0)
    rows.append(
        {
            "Ticker": ticker,
            "Qty": qty,
            "Entry": entry,
            "Current": current,
            "P&L": pl,
            "P&L%": plpc,
            "Stop Loss": stop_loss,
            "Days Held": days_held,
            "Exposure": abs(current * qty),
            "Sector": _get_sector(ticker) if "/" not in ticker else "Crypto",
            "Justification": just_lookup.get(ticker, "(no justification logged)"),
        }
    )

df = pd.DataFrame(rows)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Exposure", f"${df['Exposure'].sum():,.2f}")
with c2:
    st.metric("Open Positions", int(df.shape[0]))
with c3:
    st.metric("Net Unrealized P&L", f"${df['P&L'].sum():,.2f}")

st.dataframe(df[["Ticker", "Qty", "Entry", "Current", "P&L", "P&L%", "Stop Loss", "Days Held", "Sector", "Justification"]], use_container_width=True, hide_index=True)

st.subheader("Sector Concentration")
sector_df = df.groupby("Sector", as_index=False)["Exposure"].sum()
fig_sector = px.bar(sector_df, x="Sector", y="Exposure", title="Exposure by Sector")
st.plotly_chart(fig_sector, use_container_width=True)

st.subheader("Correlation Matrix")
tickers = [t for t in df["Ticker"].tolist() if "/" not in t]
if len(tickers) >= 2:
    prices = yf.download(tickers=tickers, period="3mo", interval="1d", auto_adjust=True, progress=False)
    closes = prices["Close"] if "Close" in prices else prices
    corr = closes.pct_change().dropna().corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Need at least 2 equity positions to compute correlation matrix.")

st.subheader("Close Position")
ticker_to_close = st.selectbox("Ticker", options=df["Ticker"].tolist())
confirm = st.checkbox("I confirm I want to close this position")
if st.button("Close Position", type="primary", disabled=not confirm):
    close_res = client.close_position(ticker_to_close)
    if close_res.get("success"):
        st.success(f"Close order submitted for {ticker_to_close}")
    else:
        st.error(f"Failed to close {ticker_to_close}: {close_res.get('error')}")
