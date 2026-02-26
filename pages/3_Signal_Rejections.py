import json
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from database import SessionLocal, TradeJustification, init_db

st.set_page_config(page_title="Signal Rejections", layout="wide")
st.title("Signal Rejections")

init_db()


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _parse_factors(raw_json: str) -> dict:
    if not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except Exception:
        return {}


@st.cache_data(ttl=900)
def outcome_24h(ticker: str, when_ts: pd.Timestamp, reference_price: float | None) -> tuple[float | None, str]:
    if "/" in ticker:
        return None, "Crypto/unsupported symbol"
    try:
        start = (when_ts - timedelta(days=1)).date().isoformat()
        end = (when_ts + timedelta(days=3)).date().isoformat()
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None, "No price data"
        close_series = hist["Close"].dropna()
        if close_series.empty:
            return None, "No close data"

        base = reference_price
        if base is None:
            base = float(close_series.iloc[0])
        target = float(close_series.iloc[-1])
        if base == 0:
            return None, "Invalid base price"

        ret_pct = (target - base) / base * 100
        verdict = "Rejection Correct" if ret_pct < 0 else "Rejection Missed Gain"
        return ret_pct, verdict
    except Exception as exc:
        return None, f"Error: {str(exc)[:80]}"


db = SessionLocal()
try:
    rows = db.query(TradeJustification).filter(
        (TradeJustification.action == "REJECT") | (TradeJustification.rejection_reason.isnot(None))
    ).order_by(TradeJustification.created_at.desc()).all()
finally:
    db.close()

if not rows:
    st.info("No rejected signals found.")
    st.stop()

data = []
for r in rows:
    factors = _parse_factors(r.decision_factors_json)
    ref_price = _safe_float(factors.get("trend_price") or factors.get("signal_price") or factors.get("price"))
    ret_24h, verdict = outcome_24h(r.ticker, pd.Timestamp(r.created_at), ref_price)
    data.append(
        {
            "Date": r.created_at,
            "Ticker": r.ticker,
            "Strategy": r.strategy_name,
            "Rejection Reason": r.rejection_reason or "unspecified",
            "Conviction": _safe_float(r.conviction_score),
            "Kelly%": _safe_float(r.kelly_pct),
            "24h Return %": ret_24h,
            "What would have happened": verdict,
            "Justification": r.justification,
        }
    )

df = pd.DataFrame(data)

reason_counts = df.groupby("Rejection Reason", as_index=False).size().rename(columns={"size": "Count"})
st.subheader("Rejection Reason Distribution")
st.plotly_chart(px.bar(reason_counts, x="Rejection Reason", y="Count"), use_container_width=True)

valid = df[df["24h Return %"].notna()].copy()
if not valid.empty:
    win_rate = (valid["24h Return %"] < 0).mean() * 100
    st.metric("Rejection Win Rate", f"{win_rate:.1f}%")
else:
    st.metric("Rejection Win Rate", "n/a")

st.subheader("Rejected Signals with 24h Outcome")
st.dataframe(
    df[["Date", "Ticker", "Strategy", "Rejection Reason", "24h Return %", "What would have happened", "Justification"]],
    use_container_width=True,
    hide_index=True,
)
