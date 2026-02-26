import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from database import SessionLocal, TradeExecution, TradeJustification, init_db

st.set_page_config(page_title="Trade Log", layout="wide")
st.title("Trade Log")

init_db()


def _normalize_factors(raw_json: str):
    if not raw_json:
        return {}
    try:
        return json.loads(raw_json)
    except Exception:
        return {}


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def load_trade_rows() -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = []
        justifications = db.query(TradeJustification).order_by(TradeJustification.created_at.desc()).all()
        linked_exec_ids = set()

        for j in justifications:
            factors = _normalize_factors(j.decision_factors_json)
            pnl = factors.get("realized_pnl")
            if pnl is None and j.trade_execution_id:
                linked_exec_ids.add(j.trade_execution_id)
            rows.append(
                {
                    "Date": j.created_at,
                    "Ticker": j.ticker,
                    "Action": j.action,
                    "Strategy": j.strategy_name,
                    "Conviction": _safe_float(j.conviction_score),
                    "Kelly%": _safe_float(j.kelly_pct),
                    "Regime": j.regime,
                    "P&L": _safe_float(pnl),
                    "Justification": j.justification,
                    "Rejection Reason": j.rejection_reason,
                    "Decision JSON": j.decision_factors_json or "{}",
                }
            )

        orphan_query = db.query(TradeExecution)
        if linked_exec_ids:
            orphan_query = orphan_query.filter(~TradeExecution.id.in_(linked_exec_ids))
        orphan_execs = orphan_query.order_by(TradeExecution.created_at.desc()).all()

        for ex in orphan_execs:
            factors = ex.decision_factors if isinstance(ex.decision_factors, dict) else {}
            kelly_fraction = factors.get("kelly_fraction")
            kelly_pct = _safe_float(kelly_fraction) * 100 if _safe_float(kelly_fraction) is not None else None
            pnl = None
            if ex.side == "sell":
                filled = _safe_float((ex.alpaca_order_data or {}).get("filled_avg_price"))
                sig = _safe_float(ex.signal_price)
                if filled is not None and sig is not None:
                    pnl = filled - sig

            rows.append(
                {
                    "Date": ex.created_at,
                    "Ticker": ex.ticker,
                    "Action": (ex.signal_type or ex.side or "").upper(),
                    "Strategy": ex.strategy_name,
                    "Conviction": _safe_float(factors.get("conviction_score")),
                    "Kelly%": kelly_pct,
                    "Regime": factors.get("macro_regime"),
                    "P&L": pnl,
                    "Justification": ex.decision_reasoning or "(no justification logged)",
                    "Rejection Reason": None,
                    "Decision JSON": json.dumps(factors, default=str),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("Date", ascending=False)
        return df
    finally:
        db.close()


def style_trades(df: pd.DataFrame):
    def row_style(row):
        if row["Action"] == "REJECT":
            return ["background-color: #fff8dc"] * len(row)
        pnl = row.get("P&L")
        if pnl is not None and not pd.isna(pnl):
            if float(pnl) > 0:
                return ["background-color: #e8f5e9"] * len(row)
            if float(pnl) < 0:
                return ["background-color: #ffebee"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


df_all = load_trade_rows()
if df_all.empty:
    st.info("No trade records found yet.")
    st.stop()

min_date = pd.to_datetime(df_all["Date"]).dt.date.min()
max_date = pd.to_datetime(df_all["Date"]).dt.date.max()
start_default = max(min_date, max_date - timedelta(days=30))

c1, c2, c3, c4 = st.columns(4)
with c1:
    date_range = st.date_input("Date Range", value=(start_default, max_date), min_value=min_date, max_value=max_date)
with c2:
    ticker_filter = st.multiselect("Ticker", options=sorted(df_all["Ticker"].dropna().unique().tolist()))
with c3:
    strategy_filter = st.multiselect("Strategy", options=sorted([s for s in df_all["Strategy"].dropna().unique().tolist() if s]))
with c4:
    action_filter = st.multiselect("Action", options=sorted(df_all["Action"].dropna().unique().tolist()))

filtered = df_all.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    if isinstance(start_date, date) and isinstance(end_date, date):
        filtered = filtered[(filtered["Date"].dt.date >= start_date) & (filtered["Date"].dt.date <= end_date)]
if ticker_filter:
    filtered = filtered[filtered["Ticker"].isin(ticker_filter)]
if strategy_filter:
    filtered = filtered[filtered["Strategy"].isin(strategy_filter)]
if action_filter:
    filtered = filtered[filtered["Action"].isin(action_filter)]

show_cols = ["Date", "Ticker", "Action", "Strategy", "Conviction", "Kelly%", "Regime", "P&L", "Justification"]
st.dataframe(style_trades(filtered[show_cols]), use_container_width=True, hide_index=True)

csv_bytes = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Export to CSV", data=csv_bytes, file_name="trade_log.csv", mime="text/csv")

st.subheader("Decision Drill-Down")
for _, row in filtered.head(100).iterrows():
    label = f"{row['Date']} | {row['Ticker']} | {row['Action']} | {row['Strategy']}"
    with st.expander(label):
        st.write(row["Justification"])
        st.code(row["Decision JSON"], language="json")
        if row.get("Rejection Reason"):
            st.warning(f"Rejection reason: {row['Rejection Reason']}")
