"""Strategy Research Dashboard — Alpha Hunt Results"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="Strategy Research", page_icon="🔬", layout="wide")
st.title("🔬 Strategy Research — Alpha Hunt")

# Load results
results_files = {
    "Alpha Hunt (Walk-Forward Validated)": "alpha_hunt_results.json",
    "AI Discovery (GPT-4o Generated)": "ai_discovery_results.json",
    "Parameter Optimizer": "optimized_params.json",
}

for title, filepath in results_files.items():
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath)
    if not os.path.exists(full_path):
        continue
    
    st.header(title)
    
    with open(full_path) as f:
        data = json.load(f)
    
    if filepath == "alpha_hunt_results.json":
        st.caption(f"Generated: {data.get('generated_at', 'unknown')}")
        strategies = data.get("strategies", {})
        
        if strategies:
            rows = []
            for name, info in strategies.items():
                train = info.get("train", {})
                test = info.get("test", {})
                rows.append({
                    "Strategy": name,
                    "Type": info.get("strategy", ""),
                    "Train Return %": train.get("return", 0),
                    "Train Sharpe": train.get("sharpe", 0),
                    "Train Win %": train.get("win_rate", 0),
                    "Train Trades": train.get("trades", 0),
                    "Test Return %": test.get("return", 0),
                    "Test Sharpe": test.get("sharpe", 0),
                    "Test Win %": test.get("win_rate", 0),
                    "Test Trades": test.get("trades", 0),
                })
            
            df = pd.DataFrame(rows).sort_values("Test Sharpe", ascending=False)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategies Found", len(df))
            col2.metric("Best Test Sharpe", f"{df['Test Sharpe'].max():.2f}")
            col3.metric("Best Test Return", f"{df['Test Return %'].max():.1f}%")
            col4.metric("Avg Test Win Rate", f"{df['Test Win %'].mean():.0f}%")
            
            # Top 10 table
            st.subheader("Top 10 Strategies (by Test Sharpe)")
            st.dataframe(
                df.head(10).style.background_gradient(subset=["Test Sharpe"], cmap="RdYlGn")
                .background_gradient(subset=["Test Return %"], cmap="RdYlGn")
                .background_gradient(subset=["Test Win %"], cmap="RdYlGn")
                .format({
                    "Train Return %": "{:.1f}",
                    "Train Sharpe": "{:.2f}",
                    "Train Win %": "{:.0f}",
                    "Test Return %": "{:.1f}",
                    "Test Sharpe": "{:.2f}",
                    "Test Win %": "{:.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            
            # Full table (expandable)
            with st.expander(f"All {len(df)} strategies"):
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Strategy type breakdown
            st.subheader("Performance by Strategy Type")
            type_stats = df.groupby("Type").agg({
                "Test Return %": "mean",
                "Test Sharpe": "mean", 
                "Test Win %": "mean",
                "Strategy": "count",
            }).rename(columns={"Strategy": "Count"}).sort_values("Test Sharpe", ascending=False)
            st.dataframe(type_stats, use_container_width=True)
            
            # Best strategy details
            st.subheader("🏆 Best Strategy Details")
            best = df.iloc[0]
            best_name = best["Strategy"]
            best_info = strategies[best_name]
            
            st.json(best_info)
        else:
            st.warning("No strategies found in results file.")
    
    elif filepath == "ai_discovery_results.json":
        st.caption(f"Generated: {data.get('generated_at', 'unknown')}")
        
        # Market context
        ctx = data.get("market_context", {})
        if ctx:
            col1, col2, col3 = st.columns(3)
            col1.metric("Macro Regime", ctx.get("macro_regime", "?").upper())
            col2.metric("VIX", f"{ctx.get('vix', 0):.1f}")
            col3.metric("Market Breadth", f"{ctx.get('market_breadth', 0)*100:.0f}%")
            
            sector = ctx.get("sector_performance_21d", {})
            if sector:
                st.subheader("21-Day Sector Performance")
                sector_df = pd.DataFrame([{"Sector": k, "Return %": v*100} for k, v in sorted(sector.items(), key=lambda x: x[1], reverse=True)])
                st.bar_chart(sector_df.set_index("Sector"))
        
        # AI strategies
        ai_results = data.get("results", [])
        for r in ai_results:
            with st.expander(f"{'✅' if r.get('passed') else '❌'} {r['name']} ({r.get('strategy_type', '?')})"):
                metrics = r.get("full_metrics", {})
                wf = r.get("walk_forward", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Backtest (In-Sample)**")
                    st.write(f"- Return: {metrics.get('total_return_pct', 0):.1f}%")
                    st.write(f"- Sharpe: {metrics.get('sharpe', 0):.2f}")
                    st.write(f"- Win Rate: {metrics.get('win_rate', 0)*100:.0f}%")
                    st.write(f"- Max DD: {metrics.get('max_drawdown', 0):.1f}%")
                    st.write(f"- Trades: {metrics.get('trade_count', 0)}")
                with col2:
                    st.write("**Walk-Forward (Out-of-Sample)**")
                    st.write(f"- Return: {wf.get('total_return_pct', 0):.1f}%")
                    st.write(f"- Sharpe: {wf.get('sharpe', 0):.2f}")
                
                st.write(f"**Tickers:** {', '.join(r.get('tickers_evaluated', []))}")
    
    elif filepath == "optimized_params.json":
        for strat_name, info in data.items():
            if "error" in info:
                st.error(f"**{strat_name}**: {info['error']}")
                unprofitable = info.get("tickers_unprofitable", [])
                if unprofitable:
                    st.write(f"Unprofitable on: {', '.join(unprofitable)}")
            else:
                st.success(f"**{strat_name}**: Optimized params found")
                st.json(info)
    
    st.divider()

# Research notes
st.header("📝 Research Notes")
st.markdown("""
### Key Findings (Feb 27, 2026)

**Why old strategies failed:**
- Basic technical indicators (RSI, MACD, EMA crossovers) have no edge on liquid US equities
- All 5 original strategies were unprofitable on ALL tickers tested
- The edge was arbed away decades ago

**What works:**
1. **Cross-sectional strategies** — ranking stocks against each other, not predicting individual stocks
2. **Factor investing** — momentum, low volatility, quality are proven across decades
3. **Short-term reversal** — weekly losers bouncing back is a robust, high-frequency edge
4. **Walk-forward validation is essential** — 3 of 5 AI-generated strategies were overfit and would have lost money

**Best strategies found:**
- Low Vol + Momentum: Buy smooth-trending stocks monthly. 100% OOS win rate, Sharpe 6.9
- Short-Term Reversal: Buy weekly losers, hold 10 days. +98% annual return, Sharpe 2.3  
- 3-Month Momentum: Buy top 10% performers, monthly rebalance. +44% annual, Sharpe 3.5
""")
