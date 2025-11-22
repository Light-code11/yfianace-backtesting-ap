"""
AI Trading Platform - Streamlit Web Interface
Interactive dashboard for strategy generation, backtesting, and portfolio optimization
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL - configurable via environment variable
# For Streamlit Cloud: Set this in Secrets management
# For local: Use localhost
API_BASE_URL = os.getenv("API_BASE_URL", st.secrets.get("API_BASE_URL", "http://localhost:8000"))

# Initialize session state for persistent values across page navigations
if 'tickers' not in st.session_state:
    st.session_state.tickers = "SPY,QQQ,AAPL"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def make_api_request(endpoint, method="GET", data=None):
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)

        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def plot_equity_curve(equity_curve):
    """Plot equity curve"""
    if not equity_curve:
        return None

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_drawdown(equity_curve):
    """Plot drawdown chart"""
    if not equity_curve:
        return None

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])

    # Calculate drawdown
    equity = df['equity'].values
    peak = equity[0]
    drawdowns = []

    for value in equity:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        drawdowns.append(dd)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=drawdowns,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#d62728', width=2)
    ))

    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_trade_distribution(trades):
    """Plot trade P&L distribution"""
    if not trades:
        return None

    df = pd.DataFrame(trades)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['profit_loss_pct'],
        nbinsx=30,
        name='Trade Returns',
        marker=dict(
            color=df['profit_loss_pct'],
            colorscale='RdYlGn',
            showscale=True
        )
    ))

    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="Profit/Loss (%)",
        yaxis_title="Number of Trades",
        template='plotly_white'
    )

    return fig


def plot_monthly_returns(equity_curve):
    """Plot monthly returns heatmap"""
    if not equity_curve:
        return None

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate daily returns
    df['returns'] = df['equity'].pct_change()

    # Group by month and calculate monthly returns
    monthly_returns = df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1) * 100

    if len(monthly_returns) == 0:
        return None

    # Create pivot table for heatmap
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[month_names[i-1] for i in pivot.index],
        colorscale='RdYlGn',
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        colorbar=dict(title="Return (%)")
    ))

    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Year",
        yaxis_title="Month",
        template='plotly_white'
    )

    return fig


def plot_win_loss_distribution(trades):
    """Plot win/loss pie chart"""
    if not trades or len(trades) == 0:
        return None

    df = pd.DataFrame(trades)

    wins = len(df[df['profit_loss_pct'] > 0])
    losses = len(df[df['profit_loss_pct'] < 0])
    breakeven = len(df[df['profit_loss_pct'] == 0])

    fig = go.Figure(data=[go.Pie(
        labels=['Winning Trades', 'Losing Trades', 'Breakeven'],
        values=[wins, losses, breakeven],
        marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6']),
        hole=0.4
    )])

    fig.update_layout(
        title=f"Trade Outcome Distribution (Total: {len(df)} trades)",
        template='plotly_white'
    )

    return fig


def plot_rolling_sharpe(equity_curve, window=30):
    """Plot rolling Sharpe ratio"""
    if not equity_curve or len(equity_curve) < window:
        return None

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Calculate daily returns
    df['returns'] = df['equity'].pct_change()

    # Calculate rolling Sharpe (annualized)
    rolling_mean = df['returns'].rolling(window=window).mean() * 252
    rolling_std = df['returns'].rolling(window=window).std() * (252 ** 0.5)
    rolling_sharpe = rolling_mean / rolling_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rolling_sharpe,
        mode='lines',
        name=f'{window}-Day Rolling Sharpe',
        line=dict(color='#9467bd', width=2)
    ))

    # Add reference line at Sharpe = 1
    fig.add_hline(y=1, line_dash="dash", line_color="green",
                  annotation_text="Sharpe = 1 (Good)")
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Sharpe = 0")

    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window} days)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_cumulative_returns(equity_curve):
    """Plot cumulative returns percentage"""
    if not equity_curve:
        return None

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])

    # Calculate cumulative return percentage
    initial_equity = df['equity'].iloc[0]
    df['cumulative_return_pct'] = ((df['equity'] - initial_equity) / initial_equity) * 100

    fig = go.Figure()

    # Cumulative returns line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_return_pct'],
        mode='lines',
        name='Cumulative Return',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Cumulative Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_portfolio_allocation(allocations):
    """Plot portfolio allocation pie chart"""
    if not allocations:
        return None

    names = list(allocations.keys())
    values = list(allocations.values())

    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Set3)
    )])

    fig.update_layout(
        title="Portfolio Allocation",
        template='plotly_white'
    )

    return fig


# =======================
# SIDEBAR NAVIGATION
# =======================

st.sidebar.title("📈 AI Trading Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Generate Strategies", "Backtest", "Paper Trading", "Portfolio Optimizer", "AI Learning", "🤖 Autonomous Agent"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-powered trading strategy platform with backtesting and portfolio optimization")


# =======================
# DASHBOARD PAGE
# =======================

if page == "Dashboard":
    st.markdown('<div class="main-header">📊 Trading Platform Dashboard</div>', unsafe_allow_html=True)

    # Fetch dashboard data
    analytics = make_api_request("/analytics/dashboard")

    if analytics:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Strategies",
                analytics['summary']['total_strategies'],
                delta=f"{analytics['summary']['active_strategies']} active"
            )

        with col2:
            st.metric(
                "Backtests Run",
                analytics['summary']['total_backtests']
            )

        with col3:
            st.metric(
                "Paper Trades",
                analytics['summary']['total_paper_trades']
            )

        with col4:
            if analytics.get('best_strategy'):
                st.metric(
                    "Best Strategy Score",
                    f"{analytics['best_strategy']['quality_score']:.1f}/100"
                )

        st.markdown("---")

        # Best performing strategy
        if analytics.get('best_strategy'):
            st.subheader("🏆 Top Performing Strategy")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Strategy:** {analytics['best_strategy']['name']}")
                st.write(f"**Total Return:** {analytics['best_strategy']['total_return_pct']:.2f}%")

            with col2:
                st.write(f"**Sharpe Ratio:** {analytics['best_strategy']['sharpe_ratio']:.2f}")
                st.write(f"**Quality Score:** {analytics['best_strategy']['quality_score']:.1f}/100")

        st.markdown("---")

        # Recent strategies
        st.subheader("📋 Recent Strategies")

        # Add toggle to show all or just active
        show_all = st.checkbox("Show archived strategies", value=False, key="show_all_strategies")
        active_param = "false" if show_all else "true"
        strategies_data = make_api_request(f"/strategies?limit=20&active_only={active_param}")

        if strategies_data and strategies_data.get('strategies'):
            strategies_df = pd.DataFrame(strategies_data['strategies'])

            # Format tickers for display
            if 'tickers' in strategies_df.columns:
                strategies_df['tickers_display'] = strategies_df['tickers'].apply(
                    lambda x: ', '.join(x) if x and isinstance(x, list) else 'N/A'
                )
                display_columns = ['tickers_display', 'name', 'strategy_type', 'created_at', 'is_active']
            else:
                display_columns = ['name', 'strategy_type', 'created_at', 'is_active']

            st.dataframe(
                strategies_df[display_columns],
                use_container_width=True,
                column_config={
                    "tickers_display": "Tickers",
                    "name": "Strategy Name",
                    "strategy_type": "Type",
                    "created_at": "Created",
                    "is_active": "Active"
                }
            )

        # Recent backtest results
        st.markdown("---")
        st.subheader("📈 Recent Backtest Results")
        backtest_data = make_api_request("/backtest/results?limit=10")

        if backtest_data and backtest_data.get('results'):
            backtest_df = pd.DataFrame(backtest_data['results'])

            # Prepare columns to display
            display_cols = [
                'strategy_name', 'total_trades', 'total_return_pct', 'sharpe_ratio',
                'win_rate', 'max_drawdown_pct', 'quality_score'
            ]

            # Add Kelly Criterion columns if available
            if 'kelly_position_pct' in backtest_df.columns:
                display_cols.extend(['kelly_position_pct', 'kelly_risk_level'])

            st.dataframe(
                backtest_df[display_cols],
                use_container_width=True,
                column_config={
                    "kelly_position_pct": "Kelly Position %",
                    "kelly_risk_level": "Kelly Risk"
                }
            )

            # Show Kelly Criterion explanation for the best strategy
            if 'kelly_position_pct' in backtest_df.columns:
                best_kelly = backtest_df.loc[backtest_df['quality_score'].idxmax()]

                if pd.notna(best_kelly.get('kelly_position_pct')):
                    st.markdown("---")
                    st.subheader("📊 Kelly Criterion - Best Strategy")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        kelly_pct = best_kelly.get('kelly_position_pct', 0)
                        st.metric("Optimal Position Size", f"{kelly_pct:.2f}%")

                    with col2:
                        risk_level = best_kelly.get('kelly_risk_level', 'UNKNOWN')
                        st.metric("Risk Level", risk_level)

                    with col3:
                        kelly_fraction = best_kelly.get('kelly_criterion', 0)
                        st.metric("Kelly Fraction", f"{kelly_fraction:.4f}")

                    with col4:
                        # Recommendation based on Kelly position
                        if kelly_pct == 0:
                            recommendation = "❌ Skip"
                        elif kelly_pct < 2:
                            recommendation = "⚠️ Very Conservative"
                        elif kelly_pct < 5:
                            recommendation = "✅ Conservative"
                        elif kelly_pct < 10:
                            recommendation = "✅ Good"
                        else:
                            recommendation = "🔥 Excellent"

                        st.metric("Recommendation", recommendation)

                    # Explanation
                    with st.expander("ℹ️ What is Kelly Criterion?"):
                        st.markdown("""
                        **Kelly Criterion** calculates the mathematically optimal position size based on:
                        - **Win Rate**: How often the strategy wins
                        - **Win/Loss Ratio**: Average win vs average loss

                        **Interpretation**:
                        - **0%**: No edge - don't trade this strategy
                        - **2-5%**: Small edge - conservative sizing
                        - **5-10%**: Good edge - moderate sizing
                        - **10-15%**: Strong edge - aggressive sizing

                        **Safety Note**: We use "Quarter Kelly" (25% of full Kelly) to reduce risk of ruin.
                        Full Kelly can be too aggressive and lead to large drawdowns.
                        """)
            else:
                st.info("💡 Tip: Run a new backtest to see Kelly Criterion recommendations!")



# =======================
# STRATEGY GENERATION PAGE
# =======================

elif page == "Generate Strategies":
    st.markdown('<div class="main-header">🤖 AI Strategy Generator</div>', unsafe_allow_html=True)

    st.write("Generate AI-powered trading strategies based on market data and historical performance.")

    # Input form
    with st.form("strategy_generation_form"):
        st.subheader("Configuration")

        col1, col2 = st.columns(2)

        with col1:
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                value=st.session_state.tickers,
                help="Enter stock tickers separated by commas"
            )

            period = st.selectbox(
                "Historical Data Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=2
            )

        with col2:
            num_strategies = st.slider(
                "Number of Strategies",
                min_value=1,
                max_value=10,
                value=3
            )

            use_past_performance = st.checkbox(
                "Use Past Performance for Learning",
                value=True,
                help="AI will learn from historical strategy performance"
            )

        submitted = st.form_submit_button("🚀 Generate Strategies", use_container_width=True)

        if submitted:
            # Parse tickers and save to session state
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            st.session_state.tickers = tickers_input  # Save for persistence

            with st.spinner("Generating AI strategies... This may take a minute."):
                # Make API request
                response = make_api_request(
                    "/strategies/generate",
                    method="POST",
                    data={
                        "tickers": tickers,
                        "period": period,
                        "num_strategies": num_strategies,
                        "use_past_performance": use_past_performance
                    }
                )

                if response and response.get('success'):
                    st.success(f"✅ Successfully generated {response['strategies_generated']} strategies!")

                    # Display strategies
                    for strategy in response['strategies']:
                        with st.expander(f"📊 {strategy['name']}", expanded=True):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(f"**Type:** {strategy['strategy_type']}")
                                st.write(f"**Tickers:** {', '.join(strategy['tickers'])}")

                            with col2:
                                st.write(f"**Stop Loss:** {strategy['risk_management']['stop_loss_pct']}%")
                                st.write(f"**Take Profit:** {strategy['risk_management']['take_profit_pct']}%")

                            st.write(f"**Description:** {strategy['description']}")

                            # Show indicators
                            if strategy.get('indicators'):
                                st.write("**Technical Indicators:**")
                                for indicator in strategy['indicators']:
                                    st.write(f"- {indicator['name']} (Period: {indicator.get('period', 'N/A')})")

                            # Action info (buttons can't be used inside forms)
                            st.info(f"💡 **Next Steps:**\n- Go to **Backtest** page to test Strategy ID: {strategy['id']}\n- Go to **Paper Trading** page to trade Strategy ID: {strategy['id']}")


# =======================
# BACKTEST PAGE
# =======================

elif page == "Backtest":
    st.markdown('<div class="main-header">🧪 Strategy Backtesting</div>', unsafe_allow_html=True)

    # Cleanup button for zero-trade strategies
    st.markdown("### 🧹 Database Cleanup")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Remove all strategies and backtest results with 0 trades to clean up your database.")
    with col2:
        if st.button("🗑️ Delete 0% Strategies", type="secondary", use_container_width=True):
            with st.spinner("Cleaning up zero-trade strategies..."):
                response = make_api_request("/strategies/cleanup", method="POST")
                if response and response.get('success'):
                    st.success(f"✅ {response['message']}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to cleanup strategies")

    st.markdown("---")

    # Get available strategies (including inactive ones to show all)
    strategies_data = make_api_request("/strategies?active_only=false")

    if strategies_data and strategies_data.get('strategies'):
        # Format: "NVDA, AAPL - Strategy Name (ID: 1)"
        def format_strategy_name(s):
            tickers = s.get('tickers', [])
            if tickers and isinstance(tickers, list) and len(tickers) > 0:
                ticker_prefix = ', '.join(tickers) + ' - '
            else:
                ticker_prefix = ''
            return f"{ticker_prefix}{s['name']} (ID: {s['id']})"

        strategy_options = {
            format_strategy_name(s): s['id']
            for s in strategies_data['strategies']
        }

        with st.form("backtest_form"):
            st.subheader("Backtest Configuration")

            selected_strategy = st.selectbox(
                "Select Strategy",
                options=list(strategy_options.keys())
            )

            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )

            submitted = st.form_submit_button("🧪 Run Backtest", use_container_width=True)

            if submitted:
                strategy_id = strategy_options[selected_strategy]

                with st.spinner("Running backtest... This may take a minute."):
                    response = make_api_request(
                        "/backtest",
                        method="POST",
                        data={
                            "strategy_id": strategy_id,
                            "initial_capital": initial_capital
                        }
                    )

                    if response and response.get('success'):
                        st.success("✅ Backtest completed successfully!")

                        # Fetch detailed results
                        backtest_id = response['backtest_id']
                        results = make_api_request(f"/backtest/results/{backtest_id}")

                        if results:
                            # Calculate total profit in dollars
                            total_profit = initial_capital * (results['metrics']['total_return_pct'] / 100)
                            final_capital = initial_capital + total_profit

                            # Display metrics
                            st.subheader("📊 Performance Metrics")

                            col1, col2, col3, col4, col5 = st.columns(5)

                            with col1:
                                st.metric("Total Return", f"{results['metrics']['total_return_pct']:.2f}%")
                                st.metric("Total Profit", f"${total_profit:,.2f}")

                            with col2:
                                st.metric("Initial Capital", f"${initial_capital:,.2f}")
                                st.metric("Final Capital", f"${final_capital:,.2f}")

                            with col3:
                                st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
                                st.metric("Sortino Ratio", f"{results['metrics']['sortino_ratio']:.2f}")

                            with col4:
                                st.metric("Max Drawdown", f"{results['metrics']['max_drawdown_pct']:.2f}%")
                                st.metric("Profit Factor", f"{results['metrics']['profit_factor']:.2f}")

                            with col5:
                                st.metric("Total Trades", results['metrics']['total_trades'])
                                st.metric("Win Rate", f"{results['metrics']['win_rate']:.2f}%")

                            # Kelly Criterion Section
                            if 'kelly_position_pct' in results['metrics'] and results['metrics']['kelly_position_pct'] is not None:
                                st.markdown("---")
                                st.subheader("📊 Kelly Criterion - Optimal Position Sizing")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    kelly_pct = results['metrics']['kelly_position_pct']
                                    st.metric("Optimal Position Size", f"{kelly_pct:.2f}%")

                                with col2:
                                    risk_level = results['metrics'].get('kelly_risk_level', 'UNKNOWN')
                                    # Color code the risk level
                                    if risk_level in ['NO EDGE', 'EXTREME']:
                                        st.metric("Risk Level", f"🔴 {risk_level}")
                                    elif risk_level in ['VERY LOW', 'LOW']:
                                        st.metric("Risk Level", f"🟢 {risk_level}")
                                    elif risk_level in ['MODERATE', 'MODERATE-HIGH']:
                                        st.metric("Risk Level", f"🟡 {risk_level}")
                                    else:
                                        st.metric("Risk Level", f"🟠 {risk_level}")

                                with col3:
                                    kelly_fraction = results['metrics'].get('kelly_criterion', 0)
                                    st.metric("Kelly Fraction", f"{kelly_fraction:.4f}")

                                with col4:
                                    # Recommendation based on Kelly position
                                    if kelly_pct == 0:
                                        recommendation = "❌ Skip This Strategy"
                                        rec_color = "red"
                                    elif kelly_pct < 2:
                                        recommendation = "⚠️ Very Conservative"
                                        rec_color = "orange"
                                    elif kelly_pct < 5:
                                        recommendation = "✅ Conservative"
                                        rec_color = "green"
                                    elif kelly_pct < 10:
                                        recommendation = "✅ Good Edge"
                                        rec_color = "green"
                                    else:
                                        recommendation = "🔥 Excellent Edge"
                                        rec_color = "green"

                                    st.metric("Recommendation", recommendation)

                                # Explanation
                                with st.expander("ℹ️ What is Kelly Criterion?"):
                                    st.markdown(f"""
                                    **Kelly Criterion** calculates the mathematically optimal position size based on:
                                    - **Win Rate**: {results['metrics']['win_rate']:.2f}% (How often the strategy wins)
                                    - **Win/Loss Ratio**: {abs(results['metrics']['avg_win'])/abs(results['metrics']['avg_loss']):.2f} (Average win vs average loss)

                                    **For This Strategy:**
                                    - **Kelly Fraction**: {kelly_fraction:.4f} (Raw Kelly recommendation)
                                    - **Position Size**: {kelly_pct:.2f}% (Quarter Kelly for safety)
                                    - **Risk Level**: {risk_level}

                                    **Interpretation**:
                                    - **0%**: No edge - don't trade this strategy
                                    - **0-2%**: Very small edge - might skip
                                    - **2-5%**: Small edge - conservative sizing
                                    - **5-10%**: Good edge - moderate sizing
                                    - **10-15%**: Strong edge - aggressive sizing

                                    **Safety Note**: We use "Quarter Kelly" (25% of full Kelly) to reduce risk of ruin.
                                    Full Kelly can be too aggressive and lead to large drawdowns (50%+).

                                    **Mathematical Formula**: f* = (p × b - q) / b
                                    - p = win probability ({results['metrics']['win_rate']/100:.4f})
                                    - q = loss probability ({1 - results['metrics']['win_rate']/100:.4f})
                                    - b = win/loss ratio ({abs(results['metrics']['avg_win'])/abs(results['metrics']['avg_loss']):.2f})
                                    """)

                            # Charts Section
                            st.markdown("---")
                            st.subheader("📊 Performance Charts")

                            # Row 1: Equity Curve and Cumulative Returns
                            col1, col2 = st.columns(2)
                            with col1:
                                equity_fig = plot_equity_curve(results['equity_curve'])
                                if equity_fig:
                                    st.plotly_chart(equity_fig, use_container_width=True)

                            with col2:
                                cumulative_fig = plot_cumulative_returns(results['equity_curve'])
                                if cumulative_fig:
                                    st.plotly_chart(cumulative_fig, use_container_width=True)

                            # Row 2: Drawdown and Rolling Sharpe
                            col1, col2 = st.columns(2)
                            with col1:
                                dd_fig = plot_drawdown(results['equity_curve'])
                                if dd_fig:
                                    st.plotly_chart(dd_fig, use_container_width=True)

                            with col2:
                                sharpe_fig = plot_rolling_sharpe(results['equity_curve'])
                                if sharpe_fig:
                                    st.plotly_chart(sharpe_fig, use_container_width=True)

                            # Row 3: Trade Distribution and Win/Loss
                            col1, col2 = st.columns(2)
                            with col1:
                                trade_fig = plot_trade_distribution(results['trades'])
                                if trade_fig:
                                    st.plotly_chart(trade_fig, use_container_width=True)

                            with col2:
                                winloss_fig = plot_win_loss_distribution(results['trades'])
                                if winloss_fig:
                                    st.plotly_chart(winloss_fig, use_container_width=True)

                            # Row 4: Monthly Returns Heatmap (full width)
                            monthly_fig = plot_monthly_returns(results['equity_curve'])
                            if monthly_fig:
                                st.plotly_chart(monthly_fig, use_container_width=True)

                            # Trade history
                            st.subheader("📝 Trade History")
                            if results['trades']:
                                trades_df = pd.DataFrame(results['trades'])

                                # Style the dataframe for better visibility
                                def highlight_profit_loss(val):
                                    """Highlight positive values in green, negative in red"""
                                    try:
                                        num = float(val)
                                        if num > 0:
                                            return 'background-color: #1e4620; color: #4ade80'  # Green
                                        elif num < 0:
                                            return 'background-color: #4a1a1a; color: #f87171'  # Red
                                        return ''
                                    except:
                                        return ''

                                # Apply styling to profit/loss columns
                                if 'profit_loss_usd' in trades_df.columns:
                                    styled_df = trades_df.style.applymap(
                                        highlight_profit_loss,
                                        subset=['profit_loss_usd']
                                    )
                                    if 'profit_loss_pct' in trades_df.columns:
                                        styled_df = styled_df.applymap(
                                            highlight_profit_loss,
                                            subset=['profit_loss_pct']
                                        )
                                    st.dataframe(styled_df, use_container_width=True)
                                else:
                                    st.dataframe(trades_df, use_container_width=True)
    else:
        st.warning("📋 No strategies found in the database!")

        # Debug info
        if strategies_data is None:
            st.error("⚠️ API connection issue - could not fetch strategies from server")
            st.code(f"API URL: {API_BASE_URL}/strategies")
        elif strategies_data.get('strategies') == []:
            st.info("The database exists but contains 0 strategies.")

        st.write("""
        **Troubleshooting:**
        1. Check the **Dashboard** page - do you see any strategies listed there?
        2. Check the **🤖 Autonomous Agent** page - is the agent status showing strategies generated?
        3. If the agent shows strategies but they're not appearing:
           - The database might be on a different server (Railway uses ephemeral storage)
           - Try running the agent again to regenerate strategies

        **Or create strategies manually:**
        - Go to the **Generate Strategies** page to create strategies with AI
        """)


# =======================
# PAPER TRADING PAGE
# =======================

elif page == "Paper Trading":
    st.markdown('<div class="main-header">📝 Paper Trading</div>', unsafe_allow_html=True)

    st.write("Test strategies with live market data without risking real money.")

    # Get available strategies
    strategies_data = make_api_request("/strategies")

    if strategies_data and strategies_data.get('strategies'):
        # Format: "NVDA, AAPL - Strategy Name (ID: 1)"
        def format_strategy_name(s):
            tickers = s.get('tickers', [])
            if tickers and isinstance(tickers, list) and len(tickers) > 0:
                ticker_prefix = ', '.join(tickers) + ' - '
            else:
                ticker_prefix = ''
            return f"{ticker_prefix}{s['name']} (ID: {s['id']})"

        strategy_options = {
            format_strategy_name(s): s['id']
            for s in strategies_data['strategies']
        }

        # Execute paper trade
        st.subheader("Execute Paper Trade")

        selected_strategy = st.selectbox(
            "Select Strategy",
            options=list(strategy_options.keys())
        )

        if st.button("▶️ Execute Strategy", use_container_width=True):
            strategy_id = strategy_options[selected_strategy]

            with st.spinner("Executing paper trade..."):
                response = make_api_request(
                    "/paper-trading/execute",
                    method="POST",
                    data={"strategy_id": strategy_id, "auto_execute": True}
                )

                if response and response.get('success'):
                    st.success("✅ Paper trade executed!")

                    # Show actions taken
                    if response.get('actions_taken'):
                        st.subheader("Actions Taken")
                        for action in response['actions_taken']:
                            st.write(f"- {action['action']} {action['ticker']} @ ${action.get('price', 'N/A'):.2f}")

                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Portfolio Value", f"${response['portfolio_value']:,.2f}")
                    with col2:
                        st.metric("Cash", f"${response['cash']:,.2f}")
                    with col3:
                        st.metric("Total Return", f"{response['total_return_pct']:.2f}%")
    else:
        st.info("📋 No strategies available for paper trading!")
        st.write("""
        **Get started:**
        - Go to the **Generate Strategies** page to create strategies manually
        - Or go to the **🤖 Autonomous Agent** page to let AI generate strategies automatically
        """)

    st.markdown("---")

    # Current positions
    st.subheader("📊 Current Positions")
    positions_data = make_api_request("/paper-trading/positions")

    if positions_data and positions_data.get('open_positions'):
        positions_df = pd.DataFrame(positions_data['open_positions'])
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions")

    # Performance summary
    st.markdown("---")
    st.subheader("📈 Performance Summary")
    performance_data = make_api_request("/paper-trading/performance")

    if performance_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Initial Capital", f"${performance_data['initial_capital']:,.2f}")

        with col2:
            st.metric("Current Cash", f"${performance_data['current_cash']:,.2f}")

        with col3:
            st.metric("Total Trades", performance_data['total_trades'])

        with col4:
            st.metric("Win Rate", f"{performance_data['win_rate']:.2f}%")


# =======================
# PORTFOLIO OPTIMIZER PAGE
# =======================

elif page == "Portfolio Optimizer":
    st.markdown('<div class="main-header">💼 Portfolio Optimizer</div>', unsafe_allow_html=True)

    st.write("Optimize capital allocation across multiple strategies.")

    # Get strategies with backtest results
    strategies_data = make_api_request("/strategies")

    if strategies_data and strategies_data.get('strategies'):
        st.subheader("Select Strategies to Include")

        # Multi-select for strategies
        # Format: "NVDA, AAPL - Strategy Name"
        def format_strategy_name_simple(s):
            tickers = s.get('tickers', [])
            if tickers and isinstance(tickers, list) and len(tickers) > 0:
                ticker_prefix = ', '.join(tickers) + ' - '
            else:
                ticker_prefix = ''
            return f"{ticker_prefix}{s['name']}"

        strategy_options = {
            format_strategy_name_simple(s): s['id']
            for s in strategies_data['strategies']
        }
        selected_strategies = st.multiselect(
            "Strategies",
            options=list(strategy_options.keys()),
            default=list(strategy_options.keys())[:3] if len(strategy_options) >= 3 else list(strategy_options.keys())
        )

        col1, col2 = st.columns(2)

        with col1:
            total_capital = st.number_input(
                "Total Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )

        with col2:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["sharpe", "min_variance", "max_return", "risk_parity"],
                format_func=lambda x: {
                    "sharpe": "Maximize Sharpe Ratio",
                    "min_variance": "Minimize Variance",
                    "max_return": "Maximize Return",
                    "risk_parity": "Risk Parity"
                }[x]
            )

        if st.button("🎯 Optimize Portfolio", use_container_width=True):
            if not selected_strategies:
                st.warning("Please select at least one strategy")
            else:
                strategy_ids = [strategy_options[name] for name in selected_strategies]

                with st.spinner("Optimizing portfolio..."):
                    response = make_api_request(
                        "/portfolio/optimize",
                        method="POST",
                        data={
                            "strategy_ids": strategy_ids,
                            "total_capital": total_capital,
                            "method": optimization_method
                        }
                    )

                    if response and response.get('success'):
                        st.success("✅ Portfolio optimized successfully!")

                        # Display results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Expected Return", f"{response['expected_return']:.2f}%")

                        with col2:
                            st.metric("Expected Volatility", f"{response['expected_volatility']:.2f}%")

                        with col3:
                            st.metric("Expected Sharpe", f"{response['expected_sharpe']:.2f}")

                        # Allocation chart
                        st.subheader("📊 Portfolio Allocation")
                        allocation_fig = plot_portfolio_allocation(response['allocations'])
                        if allocation_fig:
                            st.plotly_chart(allocation_fig, use_container_width=True)

                        # Detailed allocations
                        st.subheader("💰 Capital Allocations")
                        if response.get('strategies'):
                            allocation_df = pd.DataFrame(response['strategies'])
                            st.dataframe(allocation_df, use_container_width=True)
    else:
        st.info("📋 No strategies available for portfolio optimization!")
        st.write("""
        **Get started:**
        - Go to the **Generate Strategies** page to create strategies manually
        - Or go to the **🤖 Autonomous Agent** page to let AI generate strategies automatically
        """)


# =======================
# AI LEARNING PAGE
# =======================

elif page == "AI Learning":
    st.markdown('<div class="main-header">🧠 AI Learning & Insights</div>', unsafe_allow_html=True)

    st.write("AI analyzes strategy performance and extracts insights for continuous improvement.")

    # Trigger learning
    if st.button("🔍 Analyze & Learn from Recent Results", use_container_width=True):
        with st.spinner("AI is analyzing results..."):
            response = make_api_request("/ai/learn", method="POST")

            if response and response.get('success'):
                st.success("✅ Learning complete!")

                insights = response.get('insights', {})

                # Success patterns
                if insights.get('success_patterns'):
                    st.subheader("✅ Success Patterns")
                    for pattern in insights['success_patterns']:
                        with st.expander(pattern.get('pattern', 'Pattern'), expanded=True):
                            st.write(f"**Evidence:** {pattern.get('evidence', 'N/A')}")
                            st.write(f"**Confidence:** {pattern.get('confidence', 0) * 100:.1f}%")

                # Failure patterns
                if insights.get('failure_patterns'):
                    st.subheader("❌ Failure Patterns to Avoid")
                    for pattern in insights['failure_patterns']:
                        with st.expander(pattern.get('pattern', 'Pattern')):
                            st.write(f"**Evidence:** {pattern.get('evidence', 'N/A')}")
                            st.write(f"**Confidence:** {pattern.get('confidence', 0) * 100:.1f}%")

                # Recommendations
                if insights.get('recommendations_for_next_generation'):
                    st.subheader("💡 Recommendations for Next Generation")
                    for rec in insights['recommendations_for_next_generation']:
                        st.write(f"- {rec}")

    st.markdown("---")

    # Historical insights
    st.subheader("📚 Historical Learning Insights")
    learning_data = make_api_request("/ai/learning?limit=20")

    if learning_data and learning_data.get('insights'):
        for insight in learning_data['insights']:
            with st.expander(f"{insight['learning_type']} - {insight['created_at'][:10]}"):
                st.write(f"**Description:** {insight['description']}")
                st.write(f"**Confidence:** {insight['confidence_score'] * 100:.1f}%")

                if insight.get('recommendations'):
                    st.write("**Recommendations:**")
                    for rec in insight['recommendations']:
                        st.write(f"- {rec}")


# =======================
# AUTONOMOUS AGENT PAGE
# =======================

elif page == "🤖 Autonomous Agent":
    st.markdown('<div class="main-header">🤖 Autonomous Learning Agent</div>', unsafe_allow_html=True)

    st.write("""
    The Autonomous Learning Agent continuously generates, tests, and improves trading strategies **automatically** in the background.

    **How it works:**
    1. 🎯 Generates new strategies based on past learnings every X hours
    2. 🧪 Backtests them automatically
    3. 📊 Analyzes results and learns what works
    4. 🗑️ Archives poor performers (below quality threshold)
    5. 🔄 Repeats forever, getting smarter over time
    """)

    st.markdown("---")

    # Get status
    status_data = make_api_request("/autonomous/status")

    if status_data:
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status = "🟢 Running" if status_data['is_running'] else "🔴 Stopped"
            st.metric("Status", status)

        with col2:
            st.metric("Total Cycles", status_data['statistics']['total_cycles'])

        with col3:
            st.metric("Strategies Generated", status_data['statistics']['strategies_generated'])

        with col4:
            last_cycle = status_data['statistics']['last_cycle']
            if last_cycle:
                last_time = datetime.fromisoformat(last_cycle).strftime("%H:%M %m/%d")
                st.metric("Last Cycle", last_time)
            else:
                st.metric("Last Cycle", "Never")

        # Show current configuration if running
        if status_data['is_running'] and status_data.get('current_config'):
            st.markdown("---")
            st.subheader("📊 Current Configuration")
            config = status_data['current_config']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                tickers = config.get('tickers', [])
                st.markdown(f"**Tickers:** `{', '.join(tickers)}`")

            with col2:
                strategies = config.get('strategies_per_cycle', 'N/A')
                st.markdown(f"**Strategies/Cycle:** `{strategies}`")

            with col3:
                quality = config.get('min_quality_score', 'N/A')
                st.markdown(f"**Min Quality:** `{quality}`")

            with col4:
                if config.get('interval_hours'):
                    interval = f"{config['interval_hours']}h"
                elif config.get('interval_minutes'):
                    interval = f"{config['interval_minutes']}m"
                else:
                    interval = "N/A"
                st.markdown(f"**Interval:** `{interval}`")

        st.markdown("---")

        # Control Panel
        st.subheader("⚙️ Control Panel")

        col1, col2 = st.columns(2)

        with col1:
            # Start Agent
            st.write("**Start Autonomous Agent**")

            # Interval type selection (outside form for dynamic updates)
            interval_type = st.radio(
                "Interval Type",
                ["Hours", "Minutes"],
                horizontal=True,
                help="Choose whether to set interval in hours or minutes"
            )

            with st.form("start_agent_form"):
                tickers_input = st.text_input(
                    "Tickers (comma-separated)",
                    value=st.session_state.tickers,
                    help="Which stocks to focus on"
                )

                # Show appropriate slider based on selection
                if interval_type == "Hours":
                    interval_hours = st.slider(
                        "Learning Interval (hours)",
                        min_value=1,
                        max_value=24,
                        value=6,
                        help="How often to run learning cycles"
                    )
                    interval_minutes = None
                    interval_display = f"{interval_hours} hours"
                else:
                    interval_minutes = st.slider(
                        "Learning Interval (minutes)",
                        min_value=1,
                        max_value=120,
                        value=30,
                        help="How often to run learning cycles (1-120 minutes)"
                    )
                    interval_hours = None
                    interval_display = f"{interval_minutes} minutes"

                strategies_per_cycle = st.slider(
                    "Strategies Per Cycle",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="How many strategies to generate each cycle"
                )

                start_button = st.form_submit_button("▶️ Start Agent", use_container_width=True)

                if start_button:
                    tickers = [t.strip().upper() for t in tickers_input.split(",")]
                    st.session_state.tickers = tickers_input  # Save for persistence

                    with st.spinner("Starting autonomous agent..."):
                        response = make_api_request(
                            "/autonomous/start",
                            method="POST",
                            data={
                                "tickers": tickers,
                                "interval_hours": interval_hours,
                                "interval_minutes": interval_minutes,
                                "strategies_per_cycle": strategies_per_cycle
                            }
                        )

                        if response and response.get('success'):
                            st.success(f"✅ {response['message']}")
                            st.info(f"Agent will run every {interval_display}, generating {strategies_per_cycle} strategies per cycle")
                            st.rerun()
                        elif response:
                            st.error(f"❌ {response.get('message', 'Failed to start agent')}")
                        else:
                            st.error("❌ Failed to start agent - no response from server")

        with col2:
            # Manual Controls
            st.write("**Manual Controls**")

            # Stop Agent button
            if st.button("⏹️ Stop Agent", use_container_width=True, type="secondary"):
                with st.spinner("Stopping autonomous agent..."):
                    response = make_api_request("/autonomous/stop", method="POST")

                    if response and response.get('success'):
                        st.success("✅ " + response['message'])
                        st.info("The agent will finish its current cycle and then stop.")
                        time.sleep(2)
                        st.rerun()
                    elif response:
                        st.warning(f"⚠️ {response.get('message', 'Agent is not running')}")
                    else:
                        st.error("❌ Failed to stop agent - no response from server")

            st.write("")  # Spacer

            # Trigger cycle button
            st.write("Trigger one learning cycle manually")
            if st.button("⚡ Trigger One Cycle Now", use_container_width=True):
                with st.spinner("Running learning cycle..."):
                    response = make_api_request("/autonomous/trigger", method="POST")

                    if response and response.get('success'):
                        st.success("✅ Learning cycle started! Check status in a few minutes.")
                    elif response:
                        st.error(f"❌ {response.get('message', 'Failed to trigger cycle')}")
                    else:
                        st.error("❌ Failed to trigger cycle - no response from server")

        st.markdown("---")

        # Recent Cycles
        st.subheader("📜 Recent Learning Cycles")

        if status_data.get('recent_cycles'):
            cycles_df = pd.DataFrame(status_data['recent_cycles'])
            cycles_df['completed_at'] = pd.to_datetime(cycles_df['completed_at']).dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(
                cycles_df[['cycle_id', 'completed_at', 'strategies_tested', 'confidence_score']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No learning cycles yet. Start the agent or trigger a manual cycle to begin!")

        st.markdown("---")

        # Explanation
        with st.expander("ℹ️ How does it work?"):
            st.write("""
            **The Autonomous Learning Loop:**

            1. **Generate** - AI creates new trading strategies based on:
               - Recent market data
               - Past strategy performance
               - Previous learning insights

            2. **Backtest** - Each strategy is tested on 1 year of historical data
               - Calculates Sharpe ratio, returns, drawdown, etc.
               - Assigns quality score (0-100)

            3. **Learn** - AI analyzes results to identify:
               - What made successful strategies work
               - Common failures to avoid
               - Optimal parameter ranges

            4. **Archive** - Strategies below quality threshold are deactivated
               - Keeps database clean
               - Focuses on high performers

            5. **Repeat** - Process runs automatically every X hours
               - Continuously improving
               - Adapting to market changes

            **Result:** Over time, the AI gets better at generating winning strategies without any manual intervention!
            """)


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("AI Trading Platform v2.0")
st.sidebar.caption("Built with Streamlit + FastAPI + OpenAI")
