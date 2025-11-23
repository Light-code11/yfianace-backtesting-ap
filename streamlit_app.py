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
from strategy_visualizer import StrategyVisualizer

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
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Show detailed error for HTTP errors
        error_detail = ""
        try:
            error_detail = response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        st.error(f"BACKTEST ERROR: {response.status_code}: {error_detail}")
        return {"success": False, "error": error_detail}
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
    ["Dashboard", "Generate Strategies", "Backtest", "Paper Trading", "Portfolio Optimizer", "🤖 ML Predictions", "📊 Market Regimes", "🎯 Complete Trading System", "AI Learning", "🤖 Autonomous Agent"]
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

                            # DEBUG: Show what Kelly fields are in the response
                            with st.expander("🔍 Debug: API Response Kelly Fields"):
                                st.write("Kelly fields in API response:")
                                kelly_fields = {
                                    'kelly_criterion': results['metrics'].get('kelly_criterion', 'NOT FOUND'),
                                    'kelly_position_pct': results['metrics'].get('kelly_position_pct', 'NOT FOUND'),
                                    'kelly_risk_level': results['metrics'].get('kelly_risk_level', 'NOT FOUND')
                                }
                                st.json(kelly_fields)

                                if 'kelly_position_pct' not in results['metrics']:
                                    st.error("❌ Kelly fields are MISSING from API response. Railway may not have deployed the latest code yet.")
                                elif results['metrics']['kelly_position_pct'] is None:
                                    st.warning("⚠️ Kelly fields exist but are NULL. This is an old backtest. Run a NEW backtest.")
                                else:
                                    st.success("✅ Kelly data found! Section should appear below.")

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

                            # Advanced Risk Metrics Section
                            st.markdown("---")
                            st.subheader("📊 Advanced Risk Metrics")

                            # Check if advanced metrics are available
                            metrics = results['metrics']
                            has_advanced = 'var_95_pct' in metrics

                            if has_advanced:
                                # Value at Risk
                                st.markdown("### 📉 Value at Risk (VaR)")
                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    var_95 = metrics.get('var_95_pct', 0)
                                    st.metric("95% VaR", f"{var_95:.2f}%")
                                    st.caption("95% chance loss won't exceed this")

                                with col2:
                                    cvar_95 = metrics.get('cvar_95_pct', 0)
                                    st.metric("95% CVaR", f"{cvar_95:.2f}%")
                                    st.caption("Expected loss if VaR breached")

                                with col3:
                                    # Show interpretation
                                    if var_95 > 5:
                                        st.warning(f"⚠️ High risk: potential {var_95:.1f}% daily loss")
                                    elif var_95 > 3:
                                        st.info(f"Moderate risk: {var_95:.1f}% potential loss")
                                    else:
                                        st.success(f"✅ Low risk: {var_95:.1f}% max loss")

                                # Risk-Adjusted Returns
                                st.markdown("### 📈 Risk-Adjusted Performance")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    sortino = metrics.get('sortino_ratio', 0)
                                    st.metric("Sortino Ratio", f"{sortino:.2f}")
                                    st.caption("Return / Downside risk")

                                with col2:
                                    calmar = metrics.get('calmar_ratio', 0)
                                    st.metric("Calmar Ratio", f"{calmar:.2f}")
                                    st.caption("Return / Max drawdown")

                                with col3:
                                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                                    st.caption("Return / Total risk")

                                with col4:
                                    # Best ratio indicator
                                    best_ratio = max(sortino, calmar, metrics['sharpe_ratio'])
                                    if best_ratio > 2:
                                        st.success("🔥 Excellent!")
                                    elif best_ratio > 1:
                                        st.info("✅ Good")
                                    else:
                                        st.warning("⚠️ Poor")

                                # Drawdown Analysis
                                st.markdown("### 💧 Drawdown Analysis")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    ulcer = metrics.get('ulcer_index', 0)
                                    st.metric("Ulcer Index", f"{ulcer:.2f}")
                                    st.caption("DD depth + duration")

                                with col2:
                                    pain = metrics.get('pain_index', 0)
                                    st.metric("Pain Index", f"{pain:.2f}%")
                                    st.caption("Average drawdown")

                                with col3:
                                    max_dd_days = metrics.get('max_dd_duration_days', 0)
                                    st.metric("Max DD Duration", f"{max_dd_days} days")
                                    st.caption("Longest underwater period")

                                with col4:
                                    time_underwater = metrics.get('time_underwater_pct', 0)
                                    st.metric("Time Underwater", f"{time_underwater:.1f}%")
                                    st.caption("% of time in drawdown")

                                # Tail Risk
                                st.markdown("### 🎲 Tail Risk & Distribution")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    skewness = metrics.get('skewness', 0)
                                    st.metric("Skewness", f"{skewness:.3f}")
                                    if skewness < -0.5:
                                        st.caption("🔴 Negative skew (bad)")
                                    elif skewness > 0.5:
                                        st.caption("🟢 Positive skew (good)")
                                    else:
                                        st.caption("⚪ Neutral")

                                with col2:
                                    kurtosis = metrics.get('kurtosis', 0)
                                    st.metric("Kurtosis", f"{kurtosis:.3f}")
                                    if kurtosis > 3:
                                        st.caption("⚠️ Fat tails (extreme events)")
                                    else:
                                        st.caption("✅ Normal distribution")

                                with col3:
                                    max_win_streak = metrics.get('max_win_streak', 0)
                                    st.metric("Max Win Streak", f"{max_win_streak}")
                                    st.caption("Consecutive wins")

                                with col4:
                                    max_loss_streak = metrics.get('max_loss_streak', 0)
                                    st.metric("Max Loss Streak", f"{max_loss_streak}")
                                    st.caption("Consecutive losses")

                                # Interpretation Guide
                                with st.expander("ℹ️ How to interpret these metrics"):
                                    st.markdown("""
                                    **Value at Risk (VaR)**:
                                    - 95% VaR: There's only a 5% chance your daily loss will exceed this amount
                                    - Lower is better (less risk)

                                    **Conditional VaR (CVaR)**:
                                    - If your loss exceeds VaR, CVaR is the expected loss
                                    - Also called Expected Shortfall
                                    - Always >= VaR

                                    **Sortino Ratio**:
                                    - Like Sharpe, but only penalizes downside volatility
                                    - Higher is better (>2 is excellent, >1 is good)

                                    **Calmar Ratio**:
                                    - Return divided by maximum drawdown
                                    - Higher is better (>3 is excellent)

                                    **Ulcer Index**:
                                    - Measures both depth and duration of drawdowns
                                    - Lower is better (less painful drawdowns)

                                    **Pain Index**:
                                    - Average drawdown over the period
                                    - Lower is better

                                    **Skewness**:
                                    - Negative: More left-tail risk (large losses)
                                    - Positive: More right-tail potential (large wins)
                                    - Near 0: Symmetric distribution

                                    **Kurtosis**:
                                    - > 0: Fatter tails than normal (more extreme events)
                                    - < 0: Thinner tails than normal
                                    - = 0: Normal distribution

                                    **Time Underwater**:
                                    - % of time spent below previous peak
                                    - Lower is better (faster recovery)
                                    """)
                            else:
                                st.info("💡 Advanced risk metrics will be available after running this backtest again with the latest code.")

                            # Strategy Explanation Section
                            st.markdown("---")
                            st.subheader("📋 Strategy Details")

                            # Fetch strategy details
                            strategy_details = make_api_request(f"/backtest/strategy-details/{backtest_id}")

                            if strategy_details and not strategy_details.get('error'):
                                with st.expander("🔍 What is this strategy doing?", expanded=True):
                                    st.code(strategy_details['description'], language='text')

                                # Strategy visualization with buy/sell signals
                                if strategy_details.get('trades') and strategy_details.get('tickers_tested'):
                                    st.subheader("📈 Strategy Signals on Chart")

                                    # Let user select ticker if multiple
                                    tickers = strategy_details['tickers_tested']
                                    if len(tickers) > 1:
                                        selected_ticker = st.selectbox("Select ticker to visualize", tickers)
                                    else:
                                        selected_ticker = tickers[0]

                                    # Generate chart
                                    try:
                                        strategy_config = strategy_details['strategy_config']
                                        trades = strategy_details['trades']

                                        chart_fig = StrategyVisualizer.create_strategy_chart(
                                            ticker=selected_ticker,
                                            strategy=strategy_config,
                                            trades=trades,
                                            period='1y'
                                        )

                                        if chart_fig:
                                            st.plotly_chart(chart_fig, use_container_width=True)

                                            st.info("""
                                            **How to read this chart:**
                                            - 🟢 **Green triangles** = BUY signals (where strategy entered)
                                            - 🔴 **Red triangles** = SELL signals (where strategy exited)
                                            - Lines show technical indicators used by the strategy
                                            - Compare signals to price movement to see where it worked/failed
                                            """)
                                        else:
                                            st.warning(f"Could not generate chart for {selected_ticker}")
                                    except Exception as e:
                                        st.error(f"Error generating strategy chart: {str(e)}")

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
    st.markdown('<div class="main-header">💼 Advanced Portfolio Optimizer</div>', unsafe_allow_html=True)

    st.write("""
    **Optimize capital allocation** across multiple strategies using advanced mathematical optimization.

    Instead of picking one strategy, combine multiple strategies for:
    - 📈 Better risk-adjusted returns (higher Sharpe ratio)
    - 📉 Lower volatility through diversification
    - 🛡️ Reduced drawdowns when strategies are uncorrelated
    """)

    # Get strategies with backtest results
    strategies_data = make_api_request("/strategies")

    if strategies_data and strategies_data.get('strategies'):
        st.markdown("---")
        st.subheader("🎯 Step 1: Select Strategies")

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

        if len(strategy_options) < 2:
            st.warning("⚠️ Need at least 2 strategies with backtest results for portfolio optimization.")
            st.info("Go to **Backtest** page to create backtest results for your strategies.")
        else:
            selected_strategies = st.multiselect(
                "Select 2-5 strategies to combine (choose strategies with different types for better diversification)",
                options=list(strategy_options.keys()),
                default=list(strategy_options.keys())[:min(3, len(strategy_options))],
                help="Tip: Mix different strategy types (momentum + mean reversion + breakout) for lower correlation"
            )

            st.markdown("---")
            st.subheader("⚙️ Step 2: Configure Optimization")

            col1, col2, col3 = st.columns(3)

            with col1:
                total_capital = st.number_input(
                    "Total Capital ($)",
                    min_value=1000,
                    max_value=10000000,
                    value=100000,
                    step=10000,
                    help="Total capital to allocate across selected strategies"
                )

            with col2:
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["max_sharpe", "min_volatility", "max_return", "risk_parity"],
                    format_func=lambda x: {
                        "max_sharpe": "🏆 Maximize Sharpe Ratio",
                        "min_volatility": "🛡️ Minimize Risk",
                        "max_return": "📈 Maximize Returns",
                        "risk_parity": "⚖️ Risk Parity"
                    }[x],
                    help="max_sharpe = best risk-adjusted returns (recommended)"
                )

            with col3:
                max_allocation = st.slider(
                    "Max Allocation per Strategy",
                    min_value=10,
                    max_value=100,
                    value=40,
                    step=5,
                    help="Maximum % of capital in any single strategy (prevents over-concentration)"
                )

            # Method explanation
            with st.expander("ℹ️ What do these optimization methods do?"):
                st.markdown("""
                **🏆 Maximize Sharpe Ratio** (Recommended)
                - Best risk-adjusted returns
                - Balances return vs volatility
                - Good for most portfolios

                **🛡️ Minimize Volatility**
                - Lowest risk portfolio
                - Favors stable, low-volatility strategies
                - Good for conservative investors

                **📈 Maximize Returns**
                - Highest expected returns
                - May have higher volatility
                - Good for aggressive investors

                **⚖️ Risk Parity**
                - Equal risk contribution from each strategy
                - Balanced approach
                - Good when all strategies are quality
                """)

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
                            "method": optimization_method,
                            "constraints": {
                                "max_allocation": max_allocation / 100  # Convert % to decimal
                            }
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
# ML PREDICTIONS PAGE
# =======================

elif page == "🤖 ML Predictions":
    st.markdown('<div class="main-header">🤖 Machine Learning Price Predictions</div>', unsafe_allow_html=True)

    st.write("""
    **Train XGBoost models** to predict next-day price movements using 50+ technical indicators.

    **How it works:**
    - 📊 Trains on historical price data + technical indicators
    - 🎯 Predicts UP or DOWN for next trading day
    - 📈 Provides confidence scores for each prediction
    - 🔍 Shows which features matter most (feature importance)
    """)

    tab1, tab2, tab3 = st.tabs(["📚 Train Model", "🔮 Get Predictions", "📊 Trained Models"])

    # =======================
    # TAB 1: TRAIN MODEL
    # =======================
    with tab1:
        st.subheader("Train ML Model")

        col1, col2, col3 = st.columns(3)

        with col1:
            train_ticker = st.text_input(
                "Ticker Symbol",
                value="NVDA",
                help="Stock ticker to train model on"
            ).upper()

        with col2:
            train_period = st.selectbox(
                "Training Period",
                ["1y", "2y", "3y", "5y"],
                index=1,
                help="More data = better model, but slower training"
            )

        with col3:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="% of data reserved for testing model performance"
            )

        st.markdown("---")

        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner(f"Training XGBoost model for {train_ticker}... This may take 30-60 seconds..."):
                response = make_api_request(
                    "/ml/train",
                    method="POST",
                    data={
                        "ticker": train_ticker,
                        "period": train_period,
                        "test_size": test_size / 100,
                        "horizon": 1
                    }
                )

                if response and response.get('success'):
                    st.success(f"✅ Model trained successfully for {train_ticker}!")

                    # Display metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📊 Training Metrics")
                        train_metrics = response['train_metrics']
                        st.metric("Accuracy", f"{train_metrics['accuracy'] * 100:.1f}%")
                        st.metric("Precision", f"{train_metrics['precision'] * 100:.1f}%")
                        st.metric("Recall", f"{train_metrics['recall'] * 100:.1f}%")
                        st.metric("F1 Score", f"{train_metrics['f1_score']:.3f}")
                        if train_metrics['roc_auc'] > 0:
                            st.metric("ROC AUC", f"{train_metrics['roc_auc']:.3f}")

                    with col2:
                        st.markdown("### 🎯 Test Metrics (Out-of-Sample)")
                        test_metrics = response['test_metrics']
                        st.metric("Accuracy", f"{test_metrics['accuracy'] * 100:.1f}%")
                        st.metric("Precision", f"{test_metrics['precision'] * 100:.1f}%")
                        st.metric("Recall", f"{test_metrics['recall'] * 100:.1f}%")
                        st.metric("F1 Score", f"{test_metrics['f1_score']:.3f}")
                        if test_metrics['roc_auc'] > 0:
                            st.metric("ROC AUC", f"{test_metrics['roc_auc']:.3f}")

                    # Dataset info
                    st.markdown("### 📈 Dataset Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", response['samples']['total'])
                    with col2:
                        st.metric("Training Samples", response['samples']['train'])
                    with col3:
                        st.metric("Test Samples", response['samples']['test'])

                    # Class balance
                    st.markdown("### ⚖️ Class Balance")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Train: UP days", f"{response['class_balance']['train_positive_pct']:.1f}%")
                    with col2:
                        st.metric("Test: UP days", f"{response['class_balance']['test_positive_pct']:.1f}%")

                    # Feature importance
                    st.markdown("### 🔍 Top 10 Most Important Features")
                    features_df = pd.DataFrame(response['top_features'])

                    import plotly.express as px
                    fig = px.bar(
                        features_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance",
                        labels={'importance': 'Importance Score', 'feature': 'Feature'}
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation guide
                    with st.expander("ℹ️ How to interpret these metrics"):
                        st.markdown("""
                        **Accuracy**: % of correct predictions (UP vs DOWN)
                        - 50% = random guessing
                        - 60%+ = decent model
                        - 70%+ = strong model

                        **Precision**: When model says UP, how often is it right?
                        - Higher = fewer false positives

                        **Recall**: Of all actual UP days, how many did model catch?
                        - Higher = fewer missed opportunities

                        **F1 Score**: Balanced metric (0.0-1.0, higher is better)

                        **ROC AUC**: Overall classifier quality (0.5-1.0)
                        - 0.5 = random
                        - 0.7+ = good
                        - 0.8+ = excellent

                        **⚠️ Important**: Test metrics matter more than train metrics!
                        - If train accuracy >> test accuracy → overfitting (model memorized training data)
                        - Look for test accuracy 55%+ to beat random chance
                        """)

                elif response:
                    st.error(f"❌ Training failed: {response.get('error', 'Unknown error')}")
                else:
                    st.error("❌ Failed to connect to API")

    # =======================
    # TAB 2: GET PREDICTIONS
    # =======================
    with tab2:
        st.subheader("Get Price Prediction")

        predict_ticker = st.text_input(
            "Ticker Symbol for Prediction",
            value="NVDA",
            help="Must have a trained model for this ticker"
        ).upper()

        if st.button("🔮 Get Prediction", use_container_width=True):
            with st.spinner(f"Getting prediction for {predict_ticker}..."):
                response = make_api_request(f"/ml/predict/{predict_ticker}")

                if response and response.get('success'):
                    st.success("✅ Prediction generated!")

                    # Main prediction
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Price", f"${response['current_price']:.2f}")

                    with col2:
                        prediction = response['prediction']
                        delta_color = "normal" if prediction == "UP" else "inverse"
                        st.metric("Prediction", prediction, delta=prediction)

                    with col3:
                        confidence = response['confidence']['confidence_score']
                        st.metric("Confidence", f"{confidence * 100:.1f}%")

                    # Probability breakdown
                    st.markdown("### 📊 Probability Breakdown")
                    col1, col2 = st.columns(2)

                    with col1:
                        up_prob = response['confidence']['up_probability']
                        st.metric("UP Probability", f"{up_prob * 100:.1f}%")

                    with col2:
                        down_prob = response['confidence']['down_probability']
                        st.metric("DOWN Probability", f"{down_prob * 100:.1f}%")

                    # Visual gauge
                    import plotly.graph_objects as go

                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = up_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "UP Probability"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "green" if up_prob > 0.5 else "red"},
                            'steps' : [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))

                    st.plotly_chart(fig, use_container_width=True)

                    # Trading recommendation
                    st.markdown("### 💡 Trading Recommendation")
                    if confidence > 0.7:
                        if prediction == "UP":
                            st.success(f"🟢 **STRONG BUY SIGNAL** - Model predicts UP with {confidence*100:.1f}% confidence")
                        else:
                            st.error(f"🔴 **STRONG SELL SIGNAL** - Model predicts DOWN with {confidence*100:.1f}% confidence")
                    elif confidence > 0.6:
                        if prediction == "UP":
                            st.info(f"🟡 **MODERATE BUY** - Model predicts UP with {confidence*100:.1f}% confidence")
                        else:
                            st.warning(f"🟡 **MODERATE SELL** - Model predicts DOWN with {confidence*100:.1f}% confidence")
                    else:
                        st.warning(f"⚪ **LOW CONFIDENCE** - Prediction: {prediction} ({confidence*100:.1f}%). Consider waiting for higher confidence.")

                    st.info("""
                    **⚠️ Disclaimer**: ML predictions are not guarantees. Always:
                    - Use proper position sizing (Kelly Criterion)
                    - Set stop losses
                    - Diversify across strategies
                    - Only trade with capital you can afford to lose
                    """)

                elif response:
                    st.error(f"❌ Prediction failed: {response.get('error', 'Unknown error')}")
                    st.info("💡 Make sure you've trained a model for this ticker first (use the 'Train Model' tab)")
                else:
                    st.error("❌ Failed to connect to API")

    # =======================
    # TAB 3: TRAINED MODELS
    # =======================
    with tab3:
        st.subheader("Trained Models")

        response = make_api_request("/ml/models")

        if response and response.get('success'):
            models = response.get('models', [])

            if models:
                st.write(f"**{len(models)} trained models found**")

                # Create DataFrame
                models_df = pd.DataFrame(models)
                models_df['trained_at'] = pd.to_datetime(models_df['trained_at']).dt.strftime('%Y-%m-%d %H:%M')

                # Display as table
                st.dataframe(
                    models_df[['ticker', 'horizon', 'features_count', 'trained_at']],
                    use_container_width=True,
                    hide_index=True
                )

                # Delete model option
                st.markdown("---")
                st.subheader("Delete Model")

                ticker_to_delete = st.selectbox(
                    "Select model to delete",
                    options=[m['ticker'] for m in models]
                )

                if st.button(f"🗑️ Delete {ticker_to_delete} model", use_container_width=True):
                    response = make_api_request(
                        f"/ml/model/{ticker_to_delete}",
                        method="DELETE"
                    )

                    if response and response.get('success'):
                        st.success(f"✅ {response['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {response.get('error', 'Delete failed')}")
            else:
                st.info("📋 No trained models yet. Train your first model in the 'Train Model' tab!")
        else:
            st.error("❌ Failed to load models")


# =======================
# MARKET REGIMES PAGE
# =======================

elif page == "📊 Market Regimes":
    st.markdown('<div class="main-header">📊 Market Regime Detection (HMM)</div>', unsafe_allow_html=True)

    st.write("""
    **Hidden Markov Models (HMM)** automatically detect market regimes from price data.

    **Detects 3 regimes:**
    - 🟢 **BULL**: High returns, lower volatility (trending up)
    - 🔴 **BEAR**: Negative returns, higher volatility (trending down)
    - 🟡 **CONSOLIDATION**: Low returns, low volatility (sideways/choppy)

    **Why it matters:**
    - Different strategies work in different regimes
    - Momentum strategies excel in BULL markets
    - Mean reversion works better in CONSOLIDATION
    - Risk management critical in BEAR markets
    """)

    tab1, tab2, tab3 = st.tabs(["🔍 Detect Regimes", "📈 Regime History", "💡 Strategy Insights"])

    # =======================
    # TAB 1: DETECT REGIMES
    # =======================
    with tab1:
        st.subheader("Detect Current Market Regime")

        col1, col2 = st.columns(2)

        with col1:
            regime_ticker = st.text_input(
                "Ticker Symbol",
                value="NVDA",
                help="Stock ticker to analyze"
            ).upper()

        with col2:
            regime_period = st.selectbox(
                "Training Period",
                ["1y", "2y", "3y", "5y"],
                index=1,
                help="More data = more accurate regime detection"
            )

        st.markdown("---")

        if st.button("🔍 Detect Current Regime", use_container_width=True):
            with st.spinner(f"Analyzing market regimes for {regime_ticker}..."):
                # Train and predict
                response = make_api_request(f"/regime/predict/{regime_ticker}")

                if response and response.get('success'):
                    st.success("✅ Regime detected successfully!")

                    # Current regime
                    current_regime = response['current_regime']

                    # Main display
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Current Price", f"${response['current_price']:.2f}")

                    with col2:
                        regime_label = current_regime['label']

                        # Color-code regime
                        if regime_label == "BULL":
                            st.markdown(f"### 🟢 **{regime_label}**")
                        elif regime_label == "BEAR":
                            st.markdown(f"### 🔴 **{regime_label}**")
                        else:
                            st.markdown(f"### 🟡 **{regime_label}**")

                    with col3:
                        confidence = current_regime['confidence']
                        st.metric("Confidence", f"{confidence * 100:.1f}%")

                    # Regime probabilities
                    st.markdown("### 📊 Current Regime Probabilities")

                    regime_probs = response['regime_probabilities']

                    import plotly.graph_objects as go

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(regime_probs.keys()),
                            y=[v * 100 for v in regime_probs.values()],
                            marker=dict(
                                color=['green' if k == 'BULL' else 'red' if k == 'BEAR' else 'orange'
                                       for k in regime_probs.keys()]
                            ),
                            text=[f"{v*100:.1f}%" for v in regime_probs.values()],
                            textposition='auto'
                        )
                    ])

                    fig.update_layout(
                        title="Regime Probability Distribution",
                        xaxis_title="Market Regime",
                        yaxis_title="Probability (%)",
                        yaxis=dict(range=[0, 100]),
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Next period predictions
                    st.markdown("### 🔮 Next Period Likely Regimes")

                    next_probs = response['next_regime_probabilities']

                    col1, col2, col3 = st.columns(3)

                    prob_items = list(next_probs.items())
                    for i, (regime, prob) in enumerate(prob_items):
                        with [col1, col2, col3][i % 3]:
                            st.metric(f"{regime}", f"{prob*100:.1f}%")

                    # Trading insights
                    st.markdown("### 💡 Trading Insights")

                    regime_label = current_regime['label']

                    if regime_label == "BULL":
                        st.success("""
                        **🟢 BULL Market Detected**
                        - ✅ Momentum strategies likely to perform well
                        - ✅ Trend-following strategies (breakouts, moving average crossovers)
                        - ✅ Consider increasing position sizes (within Kelly limits)
                        - ⚠️ Watch for regime change signals (drop in confidence)
                        """)
                    elif regime_label == "BEAR":
                        st.error("""
                        **🔴 BEAR Market Detected**
                        - 🛡️ Focus on capital preservation
                        - 🛡️ Reduce position sizes or go to cash
                        - 🛡️ Consider inverse/short strategies if experienced
                        - 🛡️ Tighter stop losses recommended
                        - ⚠️ Avoid momentum longs (likely to fail)
                        """)
                    else:
                        st.warning("""
                        **🟡 CONSOLIDATION Market Detected**
                        - ⚖️ Mean reversion strategies work better
                        - ⚖️ Range-bound trading (buy support, sell resistance)
                        - ⚖️ Breakout strategies may generate false signals
                        - ⚖️ Reduce position sizes (choppy markets)
                        - ⚠️ Wait for regime change to BULL before aggressive longs
                        """)

                    # Statistical details
                    with st.expander("📊 Statistical Details"):
                        st.write(f"**Timestamp**: {response['timestamp']}")
                        st.write(f"**Current Regime State**: {current_regime['state']}")
                        st.write(f"**Confidence**: {current_regime['confidence']:.4f}")

                        st.markdown("**Regime Probabilities:**")
                        for regime, prob in regime_probs.items():
                            st.write(f"  - {regime}: {prob:.4f}")

                        st.markdown("**Next Period Transition Probabilities:**")
                        for regime, prob in next_probs.items():
                            st.write(f"  - {regime}: {prob:.4f}")

                elif response:
                    st.error(f"❌ Detection failed: {response.get('error', 'Unknown error')}")
                else:
                    st.error("❌ Failed to connect to API")

    # =======================
    # TAB 2: REGIME HISTORY
    # =======================
    with tab2:
        st.subheader("Regime History Over Time")

        col1, col2 = st.columns(2)

        with col1:
            history_ticker = st.text_input(
                "Ticker Symbol for History",
                value="NVDA",
                key="history_ticker",
                help="Stock ticker to view regime history"
            ).upper()

        with col2:
            history_period = st.selectbox(
                "Historical Period",
                ["6mo", "1y", "2y", "3y"],
                index=1,
                help="Period to visualize regime changes"
            )

        if st.button("📈 Show Regime History", use_container_width=True):
            with st.spinner(f"Loading regime history for {history_ticker}..."):
                response = make_api_request(f"/regime/history/{history_ticker}?period={history_period}")

                if response and response.get('success'):
                    st.success("✅ History loaded!")

                    timeline = response['timeline']
                    regime_labels_map = response['regime_labels']

                    # Create DataFrame
                    timeline_df = pd.DataFrame(timeline)
                    timeline_df['date'] = pd.to_datetime(timeline_df['date'])

                    # Create regime timeline chart
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Price with Market Regimes", "Regime Timeline")
                    )

                    # Price chart with regime coloring
                    colors = {
                        'BULL': 'green',
                        'BEAR': 'red',
                        'CONSOLIDATION': 'orange',
                        'HIGH_VOLATILITY': 'purple'
                    }

                    # Group consecutive regimes
                    timeline_df['regime_change'] = (timeline_df['regime_label'] != timeline_df['regime_label'].shift()).cumsum()

                    for _, group in timeline_df.groupby('regime_change'):
                        regime_label = group['regime_label'].iloc[0]
                        color = colors.get(regime_label, 'gray')

                        fig.add_trace(
                            go.Scatter(
                                x=group['date'],
                                y=group['price'],
                                mode='lines',
                                name=regime_label,
                                line=dict(color=color, width=2),
                                showlegend=False
                            ),
                            row=1, col=1
                        )

                    # Regime timeline (categorical)
                    regime_numeric = timeline_df['regime_state'].values

                    fig.add_trace(
                        go.Scatter(
                            x=timeline_df['date'],
                            y=regime_numeric,
                            mode='lines',
                            line=dict(color='blue', width=2),
                            fill='tozeroy',
                            showlegend=False
                        ),
                        row=2, col=1
                    )

                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="Regime State", row=2, col=1)

                    fig.update_layout(
                        height=700,
                        title=f"{history_ticker} Price and Market Regimes",
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Regime statistics
                    st.markdown("### 📊 Regime Statistics")

                    regime_counts = timeline_df['regime_label'].value_counts()
                    regime_pcts = (regime_counts / len(timeline_df) * 100).round(1)

                    col1, col2, col3 = st.columns(3)

                    for i, (regime, count) in enumerate(regime_counts.items()):
                        pct = regime_pcts[regime]
                        with [col1, col2, col3][i % 3]:
                            st.metric(regime, f"{pct}%", f"{count} days")

                    # Recent regime changes
                    st.markdown("### 🔄 Recent Regime Changes")

                    changes = timeline_df[timeline_df['regime_change'] != timeline_df['regime_change'].shift()].tail(10)

                    if len(changes) > 0:
                        changes_display = changes[['date', 'regime_label', 'price']].copy()
                        changes_display['date'] = changes_display['date'].dt.strftime('%Y-%m-%d')
                        changes_display.columns = ['Date', 'New Regime', 'Price']

                        st.dataframe(changes_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No regime changes detected in recent history")

                elif response:
                    st.error(f"❌ Failed: {response.get('error', 'Unknown error')}")
                else:
                    st.error("❌ Failed to connect to API")

    # =======================
    # TAB 3: STRATEGY INSIGHTS
    # =======================
    with tab3:
        st.subheader("Strategy Recommendations by Regime")

        st.markdown("""
        ### 🟢 BULL Market Strategies

        **Best Strategies:**
        - **Momentum**: Buy high, sell higher (trend is your friend)
        - **Breakout**: Enter on new highs with strong volume
        - **Moving Average Crossovers**: Golden cross signals
        - **Growth Stocks**: Focus on high-beta, growth names

        **Position Sizing:**
        - Can use higher Kelly fractions (closer to full Kelly)
        - Aggressive position sizes justified
        - Trailing stops to lock in gains

        **Risk Management:**
        - Watch for regime change signals
        - Don't chase extended moves
        - Take partial profits at resistance

        ---

        ### 🔴 BEAR Market Strategies

        **Best Strategies:**
        - **Cash is King**: Preserve capital, wait for BULL regime
        - **Short Selling**: If experienced (high risk!)
        - **Inverse ETFs**: Easier than shorting
        - **Defensive Sectors**: Utilities, consumer staples

        **Position Sizing:**
        - Minimal to zero exposure
        - Use quarter-Kelly or less
        - Small positions only

        **Risk Management:**
        - Tight stop losses (market can gap against you)
        - Don't try to catch falling knives
        - Wait for regime change confirmation

        ---

        ### 🟡 CONSOLIDATION Market Strategies

        **Best Strategies:**
        - **Mean Reversion**: Buy dips, sell rips
        - **Range Trading**: Support/resistance levels
        - **Iron Condors**: Options strategies for range-bound
        - **Pairs Trading**: Market-neutral approaches

        **Position Sizing:**
        - Conservative (half-Kelly recommended)
        - Many small positions vs few large ones
        - Quick profits, tight stops

        **Risk Management:**
        - Choppy markets = false breakouts
        - Scale in/out of positions
        - Avoid momentum strategies (whipsaw risk)
        - Wait for BULL regime for big bets

        ---

        ### 🎯 How to Use Regime Detection in Trading

        1. **Check Current Regime Daily**
           - Use the "Detect Regimes" tab
           - Look for high confidence (70%+)

        2. **Match Strategy to Regime**
           - BULL → Momentum strategies
           - BEAR → Cash / defensive
           - CONSOLIDATION → Mean reversion

        3. **Adjust Position Sizing**
           - BULL: Can be aggressive (within Kelly limits)
           - BEAR: Minimal exposure
           - CONSOLIDATION: Conservative

        4. **Watch for Regime Changes**
           - When regime probabilities shift (e.g., BULL → 40%, CONSOLIDATION → 35%, BEAR → 25%)
           - This signals potential regime transition
           - Reduce positions until new regime confirmed

        5. **Combine with Other Signals**
           - Regime + ML Prediction + Kelly Criterion = complete system
           - Don't rely on one signal alone
           - Confluence of multiple signals = higher conviction

        ---

        ### ⚠️ Important Notes

        - **Regimes are probabilistic**, not deterministic
        - **No regime lasts forever** - markets evolve
        - **Regime changes lag** - HMM detects regimes after they've started
        - **Use as a filter**, not the sole decision maker
        - **Retrain models** monthly as market conditions change

        ### 📚 Advanced Tips

        - Train regime models on different timeframes (daily vs weekly)
        - Compare regime detection across correlated assets (SPY, QQQ, etc.)
        - Use regime as portfolio-level filter (e.g., max 30% exposure in BEAR regime)
        - Backtest strategies separately by regime to validate performance
        """)


# =======================
# COMPLETE TRADING SYSTEM PAGE
# =======================

elif page == "🎯 Complete Trading System":
    st.markdown('<div class="main-header">🎯 Complete Trading System</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🚀 Build Your Optimal Portfolio Using All Advanced Features

    This page integrates **all** your advanced features into one powerful workflow:
    - 🤖 **ML Predictions** - Identify high-probability opportunities
    - 📊 **Market Regimes** - Understand current market conditions
    - 📈 **Strategy Backtesting** - Test multiple strategies on multiple tickers
    - 🎲 **Advanced Risk Metrics** - Evaluate risk-adjusted performance
    - 💼 **Portfolio Optimization** - Combine best strategies optimally
    - 🎯 **Kelly Criterion** - Determine optimal position sizes
    """)

    st.info("""
    **💡 Choose Your Optimization Goal:**

    **Risk-Adjusted (Sharpe)** 🛡️ - Focus on consistent, smooth returns
    - Best for: Conservative investors, retirement accounts
    - Target: Sharpe 0.6-1.0 (good to excellent)

    **Maximum Returns** 🚀 - Focus on highest total gains (accepts more volatility)
    - Best for: Aggressive traders, growth accounts
    - Target: 20-50%+ annual returns (with higher drawdowns)

    **Best Sortino** 📊 - Focus on downside risk only
    - Best for: Those who don't mind upside volatility

    **Best Calmar** 📉 - Focus on minimizing drawdowns
    - Best for: Risk-averse traders

    **Tip:** Enable "Vectorized Parameter Optimization" in Advanced Options!
    """)

    st.markdown("---")

    # Step 1: Configuration
    st.subheader("Step 1️⃣: Configure Your Analysis")

    col1, col2 = st.columns(2)

    with col1:
        tickers_input = st.text_input(
            "Tickers to Analyze (comma-separated)",
            value="SPY,QQQ,AAPL,MSFT,GOOGL",
            help="Enter multiple tickers to analyze"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        strategies = st.multiselect(
            "Select Strategies to Test",
            ["momentum", "mean_reversion", "breakout", "trend_following", "volatility"],
            default=["momentum", "mean_reversion", "breakout"],
            help="Choose which strategies to backtest on each ticker"
        )

    with col2:
        lookback_period = st.selectbox(
            "Backtest Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=3,
            help="Historical period for backtesting"
        )

        min_sharpe = st.number_input(
            "Quality Threshold",
            min_value=0.0,
            max_value=5.0,
            value=0.6,
            step=0.1,
            help="Minimum quality threshold - used for Sharpe/Sortino/Calmar depending on optimization goal. For 'Maximum Returns' mode, this is relaxed to 0.3."
        )

        total_capital = st.number_input(
            "Total Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Total portfolio capital for allocation"
        )

    # Advanced Options
    with st.expander("⚙️ Advanced Options"):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            use_vectorized = st.checkbox(
                "🚀 Use Vectorized Parameter Optimization",
                value=True,
                help="Automatically find optimal parameters for each strategy (100x faster). Highly recommended!"
            )

        with col_adv2:
            optimization_goal = st.selectbox(
                "Optimization Goal",
                ["Risk-Adjusted (Sharpe)", "Maximum Returns", "Best Sortino", "Best Calmar"],
                index=0,
                help="What to optimize for: risk-adjusted returns (Sharpe) or maximum absolute returns"
            )

        if use_vectorized:
            st.info("""
            **Vectorized Optimization Benefits:**
            - Tests 100-1000 parameter combinations per strategy
            - Finds optimal RSI period, MA lengths, Bollinger bands, etc.
            - 100-1000x faster than traditional grid search
            - Results in better-performing strategies
            - Takes only a few extra seconds
            """)

    st.markdown("---")

    # Step 2: Run Complete Analysis
    if st.button("🚀 Run Complete Analysis", use_container_width=True, type="primary"):

        if not tickers or not strategies:
            st.error("⚠️ Please select at least one ticker and one strategy")
        else:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_results = []
            total_steps = len(tickers) * (2 + len(strategies))  # ML + Regime + N strategies
            current_step = 0

            # Container for results
            results_container = st.container()

            with results_container:
                st.subheader("📊 Analysis Results")

                # Step 2: Market Analysis for each ticker
                st.markdown("### Step 2️⃣: Market Analysis")

                market_analysis = {}

                for ticker in tickers:
                    with st.expander(f"🔍 {ticker} - Market Analysis", expanded=True):
                        col1, col2 = st.columns(2)

                        # ML Prediction
                        with col1:
                            st.markdown("#### 🤖 ML Prediction")
                            status_text.text(f"Getting ML prediction for {ticker}...")

                            ml_response = make_api_request(f"/ml/predict/{ticker}")

                            # If no model exists, train it automatically
                            if ml_response and not ml_response.get('success') and "No trained model" in ml_response.get('error', ''):
                                with st.spinner(f"🔄 Training ML model for {ticker}... (30-60 seconds)"):
                                    status_text.text(f"🔄 Training ML model for {ticker}... (this may take 30-60 seconds)")

                                    # Train the model
                                    train_response = make_api_request(
                                        "/ml/train",
                                        method="POST",
                                        data={
                                            "ticker": ticker,
                                            "period": "2y",
                                            "test_size": 0.2,
                                            "horizon": 1
                                        }
                                    )

                                    if train_response and train_response.get('success'):
                                        st.info(f"✅ Model trained successfully (Accuracy: {train_response['test_metrics']['accuracy']*100:.1f}%)")
                                        # Now get prediction with newly trained model
                                        ml_response = make_api_request(f"/ml/predict/{ticker}")
                                    else:
                                        st.warning(f"⚠️ Failed to train model: {train_response.get('error', 'Unknown error')}")

                            current_step += 1
                            progress_bar.progress(current_step / total_steps)

                            if ml_response and ml_response.get('success'):
                                # Extract prediction and confidence from response
                                direction = ml_response['prediction']  # "UP" or "DOWN"
                                confidence = ml_response['confidence']['confidence_score'] * 100  # Convert to percentage

                                # Color-code prediction
                                if direction == "UP":
                                    st.success(f"📈 **{direction}** (Confidence: {confidence:.1f}%)")
                                else:
                                    st.warning(f"📉 **{direction}** (Confidence: {confidence:.1f}%)")

                                if confidence > 60:
                                    st.info("✅ High confidence signal")
                                else:
                                    st.caption("⚠️ Low confidence - proceed with caution")

                                market_analysis[ticker] = market_analysis.get(ticker, {})
                                market_analysis[ticker]['ml_prediction'] = direction
                                market_analysis[ticker]['ml_confidence'] = confidence
                            else:
                                st.error(f"❌ ML prediction failed: {ml_response.get('error', 'Unknown error')}")
                                market_analysis[ticker] = market_analysis.get(ticker, {})
                                market_analysis[ticker]['ml_prediction'] = "UNKNOWN"
                                market_analysis[ticker]['ml_confidence'] = 0

                        # Market Regime
                        with col2:
                            st.markdown("#### 📊 Market Regime")
                            status_text.text(f"Detecting market regime for {ticker}... (auto-training if needed)")

                            # Note: /regime/predict automatically trains on-demand
                            regime_response = make_api_request(f"/regime/predict/{ticker}")

                            current_step += 1
                            progress_bar.progress(current_step / total_steps)

                            if regime_response and regime_response.get('success'):
                                current_regime = regime_response['current_regime']['label'] if isinstance(regime_response.get('current_regime'), dict) else regime_response.get('current_regime', 'UNKNOWN')
                                regime_probs = regime_response.get('regime_probabilities', {})

                                # Color-code regime
                                if current_regime == "BULL":
                                    st.success(f"🟢 **{current_regime} MARKET**")
                                elif current_regime == "BEAR":
                                    st.error(f"🔴 **{current_regime} MARKET**")
                                else:
                                    st.info(f"🟡 **{current_regime} MARKET**")

                                # Show probabilities
                                for regime, prob in regime_probs.items():
                                    prob_pct = prob * 100 if prob < 1 else prob  # Handle both decimal and percentage
                                    st.caption(f"{regime}: {prob_pct:.1f}%")

                                market_analysis[ticker]['regime'] = current_regime
                                market_analysis[ticker]['regime_probs'] = regime_probs
                            else:
                                error_msg = regime_response.get('error', 'Unknown error') if regime_response else 'No response'
                                st.error(f"❌ Regime detection failed: {error_msg}")
                                market_analysis[ticker]['regime'] = "UNKNOWN"

                # Step 3: Strategy Backtesting
                st.markdown("---")
                st.markdown("### Step 3️⃣: Strategy Backtesting & Risk Analysis")

                # Optional: Vectorized Parameter Optimization
                optimization_results = {}
                if use_vectorized:
                    st.markdown("#### 🚀 Parameter Optimization (Vectorized)")
                    opt_status = st.empty()

                    for ticker in tickers:
                        for strategy in strategies:
                            opt_status.text(f"Optimizing {strategy} parameters for {ticker}...")

                            # Run vectorized optimization
                            opt_response = make_api_request(
                                "/vectorized/optimize",
                                method="POST",
                                data={
                                    "ticker": ticker,
                                    "strategy_type": strategy,
                                    "period": lookback_period
                                }
                            )

                            if opt_response and opt_response.get('success'):
                                key = f"{ticker}_{strategy}"
                                optimization_results[key] = {
                                    'optimal_params': opt_response['optimal_parameters'],
                                    'opt_metrics': opt_response['metrics'],
                                    'combinations_tested': opt_response['combinations_tested']
                                }

                    if optimization_results:
                        opt_status.success(f"✅ Optimized {len(optimization_results)} strategy configurations!")

                        # Show optimization summary
                        with st.expander("📊 View Optimization Results"):
                            opt_data = []
                            for key, data in optimization_results.items():
                                ticker_name, strategy_name = key.split('_', 1)
                                opt_data.append({
                                    'Ticker': ticker_name,
                                    'Strategy': strategy_name,
                                    'Optimal Params': str(data['optimal_params']),
                                    'Combinations Tested': data['combinations_tested'],
                                    'Optimized Sharpe': f"{data['opt_metrics'].get('sharpe_ratio', 0):.2f}"
                                })

                            st.dataframe(pd.DataFrame(opt_data), use_container_width=True, hide_index=True)
                            st.info("💡 These optimized parameters will improve strategy performance!")

                    st.markdown("---")

                backtest_results = []
                all_tested_strategies = []  # Track all strategies for debugging

                for ticker in tickers:
                    for strategy in strategies:
                        status_text.text(f"Backtesting {strategy} on {ticker}...")

                        # Build strategy_config from strategy type
                        # Get optimized parameters if available
                        opt_key = f"{ticker}_{strategy}"
                        opt_params = optimization_results.get(opt_key, {}).get('optimal_params', {})

                        # Build indicators list based on strategy type
                        if strategy == "momentum":
                            indicators = [
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50},
                                {"name": "RSI", "period": 14}
                            ]
                        elif strategy == "mean_reversion":
                            indicators = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "RSI", "period": 14}
                            ]
                        elif strategy == "breakout":
                            indicators = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "ATR", "period": 14}
                            ]
                        elif strategy == "trend_following":
                            indicators = [
                                {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50}
                            ]
                        else:
                            # Default indicators for unknown strategies
                            indicators = [
                                {"name": "SMA", "period": 20},
                                {"name": "RSI", "period": 14}
                            ]

                        # Create strategy config for backtest API
                        strategy_config = {
                            "name": f"{strategy.replace('_', ' ').title()} - {ticker}",
                            "tickers": [ticker],
                            "strategy_type": strategy,
                            "indicators": indicators,  # Now populated with correct indicators!
                            "risk_management": {
                                "stop_loss_pct": 15.0,     # Wider stops - avoid whipsaws (was 5%)
                                "take_profit_pct": 30.0,   # Let winners run (was 10%)
                                "position_size_pct": 95.0, # Use full capital (was 10%)
                                "max_positions": 1          # Single position focus
                            }
                        }

                        # Add optimized parameters if available
                        if opt_params:
                            strategy_config['optimized_params'] = opt_params

                        # Debug: Show what we're sending (can remove later)
                        # st.write(f"Debug - Sending to /backtest: {strategy_config}")

                        # Run backtest
                        response = make_api_request(
                            "/backtest",
                            method="POST",
                            data={
                                "strategy_config": strategy_config,
                                "initial_capital": 100000
                            }
                        )

                        current_step += 1
                        progress_bar.progress(current_step / total_steps)

                        if response and response.get('success'):
                            metrics = response['metrics']
                            sharpe = metrics['sharpe_ratio']

                            # Track all tested strategies for debugging
                            all_tested_strategies.append({
                                'ticker': ticker,
                                'strategy': strategy,
                                'sharpe': sharpe,
                                'return_pct': metrics['total_return_pct'],
                                'sortino': metrics.get('sortino_ratio', 0),
                                'calmar': metrics.get('calmar_ratio', 0)
                            })

                            # Filter based on optimization goal
                            include_strategy = False
                            if optimization_goal == "Risk-Adjusted (Sharpe)":
                                include_strategy = sharpe >= min_sharpe
                            elif optimization_goal == "Maximum Returns":
                                # For max returns, use much lower Sharpe threshold (or total return threshold)
                                include_strategy = sharpe >= 0.3  # Very permissive - just positive
                            elif optimization_goal == "Best Sortino":
                                include_strategy = metrics.get('sortino_ratio', 0) >= min_sharpe * 1.2  # Sortino usually higher
                            elif optimization_goal == "Best Calmar":
                                include_strategy = metrics.get('calmar_ratio', 0) >= min_sharpe * 0.8

                            if include_strategy:
                                # Get optimization results if available
                                opt_key = f"{ticker}_{strategy}"
                                opt_params = optimization_results.get(opt_key, {}).get('optimal_params', {})

                                result = {
                                    'ticker': ticker,
                                    'strategy': strategy,
                                    'strategy_id': f"{ticker}_{strategy}",
                                    'total_return_pct': metrics['total_return_pct'],
                                    'sharpe_ratio': sharpe,
                                    'sortino_ratio': metrics.get('sortino_ratio', 0),
                                    'calmar_ratio': metrics.get('calmar_ratio', 0),
                                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                                    'win_rate': metrics['win_rate'],
                                    'quality_score': metrics.get('quality_score', 0),
                                    'var_95_pct': metrics.get('var_95_pct', 0),
                                    'cvar_95_pct': metrics.get('cvar_95_pct', 0),
                                    'ulcer_index': metrics.get('ulcer_index', 0),
                                    'kelly_position_pct': metrics.get('kelly_position_pct', 0),
                                    'kelly_risk_level': metrics.get('kelly_risk_level', 'UNKNOWN'),
                                    'ml_prediction': market_analysis.get(ticker, {}).get('ml_prediction', 'UNKNOWN'),
                                    'ml_confidence': market_analysis.get(ticker, {}).get('ml_confidence', 0),
                                    'regime': market_analysis.get(ticker, {}).get('regime', 'UNKNOWN'),
                                    'optimized_params': str(opt_params) if opt_params else 'N/A'
                                }
                                backtest_results.append(result)

                status_text.text("✅ Analysis complete!")
                progress_bar.progress(1.0)

                # Store results in session state to persist across reruns
                st.session_state.complete_trading_results = backtest_results
                st.session_state.complete_trading_total_capital = total_capital
                st.session_state.complete_trading_optimization_goal = optimization_goal

    # Display results from session state (persists across button clicks)
    if 'complete_trading_results' in st.session_state and st.session_state.complete_trading_results:
        backtest_results = st.session_state.complete_trading_results
        total_capital = st.session_state.complete_trading_total_capital
        optimization_goal = st.session_state.complete_trading_optimization_goal

        if backtest_results:
            st.markdown("---")
            st.markdown(f"### Step 4️⃣: Results - {len(backtest_results)} Qualifying Strategies")

            # Create DataFrame
            results_df = pd.DataFrame(backtest_results)

            # Sort by optimization goal
            if optimization_goal == "Risk-Adjusted (Sharpe)":
                sort_column = 'sharpe_ratio'
                st.info(f"📊 Sorted by **Sharpe Ratio** (risk-adjusted returns)")
            elif optimization_goal == "Maximum Returns":
                sort_column = 'total_return_pct'
                st.info(f"🚀 Sorted by **Total Return %** (maximum absolute gains)")
            elif optimization_goal == "Best Sortino":
                sort_column = 'sortino_ratio'
                st.info(f"📈 Sorted by **Sortino Ratio** (downside risk-adjusted)")
            elif optimization_goal == "Best Calmar":
                sort_column = 'calmar_ratio'
                st.info(f"📉 Sorted by **Calmar Ratio** (drawdown-adjusted)")
            else:
                sort_column = 'sortino_ratio'  # Default

            results_df = results_df.sort_values(sort_column, ascending=False)

            # Highlight Top Performing Strategy
            st.markdown("---")
            st.markdown("### 🏆 Top Performing Strategy")

            top_strategy = results_df.iloc[0]

            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**Strategy:** {top_strategy['strategy'].title()} - {top_strategy['ticker']}")
                st.markdown(f"**Total Return:** {top_strategy['total_return_pct']:.2f}%")

            with col2:
                st.markdown(f"**Sharpe Ratio:** {top_strategy['sharpe_ratio']:.2f}")
                st.markdown(f"**Quality Score:** {top_strategy.get('quality_score', 0):.1f}/100")

            with col3:
                # Quick save button for top strategy
                if st.button("💾 Save Best Strategy", key="save_top_strategy", use_container_width=True, type="primary"):
                    with st.spinner(f"Saving {top_strategy['ticker']} - {top_strategy['strategy']}..."):
                        # Build strategy config
                        strategy_config = {
                            "name": f"{top_strategy['ticker']} - {top_strategy['strategy'].title()}",
                            "description": f"Top performing strategy from Complete Trading System (Return: {top_strategy['total_return_pct']:.2f}%, Sharpe: {top_strategy['sharpe_ratio']:.2f})",
                            "tickers": [top_strategy['ticker']],
                            "strategy_type": top_strategy['strategy'],
                            "indicators": [],
                            "risk_management": {
                                "stop_loss_pct": 15.0,
                                "take_profit_pct": 30.0,
                                "position_size_pct": 95.0,
                                "max_positions": 1
                            }
                        }

                        # Add indicators based on strategy type
                        if top_strategy['strategy'] == "momentum":
                            strategy_config["indicators"] = [
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50},
                                {"name": "RSI", "period": 14}
                            ]
                        elif top_strategy['strategy'] == "mean_reversion":
                            strategy_config["indicators"] = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "RSI", "period": 14}
                            ]
                        elif top_strategy['strategy'] == "breakout":
                            strategy_config["indicators"] = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "ATR", "period": 14}
                            ]
                        elif top_strategy['strategy'] == "trend_following":
                            strategy_config["indicators"] = [
                                {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50}
                            ]

                        # Save strategy via API
                        save_response = make_api_request(
                            "/strategies",
                            method="POST",
                            data=strategy_config
                        )

                        if save_response and save_response.get('success'):
                            strategy_id = save_response['strategy_id']
                            st.success(f"✅ Saved! Strategy ID: {strategy_id}")
                            st.info("🔄 Go to **Backtest** page to backtest this strategy")
                        else:
                            st.error(f"❌ Failed to save: {save_response.get('error', 'Unknown error')}")

            st.markdown("---")

            # Display all qualifying strategies
            st.markdown("#### 📊 All Qualifying Strategies")
            st.dataframe(
                results_df[[
                    'ticker', 'strategy', 'total_return_pct', 'sharpe_ratio',
                    'sortino_ratio', 'calmar_ratio', 'max_drawdown_pct',
                    'win_rate', 'var_95_pct', 'ml_prediction', 'regime'
                ]].style.format({
                    'total_return_pct': '{:.2f}%',
                    'sharpe_ratio': '{:.2f}',
                    'sortino_ratio': '{:.2f}',
                    'calmar_ratio': '{:.2f}',
                    'max_drawdown_pct': '{:.2f}%',
                    'win_rate': '{:.2f}%',
                    'var_95_pct': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

            # View detailed trades for individual strategies
            st.markdown("---")
            st.markdown("#### 📋 View Detailed Strategy Results")

            strategy_names = [f"{row['ticker']} - {row['strategy']}" for _, row in results_df.iterrows()]
            selected_detail_strategy = st.selectbox(
                "Select a strategy to view/save:",
                options=strategy_names,
                key="detail_strategy_selector"
            )

            col1, col2 = st.columns(2)

            with col1:
                view_details_btn = st.button("📊 View Detailed Results", key="view_details_btn", use_container_width=True)

            with col2:
                save_strategy_btn = st.button("💾 Save to Database", key="save_strategy_btn", use_container_width=True, type="primary")

            # Handle Save to Database button
            if save_strategy_btn:
                parts = selected_detail_strategy.split(' - ')
                if len(parts) == 2:
                    detail_ticker = parts[0]
                    detail_strategy = parts[1]

                    with st.spinner(f"Saving {selected_detail_strategy} to database..."):
                        # Build strategy config
                        strategy_config = {
                            "name": f"{detail_ticker} - {detail_strategy.title()}",
                            "description": f"Auto-generated from Complete Trading System",
                            "tickers": [detail_ticker],
                            "strategy_type": detail_strategy,
                            "indicators": [],
                            "risk_management": {
                                "stop_loss_pct": 15.0,
                                "take_profit_pct": 30.0,
                                "position_size_pct": 95.0,
                                "max_positions": 1
                            }
                        }

                        # Add indicators based on strategy type
                        if detail_strategy == "momentum":
                            strategy_config["indicators"] = [
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50},
                                {"name": "RSI", "period": 14}
                            ]
                        elif detail_strategy == "mean_reversion":
                            strategy_config["indicators"] = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "RSI", "period": 14}
                            ]
                        elif detail_strategy == "breakout":
                            strategy_config["indicators"] = [
                                {"name": "BOLLINGER_BANDS", "period": 20},
                                {"name": "ATR", "period": 14}
                            ]
                        elif detail_strategy == "trend_following":
                            strategy_config["indicators"] = [
                                {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                                {"name": "SMA", "period": 20},
                                {"name": "SMA", "period": 50}
                            ]

                        # Save strategy via API
                        save_response = make_api_request(
                            "/strategies",
                            method="POST",
                            data=strategy_config
                        )

                        if save_response and save_response.get('success'):
                            st.success(f"✅ Strategy saved! ID: {save_response['strategy_id']}")
                            st.info("🔄 Go to the **Backtest** page to run detailed backtests on this strategy.")
                        else:
                            st.error(f"❌ Failed to save strategy: {save_response.get('error', 'Unknown error')}")

            # Handle View Detailed Results button
            if view_details_btn:
                # Parse ticker and strategy from selection
                parts = selected_detail_strategy.split(' - ')
                if len(parts) == 2:
                    detail_ticker = parts[0]
                    detail_strategy = parts[1]

                    with st.spinner(f"Loading detailed results for {selected_detail_strategy}..."):
                        # Find the backtest result for this strategy
                        matching_result = None
                        for result in backtest_results:
                            if result['ticker'] == detail_ticker and result['strategy'] == detail_strategy:
                                matching_result = result
                                break

                        if matching_result:
                            # Re-run backtest to get trade details
                            strategy_config = {
                                "name": f"{detail_ticker} {detail_strategy}",
                                "tickers": [detail_ticker],
                                "strategy_type": detail_strategy,
                                "indicators": [],
                                "risk_management": {
                                    "stop_loss_pct": 15.0,
                                    "take_profit_pct": 30.0,
                                    "position_size_pct": 95.0,
                                    "max_positions": 1
                                }
                            }

                            # Add indicators based on strategy type
                            if detail_strategy == "momentum":
                                strategy_config["indicators"] = [
                                    {"name": "SMA", "period": 20},
                                    {"name": "SMA", "period": 50},
                                    {"name": "RSI", "period": 14}
                                ]
                            elif detail_strategy == "mean_reversion":
                                strategy_config["indicators"] = [
                                    {"name": "BOLLINGER_BANDS", "period": 20},
                                    {"name": "RSI", "period": 14}
                                ]
                            elif detail_strategy == "breakout":
                                strategy_config["indicators"] = [
                                    {"name": "BOLLINGER_BANDS", "period": 20},
                                    {"name": "ATR", "period": 14}
                                ]
                            elif detail_strategy == "trend_following":
                                strategy_config["indicators"] = [
                                    {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                                    {"name": "SMA", "period": 20},
                                    {"name": "SMA", "period": 50}
                                ]

                            backtest_response = make_api_request(
                                "/backtest",
                                method="POST",
                                data={
                                    "strategy_config": strategy_config,
                                    "initial_capital": total_capital
                                }
                            )

                            if backtest_response and backtest_response.get('success'):
                                backtest_id = backtest_response['backtest_id']
                                detailed_results = make_api_request(f"/backtest/results/{backtest_id}")

                                if detailed_results:
                                    st.success(f"✅ Detailed results for {selected_detail_strategy}")

                                    # Show metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Return", f"{detailed_results['metrics']['total_return_pct']:.2f}%")
                                    with col2:
                                        st.metric("Sharpe Ratio", f"{detailed_results['metrics']['sharpe_ratio']:.2f}")
                                    with col3:
                                        st.metric("Win Rate", f"{detailed_results['metrics']['win_rate']:.2f}%")
                                    with col4:
                                        st.metric("Total Trades", detailed_results['metrics']['total_trades'])

                                    # Show trades table
                                    if detailed_results.get('trades'):
                                        st.markdown("#### 📈 All Trades")
                                        trades_df = pd.DataFrame(detailed_results['trades'])
                                        st.dataframe(
                                            trades_df[[
                                                'entry_date', 'exit_date', 'entry_price', 'exit_price',
                                                'profit_loss_pct', 'profit_loss_usd', 'exit_reason'
                                            ]].style.format({
                                                'entry_price': '${:.2f}',
                                                'exit_price': '${:.2f}',
                                                'profit_loss_pct': '{:.2f}%',
                                                'profit_loss_usd': '${:,.2f}'
                                            }),
                                            use_container_width=True,
                                            height=400
                                        )

                                        # Show equity curve
                                        if detailed_results.get('equity_curve'):
                                            st.markdown("#### 📊 Equity Curve")
                                            equity_df = pd.DataFrame(detailed_results['equity_curve'])
                                            fig = px.line(
                                                equity_df,
                                                x='date',
                                                y='equity',
                                                title=f'Equity Curve - {selected_detail_strategy}'
                                            )
                                            fig.update_layout(
                                                xaxis_title="Date",
                                                yaxis_title="Portfolio Value ($)",
                                                hovermode='x unified'
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No trades were generated for this strategy")
                                else:
                                    st.error("Failed to load detailed results")
                            else:
                                st.error(f"Backtest failed: {backtest_response.get('error', 'Unknown error')}")
                        else:
                            st.error("Could not find matching strategy result")

            # Step 5: Portfolio Optimization
            st.markdown("---")
            st.markdown("### Step 5️⃣: Portfolio Optimization")

            st.write(f"**Optimizing allocation across {len(backtest_results)} strategies...**")

            col1, col2 = st.columns(2)

            with col1:
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["max_sharpe", "min_volatility", "max_return", "risk_parity"],
                    help="Method for combining strategies"
                )

            with col2:
                max_allocation = st.slider(
                    "Max Allocation per Strategy (%)",
                    min_value=5,
                    max_value=100,
                    value=30,
                    step=5,
                    help="Maximum % allocated to any single strategy"
                )

            if st.button("🎯 Optimize Portfolio", use_container_width=True):
                with st.spinner("Running portfolio optimization..."):

                    # Prepare data for optimizer
                    optimize_data = {
                        'total_capital': total_capital,
                        'method': optimization_method,
                        'constraints': {
                            'max_allocation_pct': max_allocation
                        },
                        'strategies': []
                    }

                    for _, row in results_df.iterrows():
                        optimize_data['strategies'].append({
                            'id': row['strategy_id'],
                            'name': f"{row['ticker']} - {row['strategy']}",
                            'expected_return': row['total_return_pct'],
                            'volatility': max(1, 100 / max(row['sharpe_ratio'], 0.1)),  # Approximate
                            'sharpe_ratio': row['sharpe_ratio']
                        })

                    # Call optimizer
                    opt_response = make_api_request(
                        "/portfolio/optimize",
                        method="POST",
                        data=optimize_data
                    )

                    if opt_response and opt_response.get('success'):
                        st.success("✅ Portfolio optimization complete!")

                        # Display optimal allocations
                        st.markdown("#### 💼 Optimal Portfolio Allocation")

                        allocations = opt_response['allocations']
                        portfolio_metrics = opt_response['portfolio_metrics']

                        # Create allocation chart
                        alloc_df = pd.DataFrame([
                            {'Strategy': name, 'Allocation %': weight * 100, 'Capital $': capital}
                            for name, weight, capital in zip(
                                allocations['strategy_names'],
                                allocations['weights'],
                                allocations['capital_allocation']
                            )
                        ])

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            # Pie chart
                            fig = px.pie(
                                alloc_df,
                                values='Allocation %',
                                names='Strategy',
                                title='Portfolio Allocation'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.markdown("#### 📊 Portfolio Metrics")
                            st.metric("Expected Return", f"{portfolio_metrics['expected_annual_return']:.2f}%")
                            st.metric("Expected Volatility", f"{portfolio_metrics['annual_volatility']:.2f}%")
                            st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")

                        # Detailed allocation table
                        st.markdown("#### 📋 Detailed Allocations")
                        st.dataframe(
                            alloc_df.style.format({
                                'Allocation %': '{:.2f}%',
                                'Capital $': '${:,.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

                        # Step 6: Final Recommendations
                        st.markdown("---")
                        st.markdown("### Step 6️⃣: Trading Recommendations")

                        st.success("🎯 **Your Optimized Trading System is Ready!**")

                        # Merge allocations with backtest results and market analysis
                        for idx, row in alloc_df.iterrows():
                            strategy_id = row['Strategy'].split(' - ')
                            if len(strategy_id) == 2:
                                ticker = strategy_id[0]
                                strategy = strategy_id[1]

                                # Find matching result
                                matching_result = results_df[
                                    (results_df['ticker'] == ticker) &
                                    (results_df['strategy'] == strategy)
                                ]

                                if not matching_result.empty:
                                    result = matching_result.iloc[0]

                                    with st.expander(f"📌 {row['Strategy']} - ${row['Capital $']:,.2f} ({row['Allocation %']:.1f}%)"):
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            st.markdown("**Market Context**")
                                            st.write(f"ML Prediction: **{result['ml_prediction']}** ({result['ml_confidence']:.1f}%)")
                                            st.write(f"Regime: **{result['regime']}**")

                                        with col2:
                                            st.markdown("**Performance**")
                                            st.write(f"Return: **{result['total_return_pct']:.2f}%**")
                                            st.write(f"Sortino: **{result['sortino_ratio']:.2f}**")
                                            st.write(f"Max DD: **{result['max_drawdown_pct']:.2f}%**")

                                        with col3:
                                            st.markdown("**Risk Management**")
                                            st.write(f"VaR 95%: **{result['var_95_pct']:.2f}%**")
                                            st.write(f"Kelly Size: **{result['kelly_position_pct']:.1f}%**")

                                            # Risk assessment
                                            if result['kelly_risk_level'] == 'SAFE':
                                                st.success("✅ Safe to trade")
                                            elif result['kelly_risk_level'] == 'MODERATE':
                                                st.info("⚠️ Moderate risk")
                                            else:
                                                st.warning("🔴 High risk")

                        # Summary box
                        st.markdown("---")
                        st.info(f"""
                        ### 📊 Portfolio Summary

                        - **Total Capital**: ${total_capital:,.2f}
                        - **Number of Strategies**: {len(alloc_df)}
                        - **Expected Annual Return**: {portfolio_metrics['expected_annual_return']:.2f}%
                        - **Expected Volatility**: {portfolio_metrics['annual_volatility']:.2f}%
                        - **Portfolio Sharpe**: {portfolio_metrics['sharpe_ratio']:.2f}
                        - **Optimization Method**: {optimization_method.replace('_', ' ').title()}

                        **Next Steps:**
                        1. Review each strategy's allocation and risk metrics
                        2. Consider current market regime and ML predictions
                        3. Start with paper trading to validate live performance
                        4. Monitor and rebalance monthly or when regime changes
                        """)
                    else:
                        st.error("❌ Portfolio optimization failed. Try with fewer strategies or adjust constraints.")
        else:
            # Build dynamic error message based on optimization goal
            if optimization_goal == "Maximum Returns":
                threshold_text = f"minimum quality threshold (Sharpe >= 0.3 for Maximum Returns mode)"
                suggestions = """
                **Suggestions for Maximum Returns mode:**
                - Try more volatile tickers (NVDA, TSLA, COIN)
                - Use momentum or breakout strategies
                - Ensure you have 6+ months of price data
                - Lower quality threshold to 0.2-0.3
                - Enable Vectorized Parameter Optimization
                """
            elif optimization_goal == "Risk-Adjusted (Sharpe)":
                threshold_text = f"minimum Sharpe ratio of {min_sharpe}"
                suggestions = """
                **Suggestions:**
                - Lower the minimum Sharpe requirement (try 0.4-0.5)
                - Try different strategies (mean_reversion works well)
                - Use a longer backtest period (1y or 2y)
                - Try ETFs instead of individual stocks (SPY, QQQ)
                - Enable Vectorized Parameter Optimization
                """
            elif optimization_goal == "Best Sortino":
                threshold_text = f"minimum Sortino ratio of {min_sharpe * 1.2:.2f}"
                suggestions = """
                **Suggestions:**
                - Lower the quality threshold
                - Sortino is harder to achieve than Sharpe
                - Try mean reversion strategies
                """
            else:  # Best Calmar
                threshold_text = f"minimum Calmar ratio of {min_sharpe * 0.8:.2f}"
                suggestions = """
                **Suggestions:**
                - Lower the quality threshold
                - Calmar favors strategies with low drawdowns
                - Try conservative strategies
                """

            st.warning(f"""
            ⚠️ No strategies met the {threshold_text}.

            {suggestions}

            **Debug Info:**
            - Optimization Goal: {optimization_goal}
            - Tickers tested: {', '.join(tickers)}
            - Strategies tested: {', '.join(strategies)}
            - Backtest Period: {lookback_period}
            """)

            # Show all tested strategies so user can see what they got
            if all_tested_strategies:
                st.markdown("### 📊 All Tested Strategies (Even Those That Didn't Pass)")
                tested_df = pd.DataFrame(all_tested_strategies)
                tested_df = tested_df.sort_values('sharpe', ascending=False)
                st.dataframe(
                    tested_df.style.format({
                        'sharpe': '{:.3f}',
                        'return_pct': '{:.2f}%',
                        'sortino': '{:.3f}',
                        'calmar': '{:.3f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"""
                💡 **Tip:** The highest Sharpe ratio above is **{tested_df['sharpe'].max():.3f}**.
                {"If using Maximum Returns mode, strategies with Sharpe >= 0.3 should pass." if optimization_goal == "Maximum Returns" else f"Lower your quality threshold to {tested_df['sharpe'].max() - 0.05:.2f} to see results."}
                """)

    # Quick Start Guide
    st.markdown("---")
    with st.expander("📖 Quick Start Guide", expanded=False):
        st.markdown("""
        ### How to Use the Complete Trading System

        **Prerequisites:**
        1. Train ML models for your tickers in the 🤖 ML Predictions page
        2. Train regime models in the 📊 Market Regimes page

        **Workflow:**

        **Step 1: Configuration**
        - Select multiple tickers you want to trade
        - Choose multiple strategies to test on each ticker
        - Set minimum quality threshold (Sharpe ratio)
        - Specify your total capital

        **Step 2: Market Analysis**
        - System checks ML predictions for each ticker (UP/DOWN + confidence)
        - System detects current market regime (BULL/BEAR/CONSOLIDATION)

        **Step 3: Strategy Backtesting**
        - Each strategy is backtested on each ticker
        - Advanced risk metrics calculated automatically
        - Only strategies meeting your Sharpe threshold are kept

        **Step 4: Results Review**
        - View all qualifying strategies ranked by Sortino ratio
        - See performance, risk metrics, ML signals, and regime for each

        **Step 5: Portfolio Optimization**
        - System finds optimal weights to combine strategies
        - Respects your max allocation constraints
        - Maximizes Sharpe/minimizes volatility (your choice)

        **Step 6: Trading Recommendations**
        - Get final allocations for each strategy
        - See combined market context (ML + regime)
        - Review risk management (VaR, Kelly sizing)
        - Ready to execute!

        **Tips:**
        - Start with 3-5 tickers and 3-4 strategies
        - Use min Sharpe = 1.0 for quality strategies
        - Train ML/regime models on 2y data for best results
        - Rerun analysis monthly or when market regime changes
        - Consider starting with paper trading to validate
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
