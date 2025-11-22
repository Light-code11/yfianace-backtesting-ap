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

# Page configuration
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_BASE_URL = "http://localhost:8000"

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
    ["Dashboard", "Generate Strategies", "Backtest", "Paper Trading", "Portfolio Optimizer", "AI Learning"]
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
        strategies_data = make_api_request("/strategies?limit=10")

        if strategies_data and strategies_data.get('strategies'):
            strategies_df = pd.DataFrame(strategies_data['strategies'])
            st.dataframe(
                strategies_df[['name', 'strategy_type', 'created_at', 'is_active']],
                use_container_width=True
            )

        # Recent backtest results
        st.markdown("---")
        st.subheader("📈 Recent Backtest Results")
        backtest_data = make_api_request("/backtest/results?limit=10")

        if backtest_data and backtest_data.get('results'):
            backtest_df = pd.DataFrame(backtest_data['results'])
            st.dataframe(
                backtest_df[[
                    'strategy_name', 'total_return_pct', 'sharpe_ratio',
                    'win_rate', 'max_drawdown_pct', 'quality_score'
                ]],
                use_container_width=True
            )


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
                value="AAPL, MSFT, GOOGL, TSLA, NVDA",
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
            # Parse tickers
            tickers = [t.strip().upper() for t in tickers_input.split(",")]

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

    # Get available strategies
    strategies_data = make_api_request("/strategies")

    if strategies_data and strategies_data.get('strategies'):
        strategy_options = {f"{s['name']} (ID: {s['id']})": s['id'] for s in strategies_data['strategies']}

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
                            # Display metrics
                            st.subheader("📊 Performance Metrics")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Total Return", f"{results['metrics']['total_return_pct']:.2f}%")
                                st.metric("Win Rate", f"{results['metrics']['win_rate']:.2f}%")

                            with col2:
                                st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
                                st.metric("Sortino Ratio", f"{results['metrics']['sortino_ratio']:.2f}")

                            with col3:
                                st.metric("Max Drawdown", f"{results['metrics']['max_drawdown_pct']:.2f}%")
                                st.metric("Profit Factor", f"{results['metrics']['profit_factor']:.2f}")

                            with col4:
                                st.metric("Total Trades", results['metrics']['total_trades'])
                                st.metric("Quality Score", f"{results['metrics']['quality_score']:.1f}/100")

                            # Equity curve
                            st.subheader("📈 Equity Curve")
                            equity_fig = plot_equity_curve(results['equity_curve'])
                            if equity_fig:
                                st.plotly_chart(equity_fig, use_container_width=True)

                            # Drawdown analysis
                            st.subheader("📉 Drawdown Analysis")
                            dd_fig = plot_drawdown(results['equity_curve'])
                            if dd_fig:
                                st.plotly_chart(dd_fig, use_container_width=True)

                            # Trade distribution
                            st.subheader("📊 Trade Distribution")
                            trade_fig = plot_trade_distribution(results['trades'])
                            if trade_fig:
                                st.plotly_chart(trade_fig, use_container_width=True)

                            # Trade history
                            st.subheader("📝 Trade History")
                            if results['trades']:
                                trades_df = pd.DataFrame(results['trades'])
                                st.dataframe(trades_df, use_container_width=True)


# =======================
# PAPER TRADING PAGE
# =======================

elif page == "Paper Trading":
    st.markdown('<div class="main-header">📝 Paper Trading</div>', unsafe_allow_html=True)

    st.write("Test strategies with live market data without risking real money.")

    # Get available strategies
    strategies_data = make_api_request("/strategies")

    if strategies_data and strategies_data.get('strategies'):
        strategy_options = {f"{s['name']} (ID: {s['id']})": s['id'] for s in strategies_data['strategies']}

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
        strategy_options = {s['name']: s['id'] for s in strategies_data['strategies']}
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


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("AI Trading Platform v2.0")
st.sidebar.caption("Built with Streamlit + FastAPI + OpenAI")
