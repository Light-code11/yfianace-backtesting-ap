"""
Strategy Visualization Tools
Shows buy/sell signals on price charts with technical indicators
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import yfinance as yf
from datetime import datetime, timedelta


class StrategyVisualizer:
    """Visualize trading strategies with buy/sell signals on charts"""

    @staticmethod
    def get_strategy_description(strategy: Dict) -> str:
        """
        Generate human-readable description of strategy logic

        Args:
            strategy: Strategy configuration dict

        Returns:
            Detailed description of entry/exit rules
        """
        strategy_type = strategy.get('strategy_type', '').lower()
        indicators = strategy.get('indicators', [])
        risk_mgmt = strategy.get('risk_management', {})

        # Build indicator list
        indicator_names = [ind.get('name', '') for ind in indicators]

        # Strategy-specific descriptions
        descriptions = {
            'momentum': {
                'concept': 'Momentum strategies buy when price is trending upward and gaining strength',
                'entry': [
                    'BUY when SMA_20 crosses ABOVE SMA_50 (Golden Cross)',
                    'OR BUY when RSI crosses above 50 (momentum shift)'
                ],
                'exit': [
                    'SELL when SMA_20 crosses BELOW SMA_50 (Death Cross)',
                    'OR stop loss/take profit triggers'
                ],
                'best_for': 'Trending markets with clear directional movement'
            },
            'mean_reversion': {
                'concept': 'Mean reversion strategies buy when price is oversold, expecting bounce back',
                'entry': [
                    'BUY when price touches lower Bollinger Band (oversold)',
                    'OR BUY when RSI drops below 30 (oversold territory)'
                ],
                'exit': [
                    'SELL when price returns to middle Bollinger Band (mean)',
                    'OR stop loss/take profit triggers'
                ],
                'best_for': 'Range-bound markets with oscillating prices'
            },
            'breakout': {
                'concept': 'Breakout strategies buy when price breaks resistance, expecting continuation',
                'entry': [
                    'BUY when price breaks ABOVE upper Bollinger Band',
                    'OR BUY when price breaks 20-day high'
                ],
                'exit': [
                    'SELL when price falls back to middle Bollinger Band',
                    'OR stop loss/take profit triggers'
                ],
                'best_for': 'Volatile markets with strong momentum shifts'
            },
            'trend_following': {
                'concept': 'Trend following rides established trends using momentum indicators',
                'entry': [
                    'BUY when MACD crosses ABOVE signal line (bullish crossover)',
                    'OR BUY when SMA_20 > SMA_50 (uptrend confirmed)'
                ],
                'exit': [
                    'SELL when MACD crosses BELOW signal line (bearish crossover)',
                    'OR stop loss/take profit triggers'
                ],
                'best_for': 'Strong trending markets (bull or bear)'
            }
        }

        desc = descriptions.get(strategy_type, {
            'concept': f'{strategy_type.upper()} strategy',
            'entry': ['Custom entry logic'],
            'exit': ['Custom exit logic'],
            'best_for': 'Various market conditions'
        })

        # Format output
        output = f"""
📊 STRATEGY TYPE: {strategy_type.upper()}

💡 CONCEPT:
{desc['concept']}

🟢 ENTRY SIGNALS (BUY):
{chr(10).join('   ' + rule for rule in desc['entry'])}

🔴 EXIT SIGNALS (SELL):
{chr(10).join('   ' + rule for rule in desc['exit'])}

📈 TECHNICAL INDICATORS USED:
{chr(10).join('   • ' + ind for ind in indicator_names)}

⚠️ RISK MANAGEMENT:
   • Stop Loss: {risk_mgmt.get('stop_loss_pct', 'N/A')}%
   • Take Profit: {risk_mgmt.get('take_profit_pct', 'N/A')}%
   • Position Size: {risk_mgmt.get('position_size_pct', 'N/A')}%
   • Max Positions: {risk_mgmt.get('max_positions', 'N/A')}

✅ BEST FOR:
{desc['best_for']}
"""
        return output.strip()

    @staticmethod
    def create_strategy_chart(
        ticker: str,
        strategy: Dict,
        trades: List[Dict],
        period: str = '1y'
    ) -> go.Figure:
        """
        Create interactive chart showing strategy signals and trades

        Args:
            ticker: Stock ticker symbol
            strategy: Strategy configuration
            trades: List of executed trades
            period: Time period for chart

        Returns:
            Plotly figure with price, indicators, and trade signals
        """
        # Download historical data
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return None

        # Calculate indicators used in strategy
        indicators = strategy.get('indicators', [])
        for indicator in indicators:
            name = indicator.get('name', '').upper()
            period_val = indicator.get('period', 14)

            if name == 'SMA':
                df[f'SMA_{period_val}'] = df['Close'].rolling(window=period_val).mean()
            elif name == 'EMA':
                df[f'EMA_{period_val}'] = df['Close'].ewm(span=period_val).mean()
            elif name == 'RSI':
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period_val).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period_val).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            elif name in ['BB', 'BOLLINGER']:
                sma = df['Close'].rolling(window=period_val).mean()
                std = df['Close'].rolling(window=period_val).std()
                df['BB_Upper'] = sma + (std * 2)
                df['BB_Middle'] = sma
                df['BB_Lower'] = sma - (std * 2)
            elif name == 'MACD':
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Create subplots
        strategy_type = strategy.get('strategy_type', '').lower()

        # Determine subplot layout based on indicators
        has_rsi = 'RSI' in df.columns
        has_macd = 'MACD' in df.columns

        subplot_count = 1
        if has_rsi:
            subplot_count += 1
        if has_macd:
            subplot_count += 1

        row_heights = [0.6] + [0.2] * (subplot_count - 1)

        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=['Price & Signals'] +
                          (['RSI'] if has_rsi else []) +
                          (['MACD'] if has_macd else [])
        )

        # Main price chart (candlestick)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)

        # Add moving averages if present
        for col in df.columns:
            if col.startswith('SMA_') or col.startswith('EMA_'):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    line=dict(width=1.5)
                ), row=1, col=1)

        # Add Bollinger Bands if present
        if 'BB_Upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash'),
                showlegend=True
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Middle'],
                name='BB Middle',
                line=dict(color='rgba(128, 128, 128, 0.5)'),
                showlegend=True
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(128, 128, 128, 0.5)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=True
            ), row=1, col=1)

        # Add buy/sell signals from trades
        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []

        for trade in trades:
            if trade.get('ticker') == ticker:
                entry_date = pd.to_datetime(trade.get('entry_date'))
                exit_date = pd.to_datetime(trade.get('exit_date'))

                buy_dates.append(entry_date)
                buy_prices.append(trade.get('entry_price', 0))

                sell_dates.append(exit_date)
                sell_prices.append(trade.get('exit_price', 0))

        # Plot buy signals
        if buy_dates:
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='BUY',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='#00ff00',
                    line=dict(color='darkgreen', width=2)
                )
            ), row=1, col=1)

        # Plot sell signals
        if sell_dates:
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='SELL',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='#ff0000',
                    line=dict(color='darkred', width=2)
                )
            ), row=1, col=1)

        # Add RSI if present
        current_row = 2
        if has_rsi:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=1.5)
            ), row=current_row, col=1)

            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="Overbought", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="Oversold", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)

            fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100])
            current_row += 1

        # Add MACD if present
        if has_macd:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1.5)
            ), row=current_row, col=1)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='red', width=1.5)
            ), row=current_row, col=1)

            # MACD histogram
            histogram = df['MACD'] - df['MACD_Signal']
            colors = ['green' if val >= 0 else 'red' for val in histogram]
            fig.add_trace(go.Bar(
                x=df.index,
                y=histogram,
                name='Histogram',
                marker_color=colors,
                opacity=0.3
            ), row=current_row, col=1)

            fig.update_yaxes(title_text="MACD", row=current_row, col=1)

        # Update layout
        fig.update_layout(
            title=f'{ticker} - {strategy.get("name", "Strategy")} ({strategy_type.upper()})',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            height=600 if subplot_count == 1 else 800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Remove rangeslider from candlestick
        fig.update_xaxes(rangeslider_visible=False)

        return fig


if __name__ == "__main__":
    # Example usage
    example_strategy = {
        'name': 'Momentum Golden Cross',
        'strategy_type': 'momentum',
        'indicators': [
            {'name': 'SMA', 'period': 20},
            {'name': 'SMA', 'period': 50},
            {'name': 'RSI', 'period': 14}
        ],
        'risk_management': {
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0,
            'position_size_pct': 10.0,
            'max_positions': 3
        }
    }

    print(StrategyVisualizer.get_strategy_description(example_strategy))
