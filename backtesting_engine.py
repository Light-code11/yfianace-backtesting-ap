"""
Advanced backtesting engine with technical indicators and advanced risk metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from kelly_criterion import KellyCriterion
from advanced_indicators import AdvancedIndicators
from advanced_risk_metrics import AdvancedRiskMetrics


class TechnicalIndicators:
    """Calculate technical indicators for trading strategies"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD, Signal Line, and Histogram"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K and %D)"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx


class BacktestingEngine:
    """Advanced backtesting engine for trading strategies"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.indicators = TechnicalIndicators()

    def backtest_strategy(
        self,
        strategy: Dict[str, Any],
        market_data: pd.DataFrame,
        commission: float = 0.001,  # 0.1% commission
        trailing_stop_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy

        Args:
            strategy: Strategy dictionary with entry/exit rules
            market_data: Historical OHLCV data
            commission: Transaction commission as decimal
            trailing_stop_pct: Trailing stop percentage (e.g. 5.0). None disables.

        Returns:
            Dictionary with backtest results
        """
        results = {
            "strategy_name": strategy.get("name", "Unknown"),
            "tickers": strategy.get("tickers", []),
            "trades": [],
            "equity_curve": [],
            "metrics": {}
        }

        # Extract strategy parameters
        tickers = strategy.get("tickers", [])
        indicators_config = strategy.get("indicators", [])
        stop_loss_pct = strategy.get("risk_management", {}).get("stop_loss_pct", 5.0)
        take_profit_pct = strategy.get("risk_management", {}).get("take_profit_pct", 10.0)
        position_size_pct = strategy.get("risk_management", {}).get("position_size_pct", 10.0)
        max_positions = strategy.get("risk_management", {}).get("max_positions", 3)

        # Initialize portfolio
        capital = self.initial_capital
        positions = {}  # {ticker: {qty, entry_price, entry_date}}
        trades = []

        # Pre-calculate ticker data/indicators so simulation can run date-by-date
        ticker_state = {}
        for ticker in tickers:
            if ticker not in market_data.columns.levels[1]:
                continue

            ticker_data = market_data.xs(ticker, level=1, axis=1)
            if 'Close' not in ticker_data.columns:
                continue

            indicators_data = self._calculate_indicators(ticker_data, indicators_config)
            ticker_state[ticker] = {
                'data': ticker_data,
                'indicators': indicators_data,
                'date_to_index': {dt: i for i, dt in enumerate(ticker_data.index)},
                'close_series': ticker_data['Close'].reindex(market_data.index).ffill()
            }

        # Simulate trading date-by-date across all tickers and track full portfolio equity
        for current_date in market_data.index:
            for ticker, state in ticker_state.items():
                i = state['date_to_index'].get(current_date)
                if i is None:
                    continue

                ticker_data = state['data']
                indicators_data = state['indicators']
                current_price = ticker_data['Close'].iloc[i]

                # Check exit conditions for existing positions
                if ticker in positions:
                    position = positions[ticker]
                    position['highest_price'] = max(
                        position.get('highest_price', position['entry_price']),
                        current_price
                    )
                    entry_price = position['entry_price']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    exit_reason = None
                    exit_metadata = None

                    if pnl_pct <= -stop_loss_pct:
                        exit_reason = "stop_loss"
                    elif pnl_pct >= take_profit_pct:
                        exit_reason = "take_profit"
                    elif trailing_stop_pct is not None and trailing_stop_pct > 0:
                        trailing_stop_price = position['highest_price'] * (1 - trailing_stop_pct / 100)
                        if current_price <= trailing_stop_price:
                            exit_reason = "trailing_stop"
                            exit_metadata = {
                                'trailing_high_price': position['highest_price'],
                                'trailing_stop_pct': trailing_stop_pct,
                                'trailing_stop_price': trailing_stop_price
                            }

                    if exit_reason is None and self._check_exit_signal(indicators_data, i, strategy):
                        exit_reason = "signal"

                    if exit_reason:
                        trade = self._close_position(
                            ticker, position, current_price, current_date,
                            commission, exit_reason, exit_metadata=exit_metadata
                        )
                        trades.append(trade)
                        capital += trade['exit_value']
                        del positions[ticker]

                # Check entry conditions
                if ticker not in positions and len(positions) < max_positions:
                    if self._check_entry_signal(indicators_data, i, strategy):
                        position_value = capital * (position_size_pct / 100)
                        qty = position_value / current_price
                        cost = qty * current_price * (1 + commission)

                        if cost <= capital:
                            positions[ticker] = {
                                'qty': qty,
                                'entry_price': current_price,
                                'entry_date': current_date,
                                'cost': cost,
                                'highest_price': current_price
                            }
                            capital -= cost

            # Track total portfolio equity across all open positions at this date
            portfolio_value = capital
            for pos_ticker, pos in positions.items():
                close_series = ticker_state[pos_ticker]['close_series']
                mark_price = close_series.loc[current_date]
                if pd.isna(mark_price):
                    mark_price = pos['entry_price']
                portfolio_value += pos['qty'] * mark_price

            results['equity_curve'].append({
                'date': current_date.strftime('%Y-%m-%d'),
                'equity': portfolio_value
            })

        # Close any remaining positions
        for ticker, position in list(positions.items()):
            close_series = ticker_state[ticker]['close_series']
            final_price = close_series.iloc[-1]
            if pd.isna(final_price):
                valid_closes = close_series.dropna()
                final_price = valid_closes.iloc[-1] if not valid_closes.empty else position['entry_price']
            final_date = market_data.index[-1]
            trade = self._close_position(
                ticker, position, final_price, final_date,
                commission, "end_of_test"
            )
            trades.append(trade)
            capital += trade['exit_value']  # Fixed: add exit proceeds, not profit

        if results['equity_curve']:
            # Ensure final equity reflects net liquidation value after commissions.
            results['equity_curve'][-1] = {
                'date': market_data.index[-1].strftime('%Y-%m-%d'),
                'equity': capital
            }

        # Calculate metrics
        results['trades'] = trades
        results['metrics'] = self._calculate_metrics(trades, capital, results['equity_curve'])

        return results

    def _calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators_config: List[Dict]
    ) -> pd.DataFrame:
        """Calculate technical indicators based on configuration using AdvancedIndicators"""
        df = data.copy()

        for indicator in indicators_config:
            name = indicator.get('name', '').upper()
            period = indicator.get('period', 14)

            try:
                # MOMENTUM INDICATORS
                if name == 'RSI':
                    df[f'RSI_{period}'] = AdvancedIndicators.rsi(df, period)
                    df['RSI'] = df[f'RSI_{period}']  # Keep backward compatibility

                elif name == 'STOCHASTIC' or name == 'STOCH':
                    k, d = AdvancedIndicators.stochastic(df, period)
                    df['Stoch_K'] = k
                    df['Stoch_D'] = d

                elif name == 'CCI':
                    df[f'CCI_{period}'] = AdvancedIndicators.cci(df, period)

                elif name == 'WILLIAMS_R' or name == 'WILLR':
                    df[f'Williams_R_{period}'] = AdvancedIndicators.williams_r(df, period)
                    df['Williams_R'] = df[f'Williams_R_{period}']

                elif name == 'ROC':
                    df[f'ROC_{period}'] = AdvancedIndicators.roc(df, period)

                elif name == 'MFI':
                    if 'Volume' in df.columns:
                        df[f'MFI_{period}'] = AdvancedIndicators.mfi(df, period)

                # TREND INDICATORS
                elif name == 'SMA':
                    df[f'SMA_{period}'] = AdvancedIndicators.sma(df, period)

                elif name == 'EMA':
                    df[f'EMA_{period}'] = AdvancedIndicators.ema(df, period)

                elif name == 'MACD':
                    macd_dict = AdvancedIndicators.macd(df)
                    df['MACD'] = macd_dict['MACD']
                    df['MACD_Signal'] = macd_dict['MACD_Signal']
                    df['MACD_Hist'] = macd_dict['MACD_Hist']

                elif name == 'ADX':
                    df[f'ADX_{period}'] = AdvancedIndicators.adx(df, period)
                    df['ADX'] = df[f'ADX_{period}']

                elif name == 'SUPERTREND':
                    st_dict = AdvancedIndicators.supertrend(df, period)
                    df['SuperTrend'] = st_dict['SuperTrend']
                    df['SuperTrend_Dir'] = st_dict['SuperTrend_Direction']

                elif name == 'AROON':
                    aroon_dict = AdvancedIndicators.aroon(df, period)
                    df['Aroon_Up'] = aroon_dict['Aroon_Up']
                    df['Aroon_Down'] = aroon_dict['Aroon_Down']

                # VOLATILITY INDICATORS
                elif name == 'ATR':
                    df[f'ATR_{period}'] = AdvancedIndicators.atr(df, period)
                    df['ATR'] = df[f'ATR_{period}']

                elif name == 'BB' or name == 'BOLLINGER' or name == 'BOLLINGER_BANDS':
                    bb_dict = AdvancedIndicators.bollinger_bands(df, period)
                    df['BB_Upper'] = bb_dict['BB_Upper']
                    df['BB_Middle'] = bb_dict['BB_Middle']
                    df['BB_Lower'] = bb_dict['BB_Lower']

                elif name == 'KC' or name == 'KELTNER' or name == 'KELTNER_CHANNELS':
                    kc_dict = AdvancedIndicators.keltner_channels(df, period)
                    df['KC_Upper'] = kc_dict['KC_Upper']
                    df['KC_Middle'] = kc_dict['KC_Middle']
                    df['KC_Lower'] = kc_dict['KC_Lower']

                elif name == 'DC' or name == 'DONCHIAN' or name == 'DONCHIAN_CHANNELS':
                    dc_dict = AdvancedIndicators.donchian_channels(df, period)
                    df['DC_Upper'] = dc_dict['DC_Upper']
                    df['DC_Middle'] = dc_dict['DC_Middle']
                    df['DC_Lower'] = dc_dict['DC_Lower']

                # VOLUME INDICATORS
                elif name == 'OBV':
                    if 'Volume' in df.columns:
                        df['OBV'] = AdvancedIndicators.obv(df)

                elif name == 'CMF':
                    if 'Volume' in df.columns:
                        df[f'CMF_{period}'] = AdvancedIndicators.cmf(df, period)

                elif name == 'VWAP':
                    if 'Volume' in df.columns:
                        df['VWAP'] = AdvancedIndicators.vwap(df)

                elif name == 'VOLUME_SMA':
                    if 'Volume' in df.columns:
                        df[f'Volume_SMA_{period}'] = AdvancedIndicators.volume_sma(df, period)

            except Exception as e:
                print(f"Warning: Could not calculate {name} indicator: {e}")
                continue

        return df

    def _check_entry_signal(
        self,
        data: pd.DataFrame,
        index: int,
        strategy: Dict
    ) -> bool:
        """Check if entry conditions are met"""
        if index < 50:  # Need enough data for indicators
            return False

        strategy_type = strategy.get('strategy_type', '').lower()
        row = data.iloc[index]
        prev_row = data.iloc[index - 1]

        # Simple momentum strategy
        if strategy_type == 'momentum':
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                return row['SMA_20'] > row['SMA_50'] and prev_row['SMA_20'] <= prev_row['SMA_50']
            elif 'RSI' in data.columns:
                return row['RSI'] > 50 and prev_row['RSI'] <= 50

        # Mean reversion strategy
        elif strategy_type == 'mean_reversion':
            if 'BB_Lower' in data.columns and 'BB_Middle' in data.columns:
                return row['Close'] <= row['BB_Lower']
            elif 'RSI' in data.columns:
                return row['RSI'] < 30

        # Breakout strategy
        elif strategy_type == 'breakout':
            if 'BB_Upper' in data.columns:
                return row['Close'] > row['BB_Upper']
            else:
                # Simple breakout: price breaks 20-day high
                if index >= 20:
                    return row['Close'] > data['High'].iloc[index-20:index].max()

        # Trend following
        elif strategy_type == 'trend_following':
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                return row['MACD'] > row['MACD_Signal'] and prev_row['MACD'] <= prev_row['MACD_Signal']
            elif 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                return row['SMA_20'] > row['SMA_50']

        return False

    def _check_exit_signal(
        self,
        data: pd.DataFrame,
        index: int,
        strategy: Dict
    ) -> bool:
        """Check if exit conditions are met"""
        if index < 50:
            return False

        strategy_type = strategy.get('strategy_type', '').lower()
        row = data.iloc[index]
        prev_row = data.iloc[index - 1]

        # Opposite of entry signals
        if strategy_type == 'momentum':
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                return row['SMA_20'] < row['SMA_50']

        elif strategy_type == 'mean_reversion':
            if 'BB_Middle' in data.columns:
                return row['Close'] >= row['BB_Middle']

        elif strategy_type == 'breakout':
            if 'BB_Middle' in data.columns:
                return row['Close'] < row['BB_Middle']

        elif strategy_type == 'trend_following':
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                return row['MACD'] < row['MACD_Signal']

        return False

    def _close_position(
        self,
        ticker: str,
        position: Dict,
        exit_price: float,
        exit_date: datetime,
        commission: float,
        exit_reason: str,
        exit_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """Close a position and record trade"""
        entry_price = position['entry_price']
        qty = position['qty']

        exit_value = qty * exit_price * (1 - commission)
        profit_loss = exit_value - position['cost']
        profit_loss_pct = (profit_loss / position['cost']) * 100

        trade = {
            'ticker': ticker,
            'action': 'BUY/SELL',
            'quantity': qty,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'profit_loss_usd': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'exit_reason': exit_reason,
            'exit_value': exit_value  # Add exit value for correct capital management
        }

        if exit_metadata:
            trade.update(exit_metadata)

        return trade

    def _calculate_metrics(
        self,
        trades: List[Dict],
        final_capital: float,
        equity_curve: List[Dict]
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,  # Added missing key
                'max_drawdown_pct': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'quality_score': 0  # Added missing key
            }

        # Basic metrics
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        total_trades = len(trades)

        winning_trades = [t for t in trades if t['profit_loss_usd'] > 0]
        losing_trades = [t for t in trades if t['profit_loss_usd'] <= 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        total_wins = sum(t['profit_loss_usd'] for t in winning_trades)
        total_losses = abs(sum(t['profit_loss_usd'] for t in losing_trades))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Average win/loss
        avg_win = (total_wins / len(winning_trades)) if winning_trades else 0
        avg_loss = (total_losses / len(losing_trades)) if losing_trades else 0

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        equity_values = [e['equity'] for e in equity_curve]
        peak = equity_values[0]
        max_dd = 0
        for value in equity_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            max_dd = max(max_dd, dd)

        # Calculate Kelly Criterion optimal position sizing
        # Pass profitability metrics for sanity checking
        kelly_result = KellyCriterion.calculate_kelly_from_backtest({
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return_pct': total_return_pct,  # Add profitability check
            'sharpe_ratio': sharpe_ratio  # Add risk-adjusted return check
        }, fractional_kelly=0.25)  # Use Quarter Kelly for safety

        # Calculate advanced risk metrics
        advanced_risk = AdvancedRiskMetrics.comprehensive_risk_analysis(
            equity_curve=equity_curve,
            trades=trades,
            risk_free_rate=0.02
        )

        # Extract key metrics from advanced risk analysis
        if advanced_risk.get('success'):
            var_95 = advanced_risk['value_at_risk']['var_95_pct']
            cvar_95 = advanced_risk['conditional_var']['cvar_95_pct']
            sortino_ratio = advanced_risk['risk_adjusted_returns']['sortino_ratio']
            calmar_ratio = advanced_risk['risk_adjusted_returns']['calmar_ratio']
            ulcer_index = advanced_risk['drawdown_metrics']['ulcer_index']
            pain_index = advanced_risk['drawdown_metrics']['pain_index']
            max_dd_duration = advanced_risk['drawdown_metrics']['max_drawdown_duration_days']
            time_underwater_pct = advanced_risk['drawdown_metrics']['time_underwater_pct']
            skewness = advanced_risk['tail_risk']['skewness']
            kurtosis = advanced_risk['tail_risk']['kurtosis']
            max_win_streak = advanced_risk['streaks']['max_win_streak']
            max_loss_streak = advanced_risk['streaks']['max_loss_streak']
        else:
            # Fallback values if advanced metrics fail
            var_95 = 0
            cvar_95 = 0
            sortino_ratio = sharpe_ratio * 1.2
            calmar_ratio = 0
            ulcer_index = 0
            pain_index = 0
            max_dd_duration = 0
            time_underwater_pct = 0
            skewness = 0
            kurtosis = 0
            max_win_streak = 0
            max_loss_streak = 0

        return {
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'quality_score': self._calculate_quality_score(
                sharpe_ratio, win_rate, max_dd, total_return_pct
            ),
            # Kelly Criterion optimal position sizing
            'kelly_criterion': kelly_result['kelly_fraction'],
            'kelly_position_pct': kelly_result['recommended_position_pct'],
            'kelly_risk_level': kelly_result['risk_level'],
            # Advanced Risk Metrics
            'var_95_pct': round(var_95, 2),
            'cvar_95_pct': round(cvar_95, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'ulcer_index': round(ulcer_index, 2),
            'pain_index': round(pain_index, 2),
            'max_dd_duration_days': max_dd_duration,
            'time_underwater_pct': round(time_underwater_pct, 1),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }

    def _calculate_quality_score(
        self,
        sharpe: float,
        win_rate: float,
        max_dd: float,
        total_return: float
    ) -> float:
        """Calculate composite quality score (0-100)"""
        # Normalize each metric
        sharpe_score = min(sharpe / 2 * 25, 25)  # Max 25 points
        win_rate_score = (win_rate / 100) * 25  # Max 25 points
        dd_score = max(0, (1 - max_dd / 50)) * 25  # Max 25 points, penalize high DD
        return_score = min(total_return / 20 * 25, 25)  # Max 25 points

        total_score = sharpe_score + win_rate_score + dd_score + return_score
        return round(min(total_score, 100), 2)
