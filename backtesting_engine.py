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
        slippage_pct: float = 0.0005,  # 0.05% = 5 bps
        spread_pct: float = 0.0002  # 0.02%
    ) -> Dict[str, Any]:
        """
        Backtest a trading strategy

        Args:
            strategy: Strategy dictionary with entry/exit rules
            market_data: Historical OHLCV data
            commission: Transaction commission as decimal
            slippage_pct: Per-side slippage as decimal
            spread_pct: Bid/ask spread as decimal

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

        # Process each ticker
        for ticker in tickers:
            if ticker not in market_data.columns.levels[1]:
                continue

            # Get ticker data
            ticker_data = market_data.xs(ticker, level=1, axis=1)

            # Calculate indicators
            indicators_data = self._calculate_indicators(ticker_data, indicators_config)

            # Simulate trading
            for i in range(len(ticker_data)):
                current_date = ticker_data.index[i]
                current_price = ticker_data['Close'].iloc[i]

                # Check exit conditions for existing positions
                if ticker in positions:
                    position = positions[ticker]
                    entry_price = position['entry_price']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    # Check stop loss
                    if pnl_pct <= -stop_loss_pct:
                        trade = self._close_position(
                            ticker, position, current_price, current_date,
                            commission, slippage_pct, spread_pct, "stop_loss"
                        )
                        trades.append(trade)
                        capital += trade['exit_value']  # Fixed: add exit proceeds, not profit
                        del positions[ticker]

                    # Check take profit
                    elif pnl_pct >= take_profit_pct:
                        trade = self._close_position(
                            ticker, position, current_price, current_date,
                            commission, slippage_pct, spread_pct, "take_profit"
                        )
                        trades.append(trade)
                        capital += trade['exit_value']  # Fixed: add exit proceeds, not profit
                        del positions[ticker]

                    # Check strategy exit conditions
                    elif self._check_exit_signal(indicators_data, i, strategy):
                        trade = self._close_position(
                            ticker, position, current_price, current_date,
                            commission, slippage_pct, spread_pct, "signal"
                        )
                        trades.append(trade)
                        capital += trade['exit_value']  # Fixed: add exit proceeds, not profit
                        del positions[ticker]

                # Check entry conditions
                if ticker not in positions and len(positions) < max_positions:
                    if self._check_entry_signal(indicators_data, i, strategy):
                        # Calculate position size
                        position_value = capital * (position_size_pct / 100)
                        entry_fill_price = current_price * (1 + spread_pct / 2) * (1 + slippage_pct)
                        qty = position_value / entry_fill_price
                        cost = qty * entry_fill_price * (1 + commission)

                        if cost <= capital:
                            positions[ticker] = {
                                'qty': qty,
                                'entry_price': entry_fill_price,
                                'entry_market_price': current_price,
                                'entry_date': current_date,
                                'cost': cost
                            }
                            capital -= cost

                # Track equity
                portfolio_value = capital + sum(
                    pos['qty'] * ticker_data['Close'].iloc[i]
                    for tick, pos in positions.items() if tick == ticker
                )
                results['equity_curve'].append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'equity': portfolio_value
                })

        # Close any remaining positions
        for ticker, position in list(positions.items()):
            final_price = market_data['Close'][ticker].iloc[-1]
            final_date = market_data.index[-1]
            trade = self._close_position(
                ticker, position, final_price, final_date,
                commission, slippage_pct, spread_pct, "end_of_test"
            )
            trades.append(trade)
            capital += trade['exit_value']  # Fixed: add exit proceeds, not profit

        # Calculate metrics
        results['trades'] = trades
        results['metrics'] = self._calculate_metrics(trades, capital, results['equity_curve'])

        return results

    def walk_forward_test(
        self,
        strategy: Dict[str, Any],
        market_data: pd.DataFrame,
        n_windows: int = 5,
        commission: float = 0.001,
        slippage_pct: float = 0.0005,
        spread_pct: float = 0.0002,
        degradation_threshold_pct: float = 30.0
    ) -> Dict[str, Any]:
        """
        Run anchored walk-forward validation across N windows.

        Each window optimizes on in-sample data and evaluates on the following
        out-of-sample slice.
        """
        if n_windows < 1:
            raise ValueError("n_windows must be >= 1")
        if market_data.empty:
            raise ValueError("market_data cannot be empty")

        total_rows = len(market_data)
        segment_size = total_rows // (n_windows + 1)
        if segment_size < 1:
            raise ValueError("Not enough data for requested number of walk-forward windows")

        window_results = []
        for window_idx in range(n_windows):
            in_sample_end = segment_size * (window_idx + 1)
            out_sample_start = in_sample_end
            out_sample_end = segment_size * (window_idx + 2) if window_idx < n_windows - 1 else total_rows

            in_sample_data = market_data.iloc[:in_sample_end]
            out_sample_data = market_data.iloc[out_sample_start:out_sample_end]

            if in_sample_data.empty or out_sample_data.empty:
                continue

            optimized_strategy, in_sample_metrics = self._optimize_strategy_on_insample(
                strategy=strategy,
                in_sample_data=in_sample_data,
                commission=commission,
                slippage_pct=slippage_pct,
                spread_pct=spread_pct
            )
            out_sample_result = self.backtest_strategy(
                strategy=optimized_strategy,
                market_data=out_sample_data,
                commission=commission,
                slippage_pct=slippage_pct,
                spread_pct=spread_pct
            )
            out_sample_metrics = out_sample_result["metrics"]

            return_degradation = self._calculate_metric_degradation(
                in_sample_metrics.get("total_return_pct", 0.0),
                out_sample_metrics.get("total_return_pct", 0.0)
            )
            sharpe_degradation = self._calculate_metric_degradation(
                in_sample_metrics.get("sharpe_ratio", 0.0),
                out_sample_metrics.get("sharpe_ratio", 0.0)
            )
            degraded = (
                return_degradation >= degradation_threshold_pct or
                sharpe_degradation >= degradation_threshold_pct
            )

            window_results.append({
                "window": window_idx + 1,
                "in_sample_start": in_sample_data.index[0].strftime("%Y-%m-%d"),
                "in_sample_end": in_sample_data.index[-1].strftime("%Y-%m-%d"),
                "out_sample_start": out_sample_data.index[0].strftime("%Y-%m-%d"),
                "out_sample_end": out_sample_data.index[-1].strftime("%Y-%m-%d"),
                "in_sample_metrics": in_sample_metrics,
                "out_sample_metrics": out_sample_metrics,
                "return_degradation_pct": round(return_degradation, 2),
                "sharpe_degradation_pct": round(sharpe_degradation, 2),
                "degraded_out_of_sample": degraded
            })

        if not window_results:
            return {
                "windows": [],
                "aggregate_metrics": {
                    "avg_in_sample_return_pct": 0.0,
                    "avg_out_sample_return_pct": 0.0,
                    "avg_in_sample_sharpe": 0.0,
                    "avg_out_sample_sharpe": 0.0,
                    "avg_out_sample_max_drawdown_pct": 0.0,
                    "degraded_windows": 0,
                    "degraded_window_pct": 0.0
                },
                "strategy_degrades_significantly": False
            }

        in_returns = [w["in_sample_metrics"].get("total_return_pct", 0.0) for w in window_results]
        out_returns = [w["out_sample_metrics"].get("total_return_pct", 0.0) for w in window_results]
        in_sharpes = [w["in_sample_metrics"].get("sharpe_ratio", 0.0) for w in window_results]
        out_sharpes = [w["out_sample_metrics"].get("sharpe_ratio", 0.0) for w in window_results]
        out_drawdowns = [w["out_sample_metrics"].get("max_drawdown_pct", 0.0) for w in window_results]
        degraded_count = sum(1 for w in window_results if w["degraded_out_of_sample"])
        degraded_pct = (degraded_count / len(window_results)) * 100

        return {
            "windows": window_results,
            "aggregate_metrics": {
                "avg_in_sample_return_pct": round(float(np.mean(in_returns)), 2),
                "avg_out_sample_return_pct": round(float(np.mean(out_returns)), 2),
                "avg_in_sample_sharpe": round(float(np.mean(in_sharpes)), 2),
                "avg_out_sample_sharpe": round(float(np.mean(out_sharpes)), 2),
                "avg_out_sample_max_drawdown_pct": round(float(np.mean(out_drawdowns)), 2),
                "degraded_windows": degraded_count,
                "degraded_window_pct": round(degraded_pct, 2)
            },
            "strategy_degrades_significantly": degraded_pct >= 50.0
        }

    def monte_carlo_analysis(
        self,
        trades: List[Dict[str, Any]],
        simulations: int = 1000,
        initial_capital: Optional[float] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Monte Carlo analysis by randomizing trade order.
        """
        if simulations < 1:
            raise ValueError("simulations must be >= 1")
        if not trades:
            return {
                "simulations": simulations,
                "distribution_stats": {
                    "total_return_pct": self._summarize_distribution([]),
                    "max_drawdown_pct": self._summarize_distribution([]),
                    "sharpe_ratio": self._summarize_distribution([])
                }
            }

        base_capital = float(initial_capital if initial_capital is not None else self.initial_capital)
        trade_pnls = np.array([float(t.get("profit_loss_usd", 0.0)) for t in trades], dtype=float)
        rng = np.random.default_rng(random_seed)

        total_returns = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(simulations):
            shuffled_pnls = rng.permutation(trade_pnls)
            equity = base_capital
            equity_curve = [equity]

            for pnl in shuffled_pnls:
                equity += pnl
                equity_curve.append(equity)

            total_return_pct = ((equity - base_capital) / base_capital) * 100
            max_drawdown_pct = self._calculate_max_drawdown_pct(equity_curve)

            returns_series = pd.Series(equity_curve, dtype=float).pct_change().dropna()
            sharpe_ratio = (
                (returns_series.mean() / returns_series.std()) * np.sqrt(len(returns_series))
                if len(returns_series) > 1 and returns_series.std() > 0 else 0.0
            )

            total_returns.append(total_return_pct)
            max_drawdowns.append(max_drawdown_pct)
            sharpe_ratios.append(sharpe_ratio)

        return {
            "simulations": simulations,
            "distribution_stats": {
                "total_return_pct": self._summarize_distribution(total_returns),
                "max_drawdown_pct": self._summarize_distribution(max_drawdowns),
                "sharpe_ratio": self._summarize_distribution(sharpe_ratios)
            }
        }

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
        slippage_pct: float,
        spread_pct: float,
        exit_reason: str
    ) -> Dict:
        """Close a position and record trade"""
        entry_price = position['entry_price']
        qty = position['qty']
        effective_exit_price = exit_price * (1 - spread_pct / 2) * (1 - slippage_pct)

        exit_value = qty * effective_exit_price * (1 - commission)
        profit_loss = exit_value - position['cost']
        profit_loss_pct = (profit_loss / position['cost']) * 100

        return {
            'ticker': ticker,
            'action': 'BUY/SELL',
            'quantity': qty,
            'entry_price': entry_price,
            'exit_price': effective_exit_price,
            'exit_market_price': exit_price,
            'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'profit_loss_usd': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'exit_reason': exit_reason,
            'exit_value': exit_value  # Add exit value for correct capital management
        }

    def _optimize_strategy_on_insample(
        self,
        strategy: Dict[str, Any],
        in_sample_data: pd.DataFrame,
        commission: float,
        slippage_pct: float,
        spread_pct: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Simple in-sample optimization over risk management parameters."""
        risk_cfg = strategy.get("risk_management", {})

        base_stop = float(risk_cfg.get("stop_loss_pct", 5.0))
        base_take = float(risk_cfg.get("take_profit_pct", 10.0))
        base_size = float(risk_cfg.get("position_size_pct", 10.0))

        optimization_grid = strategy.get("optimization_grid", {})
        stop_candidates = optimization_grid.get("stop_loss_pct", [base_stop * 0.8, base_stop, base_stop * 1.2])
        take_candidates = optimization_grid.get("take_profit_pct", [base_take * 0.8, base_take, base_take * 1.2])
        size_candidates = optimization_grid.get("position_size_pct", [base_size * 0.75, base_size, base_size * 1.25])

        # Prevent invalid sizing/thresholds while keeping optimization local.
        stop_candidates = sorted(set(max(0.1, float(v)) for v in stop_candidates))
        take_candidates = sorted(set(max(0.1, float(v)) for v in take_candidates))
        size_candidates = sorted(set(min(100.0, max(0.1, float(v))) for v in size_candidates))

        best_strategy = strategy
        best_metrics = None
        best_score = (float("-inf"), float("-inf"), float("-inf"))

        for stop_loss in stop_candidates:
            for take_profit in take_candidates:
                for position_size in size_candidates:
                    candidate_strategy = dict(strategy)
                    candidate_risk = dict(risk_cfg)
                    candidate_risk.update({
                        "stop_loss_pct": stop_loss,
                        "take_profit_pct": take_profit,
                        "position_size_pct": position_size
                    })
                    candidate_strategy["risk_management"] = candidate_risk

                    result = self.backtest_strategy(
                        strategy=candidate_strategy,
                        market_data=in_sample_data,
                        commission=commission,
                        slippage_pct=slippage_pct,
                        spread_pct=spread_pct
                    )
                    metrics = result.get("metrics", {})
                    score = (
                        float(metrics.get("quality_score", 0.0)),
                        float(metrics.get("sharpe_ratio", 0.0)),
                        float(metrics.get("total_return_pct", 0.0))
                    )

                    if score > best_score:
                        best_score = score
                        best_strategy = candidate_strategy
                        best_metrics = metrics

        return best_strategy, (best_metrics or {})

    @staticmethod
    def _calculate_metric_degradation(in_sample_value: float, out_sample_value: float) -> float:
        """Return degradation percentage where larger values indicate worse OOS performance."""
        baseline = abs(float(in_sample_value))
        if baseline < 1e-9:
            return 0.0 if out_sample_value >= in_sample_value else 100.0
        return max(0.0, ((in_sample_value - out_sample_value) / baseline) * 100)

    @staticmethod
    def _calculate_max_drawdown_pct(equity_values: List[float]) -> float:
        """Calculate max drawdown percentage from equity values."""
        if not equity_values:
            return 0.0
        peak = equity_values[0]
        max_dd = 0.0
        for value in equity_values:
            if value > peak:
                peak = value
            if peak > 0:
                max_dd = max(max_dd, ((peak - value) / peak) * 100)
        return max_dd

    @staticmethod
    def _summarize_distribution(values: List[float]) -> Dict[str, float]:
        """Summarize distribution with requested percentile bands and core stats."""
        if not values:
            return {
                "p5": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p95": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }

        arr = np.array(values, dtype=float)
        return {
            "p5": round(float(np.percentile(arr, 5)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2)
        }

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
