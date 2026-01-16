"""
Pair Trading Mean Reversion Strategy

A comprehensive pair trading strategy that tests for cointegration and generates
signals based on z-score mean reversion. Supports both auto-discovery of pairs
and manual pair specification.

Usage:
    # Analyze a specific pair
    python pair_trading_strategy.py --pair AAPL MSFT --analyze

    # Scan for cointegrated pairs
    python pair_trading_strategy.py --scan --universe tech

    # Backtest a pair
    python pair_trading_strategy.py --backtest AAPL MSFT --period 1y

Author: AI Trading Platform
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Statistical libraries
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

warnings.filterwarnings('ignore')


class Signal(Enum):
    """Trading signal types"""
    LONG_SPREAD = "LONG_SPREAD"    # Buy stock A, Sell stock B
    SHORT_SPREAD = "SHORT_SPREAD"  # Sell stock A, Buy stock B
    EXIT = "EXIT"                  # Close position
    HOLD = "HOLD"                  # Maintain current position


@dataclass
class CointegrationResult:
    """Result from cointegration test"""
    is_cointegrated: bool
    p_value: float
    hedge_ratio: float
    test_statistic: float
    critical_values: Dict[str, float]


@dataclass
class PairAnalysis:
    """Complete pair analysis result"""
    stock_a: str
    stock_b: str
    correlation: float
    cointegration: CointegrationResult
    adf_p_value: float
    hurst_exponent: float
    half_life: float
    quality_score: float
    spread_mean: float
    spread_std: float
    current_zscore: float
    recommendation: str


class PairTradingStatistics:
    """
    Statistical tests for pair trading cointegration analysis.

    Implements:
    - Engle-Granger cointegration test
    - Johansen cointegration test
    - ADF stationarity test
    - Hurst exponent calculation
    - Half-life calculation
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistics calculator.

        Args:
            significance_level: P-value threshold for tests (default 0.05)
        """
        self.significance_level = significance_level

    def engle_granger_test(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        trend: str = 'c'
    ) -> CointegrationResult:
        """
        Perform Engle-Granger cointegration test.

        The test checks if two time series are cointegrated using the two-step method:
        1. Regress series_a on series_b to get hedge ratio
        2. Test the residuals for stationarity

        Args:
            series_a: First price series
            series_b: Second price series
            trend: Trend assumption ('c' constant, 'ct' constant+trend, 'n' none)

        Returns:
            CointegrationResult with test details
        """
        # Align series
        series_a = series_a.dropna()
        series_b = series_b.dropna()

        # Get common index
        common_idx = series_a.index.intersection(series_b.index)
        series_a = series_a.loc[common_idx]
        series_b = series_b.loc[common_idx]

        if len(series_a) < 50:
            return CointegrationResult(
                is_cointegrated=False,
                p_value=1.0,
                hedge_ratio=0.0,
                test_statistic=0.0,
                critical_values={}
            )

        # Perform cointegration test
        test_stat, p_value, crit_values = coint(series_a, series_b, trend=trend)

        # Calculate hedge ratio using OLS
        X = add_constant(series_b)
        model = OLS(series_a, X).fit()
        hedge_ratio = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]

        return CointegrationResult(
            is_cointegrated=p_value < self.significance_level,
            p_value=float(p_value),
            hedge_ratio=float(hedge_ratio),
            test_statistic=float(test_stat),
            critical_values={
                "1%": float(crit_values[0]),
                "5%": float(crit_values[1]),
                "10%": float(crit_values[2])
            }
        )

    def johansen_test(
        self,
        series_list: List[pd.Series],
        det_order: int = 0,
        k_ar_diff: int = 1
    ) -> Dict[str, Any]:
        """
        Perform Johansen cointegration test for multiple series.

        Useful for testing cointegration among more than 2 assets.

        Args:
            series_list: List of price series
            det_order: Deterministic trend order (-1, 0, 1)
            k_ar_diff: Number of lagged differences in VAR model

        Returns:
            Dictionary with trace and eigenvalue test results
        """
        # Create DataFrame from series
        df = pd.concat(series_list, axis=1).dropna()

        if len(df) < 50:
            return {"success": False, "error": "Insufficient data"}

        try:
            result = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

            # Get number of cointegrating relations
            trace_stat = result.lr1  # Trace statistics
            trace_crit = result.cvt  # Critical values for trace test

            # Count cointegrating relationships at 5% level
            n_coint = sum(trace_stat > trace_crit[:, 1])  # Column 1 is 5% critical value

            return {
                "success": True,
                "n_cointegrating_relations": int(n_coint),
                "trace_statistics": trace_stat.tolist(),
                "trace_critical_values_5pct": trace_crit[:, 1].tolist(),
                "eigenvalues": result.eig.tolist(),
                "eigenvectors": result.evec.tolist()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def adf_test(self, spread: pd.Series) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test on spread.

        Tests if the spread is stationary (mean-reverting).

        Args:
            spread: Spread series to test

        Returns:
            Dictionary with ADF test results
        """
        spread = spread.dropna()

        if len(spread) < 20:
            return {
                "is_stationary": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "critical_values": {}
            }

        result = adfuller(spread, autolag='AIC')

        return {
            "is_stationary": result[1] < self.significance_level,
            "p_value": float(result[1]),
            "test_statistic": float(result[0]),
            "critical_values": {
                "1%": float(result[4]['1%']),
                "5%": float(result[4]['5%']),
                "10%": float(result[4]['10%'])
            },
            "n_lags": int(result[2]),
            "n_observations": int(result[3])
        }

    def calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 100) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H < 0.5: Mean reverting (good for pair trading)
        H = 0.5: Random walk (geometric Brownian motion)
        H > 0.5: Trending

        Args:
            series: Time series to analyze
            max_lag: Maximum lag for R/S calculation

        Returns:
            Hurst exponent (0 to 1)
        """
        series = series.dropna().values
        n = len(series)

        if n < 20:
            return 0.5  # Default to random walk

        max_lag = min(max_lag, n // 4)
        lags = range(2, max_lag)

        rs_values = []
        for lag in lags:
            # Divide series into chunks
            n_chunks = n // lag
            if n_chunks < 1:
                continue

            rs_chunk = []
            for i in range(n_chunks):
                chunk = series[i * lag:(i + 1) * lag]

                # Calculate mean-adjusted cumulative sum
                mean_adj = chunk - np.mean(chunk)
                cumsum = np.cumsum(mean_adj)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                S = np.std(chunk, ddof=1) if len(chunk) > 1 else 1.0

                if S > 0:
                    rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))

        if len(rs_values) < 3:
            return 0.5

        # Fit log-log regression
        lags_arr = np.array([x[0] for x in rs_values])
        rs_arr = np.array([x[1] for x in rs_values])

        # Filter out zeros and negative values
        mask = rs_arr > 0
        if not np.any(mask):
            return 0.5

        log_lags = np.log(lags_arr[mask])
        log_rs = np.log(rs_arr[mask])

        # Linear regression to get Hurst exponent
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]

        # Clip to valid range
        return float(np.clip(hurst, 0.0, 1.0))

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.

        Half-life indicates how many periods it takes for the spread to revert
        halfway to its mean.

        Args:
            spread: Spread series

        Returns:
            Half-life in number of periods (days)
        """
        spread = spread.dropna()

        if len(spread) < 10:
            return float('inf')

        # Prepare lagged spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]

        if len(spread_lag) < 10:
            return float('inf')

        # OLS regression: spread_diff = theta * spread_lag + epsilon
        X = add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()

        theta = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]

        if theta >= 0:
            return float('inf')  # Not mean reverting

        # Half-life = -ln(2) / theta
        half_life = -np.log(2) / theta

        return float(max(0, half_life))


class PairTradingStrategy:
    """
    Z-score based pair trading signal generator.

    Entry Rules:
        - LONG spread: z-score < -entry_threshold -> BUY stock A, SELL stock B
        - SHORT spread: z-score > +entry_threshold -> SELL stock A, BUY stock B

    Exit Rules:
        - Mean reversion: |z-score| < exit_threshold -> Close positions
        - Stop loss: |z-score| > stop_loss_threshold -> Emergency exit
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0,
        lookback_period: int = 100,
        rolling_window: int = 20
    ):
        """
        Initialize pair trading strategy.

        Args:
            entry_threshold: Z-score level for entry (default 2.0)
            exit_threshold: Z-score level for exit (default 0.5)
            stop_loss_threshold: Z-score level for stop loss (default 4.0)
            lookback_period: Days for cointegration test (default 100)
            rolling_window: Days for z-score calculation (default 20)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.lookback_period = lookback_period
        self.rolling_window = rolling_window
        self.stats = PairTradingStatistics()

    def calculate_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> Tuple[pd.Series, float]:
        """
        Calculate the spread between two price series.

        Spread = Price_A - hedge_ratio * Price_B

        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            hedge_ratio: Optional pre-calculated hedge ratio

        Returns:
            Tuple of (spread series, hedge ratio used)
        """
        # Align series
        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.loc[common_idx]
        prices_b = prices_b.loc[common_idx]

        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            X = add_constant(prices_b)
            model = OLS(prices_a, X).fit()
            hedge_ratio = model.params.iloc[1] if len(model.params) > 1 else model.params.iloc[0]

        # Calculate spread
        spread = prices_a - hedge_ratio * prices_b

        return spread, float(hedge_ratio)

    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of the spread.

        Args:
            spread: Spread series
            window: Rolling window size (default: self.rolling_window)

        Returns:
            Z-score series
        """
        if window is None:
            window = self.rolling_window

        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        zscore = (spread - rolling_mean) / rolling_std

        return zscore

    def generate_signals(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals based on z-score.

        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            hedge_ratio: Optional pre-calculated hedge ratio

        Returns:
            DataFrame with signals, z-scores, and positions
        """
        # Calculate spread and z-score
        spread, hedge_ratio = self.calculate_spread(prices_a, prices_b, hedge_ratio)
        zscore = self.calculate_zscore(spread)

        # Initialize signal series
        signals = pd.Series(index=zscore.index, data=Signal.HOLD.value)
        positions = pd.Series(index=zscore.index, data=0)  # 1: long spread, -1: short spread

        current_position = 0

        for i, (idx, z) in enumerate(zscore.items()):
            if pd.isna(z):
                continue

            # Check stop loss first
            if current_position != 0 and abs(z) > self.stop_loss_threshold:
                signals.iloc[i] = Signal.EXIT.value
                current_position = 0

            # Entry signals
            elif current_position == 0:
                if z < -self.entry_threshold:
                    signals.iloc[i] = Signal.LONG_SPREAD.value
                    current_position = 1
                elif z > self.entry_threshold:
                    signals.iloc[i] = Signal.SHORT_SPREAD.value
                    current_position = -1

            # Exit signals
            elif current_position == 1 and z > -self.exit_threshold:
                signals.iloc[i] = Signal.EXIT.value
                current_position = 0
            elif current_position == -1 and z < self.exit_threshold:
                signals.iloc[i] = Signal.EXIT.value
                current_position = 0

            positions.iloc[i] = current_position

        return pd.DataFrame({
            'date': zscore.index,
            'price_a': prices_a.loc[zscore.index],
            'price_b': prices_b.loc[zscore.index],
            'spread': spread,
            'zscore': zscore,
            'signal': signals,
            'position': positions,
            'hedge_ratio': hedge_ratio
        }).set_index('date')

    def get_current_signal(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series
    ) -> Dict[str, Any]:
        """
        Get current trading signal.

        Args:
            prices_a: Recent price series for stock A
            prices_b: Recent price series for stock B

        Returns:
            Dictionary with current signal and analysis
        """
        signals_df = self.generate_signals(prices_a, prices_b)

        if signals_df.empty:
            return {
                "signal": Signal.HOLD.value,
                "error": "Insufficient data"
            }

        latest = signals_df.iloc[-1]

        return {
            "signal": latest['signal'],
            "zscore": float(latest['zscore']),
            "spread": float(latest['spread']),
            "hedge_ratio": float(latest['hedge_ratio']),
            "price_a": float(latest['price_a']),
            "price_b": float(latest['price_b']),
            "position": int(latest['position']),
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        }


class PairBacktester:
    """
    Backtesting engine for pair trading strategies.

    Implements dollar-neutral position sizing with transaction costs,
    borrowing costs, and slippage modeling.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,    # 0.1% per trade
        borrowing_cost_annual: float = 0.02,  # 2% annual for shorts
        slippage: float = 0.0005              # 0.05% slippage
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital (default $100,000)
            transaction_cost: Cost per trade as decimal (default 0.1%)
            borrowing_cost_annual: Annual borrowing rate for shorts (default 2%)
            slippage: Slippage per trade as decimal (default 0.05%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.borrowing_cost_annual = borrowing_cost_annual
        self.slippage = slippage
        self.borrowing_cost_daily = borrowing_cost_annual / 252

    def backtest_pair(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        strategy: PairTradingStrategy,
        position_size_pct: float = 0.5  # Use 50% of capital per side
    ) -> Dict[str, Any]:
        """
        Backtest a pair trading strategy.

        Args:
            prices_a: Historical prices for stock A
            prices_b: Historical prices for stock B
            strategy: PairTradingStrategy instance
            position_size_pct: Percentage of capital per side (default 50%)

        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals_df = strategy.generate_signals(prices_a, prices_b)

        if signals_df.empty or len(signals_df) < 20:
            return {
                "success": False,
                "error": "Insufficient data for backtesting"
            }

        # Track portfolio
        capital = self.initial_capital
        position_a = 0
        position_b = 0
        entry_price_a = 0
        entry_price_b = 0

        equity_curve = []
        trades = []

        position_capital = self.initial_capital * position_size_pct

        for idx, row in signals_df.iterrows():
            price_a = row['price_a']
            price_b = row['price_b']
            signal = row['signal']
            hedge_ratio = row['hedge_ratio']

            # Apply borrowing costs for short positions
            if position_b < 0:  # Short stock B
                borrow_cost = abs(position_b * price_b) * self.borrowing_cost_daily
                capital -= borrow_cost
            if position_a < 0:  # Short stock A
                borrow_cost = abs(position_a * price_a) * self.borrowing_cost_daily
                capital -= borrow_cost

            # Process signals
            if signal == Signal.LONG_SPREAD.value and position_a == 0:
                # Buy A, Sell B (dollar neutral)
                shares_a = int(position_capital / price_a)
                shares_b = int((position_capital * hedge_ratio) / price_b)

                # Apply slippage
                exec_price_a = price_a * (1 + self.slippage)
                exec_price_b = price_b * (1 - self.slippage)

                cost_a = shares_a * exec_price_a * (1 + self.transaction_cost)
                proceeds_b = shares_b * exec_price_b * (1 - self.transaction_cost)

                capital = capital - cost_a + proceeds_b
                position_a = shares_a
                position_b = -shares_b  # Short
                entry_price_a = exec_price_a
                entry_price_b = exec_price_b

            elif signal == Signal.SHORT_SPREAD.value and position_a == 0:
                # Sell A, Buy B (dollar neutral)
                shares_a = int(position_capital / price_a)
                shares_b = int((position_capital * hedge_ratio) / price_b)

                # Apply slippage
                exec_price_a = price_a * (1 - self.slippage)
                exec_price_b = price_b * (1 + self.slippage)

                proceeds_a = shares_a * exec_price_a * (1 - self.transaction_cost)
                cost_b = shares_b * exec_price_b * (1 + self.transaction_cost)

                capital = capital + proceeds_a - cost_b
                position_a = -shares_a  # Short
                position_b = shares_b
                entry_price_a = exec_price_a
                entry_price_b = exec_price_b

            elif signal == Signal.EXIT.value and position_a != 0:
                # Close both positions
                if position_a > 0:  # Was long A, short B
                    exec_price_a = price_a * (1 - self.slippage)
                    exec_price_b = price_b * (1 + self.slippage)

                    proceeds_a = position_a * exec_price_a * (1 - self.transaction_cost)
                    cost_b = abs(position_b) * exec_price_b * (1 + self.transaction_cost)

                    pnl = proceeds_a - position_a * entry_price_a
                    pnl += abs(position_b) * entry_price_b - cost_b
                else:  # Was short A, long B
                    exec_price_a = price_a * (1 + self.slippage)
                    exec_price_b = price_b * (1 - self.slippage)

                    cost_a = abs(position_a) * exec_price_a * (1 + self.transaction_cost)
                    proceeds_b = position_b * exec_price_b * (1 - self.transaction_cost)

                    pnl = abs(position_a) * entry_price_a - cost_a
                    pnl += proceeds_b - position_b * entry_price_b

                capital += pnl

                trades.append({
                    "entry_date": str(entry_price_a),  # Placeholder
                    "exit_date": str(idx),
                    "position_type": "LONG_SPREAD" if position_a > 0 else "SHORT_SPREAD",
                    "pnl": float(pnl),
                    "return_pct": float(pnl / position_capital * 100)
                })

                position_a = 0
                position_b = 0

            # Calculate equity
            portfolio_value = capital
            if position_a != 0:
                portfolio_value += position_a * price_a
                portfolio_value += position_b * price_b

            equity_curve.append({
                "date": str(idx),
                "equity": float(portfolio_value),
                "position": "LONG_SPREAD" if position_a > 0 else ("SHORT_SPREAD" if position_a < 0 else "FLAT")
            })

        # Calculate metrics
        returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()

        total_return = (equity_curve[-1]['equity'] - self.initial_capital) / self.initial_capital * 100

        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Sharpe ratio (annualized)
        sharpe = 0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # Max drawdown
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdown.min())

        # Profit factor
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            "success": True,
            "metrics": {
                "total_return_pct": float(total_return),
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": float(profit_factor),
                "sharpe_ratio": float(sharpe),
                "max_drawdown_pct": float(max_drawdown),
                "final_equity": float(equity_curve[-1]['equity'])
            },
            "trades": trades,
            "equity_curve": equity_curve
        }


class PairScanner:
    """
    Auto-discovery of cointegrated trading pairs.

    Scans a universe of stocks to find pairs suitable for pair trading:
    1. Pre-filter by correlation (correlation > threshold)
    2. Test for cointegration (Engle-Granger test)
    3. Rank by quality score
    """

    # Predefined universes
    UNIVERSES = {
        "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL",
                 "ADBE", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "NOW", "SNOW", "MU", "AMAT"],
        "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
                    "PNC", "TFC", "COF", "BK", "STT", "CME", "ICE", "SPGI", "MCO", "MSCI"],
        "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "LLY",
                       "AMGN", "GILD", "MDT", "CVS", "CI", "ISRG", "SYK", "ZTS", "VRTX", "REGN"],
        "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD",
                   "DVN", "HES", "HAL", "BKR", "FANG", "WMB", "KMI", "OKE", "TRGP", "LNG"],
        "consumer": ["WMT", "HD", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT",
                     "LOW", "EL", "CL", "MDLZ", "KHC", "GIS", "K", "SYY", "KR", "DG"],
        "etf": ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLI", "XLP",
                "XLY", "XLB", "XLU", "GLD", "SLV", "TLT", "IEF", "HYG", "LQD", "VXX"],
        "gold": ["GLD", "GDX", "GOLD", "NEM", "AEM", "FNV", "WPM", "RGLD", "KGC", "AU"]
    }

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        significance_level: float = 0.05,
        max_workers: int = 10
    ):
        """
        Initialize pair scanner.

        Args:
            correlation_threshold: Minimum correlation for pre-filter (default 0.7)
            significance_level: P-value threshold for cointegration (default 0.05)
            max_workers: Number of parallel threads (default 10)
        """
        self.correlation_threshold = correlation_threshold
        self.significance_level = significance_level
        self.max_workers = max_workers
        self.stats = PairTradingStatistics(significance_level)

    def scan_for_pairs(
        self,
        universe: Optional[List[str]] = None,
        universe_name: Optional[str] = None,
        period: str = "1y",
        min_quality_score: float = 50.0
    ) -> List[PairAnalysis]:
        """
        Scan universe for cointegrated pairs.

        Args:
            universe: List of tickers to scan
            universe_name: Name of predefined universe (tech, finance, etc.)
            period: Historical data period (default 1 year)
            min_quality_score: Minimum quality score to include (default 50)

        Returns:
            List of PairAnalysis objects, sorted by quality score
        """
        # Get universe
        if universe_name and universe_name.lower() in self.UNIVERSES:
            tickers = self.UNIVERSES[universe_name.lower()]
        elif universe:
            tickers = universe
        else:
            tickers = self.UNIVERSES["tech"]  # Default to tech

        print(f"Scanning {len(tickers)} stocks for cointegrated pairs...")

        # Download all price data
        prices = {}
        for ticker in tickers:
            try:
                data = yf.download(ticker, period=period, progress=False)
                if not data.empty and len(data) > 60:
                    prices[ticker] = data['Close']
            except Exception:
                continue

        print(f"Successfully downloaded data for {len(prices)} stocks")

        if len(prices) < 2:
            return []

        # Generate all pairs
        tickers_with_data = list(prices.keys())
        pairs = []
        for i in range(len(tickers_with_data)):
            for j in range(i + 1, len(tickers_with_data)):
                pairs.append((tickers_with_data[i], tickers_with_data[j]))

        print(f"Testing {len(pairs)} pairs...")

        # Test pairs in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pair = {
                executor.submit(
                    self._analyze_pair,
                    prices[pair[0]],
                    prices[pair[1]],
                    pair[0],
                    pair[1]
                ): pair
                for pair in pairs
            }

            for future in as_completed(future_to_pair):
                try:
                    analysis = future.result()
                    if analysis and analysis.quality_score >= min_quality_score:
                        results.append(analysis)
                except Exception:
                    continue

        # Sort by quality score
        results.sort(key=lambda x: x.quality_score, reverse=True)

        print(f"Found {len(results)} cointegrated pairs with quality >= {min_quality_score}")

        return results

    def _analyze_pair(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        ticker_a: str,
        ticker_b: str
    ) -> Optional[PairAnalysis]:
        """
        Analyze a single pair.

        Args:
            prices_a: Price series for stock A
            prices_b: Price series for stock B
            ticker_a: Ticker symbol for stock A
            ticker_b: Ticker symbol for stock B

        Returns:
            PairAnalysis if pair passes pre-filter, None otherwise
        """
        # Align series
        common_idx = prices_a.index.intersection(prices_b.index)
        prices_a = prices_a.loc[common_idx]
        prices_b = prices_b.loc[common_idx]

        if len(prices_a) < 60:
            return None

        # Pre-filter by correlation
        correlation = prices_a.corr(prices_b)
        if abs(correlation) < self.correlation_threshold:
            return None

        # Cointegration test
        coint_result = self.stats.engle_granger_test(prices_a, prices_b)

        if not coint_result.is_cointegrated:
            return None

        # Calculate spread
        spread = prices_a - coint_result.hedge_ratio * prices_b

        # ADF test on spread
        adf_result = self.stats.adf_test(spread)

        # Hurst exponent
        hurst = self.stats.calculate_hurst_exponent(spread)

        # Half-life
        half_life = self.stats.calculate_half_life(spread)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            coint_p_value=coint_result.p_value,
            adf_p_value=adf_result['p_value'],
            hurst=hurst,
            half_life=half_life
        )

        # Current z-score
        rolling_mean = spread.rolling(20).mean()
        rolling_std = spread.rolling(20).std()
        current_zscore = (spread.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        # Generate recommendation
        recommendation = "WAIT"
        if current_zscore < -2.0:
            recommendation = "LONG_SPREAD"
        elif current_zscore > 2.0:
            recommendation = "SHORT_SPREAD"
        elif abs(current_zscore) < 0.5:
            recommendation = "SPREAD_AT_MEAN"

        return PairAnalysis(
            stock_a=ticker_a,
            stock_b=ticker_b,
            correlation=float(correlation),
            cointegration=coint_result,
            adf_p_value=float(adf_result['p_value']),
            hurst_exponent=float(hurst),
            half_life=float(half_life),
            quality_score=float(quality_score),
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            current_zscore=float(current_zscore) if not np.isnan(current_zscore) else 0.0,
            recommendation=recommendation
        )

    def _calculate_quality_score(
        self,
        coint_p_value: float,
        adf_p_value: float,
        hurst: float,
        half_life: float
    ) -> float:
        """
        Calculate quality score (0-100).

        | Factor         | Max Points | Criteria                           |
        |----------------|------------|------------------------------------|
        | Cointegration  | 30         | p < 0.01 = 30, p < 0.05 = 20       |
        | Hurst Exponent | 25         | H < 0.3 = 25, H < 0.4 = 20, H < 0.5 = 15 |
        | Half-Life      | 25         | 5-20 days = 25, 20-40 = 20, 40-60 = 10 |
        | ADF Stationarity | 20       | p < 0.01 = 20, p < 0.05 = 15       |
        """
        score = 0.0

        # Cointegration score (30 max)
        if coint_p_value < 0.01:
            score += 30
        elif coint_p_value < 0.05:
            score += 20
        elif coint_p_value < 0.10:
            score += 10

        # Hurst exponent score (25 max)
        if hurst < 0.3:
            score += 25
        elif hurst < 0.4:
            score += 20
        elif hurst < 0.5:
            score += 15
        elif hurst < 0.6:
            score += 5

        # Half-life score (25 max)
        if 5 <= half_life <= 20:
            score += 25
        elif 20 < half_life <= 40:
            score += 20
        elif 40 < half_life <= 60:
            score += 10
        elif 3 <= half_life < 5:
            score += 15
        elif half_life > 60:
            score += 5

        # ADF stationarity score (20 max)
        if adf_p_value < 0.01:
            score += 20
        elif adf_p_value < 0.05:
            score += 15
        elif adf_p_value < 0.10:
            score += 10

        return score


def analyze_pair_cli(ticker_a: str, ticker_b: str, period: str = "1y") -> Dict[str, Any]:
    """CLI function to analyze a specific pair."""
    print(f"\nAnalyzing pair: {ticker_a} / {ticker_b}")
    print("=" * 50)

    # Download data (squeeze to convert DataFrame to Series)
    data_a = yf.download(ticker_a, period=period, progress=False)['Close'].squeeze()
    data_b = yf.download(ticker_b, period=period, progress=False)['Close'].squeeze()

    if len(data_a) == 0 or len(data_b) == 0:
        return {"error": "Failed to download data"}

    stats = PairTradingStatistics()
    strategy = PairTradingStrategy()

    # Cointegration test
    coint_result = stats.engle_granger_test(data_a, data_b)
    print(f"\nCointegration Test:")
    print(f"  Is Cointegrated: {coint_result.is_cointegrated}")
    print(f"  P-Value: {coint_result.p_value:.4f}")
    print(f"  Hedge Ratio: {coint_result.hedge_ratio:.4f}")

    # Calculate spread
    spread, hedge_ratio = strategy.calculate_spread(data_a, data_b, coint_result.hedge_ratio)

    # ADF test
    adf_result = stats.adf_test(spread)
    print(f"\nADF Test (Spread Stationarity):")
    print(f"  Is Stationary: {adf_result['is_stationary']}")
    print(f"  P-Value: {adf_result['p_value']:.4f}")

    # Hurst exponent
    hurst = stats.calculate_hurst_exponent(spread)
    print(f"\nHurst Exponent: {hurst:.4f}")
    print(f"  Interpretation: {'Mean Reverting' if hurst < 0.5 else 'Trending' if hurst > 0.5 else 'Random Walk'}")

    # Half-life
    half_life = stats.calculate_half_life(spread)
    print(f"\nHalf-Life: {half_life:.1f} days")

    # Current signal
    signal = strategy.get_current_signal(data_a, data_b)
    print(f"\nCurrent Signal:")
    print(f"  Signal: {signal['signal']}")
    print(f"  Z-Score: {signal['zscore']:.2f}")

    return {
        "cointegration": {
            "is_cointegrated": coint_result.is_cointegrated,
            "p_value": coint_result.p_value,
            "hedge_ratio": coint_result.hedge_ratio
        },
        "adf_test": adf_result,
        "hurst_exponent": hurst,
        "half_life": half_life,
        "current_signal": signal
    }


def scan_pairs_cli(universe: str = "tech", period: str = "1y") -> List[Dict]:
    """CLI function to scan for cointegrated pairs."""
    scanner = PairScanner()
    results = scanner.scan_for_pairs(universe_name=universe, period=period)

    print(f"\nTop Cointegrated Pairs ({universe.upper()}):")
    print("=" * 80)

    for i, pair in enumerate(results[:10], 1):
        print(f"\n{i}. {pair.stock_a} / {pair.stock_b}")
        print(f"   Quality Score: {pair.quality_score:.1f}")
        print(f"   Correlation: {pair.correlation:.3f}")
        print(f"   Cointegration p-value: {pair.cointegration.p_value:.4f}")
        print(f"   Hurst Exponent: {pair.hurst_exponent:.3f}")
        print(f"   Half-Life: {pair.half_life:.1f} days")
        print(f"   Current Z-Score: {pair.current_zscore:.2f}")
        print(f"   Recommendation: {pair.recommendation}")

    return [
        {
            "stock_a": p.stock_a,
            "stock_b": p.stock_b,
            "quality_score": p.quality_score,
            "correlation": p.correlation,
            "cointegration_pvalue": p.cointegration.p_value,
            "hedge_ratio": p.cointegration.hedge_ratio,
            "hurst_exponent": p.hurst_exponent,
            "half_life": p.half_life,
            "current_zscore": p.current_zscore,
            "recommendation": p.recommendation
        }
        for p in results
    ]


def backtest_pair_cli(
    ticker_a: str,
    ticker_b: str,
    period: str = "1y",
    initial_capital: float = 100000
) -> Dict[str, Any]:
    """CLI function to backtest a pair."""
    print(f"\nBacktesting pair: {ticker_a} / {ticker_b}")
    print(f"Period: {period}, Initial Capital: ${initial_capital:,.0f}")
    print("=" * 50)

    # Download data (squeeze to convert DataFrame to Series)
    data_a = yf.download(ticker_a, period=period, progress=False)['Close'].squeeze()
    data_b = yf.download(ticker_b, period=period, progress=False)['Close'].squeeze()

    if len(data_a) == 0 or len(data_b) == 0:
        return {"error": "Failed to download data"}

    strategy = PairTradingStrategy()
    backtester = PairBacktester(initial_capital=initial_capital)

    results = backtester.backtest_pair(data_a, data_b, strategy)

    if results['success']:
        metrics = results['metrics']
        print(f"\nBacktest Results:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Final Equity: ${metrics['final_equity']:,.0f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pair Trading Strategy")
    parser.add_argument("--pair", nargs=2, metavar=("TICKER_A", "TICKER_B"),
                        help="Analyze a specific pair")
    parser.add_argument("--scan", action="store_true",
                        help="Scan for cointegrated pairs")
    parser.add_argument("--backtest", nargs=2, metavar=("TICKER_A", "TICKER_B"),
                        help="Backtest a pair")
    parser.add_argument("--universe", type=str, default="tech",
                        choices=["tech", "finance", "healthcare", "energy", "consumer", "etf", "gold"],
                        help="Universe for scanning (default: tech)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Historical period (default: 1y)")
    parser.add_argument("--capital", type=float, default=100000,
                        help="Initial capital for backtest (default: 100000)")

    args = parser.parse_args()

    if args.pair:
        analyze_pair_cli(args.pair[0], args.pair[1], args.period)
    elif args.scan:
        scan_pairs_cli(args.universe, args.period)
    elif args.backtest:
        backtest_pair_cli(args.backtest[0], args.backtest[1], args.period, args.capital)
    else:
        # Default: show example usage
        print("Pair Trading Strategy - Examples:")
        print("\n1. Analyze a specific pair:")
        print("   python pair_trading_strategy.py --pair AAPL MSFT --analyze")
        print("\n2. Scan for cointegrated pairs:")
        print("   python pair_trading_strategy.py --scan --universe tech")
        print("\n3. Backtest a pair:")
        print("   python pair_trading_strategy.py --backtest GLD GDX --period 2y")

        # Run a quick demo
        print("\n" + "=" * 50)
        print("Running demo with GLD/GDX (classic gold pair)...")
        print("=" * 50)
        analyze_pair_cli("GLD", "GDX", "1y")
