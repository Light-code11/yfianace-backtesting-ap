"""
Advanced Risk Metrics Calculator
Professional-grade risk analysis for trading strategies
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats

try:
    import empyrical as ep
    EMPYRICAL_AVAILABLE = True
except ImportError:
    EMPYRICAL_AVAILABLE = False
    print("Warning: empyrical not available. Using basic risk metrics only.")


class AdvancedRiskMetrics:
    """
    Advanced risk metrics calculator using empyrical and custom calculations

    Metrics included:
    - Value at Risk (VaR): Parametric and Historical
    - Conditional VaR (CVaR/Expected Shortfall)
    - Sortino Ratio: Return/downside deviation
    - Calmar Ratio: Return/max drawdown
    - Ulcer Index: Drawdown depth and duration
    - Pain Index: Average drawdown
    - Max Drawdown: Largest peak-to-trough decline
    - Drawdown Duration: Time spent in drawdown
    - Win/Loss Streaks: Consecutive wins/losses
    - Tail Risk Metrics: Kurtosis, Skewness
    """

    @staticmethod
    def calculate_returns(equity_curve: List[Dict[str, Any]]) -> pd.Series:
        """
        Extract returns from equity curve

        Args:
            equity_curve: List of dicts with 'equity' values

        Returns:
            Series of returns
        """
        if not equity_curve or len(equity_curve) < 2:
            return pd.Series([])

        equity_values = [e['equity'] for e in equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()

        return returns

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        VaR = "There is a (1-confidence_level)% chance of losing more than X%"

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 0.95 = 95%)
            method: "historical" or "parametric"

        Returns:
            VaR as a positive percentage (e.g., 5.0 means 5% potential loss)
        """
        if len(returns) < 2:
            return 0.0

        if method == "historical":
            # Historical VaR: Use actual return distribution
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        else:
            # Parametric VaR: Assume normal distribution
            mean = returns.mean()
            std = returns.std()
            var = -(mean - stats.norm.ppf(1 - confidence_level) * std)

        return max(float(var), 0.0)

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall

        CVaR = "If the VaR threshold is breached, expected loss is X%"
        CVaR is the average of all losses beyond VaR threshold

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 0.95)
            method: "historical" or "parametric"

        Returns:
            CVaR as a positive percentage
        """
        if len(returns) < 2:
            return 0.0

        var = -AdvancedRiskMetrics.value_at_risk(returns, confidence_level, method)

        if method == "historical":
            # CVaR = average of returns worse than VaR
            worst_returns = returns[returns <= var]
            if len(worst_returns) > 0:
                cvar = -worst_returns.mean()
            else:
                cvar = -var
        else:
            # Parametric CVaR
            mean = returns.mean()
            std = returns.std()
            z = stats.norm.ppf(1 - confidence_level)
            cvar = -(mean - std * stats.norm.pdf(z) / (1 - confidence_level))

        return max(float(cvar), 0.0)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio

        Sortino = (Return - RFR) / Downside Deviation
        Like Sharpe, but only penalizes downside volatility

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for daily)

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        if EMPYRICAL_AVAILABLE:
            try:
                sortino = ep.sortino_ratio(
                    returns,
                    required_return=risk_free_rate / periods_per_year
                )
                return float(sortino) if not np.isnan(sortino) else 0.0
            except:
                pass

        # Fallback calculation
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        annualized_return = excess_returns.mean() * periods_per_year
        annualized_downside = downside_std * np.sqrt(periods_per_year)

        sortino = annualized_return / annualized_downside

        return float(sortino)

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio

        Calmar = Annual Return / Max Drawdown
        Measures return per unit of downside risk

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        if EMPYRICAL_AVAILABLE:
            try:
                calmar = ep.calmar_ratio(returns, period='daily')
                return float(calmar) if not np.isnan(calmar) else 0.0
            except:
                pass

        # Fallback calculation
        annual_return = returns.mean() * periods_per_year
        max_dd = AdvancedRiskMetrics.max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        calmar = annual_return / abs(max_dd)

        return float(calmar)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Series of returns

        Returns:
            Max drawdown as negative percentage
        """
        if len(returns) < 2:
            return 0.0

        if EMPYRICAL_AVAILABLE:
            try:
                max_dd = ep.max_drawdown(returns)
                return float(max_dd) if not np.isnan(max_dd) else 0.0
            except:
                pass

        # Fallback calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return float(drawdown.min())

    @staticmethod
    def ulcer_index(returns: pd.Series) -> float:
        """
        Calculate Ulcer Index

        Ulcer Index measures depth and duration of drawdowns
        Lower is better (less painful drawdowns)

        Args:
            returns: Series of returns

        Returns:
            Ulcer index
        """
        if len(returns) < 2:
            return 0.0

        # Calculate drawdown series
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown_pct = ((cumulative - running_max) / running_max) * 100  # As percentage

        # Ulcer Index = sqrt(mean(drawdown²))
        ulcer = np.sqrt((drawdown_pct ** 2).mean())

        return float(ulcer)

    @staticmethod
    def pain_index(returns: pd.Series) -> float:
        """
        Calculate Pain Index (average drawdown)

        Args:
            returns: Series of returns

        Returns:
            Pain index as positive percentage
        """
        if len(returns) < 2:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        pain = -drawdown.mean()  # Make positive

        return float(pain) * 100  # As percentage

    @staticmethod
    def tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate tail risk metrics (skewness and kurtosis)

        Args:
            returns: Series of returns

        Returns:
            Dict with skewness and excess kurtosis
        """
        if len(returns) < 3:
            return {"skewness": 0.0, "kurtosis": 0.0}

        skew = float(returns.skew())
        kurt = float(returns.kurtosis())  # Excess kurtosis (normal = 0)

        return {
            "skewness": skew,
            "kurtosis": kurt
        }

    @staticmethod
    def drawdown_analysis(returns: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis

        Args:
            returns: Series of returns

        Returns:
            Dict with drawdown statistics
        """
        if len(returns) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "avg_drawdown_pct": 0.0,
                "max_drawdown_duration_days": 0,
                "avg_drawdown_duration_days": 0,
                "current_drawdown_pct": 0.0,
                "time_underwater_pct": 0.0
            }

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Max drawdown
        max_dd = float(drawdown.min()) * 100

        # Average drawdown (when in drawdown)
        in_drawdown = drawdown[drawdown < 0]
        avg_dd = float(in_drawdown.mean()) * 100 if len(in_drawdown) > 0 else 0.0

        # Drawdown durations
        underwater = (drawdown < -0.001).astype(int)  # Use small threshold to avoid noise

        # Find drawdown periods
        drawdown_periods = []
        in_dd = False
        start = 0

        for i, val in enumerate(underwater):
            if val == 1 and not in_dd:
                start = i
                in_dd = True
            elif val == 0 and in_dd:
                drawdown_periods.append(i - start)
                in_dd = False

        if in_dd:  # Still in drawdown at end
            drawdown_periods.append(len(underwater) - start)

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_dd_duration = int(np.mean(drawdown_periods)) if drawdown_periods else 0

        # Current drawdown
        current_dd = float(drawdown.iloc[-1]) * 100

        # Time underwater
        time_underwater = (underwater.sum() / len(underwater)) * 100

        return {
            "max_drawdown_pct": max_dd,
            "avg_drawdown_pct": avg_dd,
            "max_drawdown_duration_days": max_dd_duration,
            "avg_drawdown_duration_days": avg_dd_duration,
            "current_drawdown_pct": current_dd,
            "time_underwater_pct": float(time_underwater)
        }

    @staticmethod
    def win_loss_streaks(trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate win and loss streaks

        Args:
            trades: List of trade dicts with 'profit' or 'pnl' field

        Returns:
            Dict with max win/loss streaks
        """
        if not trades:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "current_streak": 0,
                "current_streak_type": "none"
            }

        # Extract profits
        profits = [t.get('profit', t.get('pnl', 0)) for t in trades]

        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        current_is_win = None

        for profit in profits:
            is_win = profit > 0

            if current_is_win is None:
                current_is_win = is_win
                current_streak = 1
            elif is_win == current_is_win:
                current_streak += 1
            else:
                # Streak ended
                if current_is_win:
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)

                current_is_win = is_win
                current_streak = 1

        # Update final streak
        if current_is_win is not None:
            if current_is_win:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_streak": current_streak,
            "current_streak_type": "win" if current_is_win else "loss" if current_is_win is not None else "none"
        }

    @staticmethod
    def comprehensive_risk_analysis(
        equity_curve: List[Dict[str, Any]],
        trades: Optional[List[Dict[str, Any]]] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Complete risk analysis with all metrics

        Args:
            equity_curve: List of equity curve points
            trades: Optional list of trades for streak analysis
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict with all risk metrics
        """
        if not equity_curve or len(equity_curve) < 2:
            return {
                "success": False,
                "error": "Insufficient equity curve data"
            }

        # Extract returns
        returns = AdvancedRiskMetrics.calculate_returns(equity_curve)

        if len(returns) < 2:
            return {
                "success": False,
                "error": "Insufficient returns data"
            }

        # Value at Risk
        var_95 = AdvancedRiskMetrics.value_at_risk(returns, 0.95, "historical")
        var_99 = AdvancedRiskMetrics.value_at_risk(returns, 0.99, "historical")

        # Conditional VaR
        cvar_95 = AdvancedRiskMetrics.conditional_var(returns, 0.95, "historical")
        cvar_99 = AdvancedRiskMetrics.conditional_var(returns, 0.99, "historical")

        # Ratios
        sortino = AdvancedRiskMetrics.sortino_ratio(returns, risk_free_rate)
        calmar = AdvancedRiskMetrics.calmar_ratio(returns)

        # Drawdown metrics
        ulcer = AdvancedRiskMetrics.ulcer_index(returns)
        pain = AdvancedRiskMetrics.pain_index(returns)
        drawdown_stats = AdvancedRiskMetrics.drawdown_analysis(returns)

        # Tail risk
        tail_metrics = AdvancedRiskMetrics.tail_risk_metrics(returns)

        # Win/loss streaks
        if trades:
            streak_stats = AdvancedRiskMetrics.win_loss_streaks(trades)
        else:
            streak_stats = {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "current_streak": 0,
                "current_streak_type": "none"
            }

        return {
            "success": True,
            "value_at_risk": {
                "var_95_pct": round(var_95 * 100, 2),
                "var_99_pct": round(var_99 * 100, 2),
                "interpretation_95": f"95% confidence: Daily loss won't exceed {var_95*100:.2f}%",
                "interpretation_99": f"99% confidence: Daily loss won't exceed {var_99*100:.2f}%"
            },
            "conditional_var": {
                "cvar_95_pct": round(cvar_95 * 100, 2),
                "cvar_99_pct": round(cvar_99 * 100, 2),
                "interpretation_95": f"If VaR breached, expected loss is {cvar_95*100:.2f}%",
                "interpretation_99": f"In worst 1% scenarios, expected loss is {cvar_99*100:.2f}%"
            },
            "risk_adjusted_returns": {
                "sortino_ratio": round(sortino, 2),
                "calmar_ratio": round(calmar, 2),
                "interpretation_sortino": "Higher is better (return per unit downside risk)",
                "interpretation_calmar": "Higher is better (return per unit max drawdown)"
            },
            "drawdown_metrics": {
                "ulcer_index": round(ulcer, 2),
                "pain_index": round(pain, 2),
                "max_drawdown_pct": round(drawdown_stats['max_drawdown_pct'], 2),
                "avg_drawdown_pct": round(drawdown_stats['avg_drawdown_pct'], 2),
                "max_drawdown_duration_days": drawdown_stats['max_drawdown_duration_days'],
                "avg_drawdown_duration_days": drawdown_stats['avg_drawdown_duration_days'],
                "current_drawdown_pct": round(drawdown_stats['current_drawdown_pct'], 2),
                "time_underwater_pct": round(drawdown_stats['time_underwater_pct'], 1)
            },
            "tail_risk": {
                "skewness": round(tail_metrics['skewness'], 3),
                "kurtosis": round(tail_metrics['kurtosis'], 3),
                "interpretation_skewness": "Negative = more left tail risk (bad)",
                "interpretation_kurtosis": "Positive = fatter tails, more extreme events"
            },
            "streaks": streak_stats
        }


if __name__ == "__main__":
    # Test risk metrics
    print("=" * 70)
    print("TESTING ADVANCED RISK METRICS")
    print("=" * 70)
    print(f"Empyrical available: {EMPYRICAL_AVAILABLE}")

    # Create sample equity curve
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    # Simulate returns with some negative skew and fat tails
    returns = np.random.normal(0.001, 0.02, 252)
    returns[::20] = -0.05  # Add some large losses

    equity = (1 + pd.Series(returns)).cumprod() * 100000

    equity_curve = [
        {'date': str(d), 'equity': e}
        for d, e in zip(dates, equity)
    ]

    # Sample trades
    trades = [
        {'profit': 100 * (1 if i % 3 != 0 else -1)}
        for i in range(50)
    ]

    # Run analysis
    print("\nRunning comprehensive risk analysis...")
    result = AdvancedRiskMetrics.comprehensive_risk_analysis(
        equity_curve=equity_curve,
        trades=trades,
        risk_free_rate=0.02
    )

    if result['success']:
        print("\n✅ Analysis successful!")

        print("\n📊 Value at Risk (VaR):")
        var = result['value_at_risk']
        print(f"  95% VaR: {var['var_95_pct']:.2f}%")
        print(f"  99% VaR: {var['var_99_pct']:.2f}%")
        print(f"  → {var['interpretation_95']}")

        print("\n📉 Conditional VaR (CVaR):")
        cvar = result['conditional_var']
        print(f"  95% CVaR: {cvar['cvar_95_pct']:.2f}%")
        print(f"  99% CVaR: {cvar['cvar_99_pct']:.2f}%")
        print(f"  → {cvar['interpretation_95']}")

        print("\n📈 Risk-Adjusted Returns:")
        ratios = result['risk_adjusted_returns']
        print(f"  Sortino Ratio: {ratios['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio: {ratios['calmar_ratio']:.2f}")

        print("\n💧 Drawdown Metrics:")
        dd = result['drawdown_metrics']
        print(f"  Ulcer Index: {dd['ulcer_index']:.2f}")
        print(f"  Pain Index: {dd['pain_index']:.2f}%")
        print(f"  Max Drawdown: {dd['max_drawdown_pct']:.2f}%")
        print(f"  Avg Drawdown: {dd['avg_drawdown_pct']:.2f}%")
        print(f"  Max DD Duration: {dd['max_drawdown_duration_days']} days")
        print(f"  Time Underwater: {dd['time_underwater_pct']:.1f}%")

        print("\n🎲 Tail Risk:")
        tail = result['tail_risk']
        print(f"  Skewness: {tail['skewness']:.3f}")
        print(f"  Kurtosis: {tail['kurtosis']:.3f}")

        print("\n🔥 Win/Loss Streaks:")
        streaks = result['streaks']
        print(f"  Max Win Streak: {streaks['max_win_streak']}")
        print(f"  Max Loss Streak: {streaks['max_loss_streak']}")
        print(f"  Current Streak: {streaks['current_streak']} ({streaks['current_streak_type']})")
    else:
        print(f"\n❌ Analysis failed: {result['error']}")
