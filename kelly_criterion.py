"""
Kelly Criterion Position Sizing
Optimal position sizing based on win rate and win/loss ratio
"""
import numpy as np


class KellyCriterion:
    """
    Calculate optimal position sizing using Kelly Criterion

    Formula: f* = (p * b - q) / b
    Where:
        p = win probability (win rate)
        q = loss probability (1 - win rate)
        b = win/loss ratio (avg_win / avg_loss)
        f* = optimal fraction of capital to risk
    """

    @staticmethod
    def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float,
                        fractional_kelly: float = 0.25) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade size (absolute value)
            avg_loss: Average losing trade size (absolute value)
            fractional_kelly: Fraction of full Kelly to use (default 0.25 = Quarter Kelly)
                             Full Kelly can be aggressive, fractional is safer

        Returns:
            Optimal position size as fraction of capital (0-1)
        """
        # Input validation
        if not (0 < win_rate < 1):
            raise ValueError("Win rate must be between 0 and 1")

        if avg_win <= 0 or avg_loss <= 0:
            raise ValueError("Average win and loss must be positive")

        # Calculate components
        p = win_rate  # Win probability
        q = 1 - win_rate  # Loss probability
        b = avg_win / avg_loss  # Win/loss ratio

        # Full Kelly Criterion
        kelly_fraction = (p * b - q) / b

        # Apply fractional Kelly (Quarter Kelly by default for safety)
        # Full Kelly can lead to large drawdowns, fractional is more conservative
        optimal_position = kelly_fraction * fractional_kelly

        # Ensure position is between 0 and 1
        # Negative Kelly means negative edge - don't trade
        optimal_position = max(0.0, min(1.0, optimal_position))

        return optimal_position

    @staticmethod
    def calculate_kelly_from_backtest(backtest_results: dict,
                                     fractional_kelly: float = 0.25) -> dict:
        """
        Calculate Kelly position size from backtest results

        Args:
            backtest_results: Dict with keys: win_rate, avg_win, avg_loss
            fractional_kelly: Fraction of full Kelly to use

        Returns:
            Dict with kelly_fraction, recommended_position_pct, and analysis
        """
        win_rate = backtest_results.get('win_rate', 0)
        avg_win = abs(backtest_results.get('avg_win', 0))
        avg_loss = abs(backtest_results.get('avg_loss', 1))  # Avoid division by zero

        # Handle edge cases
        if win_rate == 0 or avg_win == 0:
            return {
                'kelly_fraction': 0.0,
                'recommended_position_pct': 0.0,
                'analysis': 'No winning trades - do not trade this strategy',
                'risk_level': 'EXTREME'
            }

        if avg_loss == 0:  # All winning trades
            avg_loss = avg_win * 0.1  # Conservative assumption

        try:
            kelly = KellyCriterion.calculate_kelly(
                win_rate=win_rate / 100,  # Convert from percentage
                avg_win=avg_win,
                avg_loss=avg_loss,
                fractional_kelly=fractional_kelly
            )

            recommended_pct = kelly * 100  # Convert to percentage

            # Risk level assessment
            if kelly == 0:
                risk_level = "NO EDGE"
                analysis = "Strategy has negative edge - do not trade"
            elif recommended_pct < 2:
                risk_level = "VERY LOW"
                analysis = "Very small edge - consider skipping"
            elif recommended_pct < 5:
                risk_level = "LOW"
                analysis = "Conservative position sizing recommended"
            elif recommended_pct < 10:
                risk_level = "MODERATE"
                analysis = "Good edge with manageable risk"
            elif recommended_pct < 15:
                risk_level = "MODERATE-HIGH"
                analysis = "Strong edge but requires discipline"
            else:
                risk_level = "HIGH"
                analysis = "Very strong edge but monitor closely for overtrading"
                recommended_pct = min(recommended_pct, 15)  # Cap at 15%

            return {
                'kelly_fraction': round(kelly, 4),
                'recommended_position_pct': round(recommended_pct, 2),
                'fractional_kelly_used': fractional_kelly,
                'risk_level': risk_level,
                'analysis': analysis,
                'win_loss_ratio': round(avg_win / avg_loss, 2)
            }

        except Exception as e:
            return {
                'kelly_fraction': 0.0,
                'recommended_position_pct': 0.0,
                'analysis': f'Error calculating Kelly: {str(e)}',
                'risk_level': 'ERROR'
            }

    @staticmethod
    def get_recommendation_text(kelly_result: dict) -> str:
        """
        Get human-readable recommendation

        Args:
            kelly_result: Result from calculate_kelly_from_backtest

        Returns:
            Formatted recommendation string
        """
        rec = kelly_result['recommended_position_pct']
        risk = kelly_result['risk_level']
        analysis = kelly_result['analysis']

        text = f"""
📊 Kelly Criterion Analysis
━━━━━━━━━━━━━━━━━━━━━━
Recommended Position: {rec:.2f}%
Risk Level: {risk}
Win/Loss Ratio: {kelly_result.get('win_loss_ratio', 'N/A')}

{analysis}

Note: Using {int(kelly_result.get('fractional_kelly_used', 0.25) * 100)}% of Full Kelly for safety
Full Kelly can be aggressive - fractional Kelly reduces risk of ruin
"""
        return text.strip()


# Example usage
if __name__ == "__main__":
    # Example 1: Direct calculation
    print("Example 1: Direct Kelly Calculation")
    print("=" * 50)

    kelly = KellyCriterion.calculate_kelly(
        win_rate=0.55,  # 55% win rate
        avg_win=100,    # Average win $100
        avg_loss=60,    # Average loss $60
        fractional_kelly=0.25  # Quarter Kelly
    )

    print(f"Optimal position size: {kelly * 100:.2f}%\n")

    # Example 2: From backtest results
    print("Example 2: From Backtest Results")
    print("=" * 50)

    backtest_results = {
        'win_rate': 58,  # 58%
        'avg_win': 120,
        'avg_loss': 80,
        'total_trades': 100
    }

    recommendation = KellyCriterion.calculate_kelly_from_backtest(backtest_results)
    print(KellyCriterion.get_recommendation_text(recommendation))
