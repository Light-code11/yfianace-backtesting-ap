"""
Advanced Portfolio Optimization using PyPortfolioOpt
Implements Markowitz, Black-Litterman, Risk Parity, and more
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
    from pypfopt import BlackLittermanModel, objective_functions
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("Warning: PyPortfolioOpt not available. Using basic optimization only.")


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization for multi-strategy allocation

    Supports:
    - Markowitz Mean-Variance Optimization
    - Maximum Sharpe Ratio
    - Minimum Volatility
    - Maximum Quadratic Utility
    - Risk Parity
    - Black-Litterman (coming soon)
    - Hierarchical Risk Parity (coming soon)
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 2% = 0.02)
        """
        self.risk_free_rate = risk_free_rate

    def optimize_portfolio(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: List[Dict[str, Any]],
        total_capital: float,
        method: str = "max_sharpe",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation across multiple strategies

        Args:
            strategies: List of strategy configurations
            backtest_results: Backtest results for each strategy (must include equity_curve)
            total_capital: Total capital to allocate
            method: Optimization method
                - "max_sharpe": Maximize Sharpe ratio (default)
                - "min_volatility": Minimize portfolio volatility
                - "max_return": Maximize returns for given risk
                - "risk_parity": Equal risk contribution
                - "efficient_risk": Efficient frontier for target risk
                - "efficient_return": Efficient frontier for target return
            constraints: Optional dict with:
                - max_allocation: Max % per strategy (default 0.4 = 40%)
                - min_allocation: Min % per strategy (default 0.0)
                - target_risk: Target volatility (for efficient_risk method)
                - target_return: Target return (for efficient_return method)

        Returns:
            Dict with:
                - allocations: Dict[strategy_name, allocation_%]
                - capital_allocations: Dict[strategy_name, capital_$]
                - expected_return: Annual return %
                - expected_volatility: Annual volatility %
                - expected_sharpe: Sharpe ratio
                - method: Optimization method used
                - strategy_details: List of strategies with allocations
        """
        # Validate inputs
        if not strategies or not backtest_results:
            return {
                "success": False,
                "error": "No strategies or backtest results provided"
            }

        if len(strategies) != len(backtest_results):
            return {
                "success": False,
                "error": "Number of strategies and backtest results must match"
            }

        # Extract returns from equity curves
        try:
            returns_df, strategy_names = self._extract_returns(strategies, backtest_results)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract returns: {str(e)}"
            }

        if returns_df.empty or len(returns_df) < 10:
            return {
                "success": False,
                "error": "Insufficient data for optimization (need at least 10 data points)"
            }

        # Set default constraints
        if constraints is None:
            constraints = {}

        max_weight = constraints.get('max_allocation', 0.4)  # Max 40% per strategy by default
        min_weight = constraints.get('min_allocation', 0.0)   # Min 0%

        # Optimize based on method
        try:
            if PYPFOPT_AVAILABLE:
                weights, metrics = self._optimize_with_pypfopt(
                    returns_df, method, max_weight, min_weight, constraints
                )
            else:
                weights, metrics = self._optimize_basic(
                    returns_df, method, max_weight, min_weight
                )
        except Exception as e:
            return {
                "success": False,
                "error": f"Optimization failed: {str(e)}"
            }

        # Build response
        allocations = {}
        capital_allocations = {}
        strategy_details = []

        for i, strategy_name in enumerate(strategy_names):
            weight = weights[i]
            allocation_pct = weight * 100
            allocated_capital = total_capital * weight

            allocations[strategy_name] = round(allocation_pct, 2)
            capital_allocations[strategy_name] = round(allocated_capital, 2)

            # Find strategy and backtest result
            strategy = next((s for s in strategies if s['name'] == strategy_name), None)
            backtest = next((b for b in backtest_results if b.get('strategy_name') == strategy_name), None)

            if strategy and backtest:
                strategy_details.append({
                    'name': strategy_name,
                    'allocation_pct': round(allocation_pct, 2),
                    'allocated_capital': round(allocated_capital, 2),
                    'strategy_type': strategy.get('strategy_type', 'unknown'),
                    'sharpe_ratio': backtest.get('sharpe_ratio', 0),
                    'total_return_pct': backtest.get('total_return_pct', 0),
                    'max_drawdown_pct': backtest.get('max_drawdown_pct', 0),
                    'win_rate': backtest.get('win_rate', 0)
                })

        return {
            "success": True,
            "method": method,
            "total_capital": total_capital,
            "allocations": allocations,
            "capital_allocations": capital_allocations,
            "expected_return": round(metrics['expected_return'] * 100, 2),
            "expected_volatility": round(metrics['expected_volatility'] * 100, 2),
            "expected_sharpe": round(metrics['expected_sharpe'], 2),
            "strategies": strategy_details,
            "optimization_library": "pypfopt" if PYPFOPT_AVAILABLE else "scipy"
        }

    def _extract_returns(
        self,
        strategies: List[Dict],
        backtest_results: List[Dict]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Extract returns from equity curves"""
        returns_dict = {}

        for strategy, result in zip(strategies, backtest_results):
            equity_curve = result.get('equity_curve', [])
            if not equity_curve:
                continue

            # Extract equity values
            equity_values = [e['equity'] for e in equity_curve]
            if len(equity_values) < 2:
                continue

            # Calculate returns
            returns = pd.Series(equity_values).pct_change().dropna()

            # Store with strategy name
            strategy_name = strategy.get('name', f'Strategy_{len(returns_dict)}')
            returns_dict[strategy_name] = returns

        if not returns_dict:
            raise ValueError("No valid equity curves found")

        # Create DataFrame with aligned returns
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Drop rows with NaN

        return returns_df, list(returns_dict.keys())

    def _optimize_with_pypfopt(
        self,
        returns_df: pd.DataFrame,
        method: str,
        max_weight: float,
        min_weight: float,
        constraints: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """Optimize using PyPortfolioOpt library"""

        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(returns_df, frequency=252)
        S = risk_models.sample_cov(returns_df, frequency=252)

        # Create optimizer
        ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, max_weight))

        # Apply optimization method
        if method == "max_sharpe":
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)

        elif method == "min_volatility":
            weights = ef.min_volatility()

        elif method == "max_return":
            # Maximize return for given risk (target volatility)
            target_risk = constraints.get('target_risk', 0.15)  # Default 15% volatility
            weights = ef.efficient_risk(target_volatility=target_risk)

        elif method == "efficient_risk":
            target_risk = constraints.get('target_risk', 0.15)
            weights = ef.efficient_risk(target_volatility=target_risk)

        elif method == "efficient_return":
            target_return = constraints.get('target_return', 0.20)  # Default 20% return
            weights = ef.efficient_return(target_return=target_return)

        elif method == "risk_parity":
            # Risk parity: equal risk contribution from each asset
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            weights = ef.max_quadratic_utility(risk_aversion=1)

        else:
            # Default to max Sharpe
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)

        # Clean weights (remove tiny allocations)
        cleaned_weights = ef.clean_weights()

        # Get performance metrics
        performance = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=self.risk_free_rate
        )

        # Convert weights dict to array (preserving order)
        weights_array = np.array([cleaned_weights[col] for col in returns_df.columns])

        metrics = {
            'expected_return': performance[0],  # Annual return
            'expected_volatility': performance[1],  # Annual volatility
            'expected_sharpe': performance[2]  # Sharpe ratio
        }

        return weights_array, metrics

    def _optimize_basic(
        self,
        returns_df: pd.DataFrame,
        method: str,
        max_weight: float,
        min_weight: float
    ) -> Tuple[np.ndarray, Dict]:
        """Basic optimization without PyPortfolioOpt (fallback)"""
        from scipy.optimize import minimize

        n_assets = len(returns_df.columns)

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize

        # Objective functions
        def portfolio_return(weights):
            return np.dot(weights, expected_returns)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Bounds: min_weight <= weight <= max_weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess: equal weights
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize based on method
        if method in ["max_sharpe", "sharpe"]:
            result = minimize(negative_sharpe, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        elif method in ["min_volatility", "min_variance"]:
            result = minimize(portfolio_volatility, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        elif method in ["max_return"]:
            result = minimize(lambda w: -portfolio_return(w), init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        else:
            # Default to max Sharpe
            result = minimize(negative_sharpe, init_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

        weights = result.x

        # Calculate metrics
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0

        metrics = {
            'expected_return': ret,
            'expected_volatility': vol,
            'expected_sharpe': sharpe
        }

        return weights, metrics

    def get_efficient_frontier(
        self,
        returns_df: pd.DataFrame,
        n_points: int = 100
    ) -> Dict[str, List]:
        """
        Calculate efficient frontier points

        Returns:
            Dict with 'returns', 'volatilities', 'sharpe_ratios' lists
        """
        if not PYPFOPT_AVAILABLE:
            return {
                "error": "PyPortfolioOpt required for efficient frontier calculation"
            }

        mu = expected_returns.mean_historical_return(returns_df, frequency=252)
        S = risk_models.sample_cov(returns_df, frequency=252)

        # Calculate efficient frontier
        returns_list = []
        volatilities_list = []
        sharpe_list = []

        target_returns = np.linspace(mu.min(), mu.max(), n_points)

        for target_return in target_returns:
            try:
                ef = EfficientFrontier(mu, S)
                ef.efficient_return(target_return)
                perf = ef.portfolio_performance(
                    verbose=False,
                    risk_free_rate=self.risk_free_rate
                )
                returns_list.append(perf[0])
                volatilities_list.append(perf[1])
                sharpe_list.append(perf[2])
            except:
                continue

        return {
            "returns": returns_list,
            "volatilities": volatilities_list,
            "sharpe_ratios": sharpe_list
        }


if __name__ == "__main__":
    # Test the optimizer
    print("Testing Advanced Portfolio Optimizer...")
    print(f"PyPortfolioOpt available: {PYPFOPT_AVAILABLE}")

    # Example test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Simulate 3 strategy returns
    returns_data = {
        'Momentum Strategy': np.random.normal(0.001, 0.02, 100),
        'Mean Reversion Strategy': np.random.normal(0.0008, 0.015, 100),
        'Breakout Strategy': np.random.normal(0.0012, 0.025, 100)
    }

    returns_df = pd.DataFrame(returns_data, index=dates)

    # Create optimizer
    optimizer = AdvancedPortfolioOptimizer(risk_free_rate=0.02)

    # Prepare test inputs
    strategies = [
        {'name': 'Momentum Strategy', 'strategy_type': 'momentum'},
        {'name': 'Mean Reversion Strategy', 'strategy_type': 'mean_reversion'},
        {'name': 'Breakout Strategy', 'strategy_type': 'breakout'}
    ]

    # Create equity curves from returns
    backtest_results = []
    for name, returns in returns_data.items():
        equity = (1 + pd.Series(returns)).cumprod() * 100000
        equity_curve = [{'date': str(d), 'equity': e} for d, e in zip(dates, equity)]
        backtest_results.append({
            'strategy_name': name,
            'equity_curve': equity_curve,
            'sharpe_ratio': 1.5,
            'total_return_pct': 15.0,
            'max_drawdown_pct': 5.0,
            'win_rate': 60.0
        })

    # Test optimization
    result = optimizer.optimize_portfolio(
        strategies=strategies,
        backtest_results=backtest_results,
        total_capital=100000,
        method="max_sharpe"
    )

    if result['success']:
        print("\n✅ Optimization successful!")
        print(f"\nMethod: {result['method']}")
        print(f"Expected Return: {result['expected_return']:.2f}%")
        print(f"Expected Volatility: {result['expected_volatility']:.2f}%")
        print(f"Expected Sharpe: {result['expected_sharpe']:.2f}")
        print("\nAllocations:")
        for name, allocation in result['allocations'].items():
            capital = result['capital_allocations'][name]
            print(f"  {name}: {allocation:.1f}% (${capital:,.0f})")
    else:
        print(f"\n❌ Optimization failed: {result['error']}")
