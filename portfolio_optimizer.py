"""
Portfolio optimization for multi-strategy allocation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """Optimize portfolio allocation across multiple strategies"""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    def optimize_allocations(
        self,
        strategies: List[Dict[str, Any]],
        backtest_results: List[Dict[str, Any]],
        total_capital: float,
        method: str = "sharpe",
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation across strategies

        Args:
            strategies: List of strategy configurations
            backtest_results: Backtest results for each strategy
            total_capital: Total capital to allocate
            method: Optimization method (sharpe, min_variance, max_return, risk_parity)
            constraints: Optional constraints (max_allocation, min_allocation, etc.)

        Returns:
            Optimization results with allocations
        """
        if not strategies or not backtest_results:
            return {"error": "No strategies or backtest results provided"}

        # Extract returns data from equity curves
        returns_data = []
        for result in backtest_results:
            equity_curve = result.get('equity_curve', [])
            if equity_curve:
                equity_values = pd.Series([e['equity'] for e in equity_curve])
                returns = equity_values.pct_change().dropna()
                returns_data.append(returns)

        if not returns_data:
            return {"error": "No valid equity curves found"}

        # Align returns to same length
        min_length = min(len(r) for r in returns_data)
        returns_matrix = np.array([r.iloc[:min_length].values for r in returns_data]).T
        returns_df = pd.DataFrame(returns_matrix, columns=[s['name'] for s in strategies])

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize

        # Set default constraints
        if constraints is None:
            constraints = {}

        max_allocation = constraints.get('max_allocation', 0.5)  # Max 50% per strategy
        min_allocation = constraints.get('min_allocation', 0.0)  # Min 0% per strategy

        # Optimize based on method
        if method == "sharpe":
            result = self._optimize_sharpe(expected_returns, cov_matrix, min_allocation, max_allocation)
        elif method == "min_variance":
            result = self._optimize_min_variance(expected_returns, cov_matrix, min_allocation, max_allocation)
        elif method == "max_return":
            result = self._optimize_max_return(expected_returns, cov_matrix, min_allocation, max_allocation)
        elif method == "risk_parity":
            result = self._optimize_risk_parity(expected_returns, cov_matrix, min_allocation, max_allocation)
        else:
            return {"error": f"Unknown optimization method: {method}"}

        # Build response
        allocations = {}
        capital_allocations = {}

        for i, strategy in enumerate(strategies):
            allocation_pct = result['weights'][i] * 100
            allocations[strategy['name']] = round(allocation_pct, 2)
            capital_allocations[strategy['name']] = round(total_capital * result['weights'][i], 2)

        return {
            "method": method,
            "total_capital": total_capital,
            "allocations": allocations,
            "capital_allocations": capital_allocations,
            "expected_return": round(result['expected_return'] * 100, 2),
            "expected_volatility": round(result['expected_volatility'] * 100, 2),
            "expected_sharpe": round(result['sharpe_ratio'], 2),
            "strategies": [
                {
                    "name": strategies[i]['name'],
                    "allocation_pct": round(result['weights'][i] * 100, 2),
                    "capital": round(total_capital * result['weights'][i], 2),
                    "expected_return_pct": round(expected_returns.iloc[i] * 100, 2),
                    "volatility_pct": round(np.sqrt(cov_matrix.iloc[i, i]) * 100, 2)
                }
                for i in range(len(strategies))
                if result['weights'][i] > 0.01  # Only include allocations > 1%
            ]
        }

    def _optimize_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        min_alloc: float,
        max_alloc: float
    ) -> Dict:
        """Maximize Sharpe ratio"""
        n = len(expected_returns)

        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe  # Minimize negative Sharpe

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((min_alloc, max_alloc) for _ in range(n))
        initial_guess = np.array([1/n] * n)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def _optimize_min_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        min_alloc: float,
        max_alloc: float
    ) -> Dict:
        """Minimize portfolio variance"""
        n = len(expected_returns)

        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((min_alloc, max_alloc) for _ in range(n))
        initial_guess = np.array([1/n] * n)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def _optimize_max_return(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        min_alloc: float,
        max_alloc: float
    ) -> Dict:
        """Maximize expected return"""
        n = len(expected_returns)

        def objective(weights):
            return -np.sum(expected_returns * weights)  # Minimize negative return

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((min_alloc, max_alloc) for _ in range(n))
        initial_guess = np.array([1/n] * n)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def _optimize_risk_parity(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        min_alloc: float,
        max_alloc: float
    ) -> Dict:
        """Risk parity allocation (equal risk contribution)"""
        n = len(expected_returns)

        def objective(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
            risk_contrib = weights * marginal_contrib
            # Minimize difference from equal risk
            return np.sum((risk_contrib - np.mean(risk_contrib)) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((min_alloc, max_alloc) for _ in range(n))
        initial_guess = np.array([1/n] * n)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }

    def calculate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        num_points: int = 50
    ) -> List[Dict]:
        """Calculate efficient frontier points"""
        n = len(expected_returns)
        frontier = []

        # Get min and max return
        min_var_result = self._optimize_min_variance(expected_returns, cov_matrix, 0, 1)
        max_ret_result = self._optimize_max_return(expected_returns, cov_matrix, 0, 1)

        min_return = min_var_result['expected_return']
        max_return = max_ret_result['expected_return']

        # Calculate points along frontier
        for target_return in np.linspace(min_return, max_return, num_points):
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(expected_returns * w) - target_return}
            )
            bounds = tuple((0, 1) for _ in range(n))
            initial_guess = np.array([1/n] * n)

            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )

                if result.success:
                    volatility = np.sqrt(result.fun)
                    sharpe = (target_return - self.risk_free_rate) / volatility

                    frontier.append({
                        'return': target_return,
                        'volatility': volatility,
                        'sharpe': sharpe
                    })
            except:
                continue

        return frontier
