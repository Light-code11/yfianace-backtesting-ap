"""
Auto Strategy Generator - Automatically creates, tests, and deploys strategies

Evolutionary approach:
1. Generate random strategy variations
2. Backtest all of them
3. Keep only the best performers
4. Deploy winners to live trading
5. Generate new variations from winners
6. Repeat weekly

This allows the system to discover new profitable strategies without human input!
"""
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from backtesting_engine import BacktestingEngine
from database import SessionLocal, Strategy, BacktestResult, StrategyPerformance


class AutoStrategyGenerator:
    """
    Automatically generates and tests trading strategies

    Uses evolutionary approach:
    - Generate 20 random strategy variations
    - Backtest all on multiple tickers
    - Keep top 3 with Sharpe > 1.5
    - Deploy to live trading
    - Track performance
    - Generate new variations from best performers
    """

    # Strategy templates
    STRATEGY_TYPES = ['momentum', 'mean_reversion', 'breakout', 'trend_following']

    # Indicator options
    INDICATOR_POOL = [
        {'name': 'SMA', 'period_options': [10, 20, 50, 100, 200]},
        {'name': 'EMA', 'period_options': [9, 12, 20, 26, 50]},
        {'name': 'RSI', 'period_options': [7, 14, 21, 28]},
        {'name': 'MACD', 'period_options': [None]},  # Fixed parameters
        {'name': 'BB', 'period_options': [None]},  # Fixed parameters
        {'name': 'ATR', 'period_options': [14, 20]},
    ]

    # Ticker universe for testing
    TEST_UNIVERSE = [
        # Tech
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL', 'META', 'TSLA',
        # Finance
        'JPM', 'BAC', 'V', 'MA',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE',
        # Energy
        'XOM', 'CVX',
        # Crypto
        'COIN', 'MARA',
        # ETFs
        'SPY', 'QQQ', 'IWM'
    ]

    def __init__(self):
        self.db = SessionLocal()
        self.backtest_engine = BacktestingEngine()

    def generate_strategies_batch(self, num_strategies: int = 20) -> List[Dict]:
        """
        Generate a batch of random strategy variations

        Returns list of strategy configurations ready for backtesting
        """
        print(f"\n🧬 Generating {num_strategies} random strategy variations...")

        strategies = []

        for i in range(num_strategies):
            strategy = self._create_random_strategy(f"AutoGen_{i+1}")
            strategies.append(strategy)
            print(f"   {i+1}. {strategy['name']} ({strategy['strategy_type']})")

        return strategies

    def _create_random_strategy(self, name_suffix: str) -> Dict:
        """Create a single random strategy configuration"""

        # Random strategy type
        strategy_type = random.choice(self.STRATEGY_TYPES)

        # Random tickers (2-4 tickers)
        num_tickers = random.randint(2, 4)
        tickers = random.sample(self.TEST_UNIVERSE, num_tickers)

        # Random indicators (2-4 indicators)
        num_indicators = random.randint(2, 4)
        indicators = []

        selected_indicator_types = random.sample(self.INDICATOR_POOL, num_indicators)

        for ind_config in selected_indicator_types:
            indicator = {'name': ind_config['name']}

            # Add period if applicable
            if ind_config['period_options'] and ind_config['period_options'][0] is not None:
                indicator['period'] = random.choice(ind_config['period_options'])

            indicators.append(indicator)

        # Random risk management (conservative to aggressive)
        stop_loss_pct = round(random.uniform(3.0, 15.0), 1)
        take_profit_pct = round(random.uniform(stop_loss_pct * 1.5, stop_loss_pct * 3.0), 1)
        position_size_pct = round(random.uniform(10.0, 30.0), 1)

        # Random holding period (swing to position trading)
        holding_period_days = random.choice([5, 10, 20, 30, 60])

        return {
            'name': f"{strategy_type.title()} - {name_suffix}",
            'description': f"Auto-generated {strategy_type} strategy",
            'strategy_type': strategy_type,
            'tickers': tickers,
            'indicators': indicators,
            'risk_management': {
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'position_size_pct': position_size_pct
            },
            'holding_period_days': holding_period_days,
            'entry_conditions': self._generate_entry_conditions(strategy_type, indicators),
            'exit_conditions': self._generate_exit_conditions(strategy_type),
            'rationale': f"Automatically generated {strategy_type} strategy using {len(indicators)} indicators",
            'market_analysis': "Auto-generated strategy",
            'risk_assessment': f"Stop loss: {stop_loss_pct}%, Take profit: {take_profit_pct}%"
        }

    def _generate_entry_conditions(self, strategy_type: str, indicators: List[Dict]) -> List[Dict]:
        """Generate entry conditions based on strategy type and indicators"""

        conditions = []

        if strategy_type == 'momentum':
            # Look for trend + momentum confirmation
            conditions.append({
                "type": "indicator_crossover",
                "description": "Fast MA crosses above slow MA"
            })
            conditions.append({
                "type": "momentum_confirmation",
                "description": "RSI not overbought (< 70)"
            })

        elif strategy_type == 'mean_reversion':
            # Look for oversold + reversal
            conditions.append({
                "type": "oversold",
                "description": "Price at lower Bollinger Band"
            })
            conditions.append({
                "type": "reversal_signal",
                "description": "RSI < 30 (oversold)"
            })

        elif strategy_type == 'breakout':
            # Look for consolidation breakout
            conditions.append({
                "type": "breakout",
                "description": "Price breaks above 20-day high"
            })
            conditions.append({
                "type": "volume_confirmation",
                "description": "Increased volume"
            })

        elif strategy_type == 'trend_following':
            # Look for established trend
            conditions.append({
                "type": "trend_confirmation",
                "description": "Price above long-term MA"
            })
            conditions.append({
                "type": "pullback_entry",
                "description": "Short-term pullback to MA"
            })

        return conditions

    def _generate_exit_conditions(self, strategy_type: str) -> List[Dict]:
        """Generate exit conditions"""
        return [
            {
                "type": "stop_loss",
                "description": "Stop loss hit"
            },
            {
                "type": "take_profit",
                "description": "Take profit target reached"
            },
            {
                "type": "time_based",
                "description": "Holding period exceeded"
            }
        ]

    def evaluate_and_select_winners(
        self,
        strategy_configs: List[Dict],
        min_sharpe: float = 1.5,
        min_trades: int = 30,
        max_drawdown: float = 20.0,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Backtest all strategies and select the best performers

        Criteria:
        - Sharpe ratio > min_sharpe (default 1.5)
        - Total trades > min_trades (default 30)
        - Max drawdown < max_drawdown (default 20%)
        - Win rate > 45%

        Returns top N strategies sorted by Sharpe ratio
        """
        print(f"\n🔬 Backtesting {len(strategy_configs)} strategies...")
        print(f"   Criteria: Sharpe > {min_sharpe}, Trades > {min_trades}, Drawdown < {max_drawdown}%")

        results = []

        for i, strategy_config in enumerate(strategy_configs, 1):
            print(f"\n   [{i}/{len(strategy_configs)}] Testing {strategy_config['name']}...")

            # Run backtest
            backtest_result = self.backtest_engine.backtest_strategy(
                tickers=strategy_config['tickers'],
                strategy_type=strategy_config['strategy_type'],
                indicators=strategy_config['indicators'],
                stop_loss_pct=strategy_config['risk_management']['stop_loss_pct'],
                take_profit_pct=strategy_config['risk_management']['take_profit_pct'],
                position_size_pct=strategy_config['risk_management']['position_size_pct'],
                holding_period_days=strategy_config['holding_period_days'],
                start_date=datetime.now() - timedelta(days=730),  # 2 years
                end_date=datetime.now()
            )

            if not backtest_result.get('success'):
                print(f"      ❌ Backtest failed: {backtest_result.get('error')}")
                continue

            metrics = backtest_result

            # Check if meets criteria
            sharpe = metrics.get('sharpe_ratio', 0)
            total_trades = metrics.get('total_trades', 0)
            max_dd = abs(metrics.get('max_drawdown_pct', 100))
            win_rate = metrics.get('win_rate', 0)

            print(f"      Sharpe: {sharpe:.2f}, Trades: {total_trades}, DD: {max_dd:.1f}%, WR: {win_rate:.1f}%")

            passes = (
                sharpe >= min_sharpe and
                total_trades >= min_trades and
                max_dd <= max_drawdown and
                win_rate >= 45.0
            )

            if passes:
                print(f"      ✅ WINNER! Meets all criteria")
                results.append({
                    'strategy_config': strategy_config,
                    'backtest_result': metrics,
                    'score': sharpe  # Use Sharpe for ranking
                })
            else:
                print(f"      ❌ Rejected")

        # Sort by score (Sharpe ratio) descending
        results.sort(key=lambda x: x['score'], reverse=True)

        # Return top N
        winners = results[:top_n]

        print(f"\n🏆 Selected {len(winners)} winning strategies:")
        for i, winner in enumerate(winners, 1):
            print(f"   {i}. {winner['strategy_config']['name']} (Sharpe: {winner['score']:.2f})")

        return winners

    def deploy_strategies(self, winning_strategies: List[Dict]) -> List[int]:
        """
        Deploy winning strategies to database for live trading

        Returns list of deployed strategy IDs
        """
        print(f"\n🚀 Deploying {len(winning_strategies)} strategies to live trading...")

        deployed_ids = []

        for winner in winning_strategies:
            config = winner['strategy_config']
            backtest = winner['backtest_result']

            # Check if strategy with same name exists
            existing = self.db.query(Strategy).filter(
                Strategy.name == config['name']
            ).first()

            if existing:
                print(f"   ⚠️  Strategy '{config['name']}' already exists, skipping")
                continue

            # Create strategy in database
            strategy = Strategy(
                name=config['name'],
                description=config['description'],
                strategy_type=config['strategy_type'],
                tickers=config['tickers'],
                indicators=config['indicators'],
                entry_conditions=config['entry_conditions'],
                exit_conditions=config['exit_conditions'],
                stop_loss_pct=config['risk_management']['stop_loss_pct'],
                take_profit_pct=config['risk_management']['take_profit_pct'],
                position_size_pct=config['risk_management']['position_size_pct'],
                holding_period_days=config['holding_period_days'],
                rationale=config['rationale'],
                market_analysis=config['market_analysis'],
                risk_assessment=config['risk_assessment'],
                is_active=True
            )
            self.db.add(strategy)
            self.db.commit()
            self.db.refresh(strategy)

            # Save backtest result
            backtest_record = BacktestResult(
                strategy_id=strategy.id,
                strategy_name=strategy.name,
                start_date=backtest.get('start_date'),
                end_date=backtest.get('end_date'),
                initial_capital=backtest.get('initial_capital', 100000),
                tickers_tested=config['tickers'],
                total_return_pct=backtest.get('total_return_pct'),
                total_trades=backtest.get('total_trades'),
                winning_trades=backtest.get('winning_trades'),
                losing_trades=backtest.get('losing_trades'),
                win_rate=backtest.get('win_rate'),
                sharpe_ratio=backtest.get('sharpe_ratio'),
                sortino_ratio=backtest.get('sortino_ratio'),
                max_drawdown_pct=backtest.get('max_drawdown_pct'),
                profit_factor=backtest.get('profit_factor'),
                avg_win=backtest.get('avg_win'),
                avg_loss=backtest.get('avg_loss'),
                trades=backtest.get('trades', []),
                equity_curve=backtest.get('equity_curve', []),
                quality_score=backtest.get('quality_score', 0)
            )
            self.db.add(backtest_record)

            # Initialize performance tracking
            perf = StrategyPerformance(
                strategy_id=strategy.id,
                strategy_name=strategy.name,
                backtest_sharpe=backtest.get('sharpe_ratio'),
                backtest_win_rate=backtest.get('win_rate'),
                backtest_avg_return=backtest.get('avg_win', 0),
                backtest_max_drawdown=abs(backtest.get('max_drawdown_pct', 0)),
                allocation_weight=1.0,  # Start with full weight
                is_deprecated=False
            )
            self.db.add(perf)

            self.db.commit()

            deployed_ids.append(strategy.id)
            print(f"   ✅ Deployed: {config['name']} (ID: {strategy.id})")

        print(f"\n✨ Successfully deployed {len(deployed_ids)} new strategies!")
        return deployed_ids

    def run_full_cycle(
        self,
        num_strategies: int = 20,
        min_sharpe: float = 1.5,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Run complete auto-generation cycle

        Steps:
        1. Generate N random strategies
        2. Backtest all of them
        3. Select top performers
        4. Deploy winners to live trading

        Returns summary of results
        """
        print("=" * 70)
        print("🤖 AUTO STRATEGY GENERATOR - FULL CYCLE")
        print("=" * 70)
        print(f"   Generating: {num_strategies} strategies")
        print(f"   Min Sharpe: {min_sharpe}")
        print(f"   Top N: {top_n}")

        try:
            # Step 1: Generate
            strategy_configs = self.generate_strategies_batch(num_strategies)

            # Step 2 & 3: Evaluate and select
            winners = self.evaluate_and_select_winners(
                strategy_configs,
                min_sharpe=min_sharpe,
                top_n=top_n
            )

            # Step 4: Deploy
            if winners:
                deployed_ids = self.deploy_strategies(winners)
            else:
                print("\n⚠️  No strategies met the criteria. Try again with different parameters.")
                deployed_ids = []

            return {
                "success": True,
                "strategies_generated": num_strategies,
                "strategies_tested": len(strategy_configs),
                "winners_found": len(winners),
                "strategies_deployed": len(deployed_ids),
                "deployed_strategy_ids": deployed_ids
            }

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.db.close()


if __name__ == "__main__":
    # Test the auto-generator
    generator = AutoStrategyGenerator()

    # Run with conservative settings (easier to pass)
    results = generator.run_full_cycle(
        num_strategies=10,  # Test with fewer for speed
        min_sharpe=1.0,     # Lower threshold for testing
        top_n=2             # Deploy top 2
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS:")
    print(f"✅ Success: {results['success']}")
    print(f"📊 Generated: {results.get('strategies_generated', 0)}")
    print(f"🔬 Tested: {results.get('strategies_tested', 0)}")
    print(f"🏆 Winners: {results.get('winners_found', 0)}")
    print(f"🚀 Deployed: {results.get('strategies_deployed', 0)}")
    if results.get('deployed_strategy_ids'):
        print(f"📋 Strategy IDs: {results['deployed_strategy_ids']}")
    print("=" * 70)
