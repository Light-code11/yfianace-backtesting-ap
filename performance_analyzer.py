"""
Performance Analyzer - The learning brain of the autonomous system

Tracks:
- Live performance vs backtest predictions
- Strategy degradation over time
- Adjusts position sizing based on results
- Auto-deprecates failing strategies
- Identifies which strategy types work best

This is how the system "learns" and improves over time!
"""
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv

from database import (
    SessionLocal, Strategy, StrategyPerformance,
    TradeExecution, BacktestResult, LivePosition
)

load_dotenv()


class PerformanceAnalyzer:
    """
    Analyzes strategy performance and makes learning decisions

    Key Functions:
    1. Compare live vs backtest performance
    2. Adjust allocation weights (reward winners, penalize losers)
    3. Deprecate strategies that fail in live trading
    4. Identify patterns in what works
    """

    def __init__(self):
        self.db = SessionLocal()

        # Configuration
        self.min_trades_for_eval = int(os.getenv('MIN_TRADES_FOR_EVALUATION', 10))
        self.performance_threshold = float(os.getenv('PERFORMANCE_THRESHOLD_SHARPE', 0.8))
        self.auto_deprecate = os.getenv('AUTO_DEPRECATE_STRATEGIES', 'true').lower() == 'true'

        print(f"📊 Performance Analyzer initialized")
        print(f"   Min trades for evaluation: {self.min_trades_for_eval}")
        print(f"   Performance threshold: {self.performance_threshold}")
        print(f"   Auto-deprecate: {self.auto_deprecate}")

    def update_daily_metrics(self) -> Dict[str, Any]:
        """
        Update performance metrics for all active strategies

        Called daily after market close to track performance
        """
        print("\n📈 Updating daily performance metrics...")

        strategies_analyzed = 0
        weights_adjusted = 0

        # Get all active strategies
        active_strategies = self.db.query(Strategy).filter(
            Strategy.is_active == True
        ).all()

        for strategy in active_strategies:
            # Get performance record (create if doesn't exist)
            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == strategy.id
            ).first()

            if not perf:
                # Initialize performance tracking
                backtest = self.db.query(BacktestResult).filter(
                    BacktestResult.strategy_id == strategy.id
                ).first()

                perf = StrategyPerformance(
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    backtest_sharpe=backtest.sharpe_ratio if backtest else 0.0,
                    backtest_win_rate=backtest.win_rate if backtest else 0.0,
                    backtest_avg_return=backtest.avg_win if backtest else 0.0,
                    backtest_max_drawdown=abs(backtest.max_drawdown_pct) if backtest else 0.0,
                    allocation_weight=1.0
                )
                self.db.add(perf)

            # Update live metrics
            self._update_strategy_live_metrics(strategy, perf)

            # Adjust allocation weight based on performance
            weight_changed = self._adjust_allocation_weight(perf)
            if weight_changed:
                weights_adjusted += 1

            strategies_analyzed += 1

        self.db.commit()

        print(f"   ✅ Analyzed {strategies_analyzed} strategies")
        print(f"   ⚖️  Adjusted {weights_adjusted} allocation weights")

        return {
            "success": True,
            "strategies_analyzed": strategies_analyzed,
            "weights_adjusted": weights_adjusted
        }

    def _update_strategy_live_metrics(self, strategy: Strategy, perf: StrategyPerformance):
        """Calculate live performance metrics for a strategy"""

        # Get all filled trades for this strategy
        trades = self.db.query(TradeExecution).filter(
            TradeExecution.strategy_id == strategy.id,
            TradeExecution.order_status == 'filled'
        ).all()

        if len(trades) < self.min_trades_for_eval:
            # Not enough data yet
            return

        # Calculate metrics
        perf.live_total_trades = len(trades)

        # TODO: Calculate actual P&L from filled trades
        # For now, using simplified approach

        # Calculate win rate (if we have position exit data)
        closed_positions = self.db.query(LivePosition).filter(
            LivePosition.strategy_id == strategy.id,
            LivePosition.is_open == False
        ).all()

        if closed_positions:
            winning = sum(1 for p in closed_positions if p.unrealized_pl > 0)
            perf.live_winning_trades = winning
            perf.live_losing_trades = len(closed_positions) - winning
            perf.live_win_rate = (winning / len(closed_positions)) * 100 if closed_positions else 0.0

            # Calculate average return
            avg_return_pct = np.mean([p.unrealized_plpc for p in closed_positions])
            perf.live_avg_return = float(avg_return_pct)

        # Calculate performance delta
        if perf.backtest_win_rate and perf.live_win_rate:
            perf.win_rate_delta = perf.live_win_rate - perf.backtest_win_rate

        # Update timestamps
        perf.last_evaluated_at = datetime.utcnow()
        perf.updated_at = datetime.utcnow()

    def _adjust_allocation_weight(self, perf: StrategyPerformance) -> bool:
        """
        Adjust strategy allocation weight based on performance

        Good performance → Increase weight (up to 1.5x)
        Poor performance → Decrease weight (down to 0.2x)
        """
        if not perf.live_win_rate or perf.live_total_trades < self.min_trades_for_eval:
            return False

        old_weight = perf.allocation_weight

        # Calculate performance score (0-1)
        # Based on win rate delta and live win rate
        win_rate_score = 0.0
        if perf.backtest_win_rate > 0:
            # How close is live win rate to backtest?
            win_rate_ratio = perf.live_win_rate / perf.backtest_win_rate
            win_rate_score = min(win_rate_ratio, 1.5)  # Cap at 1.5

        # Adjust weight
        if win_rate_score >= 1.1:
            # Performing better than backtest!
            perf.allocation_weight = min(1.5, old_weight * 1.1)
        elif win_rate_score >= 0.9:
            # Performing as expected
            perf.allocation_weight = 1.0
        elif win_rate_score >= 0.7:
            # Underperforming but acceptable
            perf.allocation_weight = 0.7
        elif win_rate_score >= 0.5:
            # Significantly underperforming
            perf.allocation_weight = 0.3
        else:
            # Failing badly
            perf.allocation_weight = 0.1

        weight_changed = abs(perf.allocation_weight - old_weight) > 0.05

        if weight_changed:
            print(f"   ⚖️  {perf.strategy_name}: Weight {old_weight:.2f} → {perf.allocation_weight:.2f}")

        return weight_changed

    def weekly_review(self) -> Dict[str, Any]:
        """
        Weekly performance review and strategy deprecation

        Decisions:
        - Which strategies to deprecate
        - Which strategies are top performers
        - Learning insights
        """
        print("\n📊 Weekly Performance Review...")

        strategies_reviewed = 0
        strategies_deprecated = 0
        top_performers = []

        # Get all strategies with performance data
        all_perf = self.db.query(StrategyPerformance).all()

        for perf in all_perf:
            strategies_reviewed += 1

            # Check if should be deprecated
            should_deprecate = self._should_deprecate_strategy(perf)

            if should_deprecate and self.auto_deprecate and not perf.is_deprecated:
                # Deprecate the strategy
                perf.is_deprecated = True
                perf.deprecation_reason = should_deprecate

                # Mark strategy as inactive
                strategy = self.db.query(Strategy).filter(
                    Strategy.id == perf.strategy_id
                ).first()
                if strategy:
                    strategy.is_active = False

                strategies_deprecated += 1
                print(f"   ❌ Deprecated: {perf.strategy_name} - {should_deprecate}")

            # Track top performers
            if not perf.is_deprecated and perf.live_total_trades >= self.min_trades_for_eval:
                if perf.live_win_rate and perf.live_win_rate > 55:
                    top_performers.append({
                        "name": perf.strategy_name,
                        "win_rate": perf.live_win_rate,
                        "weight": perf.allocation_weight
                    })

        # Sort top performers
        top_performers.sort(key=lambda x: x['win_rate'], reverse=True)
        top_performers = top_performers[:5]  # Top 5

        self.db.commit()

        print(f"\n   ✅ Reviewed {strategies_reviewed} strategies")
        print(f"   ❌ Deprecated {strategies_deprecated} underperformers")
        print(f"   ⭐ Top {len(top_performers)} performers identified")

        if top_performers:
            print("\n   🏆 Top Performers:")
            for i, perf in enumerate(top_performers, 1):
                print(f"      {i}. {perf['name']}: {perf['win_rate']:.1f}% win rate (weight: {perf['weight']:.2f})")

        return {
            "success": True,
            "strategies_reviewed": strategies_reviewed,
            "strategies_deprecated": strategies_deprecated,
            "top_performers": [p['name'] for p in top_performers]
        }

    def _should_deprecate_strategy(self, perf: StrategyPerformance) -> str:
        """
        Determine if a strategy should be deprecated

        Returns deprecation reason or empty string if should keep
        """
        # Need minimum trades to evaluate
        if perf.live_total_trades < self.min_trades_for_eval:
            return ""

        # Check win rate
        if perf.live_win_rate and perf.live_win_rate < 35:
            return f"Low win rate: {perf.live_win_rate:.1f}% (threshold: 35%)"

        # Check if massively underperforming backtest
        if perf.backtest_win_rate and perf.live_win_rate:
            if perf.live_win_rate < perf.backtest_win_rate * 0.5:
                return f"Live win rate ({perf.live_win_rate:.1f}%) < 50% of backtest ({perf.backtest_win_rate:.1f}%)"

        # Check allocation weight (if consistently low, it's not trusted)
        if perf.allocation_weight < 0.2:
            return "Consistently poor performance (weight < 0.2)"

        return ""

    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Extract learning insights from performance data

        What strategy types work best?
        What indicators are most common in winners?
        What risk parameters are optimal?
        """
        print("\n🧠 Generating learning insights...")

        insights = {
            "strategy_type_performance": {},
            "top_strategies": [],
            "common_patterns": {}
        }

        # Get all non-deprecated strategies with good performance
        good_performers = self.db.query(StrategyPerformance).filter(
            StrategyPerformance.is_deprecated == False,
            StrategyPerformance.live_total_trades >= self.min_trades_for_eval,
            StrategyPerformance.live_win_rate >= 50.0
        ).all()

        if not good_performers:
            print("   ⚠️  No strategies with sufficient performance data yet")
            return insights

        # Analyze by strategy type
        strategy_types = {}
        for perf in good_performers:
            strategy = self.db.query(Strategy).filter(
                Strategy.id == perf.strategy_id
            ).first()

            if strategy:
                stype = strategy.strategy_type
                if stype not in strategy_types:
                    strategy_types[stype] = []
                strategy_types[stype].append(perf.live_win_rate)

        # Calculate average win rate by type
        for stype, win_rates in strategy_types.items():
            insights["strategy_type_performance"][stype] = {
                "avg_win_rate": np.mean(win_rates),
                "count": len(win_rates)
            }

        print(f"   📊 Strategy Type Performance:")
        for stype, data in insights["strategy_type_performance"].items():
            print(f"      {stype}: {data['avg_win_rate']:.1f}% avg win rate ({data['count']} strategies)")

        # Top strategies
        top_5 = sorted(good_performers, key=lambda x: x.live_win_rate or 0, reverse=True)[:5]
        for perf in top_5:
            insights["top_strategies"].append({
                "name": perf.strategy_name,
                "win_rate": perf.live_win_rate,
                "trades": perf.live_total_trades
            })

        return insights


if __name__ == "__main__":
    # Test the analyzer
    analyzer = PerformanceAnalyzer()

    print("\n1. Testing daily metrics update...")
    result1 = analyzer.update_daily_metrics()
    print(f"   Result: {result1}")

    print("\n2. Testing weekly review...")
    result2 = analyzer.weekly_review()
    print(f"   Result: {result2}")

    print("\n3. Getting learning insights...")
    insights = analyzer.get_learning_insights()
    print(f"   Insights: {insights}")
