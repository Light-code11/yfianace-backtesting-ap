"""
Autonomous AI Learning System
Runs in background, continuously generating, testing, and improving strategies
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import asyncio
from sqlalchemy.orm import Session
import numpy as np

from database import SessionLocal, Strategy, BacktestResult, AILearning, PerformanceLog
from ai_strategy_generator import AIStrategyGenerator
from backtesting_engine import BacktestingEngine
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for PostgreSQL"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class AutonomousLearningAgent:
    """Autonomous agent that learns and improves trading strategies over time"""

    def __init__(self,
                 tickers: List[str] = None,
                 learning_interval_minutes: int = None,
                 learning_interval_hours: int = None,
                 strategies_per_cycle: int = 3,
                 min_quality_score: float = 70.0):
        """
        Initialize the autonomous learning agent

        Args:
            tickers: List of tickers to focus on (default: ['SPY', 'QQQ', 'AAPL'])
            learning_interval_minutes: How often to run learning cycle (in minutes)
            learning_interval_hours: How often to run learning cycle (in hours)
            strategies_per_cycle: Number of strategies to generate each cycle
            min_quality_score: Minimum quality score to keep a strategy
        """
        self.tickers = tickers or ['SPY', 'QQQ', 'AAPL']

        # Support both minutes and hours, prioritize minutes if provided
        if learning_interval_minutes is not None:
            self.interval_seconds = learning_interval_minutes * 60
            self.interval_display = f"{learning_interval_minutes} minutes"
        elif learning_interval_hours is not None:
            self.interval_seconds = learning_interval_hours * 3600
            self.interval_display = f"{learning_interval_hours} hours"
        else:
            # Default to 6 hours
            self.interval_seconds = 6 * 3600
            self.interval_display = "6 hours"

        self.strategies_per_cycle = strategies_per_cycle
        self.min_quality_score = min_quality_score
        self.ai_generator = AIStrategyGenerator()
        self.backtest_engine = BacktestingEngine()
        self.should_stop = False  # Flag to stop the agent

        logger.info(f"Autonomous Learning Agent initialized")
        logger.info(f"Tickers: {self.tickers}")
        logger.info(f"Learning interval: {self.interval_display}")
        logger.info(f"Strategies per cycle: {strategies_per_cycle}")

    def fetch_market_data(self, tickers: List[str], period: str = "1y"):
        """Fetch market data for backtesting, including benchmark ETFs for relative strength analysis"""
        try:
            # Add benchmark tickers for relative strength analysis
            benchmarks = {'SPY', 'QQQ', 'SOXX'}  # Market and sector benchmarks (SOXX = iShares Semiconductor ETF)
            all_tickers = list(set(tickers) | benchmarks)  # Combine and remove duplicates

            ticker_string = " ".join(all_tickers)
            data = yf.download(ticker_string, period=period, progress=False)
            logger.info(f"Fetched market data for {all_tickers}: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def learning_cycle(self):
        """Execute one complete learning cycle"""
        logger.info("="*60)
        logger.info("STARTING NEW LEARNING CYCLE")
        logger.info("="*60)

        db = SessionLocal()

        try:
            # Step 1: Gather past performance data
            logger.info("Step 1: Gathering past performance data...")
            past_results = db.query(BacktestResult).order_by(
                BacktestResult.created_at.desc()
            ).limit(20).all()

            past_performance = [
                {
                    "strategy_name": r.strategy_name,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_return_pct": r.total_return_pct,
                    "win_rate": r.win_rate,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "quality_score": r.quality_score
                }
                for r in past_results
            ]

            # Step 2: Get existing learning insights
            logger.info("Step 2: Loading learning insights...")
            learnings = db.query(AILearning).order_by(
                AILearning.created_at.desc()
            ).limit(5).all()

            learning_insights = [
                {
                    "type": l.learning_type,
                    "description": l.description,
                    "insights": l.key_insights
                }
                for l in learnings
            ]

            # Step 3: Fetch market data
            logger.info(f"Step 3: Fetching market data for {self.tickers}...")
            market_data = self.fetch_market_data(self.tickers, period="1y")

            if market_data is None or market_data.empty:
                logger.error("Failed to fetch market data. Aborting cycle.")
                return

            # Step 4: Generate new strategies using AI
            logger.info(f"Step 4: Generating {self.strategies_per_cycle} new strategies...")
            strategies = self.ai_generator.generate_strategies(
                market_data=market_data,
                tickers=self.tickers,
                num_strategies=self.strategies_per_cycle,
                past_performance=past_performance,
                learning_insights=learning_insights
            )

            logger.info(f"Generated {len(strategies)} strategies")

            # Step 5: Backtest each strategy
            logger.info("Step 5: Backtesting all strategies...")
            backtest_results = []

            for i, strategy in enumerate(strategies, 1):
                logger.info(f"  Backtesting strategy {i}/{len(strategies)}: {strategy['name']}")

                # Make strategy name unique by adding timestamp
                unique_name = f"{strategy['name']} [{datetime.now().strftime('%Y%m%d_%H%M%S')}]"

                # Save strategy to database first
                db_strategy = Strategy(
                    name=unique_name,
                    description=strategy.get('description', ''),
                    tickers=strategy.get('tickers', self.tickers),
                    entry_conditions=strategy.get('entry_conditions', {}),
                    exit_conditions=strategy.get('exit_conditions', {}),
                    stop_loss_pct=strategy.get('risk_management', {}).get('stop_loss_pct', 5.0),
                    take_profit_pct=strategy.get('risk_management', {}).get('take_profit_pct', 10.0),
                    position_size_pct=strategy.get('risk_management', {}).get('position_size_pct', 10.0),
                    holding_period_days=strategy.get('holding_period_days', 5),
                    rationale=strategy.get('rationale', ''),
                    market_analysis=strategy.get('market_analysis', ''),
                    risk_assessment=strategy.get('risk_assessment', ''),
                    strategy_type=strategy.get('strategy_type', 'unknown'),
                    indicators=strategy.get('indicators', []),
                    is_active=True
                )
                db.add(db_strategy)
                db.commit()
                db.refresh(db_strategy)

                # Backtest
                try:
                    strategy_config = {
                        "id": db_strategy.id,
                        "name": db_strategy.name,
                        "tickers": db_strategy.tickers,
                        "indicators": db_strategy.indicators,
                        "strategy_type": db_strategy.strategy_type,
                        "risk_management": {
                            "stop_loss_pct": db_strategy.stop_loss_pct,
                            "take_profit_pct": db_strategy.take_profit_pct,
                            "position_size_pct": db_strategy.position_size_pct
                        }
                    }

                    results = self.backtest_engine.backtest_strategy(strategy_config, market_data)

                    # Convert numpy types to Python types for PostgreSQL compatibility
                    trades_clean = convert_numpy_types(results['trades'])
                    equity_curve_clean = convert_numpy_types(results['equity_curve'])
                    metrics_clean = convert_numpy_types(results['metrics'])

                    # Save backtest results
                    backtest_result = BacktestResult(
                        strategy_id=db_strategy.id,
                        strategy_name=results['strategy_name'],
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                        initial_capital=100000,
                        tickers_tested=results['tickers'],
                        total_return_pct=metrics_clean['total_return_pct'],
                        total_trades=metrics_clean['total_trades'],
                        winning_trades=metrics_clean['winning_trades'],
                        losing_trades=metrics_clean['losing_trades'],
                        win_rate=metrics_clean['win_rate'],
                        sharpe_ratio=metrics_clean['sharpe_ratio'],
                        sortino_ratio=metrics_clean['sortino_ratio'],
                        max_drawdown_pct=metrics_clean['max_drawdown_pct'],
                        profit_factor=metrics_clean['profit_factor'],
                        avg_win=metrics_clean['avg_win'],
                        avg_loss=metrics_clean['avg_loss'],
                        trades=trades_clean,
                        equity_curve=equity_curve_clean,
                        quality_score=metrics_clean['quality_score'],
                        kelly_criterion=metrics_clean.get('kelly_criterion'),
                        kelly_position_pct=metrics_clean.get('kelly_position_pct'),
                        kelly_risk_level=metrics_clean.get('kelly_risk_level')
                    )
                    db.add(backtest_result)
                    db.commit()

                    backtest_results.append(backtest_result)

                    logger.info(f"    ✅ Quality Score: {results['metrics']['quality_score']:.1f}/100, "
                              f"Sharpe: {results['metrics']['sharpe_ratio']:.2f}, "
                              f"Return: {results['metrics']['total_return_pct']:.1f}%")

                except Exception as e:
                    logger.error(f"    ❌ Backtest failed: {e}")
                    continue

            # Step 6: Analyze results and learn
            logger.info("Step 6: Analyzing results and extracting insights...")

            if backtest_results:
                strategies_data = [
                    {
                        "name": s.name,
                        "strategy_type": s.strategy_type,
                        "indicators": s.indicators,
                        "stop_loss_pct": s.stop_loss_pct,
                        "take_profit_pct": s.take_profit_pct
                    }
                    for s in [db.query(Strategy).get(br.strategy_id) for br in backtest_results]
                ]

                results_data = [
                    {
                        "strategy_name": r.strategy_name,
                        "sharpe_ratio": r.sharpe_ratio,
                        "total_return_pct": r.total_return_pct,
                        "win_rate": r.win_rate,
                        "max_drawdown_pct": r.max_drawdown_pct,
                        "quality_score": r.quality_score
                    }
                    for r in backtest_results
                ]

                # Use AI to learn from results
                learning = self.ai_generator.learn_from_results(strategies_data, results_data)

                # Save learning insights
                ai_learning = AILearning(
                    learning_type="autonomous_cycle",
                    description=f"Autonomous learning cycle completed at {datetime.now().isoformat()}",
                    strategy_ids=[r.strategy_id for r in backtest_results],
                    performance_data=results_data,
                    key_insights=learning,
                    recommendations=learning.get('recommendations_for_next_generation', []),
                    confidence_score=0.85
                )
                db.add(ai_learning)
                db.commit()

                logger.info("✅ Learning insights saved to database")

            # Step 7: Archive poor performers
            logger.info("Step 7: Archiving poor performing strategies...")
            poor_performers = [
                br for br in backtest_results
                if br.quality_score < self.min_quality_score
            ]

            for br in poor_performers:
                strategy = db.query(Strategy).get(br.strategy_id)
                if strategy:
                    strategy.is_active = False
                    logger.info(f"  Archived: {strategy.name} (Quality: {br.quality_score:.1f})")

            db.commit()

            # Step 8: Summary
            logger.info("="*60)
            logger.info("LEARNING CYCLE COMPLETE")
            logger.info(f"✅ Generated: {len(strategies)} strategies")
            logger.info(f"✅ Backtested: {len(backtest_results)} strategies")
            logger.info(f"✅ Archived: {len(poor_performers)} poor performers")
            logger.info(f"✅ Active strategies: {len(backtest_results) - len(poor_performers)}")

            if backtest_results:
                best = max(backtest_results, key=lambda x: x.quality_score)
                logger.info(f"🏆 Best strategy: {best.strategy_name} (Quality: {best.quality_score:.1f})")

            logger.info("="*60)

        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

        finally:
            db.close()

    def stop(self):
        """Stop the learning agent"""
        self.should_stop = True
        logger.info("🛑 Stop signal received. Agent will stop after current cycle.")

    def run_forever(self):
        """Run the learning agent continuously"""
        logger.info("🤖 Autonomous Learning Agent started!")
        logger.info(f"Will run learning cycle every {self.interval_display}")

        cycle_count = 0

        while not self.should_stop:
            try:
                cycle_count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE #{cycle_count}")
                logger.info(f"{'='*60}")

                self.learning_cycle()

                if self.should_stop:
                    break

                # Wait for next cycle
                logger.info(f"\n⏰ Next cycle in {self.interval_display}...")
                logger.info(f"Next run at: {(datetime.now() + timedelta(seconds=self.interval_seconds)).strftime('%Y-%m-%d %H:%M:%S')}")

                # Sleep in small intervals to check stop flag frequently
                sleep_intervals = int(self.interval_seconds / 10)  # Check every 1/10th of interval
                for _ in range(sleep_intervals):
                    if self.should_stop:
                        break
                    time.sleep(10)

            except KeyboardInterrupt:
                logger.info("\n🛑 Autonomous Learning Agent stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                logger.info("Will retry in 1 hour...")
                time.sleep(3600)

        logger.info("🛑 Autonomous Learning Agent stopped.")


if __name__ == "__main__":
    # Example usage
    agent = AutonomousLearningAgent(
        tickers=['NVDA', 'AAPL', 'MSFT'],  # Tickers to focus on
        learning_interval_hours=6,          # Run every 6 hours
        strategies_per_cycle=3,             # Generate 3 strategies per cycle
        min_quality_score=70.0              # Archive strategies below 70 quality
    )

    agent.run_forever()
