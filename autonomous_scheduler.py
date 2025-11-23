"""
Autonomous Scheduler - Runs the system automatically without human intervention

Schedules:
- Daily (10:00 AM ET): Run trading cycle
- Weekly (Sunday 6:00 PM ET): Generate new strategies
- Daily (5:00 PM ET): Update performance metrics
- Weekly (Sunday 7:00 PM ET): Deprecate underperformers

This is the "brain" that keeps the system running 24/7
"""
import os
from datetime import datetime, time
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from autonomous_trading_engine import AutonomousTradingEngine
from auto_strategy_generator import AutoStrategyGenerator
from performance_analyzer import PerformanceAnalyzer

load_dotenv()


class AutonomousScheduler:
    """
    Manages all autonomous system tasks

    Daily Tasks:
    - 10:00 AM ET: Trading cycle (scan market, execute trades)
    - 5:00 PM ET: Performance update

    Weekly Tasks:
    - Sunday 6:00 PM ET: Generate new strategies
    - Sunday 7:00 PM ET: Analyze performance, deprecate failures
    """

    def __init__(self):
        self.scheduler = BlockingScheduler(timezone=pytz.timezone('US/Eastern'))
        self.trading_engine = None
        self.strategy_generator = None
        self.performance_analyzer = None

        # Configuration
        self.auto_trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
        self.auto_generate_enabled = os.getenv('AUTO_GENERATE_STRATEGIES', 'false').lower() == 'true'

        print("🤖 Autonomous Scheduler Initialized")
        print(f"   Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"   Auto-generation: {'ENABLED' if self.auto_generate_enabled else 'DISABLED'}")

    def daily_trading_cycle(self):
        """Run daily trading cycle (10:00 AM ET)"""
        print("\n" + "=" * 70)
        print(f"📅 DAILY TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if not self.auto_trading_enabled:
            print("⚠️  Auto-trading is DISABLED. Skipping trading cycle.")
            return

        try:
            self.trading_engine = AutonomousTradingEngine()
            results = self.trading_engine.run_daily_cycle()

            print(f"\n✅ Trading cycle complete:")
            print(f"   Signals: {results.get('signals_generated', 0)}")
            print(f"   Executed: {results.get('trades_executed', 0)}")
            print(f"   Rejected: {results.get('trades_rejected', 0)}")

        except Exception as e:
            print(f"\n❌ Trading cycle failed: {str(e)}")

    def daily_performance_update(self):
        """Update performance metrics (5:00 PM ET - after market close)"""
        print("\n" + "=" * 70)
        print(f"📊 DAILY PERFORMANCE UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        try:
            self.performance_analyzer = PerformanceAnalyzer()
            results = self.performance_analyzer.update_daily_metrics()

            print(f"\n✅ Performance update complete:")
            print(f"   Strategies analyzed: {results.get('strategies_analyzed', 0)}")
            print(f"   Weights adjusted: {results.get('weights_adjusted', 0)}")

        except Exception as e:
            print(f"\n❌ Performance update failed: {str(e)}")

    def weekly_strategy_generation(self):
        """Generate new strategies (Sunday 6:00 PM ET)"""
        print("\n" + "=" * 70)
        print(f"🧬 WEEKLY STRATEGY GENERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        if not self.auto_generate_enabled:
            print("⚠️  Auto-generation is DISABLED. Skipping strategy generation.")
            return

        try:
            self.strategy_generator = AutoStrategyGenerator()

            num_strategies = int(os.getenv('STRATEGIES_PER_BATCH', 20))
            min_sharpe = float(os.getenv('MIN_SHARPE_FOR_DEPLOYMENT', 1.5))
            top_n = int(os.getenv('TOP_N_STRATEGIES_TO_DEPLOY', 3))

            results = self.strategy_generator.run_full_cycle(
                num_strategies=num_strategies,
                min_sharpe=min_sharpe,
                top_n=top_n
            )

            print(f"\n✅ Strategy generation complete:")
            print(f"   Generated: {results.get('strategies_generated', 0)}")
            print(f"   Winners: {results.get('winners_found', 0)}")
            print(f"   Deployed: {results.get('strategies_deployed', 0)}")

        except Exception as e:
            print(f"\n❌ Strategy generation failed: {str(e)}")

    def weekly_performance_review(self):
        """Analyze performance and deprecate bad strategies (Sunday 7:00 PM ET)"""
        print("\n" + "=" * 70)
        print(f"📈 WEEKLY PERFORMANCE REVIEW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        try:
            self.performance_analyzer = PerformanceAnalyzer()
            results = self.performance_analyzer.weekly_review()

            print(f"\n✅ Performance review complete:")
            print(f"   Strategies reviewed: {results.get('strategies_reviewed', 0)}")
            print(f"   Deprecated: {results.get('strategies_deprecated', 0)}")
            print(f"   Top performers: {results.get('top_performers', [])}")

        except Exception as e:
            print(f"\n❌ Performance review failed: {str(e)}")

    def start(self):
        """Start the autonomous scheduler"""
        print("\n" + "=" * 70)
        print("🚀 STARTING AUTONOMOUS SYSTEM")
        print("=" * 70)

        # Daily: Trading cycle at 10:00 AM ET (after market opens)
        self.scheduler.add_job(
            self.daily_trading_cycle,
            CronTrigger(hour=10, minute=0, day_of_week='mon-fri', timezone='US/Eastern'),
            id='daily_trading',
            name='Daily Trading Cycle',
            replace_existing=True
        )
        print("✅ Scheduled: Daily trading cycle (Mon-Fri 10:00 AM ET)")

        # Daily: Performance update at 5:00 PM ET (after market closes)
        self.scheduler.add_job(
            self.daily_performance_update,
            CronTrigger(hour=17, minute=0, day_of_week='mon-fri', timezone='US/Eastern'),
            id='daily_performance',
            name='Daily Performance Update',
            replace_existing=True
        )
        print("✅ Scheduled: Daily performance update (Mon-Fri 5:00 PM ET)")

        # Weekly: Strategy generation on Sunday 6:00 PM ET
        self.scheduler.add_job(
            self.weekly_strategy_generation,
            CronTrigger(hour=18, minute=0, day_of_week='sun', timezone='US/Eastern'),
            id='weekly_generation',
            name='Weekly Strategy Generation',
            replace_existing=True
        )
        print("✅ Scheduled: Weekly strategy generation (Sunday 6:00 PM ET)")

        # Weekly: Performance review on Sunday 7:00 PM ET
        self.scheduler.add_job(
            self.weekly_performance_review,
            CronTrigger(hour=19, minute=0, day_of_week='sun', timezone='US/Eastern'),
            id='weekly_review',
            name='Weekly Performance Review',
            replace_existing=True
        )
        print("✅ Scheduled: Weekly performance review (Sunday 7:00 PM ET)")

        print("\n📅 Upcoming scheduled jobs:")
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            print(f"   {job.name}: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z') if next_run else 'N/A'}")

        print("\n🎯 System is now running autonomously!")
        print("   Press Ctrl+C to stop")
        print("=" * 70)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("\n\n🛑 Shutting down autonomous system...")
            self.scheduler.shutdown()
            print("✅ Shutdown complete")


def run_manual_test():
    """Run all tasks once for testing (doesn't start scheduler)"""
    print("=" * 70)
    print("🧪 MANUAL TEST MODE - Running all tasks once")
    print("=" * 70)

    scheduler = AutonomousScheduler()

    print("\n1. Testing Daily Trading Cycle...")
    scheduler.daily_trading_cycle()

    print("\n2. Testing Daily Performance Update...")
    scheduler.daily_performance_update()

    print("\n3. Testing Weekly Strategy Generation...")
    scheduler.weekly_strategy_generation()

    print("\n4. Testing Weekly Performance Review...")
    scheduler.weekly_performance_review()

    print("\n" + "=" * 70)
    print("✅ Manual test complete!")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Manual test mode
        run_manual_test()
    else:
        # Start autonomous scheduler
        scheduler = AutonomousScheduler()
        scheduler.start()
