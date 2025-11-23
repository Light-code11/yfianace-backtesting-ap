"""
Autonomous Trading Engine - Makes trading decisions and executes without human input

This is the brain of the autonomous system that:
1. Scans market for signals
2. Evaluates signals based on strategy performance
3. Calculates position sizes
4. Executes trades via Alpaca
5. Monitors positions
6. Learns from results
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca_client import AlpacaClient
from market_scanner import MarketScanner
from live_signal_generator import LiveSignalGenerator
from database import (
    SessionLocal, Strategy, StrategyPerformance, LivePosition,
    TradeExecution, AutoTradingState
)

load_dotenv()


class AutonomousTradingEngine:
    """
    Fully autonomous trading system

    Makes decisions based on:
    - Strategy backtest performance
    - Live performance tracking
    - Market regime
    - Risk management rules
    - Position correlations
    """

    def __init__(self):
        self.alpaca = AlpacaClient()
        self.db = SessionLocal()

        # Configuration from environment
        self.auto_trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
        self.max_position_size_pct = float(os.getenv('MAX_POSITION_SIZE_PCT', 20))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', 5))
        self.max_portfolio_positions = int(os.getenv('MAX_PORTFOLIO_POSITIONS', 10))
        self.min_signal_confidence = os.getenv('MIN_SIGNAL_CONFIDENCE', 'HIGH')

        print(f"🤖 Autonomous Trading Engine initialized")
        print(f"   Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"   Max position size: {self.max_position_size_pct}%")
        print(f"   Max daily loss: {self.max_daily_loss_pct}%")

    def run_daily_cycle(self) -> Dict[str, Any]:
        """
        Main execution loop - runs once per day

        Steps:
        1. Check if trading is enabled
        2. Check risk limits (circuit breakers)
        3. Sync positions from Alpaca
        4. Scan market for signals
        5. Evaluate signals
        6. Execute trades
        7. Update performance tracking
        8. Log results
        """
        print("\n" + "=" * 70)
        print(f"🚀 AUTONOMOUS TRADING CYCLE STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "errors": []
        }

        try:
            # Step 1: Check if enabled
            if not self.auto_trading_enabled:
                results['error'] = "Auto-trading is DISABLED. Set AUTO_TRADING_ENABLED=true in .env"
                print(f"⚠️  {results['error']}")
                return results

            # Step 2: Check circuit breakers
            breaker_check = self._check_circuit_breakers()
            if not breaker_check['can_trade']:
                results['error'] = f"Circuit breaker triggered: {breaker_check['reason']}"
                print(f"🛑 {results['error']}")
                return results

            # Step 3: Sync positions
            print("\n📊 Syncing positions from Alpaca...")
            self._sync_positions()

            # Step 4: Generate signals
            print("\n🔍 Scanning market for trading signals...")
            signals = self._generate_signals()
            results['signals_generated'] = len(signals)
            print(f"   Found {len(signals)} potential signals")

            # Step 5: Evaluate and filter signals
            print("\n🧠 Evaluating signals...")
            actionable_signals = self._evaluate_signals(signals)
            print(f"   {len(actionable_signals)} signals passed evaluation")

            # Step 6: Execute trades
            print("\n💰 Executing trades...")
            for signal in actionable_signals:
                execution_result = self._execute_signal(signal)
                if execution_result['success']:
                    results['trades_executed'] += 1
                else:
                    results['trades_rejected'] += 1
                    results['errors'].append(execution_result.get('error'))

            # Step 7: Update performance
            print("\n📈 Updating performance metrics...")
            self._update_performance()

            # Step 8: Update state
            self._update_system_state(results)

            results['success'] = True
            print(f"\n✅ CYCLE COMPLETE")
            print(f"   Signals: {results['signals_generated']}")
            print(f"   Executed: {results['trades_executed']}")
            print(f"   Rejected: {results['trades_rejected']}")

        except Exception as e:
            results['error'] = str(e)
            results['errors'].append(str(e))
            print(f"\n❌ ERROR: {str(e)}")

        finally:
            self.db.close()

        return results

    def _check_circuit_breakers(self) -> Dict[str, Any]:
        """
        Check if trading should be halted

        Reasons to halt:
        - Daily loss limit exceeded
        - Market is closed
        - System errors
        """
        # Check market hours
        if not self.alpaca.is_market_open():
            return {
                "can_trade": False,
                "reason": "Market is closed"
            }

        # Check daily loss limit
        account = self.alpaca.get_account()
        if account['success']:
            acc = account['account']
            equity = float(acc['equity'])
            last_equity = float(acc['last_equity'])

            if last_equity > 0:
                daily_pnl_pct = ((equity - last_equity) / last_equity) * 100

                if daily_pnl_pct < -self.max_daily_loss_pct:
                    return {
                        "can_trade": False,
                        "reason": f"Daily loss limit exceeded: {daily_pnl_pct:.2f}%"
                    }

        return {"can_trade": True, "reason": None}

    def _sync_positions(self):
        """Sync positions from Alpaca to database"""
        positions = self.alpaca.get_positions()

        if not positions['success']:
            print(f"   ⚠️  Failed to sync positions: {positions['error']}")
            return

        # Mark all positions as closed first
        self.db.query(LivePosition).filter(LivePosition.is_open == True).update({
            "is_open": False,
            "exit_reason": "sync_closed"
        })

        # Add/update current positions
        for pos in positions.get('positions', []):
            symbol = pos['symbol']
            alpaca_id = f"{symbol}_{pos['asset_id']}"

            # Check if position exists
            db_pos = self.db.query(LivePosition).filter(
                LivePosition.alpaca_position_id == alpaca_id
            ).first()

            if db_pos:
                # Update existing
                db_pos.qty = float(pos['qty'])
                db_pos.current_price = float(pos['current_price'])
                db_pos.unrealized_pl = float(pos['unrealized_pl'])
                db_pos.unrealized_plpc = float(pos['unrealized_plpc'])
                db_pos.is_open = True
                db_pos.updated_at = datetime.utcnow()
            else:
                # Create new
                db_pos = LivePosition(
                    ticker=symbol,
                    qty=float(pos['qty']),
                    entry_price=float(pos['avg_entry_price']),
                    current_price=float(pos['current_price']),
                    unrealized_pl=float(pos['unrealized_pl']),
                    unrealized_plpc=float(pos['unrealized_plpc']),
                    alpaca_position_id=alpaca_id,
                    alpaca_data=pos,
                    is_open=True
                )
                self.db.add(db_pos)

        self.db.commit()
        print(f"   ✅ Synced {len(positions.get('positions', []))} positions")

    def _generate_signals(self) -> List[Dict]:
        """Generate trading signals from all active strategies"""
        # Get active strategies
        strategies = self.db.query(Strategy).filter(
            Strategy.is_active == True
        ).all()

        if not strategies:
            print("   ⚠️  No active strategies found")
            return []

        # Prepare strategy configs
        strategy_configs = []
        for strat in strategies:
            strategy_configs.append({
                'id': strat.id,
                'name': strat.name,
                'strategy_type': strat.strategy_type,
                'indicators': strat.indicators,
                'risk_management': {
                    'stop_loss_pct': strat.stop_loss_pct,
                    'take_profit_pct': strat.take_profit_pct,
                    'position_size_pct': strat.position_size_pct
                }
            })

        # Run market scanner
        scan_results = MarketScanner.scan_market(
            strategies=strategy_configs,
            universe=None,  # Use default universe
            max_workers=10,
            min_confidence=self.min_signal_confidence
        )

        all_signals = scan_results.get('all_signals', [])

        # Filter for BUY/SELL only (no HOLD)
        return [s for s in all_signals if s['signal'] in ['BUY', 'SELL']]

    def _evaluate_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Evaluate signals and decide which to execute

        Factors:
        - Signal confidence
        - Strategy live performance vs backtest
        - Current positions (don't over-concentrate)
        - Position size limits
        - Correlation with existing positions
        """
        actionable = []

        for signal in signals:
            # Check confidence
            if signal.get('confidence') != 'HIGH' and self.min_signal_confidence == 'HIGH':
                continue

            # Check if we already have a position in this ticker
            existing_pos = self.db.query(LivePosition).filter(
                LivePosition.ticker == signal['ticker'],
                LivePosition.is_open == True
            ).first()

            if existing_pos:
                continue  # Skip if already in position

            # Check position count limit
            open_positions_count = self.db.query(LivePosition).filter(
                LivePosition.is_open == True
            ).count()

            if open_positions_count >= self.max_portfolio_positions:
                print(f"   ⚠️  Max positions reached ({self.max_portfolio_positions})")
                break

            # Get strategy performance
            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == signal.get('strategy_id')
            ).first()

            # If strategy is deprecated, skip
            if perf and perf.is_deprecated:
                continue

            # Adjust position size based on strategy performance
            if perf and perf.allocation_weight:
                signal['adjusted_position_size_pct'] = signal.get('position_size_pct', 25) * perf.allocation_weight
            else:
                signal['adjusted_position_size_pct'] = signal.get('position_size_pct', 25)

            # Cap at max position size
            signal['adjusted_position_size_pct'] = min(
                signal['adjusted_position_size_pct'],
                self.max_position_size_pct
            )

            actionable.append(signal)

        # Sort by quality score (highest first)
        actionable.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        return actionable

    def _execute_signal(self, signal: Dict) -> Dict[str, Any]:
        """Execute a trading signal via Alpaca"""
        try:
            # Get account to calculate position size
            account = self.alpaca.get_account()
            if not account['success']:
                return {"success": False, "error": "Failed to get account"}

            equity = float(account['account']['equity'])
            position_size_usd = equity * (signal['adjusted_position_size_pct'] / 100)

            # Place order
            if signal['signal'] == 'BUY':
                order = self.alpaca.place_order(
                    symbol=signal['ticker'],
                    notional=position_size_usd,
                    side='buy',
                    order_type='market',
                    time_in_force='day',
                    take_profit=signal.get('take_profit'),
                    stop_loss=signal.get('stop_loss')
                )
            else:  # SELL
                # Check if we have a position to sell
                pos = self.alpaca.get_position(signal['ticker'])
                if not pos['success'] or not pos['position']:
                    return {"success": False, "error": "No position to sell"}

                order = self.alpaca.close_position(signal['ticker'])

            if not order['success']:
                return {"success": False, "error": order['error']}

            # Log execution
            execution = TradeExecution(
                ticker=signal['ticker'],
                strategy_id=signal.get('strategy_id'),
                strategy_name=signal.get('strategy_name'),
                signal_type=signal['signal'],
                signal_confidence=signal.get('confidence'),
                signal_price=signal.get('current_price'),
                order_type='market',
                side='buy' if signal['signal'] == 'BUY' else 'sell',
                notional=position_size_usd if signal['signal'] == 'BUY' else None,
                order_status='submitted',
                alpaca_order_id=order['order']['id'],
                alpaca_order_data=order['order'],
                decision_reasoning=signal.get('reasoning'),
                decision_factors={
                    'quality_score': signal.get('quality_score'),
                    'confidence': signal.get('confidence'),
                    'adjusted_position_size_pct': signal.get('adjusted_position_size_pct')
                }
            )
            self.db.add(execution)
            self.db.commit()

            print(f"   ✅ {signal['signal']} {signal['ticker']} @ ${signal.get('current_price')} (${position_size_usd:,.0f})")

            return {"success": True, "order": order['order']}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_performance(self):
        """Update strategy performance metrics"""
        # Get all strategies with live trades
        strategies = self.db.query(Strategy).filter(Strategy.is_active == True).all()

        for strategy in strategies:
            # Get live trade history
            executions = self.db.query(TradeExecution).filter(
                TradeExecution.strategy_id == strategy.id,
                TradeExecution.order_status == 'filled'
            ).all()

            if len(executions) < 10:
                continue  # Not enough data yet

            # Calculate live metrics
            # TODO: Implement full performance calculation
            # For now, just update trade counts

            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == strategy.id
            ).first()

            if not perf:
                # Create performance record
                perf = StrategyPerformance(
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    backtest_sharpe=0.0,  # TODO: Get from backtest
                    backtest_win_rate=0.0,
                    backtest_avg_return=0.0,
                    backtest_max_drawdown=0.0
                )
                self.db.add(perf)

            perf.live_total_trades = len(executions)
            perf.updated_at = datetime.utcnow()

        self.db.commit()

    def _update_system_state(self, results: Dict):
        """Update autonomous trading system state"""
        state = self.db.query(AutoTradingState).first()

        if not state:
            state = AutoTradingState(
                is_enabled=self.auto_trading_enabled,
                portfolio_value=0.0,
                cash_balance=0.0,
                buying_power=0.0
            )
            self.db.add(state)

        # Update state
        state.is_enabled = self.auto_trading_enabled
        state.last_run_at = datetime.utcnow()
        state.daily_trades = results.get('trades_executed', 0)
        state.total_signals_generated += results.get('signals_generated', 0)
        state.total_trades_executed += results.get('trades_executed', 0)
        state.total_trades_rejected += results.get('trades_rejected', 0)

        # Update portfolio values
        account = self.alpaca.get_account()
        if account['success']:
            acc = account['account']
            state.portfolio_value = float(acc['equity'])
            state.cash_balance = float(acc['cash'])
            state.buying_power = float(acc['buying_power'])

        self.db.commit()


if __name__ == "__main__":
    # Test the engine
    engine = AutonomousTradingEngine()
    results = engine.run_daily_cycle()

    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Success: {results['success']}")
    print(f"Signals: {results['signals_generated']}")
    print(f"Executed: {results['trades_executed']}")
    print(f"Rejected: {results['trades_rejected']}")
    if results.get('error'):
        print(f"Error: {results['error']}")
    print("=" * 70)
