
"""
Autonomous Trading Engine - Makes trading decisions and executes without human input.
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv

from alpaca_client import AlpacaClient
from market_scanner import MarketScanner
from database import (
    SessionLocal,
    Strategy,
    StrategyPerformance,
    LivePosition,
    TradeExecution,
    AutoTradingState,
    RLAllocationLog,
    RiskEventLog,
)
from risk_config import RiskConfig
from risk_manager import RiskManager
from rl_strategy_allocator import RLStrategyAllocator

try:
    from ml_price_predictor import MLPricePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML Price Predictor not available")

try:
    from hmm_regime_detector import HMMRegimeDetector
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: HMM Regime Detector not available")

try:
    from daily_report import generate_daily_report
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

load_dotenv()


class AutonomousTradingEngine:
    """
    Fully autonomous trading system.
    """

    def __init__(self):
        self.alpaca = AlpacaClient()
        self.db = SessionLocal()

        self.auto_trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
        self.max_position_size_pct = float(os.getenv('MAX_POSITION_SIZE_PCT', 20))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', 5))
        self.max_portfolio_positions = int(os.getenv('MAX_PORTFOLIO_POSITIONS', 10))
        self.min_signal_confidence = os.getenv('MIN_SIGNAL_CONFIDENCE', 'HIGH')

        self.use_ml_validation = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
        self.use_regime_detection = os.getenv('USE_REGIME_DETECTION', 'true').lower() == 'true'
        self.ml_min_confidence = float(os.getenv('ML_MIN_CONFIDENCE', 0.6))

        risk_cfg = RiskConfig(
            max_drawdown_pct=float(os.getenv('RISK_MAX_DRAWDOWN_PCT', 10.0)),
            max_daily_loss_pct=float(os.getenv('RISK_MAX_DAILY_LOSS_PCT', 2.0)),
            max_position_pct=float(os.getenv('RISK_MAX_POSITION_PCT', 5.0)),
            max_exposure_pct=float(os.getenv('RISK_MAX_EXPOSURE_PCT', 80.0)),
            max_correlation=float(os.getenv('RISK_MAX_CORRELATION', 0.8)),
            consecutive_loss_limit=int(os.getenv('RISK_CONSECUTIVE_LOSS_LIMIT', 3)),
            pause_hours=int(os.getenv('RISK_PAUSE_HOURS', 4)),
            max_positions=int(os.getenv('RISK_MAX_POSITIONS', 10)),
            vix_threshold=float(os.getenv('RISK_VIX_THRESHOLD', 25.0)),
            max_sector_pct=float(os.getenv('RISK_MAX_SECTOR_PCT', 30.0)),
        )
        self.risk_manager = RiskManager(risk_cfg)

        self.ml_predictor = None
        self.regime_detector = None
        self.current_regime = None

        if ML_AVAILABLE and self.use_ml_validation:
            self.ml_predictor = MLPricePredictor()
            print("   XGBoost ML Predictor loaded")

        if HMM_AVAILABLE and self.use_regime_detection:
            self.regime_detector = HMMRegimeDetector()
            print("   HMM Regime Detector loaded")

        self.rl_allocator = self._initialize_rl_allocator()
        self.last_rl_allocations: Dict[str, float] = {}
        self.last_rl_retrain_at: Optional[datetime] = None

        print("Autonomous Trading Engine initialized")
        print(f"   Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"   Max position size: {self.max_position_size_pct}%")
        print(f"   Max daily loss: {self.max_daily_loss_pct}%")
        print(f"   ML Validation: {'ENABLED' if self.use_ml_validation and ML_AVAILABLE else 'DISABLED'}")
        print(f"   Regime Detection: {'ENABLED' if self.use_regime_detection and HMM_AVAILABLE else 'DISABLED'}")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _initialize_rl_allocator(self) -> RLStrategyAllocator:
        strategy_types = self.db.query(Strategy.strategy_type).filter(Strategy.is_active == True).all()
        names = []
        for item in strategy_types:
            if isinstance(item, tuple):
                stype = item[0]
            else:
                stype = item.strategy_type
            if stype:
                names.append(stype)

        unique_names = sorted(set(names)) if names else ["momentum", "mean_reversion", "breakout"]
        return RLStrategyAllocator(strategy_names=unique_names)

    def _refresh_risk_state(self):
        account = self.alpaca.get_account()
        if not account.get('success'):
            return

        acc = account.get('account', {})
        equity = self._safe_float(acc.get('equity'))
        cash = self._safe_float(acc.get('cash'))

        positions = []
        for pos in self.db.query(LivePosition).filter(LivePosition.is_open == True).all():
            positions.append(
                {
                    'ticker': pos.ticker,
                    'qty': self._safe_float(pos.qty),
                    'current_price': self._safe_float(pos.current_price),
                    'market_value': abs(self._safe_float(pos.qty) * self._safe_float(pos.current_price)),
                    'sector': 'UNKNOWN',
                }
            )

        self.risk_manager.update_portfolio_state(account_value=equity, cash=cash, positions=positions)

        try:
            import yfinance as yf

            vix = yf.download('^VIX', period='5d', progress=False, auto_adjust=True)
            if not vix.empty:
                self.risk_manager.update_market_context(vix=self._safe_float(vix['Close'].iloc[-1], 20.0))
        except Exception:
            pass

    def _log_risk_event(self, event_type: str, message: str, severity: str = "warning", ticker: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        try:
            evt = RiskEventLog(
                event_type=event_type,
                severity=severity,
                message=message,
                ticker=ticker,
                context=context or {},
            )
            self.db.add(evt)
            self.db.commit()
        except Exception:
            self.db.rollback()

    def _log_rl_allocations(self, allocations: Dict[str, float], state_snapshot: Dict[str, Any]):
        try:
            for strategy_name, allocation in allocations.items():
                row = RLAllocationLog(
                    strategy_name=strategy_name,
                    regime=self.current_regime.get('label') if self.current_regime else None,
                    allocations={strategy_name: allocation},
                    state_snapshot=state_snapshot,
                    source='autonomous_engine',
                )
                self.db.add(row)
            self.db.commit()
        except Exception:
            self.db.rollback()

    def _build_rl_state(self) -> Dict[str, Any]:
        regime = self.current_regime or {}
        probs = regime.get('probabilities') or {}

        regime_vector = [
            self._safe_float(probs.get('BULL', 0.33)),
            self._safe_float(probs.get('BEAR', 0.33)),
            self._safe_float(probs.get('CONSOLIDATION', 0.34)),
        ]

        strategy_vector = []
        for strategy_name in self.rl_allocator.strategy_names:
            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_name == strategy_name
            ).first()
            if perf:
                strategy_vector.extend(
                    [
                        self._safe_float(perf.live_win_rate, 50.0) / 100.0,
                        self._safe_float(perf.live_avg_return, 0.0),
                        self._safe_float(perf.live_sharpe, 0.0),
                        -abs(self._safe_float(perf.live_max_drawdown, 0.0)) / 100.0,
                    ]
                )
            else:
                strategy_vector.extend([0.5, 0.0, 0.0, 0.0])

        account = self.alpaca.get_account()
        portfolio_metrics = [0.0] * 5
        if account.get('success'):
            acc = account.get('account', {})
            equity = self._safe_float(acc.get('equity'))
            last_equity = self._safe_float(acc.get('last_equity'))
            cash = self._safe_float(acc.get('cash'))
            pnl = equity - last_equity
            drawdown = self.risk_manager.get_risk_report().get('drawdown_pct', 0.0)
            position_count = self.db.query(LivePosition).filter(LivePosition.is_open == True).count()
            cash_ratio = (cash / equity) if equity > 0 else 0.0
            portfolio_metrics = [pnl, drawdown, cash_ratio, float(position_count), 0.0]

        market_features = [0.0] * 10

        state_vector = regime_vector + strategy_vector + portfolio_metrics + market_features
        return {
            'state_vector': state_vector,
            'strategy_returns': {name: 0.0 for name in self.rl_allocator.strategy_names},
        }

    def _maybe_retrain_rl(self):
        retrain_interval = timedelta(days=7)
        if self.last_rl_retrain_at and (datetime.utcnow() - self.last_rl_retrain_at) < retrain_interval:
            return

        executions = self.db.query(TradeExecution).order_by(TradeExecution.created_at.desc()).limit(500).all()
        if len(executions) < 50:
            return

        rows = []
        for ex in executions:
            state = self._build_rl_state()
            strategy_name = ex.strategy_name or ex.strategy_id or 'unknown'
            state['date'] = ex.created_at.date().isoformat() if ex.created_at else datetime.utcnow().date().isoformat()
            state['strategy_returns'] = {
                name: (0.001 if str(strategy_name).lower() == str(name).lower() else 0.0)
                for name in self.rl_allocator.strategy_names
            }
            rows.append(state)

        result = self.rl_allocator.retrain_on_last_two_years(rows, episodes=200, eval_freq=50)
        if result.get('success'):
            self.last_rl_retrain_at = datetime.utcnow()

    def run_daily_cycle(self) -> Dict[str, Any]:
        """
        Main execution loop - runs once per day.
        """
        print("\n" + "=" * 70)
        print("AUTONOMOUS TRADING CYCLE STARTED")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        results = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "signals_generated": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "errors": [],
        }

        try:
            if not self.auto_trading_enabled:
                results['error'] = "Auto-trading is DISABLED. Set AUTO_TRADING_ENABLED=true in .env"
                print(f"WARNING: {results['error']}")
                return results

            breaker_check = self._check_circuit_breakers()
            if not breaker_check['can_trade']:
                results['error'] = f"Circuit breaker triggered: {breaker_check['reason']}"
                self._log_risk_event("can_trade_block", results['error'], severity="critical")
                print(results['error'])
                return results

            print("\nSyncing positions from Alpaca...")
            self._sync_positions()
            self._refresh_risk_state()

            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                results['error'] = f"Risk manager blocked trading: {reason}"
                self._log_risk_event("can_trade_block", results['error'], severity="critical")
                print(results['error'])
                return results

            if self.regime_detector and self.use_regime_detection:
                print("\nDetecting market regime (HMM)...")
                self.current_regime = self._detect_market_regime()
                if self.current_regime:
                    regime_label = self.current_regime.get('label', 'UNKNOWN')
                    regime_confidence = self.current_regime.get('confidence', 0) * 100
                    results['market_regime'] = regime_label
                    results['regime_confidence'] = regime_confidence
                    print(f"   Market Regime: {regime_label} ({regime_confidence:.1f}% confidence)")

            self._maybe_retrain_rl()

            print("\nScanning market for trading signals...")
            signals = self._generate_signals()
            results['signals_generated'] = len(signals)
            print(f"   Found {len(signals)} potential signals")

            if self.ml_predictor and self.use_ml_validation:
                print("\nValidating signals with XGBoost ML...")
                signals = self._validate_signals_with_ml(signals)
                results['ml_validated_signals'] = len(signals)
                print(f"   {len(signals)} signals passed ML validation")

            print("\nEvaluating signals...")
            actionable_signals = self._evaluate_signals(signals)
            print(f"   {len(actionable_signals)} signals passed evaluation")

            print("\nExecuting trades...")
            for signal in actionable_signals:
                execution_result = self._execute_signal(signal)
                if execution_result['success']:
                    results['trades_executed'] += 1
                else:
                    results['trades_rejected'] += 1
                    results['errors'].append(execution_result.get('error'))

            print("\nUpdating performance metrics...")
            self._update_performance()

            self._update_system_state(results)

            if REPORT_AVAILABLE:
                try:
                    report_payload = generate_daily_report()
                    results['daily_report_summary'] = report_payload.get('summary')
                except Exception as exc:
                    results['errors'].append(f"daily_report_failed: {str(exc)}")

            results['success'] = True
            print("\nCYCLE COMPLETE")
            print(f"   Signals: {results['signals_generated']}")
            print(f"   Executed: {results['trades_executed']}")
            print(f"   Rejected: {results['trades_rejected']}")

        except Exception as e:
            results['error'] = str(e)
            results['errors'].append(str(e))
            print(f"\nERROR: {str(e)}")

        finally:
            self.db.close()

        return results

    def _check_circuit_breakers(self) -> Dict[str, Any]:
        if not self.alpaca.is_market_open():
            return {
                "can_trade": False,
                "reason": "Market is closed",
            }

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
                        "reason": f"Daily loss limit exceeded: {daily_pnl_pct:.2f}%",
                    }

        return {"can_trade": True, "reason": None}

    def _sync_positions(self):
        positions = self.alpaca.get_positions()

        if not positions['success']:
            print(f"   Failed to sync positions: {positions['error']}")
            return

        self.db.query(LivePosition).filter(LivePosition.is_open == True).update(
            {
                "is_open": False,
                "exit_reason": "sync_closed",
            }
        )

        for pos in positions.get('positions', []):
            symbol = pos['symbol']
            alpaca_id = f"{symbol}_{pos['asset_id']}"

            db_pos = self.db.query(LivePosition).filter(
                LivePosition.alpaca_position_id == alpaca_id
            ).first()

            if db_pos:
                db_pos.qty = float(pos['qty'])
                db_pos.current_price = float(pos['current_price'])
                db_pos.unrealized_pl = float(pos['unrealized_pl'])
                db_pos.unrealized_plpc = float(pos['unrealized_plpc'])
                db_pos.is_open = True
                db_pos.updated_at = datetime.utcnow()
            else:
                db_pos = LivePosition(
                    ticker=symbol,
                    qty=float(pos['qty']),
                    entry_price=float(pos['avg_entry_price']),
                    current_price=float(pos['current_price']),
                    unrealized_pl=float(pos['unrealized_pl']),
                    unrealized_plpc=float(pos['unrealized_plpc']),
                    alpaca_position_id=alpaca_id,
                    alpaca_data=pos,
                    is_open=True,
                )
                self.db.add(db_pos)

        self.db.commit()
        print(f"   Synced {len(positions.get('positions', []))} positions")

    def _detect_market_regime(self, reference_ticker: str = "SPY") -> Optional[Dict]:
        if not self.regime_detector:
            return None

        try:
            result = self.regime_detector.train(reference_ticker, period="2y")

            if not result.get('success'):
                print(f"   Regime detection failed: {result.get('error')}")
                return None

            current_regime = result.get('current_regime', {})
            probabilities = current_regime.get('probabilities', {})

            return {
                'label': current_regime.get('label', 'UNKNOWN'),
                'confidence': current_regime.get('confidence', 0),
                'state': current_regime.get('state', -1),
                'probabilities': probabilities,
                'characteristics': result.get('regime_characteristics', {}),
            }

        except Exception as e:
            print(f"   Regime detection error: {str(e)}")
            return None

    def _validate_signals_with_ml(self, signals: List[Dict]) -> List[Dict]:
        if not self.ml_predictor or not signals:
            return signals

        validated_signals = []

        for signal in signals:
            ticker = signal.get('ticker')
            signal_type = signal.get('signal')

            try:
                model_path = self.ml_predictor.model_dir / f"{ticker}_model.pkl"

                if not model_path.exists():
                    print(f"      Training ML model for {ticker}...")
                    train_result = self.ml_predictor.train_model(ticker, period="1y")
                    if not train_result.get('success'):
                        validated_signals.append(signal)
                        continue

                prediction = self.ml_predictor.predict(ticker)

                if not prediction.get('success'):
                    validated_signals.append(signal)
                    continue

                ml_direction = prediction.get('prediction')
                confidence_obj = prediction.get('confidence', {})
                if isinstance(confidence_obj, dict):
                    ml_confidence = self._safe_float(confidence_obj.get('confidence_score'), 0.0)
                else:
                    ml_confidence = self._safe_float(confidence_obj, 0.0)

                signal_agrees = False

                if signal_type == 'BUY' and ml_direction == 'UP':
                    signal_agrees = True
                elif signal_type == 'SELL' and ml_direction == 'DOWN':
                    signal_agrees = True

                signal['ml_prediction'] = ml_direction
                signal['ml_confidence'] = ml_confidence
                signal['ml_agrees'] = signal_agrees

                if signal_agrees and ml_confidence >= self.ml_min_confidence:
                    original_score = signal.get('quality_score', 50)
                    ml_bonus = (ml_confidence - 0.5) * 40
                    signal['quality_score'] = min(original_score + ml_bonus, 100)
                    signal['ml_validated'] = True

                    print(f"      {ticker}: {signal_type} agrees with ML ({ml_direction}, {ml_confidence:.1%})")
                    validated_signals.append(signal)
                else:
                    print(f"      {ticker}: {signal_type} rejected by ML ({ml_direction}, {ml_confidence:.1%})")

            except Exception as e:
                print(f"      ML validation error for {ticker}: {str(e)[:50]}")
                validated_signals.append(signal)

        return validated_signals

    def _get_regime_strategy_boost(self, strategy_type: str) -> float:
        if not self.current_regime:
            return 1.0

        regime = self.current_regime.get('label', 'UNKNOWN')
        strategy_type = strategy_type.lower()

        alignments = {
            'BULL': {
                'momentum': 1.5,
                'trend_following': 1.4,
                'breakout': 1.3,
                'mean_reversion': 0.7,
            },
            'BEAR': {
                'momentum': 0.6,
                'trend_following': 0.7,
                'breakout': 0.5,
                'mean_reversion': 1.3,
            },
            'CONSOLIDATION': {
                'momentum': 0.8,
                'trend_following': 0.7,
                'breakout': 0.6,
                'mean_reversion': 1.4,
            },
        }

        regime_boosts = alignments.get(regime, {})
        return regime_boosts.get(strategy_type, 1.0)

    def _generate_signals(self) -> List[Dict]:
        strategies = self.db.query(Strategy).filter(Strategy.is_active == True).all()

        if not strategies:
            print("   No active strategies found")
            return []

        strategy_configs = []
        for strat in strategies:
            strategy_configs.append(
                {
                    'id': strat.id,
                    'name': strat.name,
                    'strategy_type': strat.strategy_type,
                    'indicators': strat.indicators,
                    'risk_management': {
                        'stop_loss_pct': strat.stop_loss_pct,
                        'take_profit_pct': strat.take_profit_pct,
                        'position_size_pct': strat.position_size_pct,
                    },
                }
            )

        scan_results = MarketScanner.scan_market(
            strategies=strategy_configs,
            universe=None,
            max_workers=10,
            min_confidence=self.min_signal_confidence,
        )

        all_signals = scan_results.get('all_signals', [])
        return [s for s in all_signals if s['signal'] in ['BUY', 'SELL']]

    def _evaluate_signals(self, signals: List[Dict]) -> List[Dict]:
        actionable = []

        rl_state = self._build_rl_state()
        rl_allocations = self.rl_allocator.get_allocation(rl_state)
        self.last_rl_allocations = rl_allocations
        self._log_rl_allocations(rl_allocations, rl_state)

        for signal in signals:
            if signal.get('confidence') != 'HIGH' and self.min_signal_confidence == 'HIGH':
                continue

            existing_pos = self.db.query(LivePosition).filter(
                LivePosition.ticker == signal['ticker'],
                LivePosition.is_open == True,
            ).first()

            if existing_pos:
                continue

            open_positions_count = self.db.query(LivePosition).filter(
                LivePosition.is_open == True
            ).count()

            if open_positions_count >= self.max_portfolio_positions:
                print(f"   Max positions reached ({self.max_portfolio_positions})")
                break

            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == signal.get('strategy_id')
            ).first()

            if perf and perf.is_deprecated:
                continue

            if perf and perf.allocation_weight:
                signal['adjusted_position_size_pct'] = signal.get('position_size_pct', 25) * perf.allocation_weight
            else:
                signal['adjusted_position_size_pct'] = signal.get('position_size_pct', 25)

            strategy_type = signal.get('strategy_type', 'unknown')
            regime_multiplier = self._get_regime_strategy_boost(strategy_type)
            signal['adjusted_position_size_pct'] *= regime_multiplier
            signal['regime_multiplier'] = regime_multiplier

            rl_weight = self._safe_float(rl_allocations.get(strategy_type), 0.0)
            signal['rl_allocation_weight'] = rl_weight

            if rl_weight <= 0:
                continue

            signal['quality_score'] = self._safe_float(signal.get('quality_score', 0.0), 0.0) * rl_weight
            signal['adjusted_position_size_pct'] *= rl_weight

            signal['adjusted_position_size_pct'] = min(
                signal['adjusted_position_size_pct'],
                self.max_position_size_pct,
            )

            actionable.append(signal)

        actionable.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        return actionable

    def _execute_signal(self, signal: Dict) -> Dict[str, Any]:
        try:
            account = self.alpaca.get_account()
            if not account['success']:
                return {"success": False, "error": "Failed to get account"}

            equity = float(account['account']['equity'])
            signal_strength = min(1.0, max(0.0, self._safe_float(signal.get('quality_score', 50.0), 50.0) / 100.0))
            position_size_usd = self.risk_manager.calculate_position_size(
                ticker=signal['ticker'],
                signal_strength=signal_strength,
                account_value=equity,
            )

            price = self._safe_float(signal.get('current_price'), 0.0)
            if price <= 0 and signal['signal'] == 'BUY':
                return {"success": False, "error": "Invalid signal price"}

            if signal['signal'] == 'BUY' and position_size_usd <= 0:
                return {"success": False, "error": "Risk manager reduced position size to 0"}

            qty_for_risk = (position_size_usd / price) if price > 0 else 0.0
            is_valid, reason = self.risk_manager.validate_order(
                ticker=signal['ticker'],
                side='buy' if signal['signal'] == 'BUY' else 'sell',
                qty=qty_for_risk if signal['signal'] == 'BUY' else 1,
                price=price if price > 0 else 1.0,
            )
            if not is_valid:
                self._log_risk_event(
                    event_type="order_rejected",
                    message=reason,
                    severity="warning",
                    ticker=signal.get('ticker'),
                    context={"signal": signal},
                )
                return {"success": False, "error": reason}

            if signal['signal'] == 'BUY':
                order = self.alpaca.place_order(
                    symbol=signal['ticker'],
                    notional=position_size_usd,
                    side='buy',
                    order_type='market',
                    time_in_force='day',
                    take_profit=signal.get('take_profit'),
                    stop_loss=signal.get('stop_loss'),
                )
            else:
                pos = self.alpaca.get_position(signal['ticker'])
                if not pos['success'] or not pos['position']:
                    return {"success": False, "error": "No position to sell"}

                order = self.alpaca.close_position(signal['ticker'])

            if not order['success']:
                return {"success": False, "error": order['error']}

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
                    'adjusted_position_size_pct': signal.get('adjusted_position_size_pct'),
                    'rl_allocation_weight': signal.get('rl_allocation_weight'),
                },
            )
            self.db.add(execution)
            self.db.commit()

            self.risk_manager.update_state(
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'ticker': signal['ticker'],
                    'pnl': 0.0,
                }
            )

            print(
                f"   {signal['signal']} {signal['ticker']} @ ${signal.get('current_price')} "
                f"(${position_size_usd:,.0f})"
            )

            return {"success": True, "order": order['order']}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_performance(self):
        strategies = self.db.query(Strategy).filter(Strategy.is_active == True).all()

        for strategy in strategies:
            executions = self.db.query(TradeExecution).filter(
                TradeExecution.strategy_id == strategy.id,
                TradeExecution.order_status == 'filled',
            ).all()

            if len(executions) < 10:
                continue

            perf = self.db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == strategy.id
            ).first()

            if not perf:
                perf = StrategyPerformance(
                    strategy_id=strategy.id,
                    strategy_name=strategy.name,
                    backtest_sharpe=0.0,
                    backtest_win_rate=0.0,
                    backtest_avg_return=0.0,
                    backtest_max_drawdown=0.0,
                )
                self.db.add(perf)

            perf.live_total_trades = len(executions)
            perf.updated_at = datetime.utcnow()

        self.db.commit()

    def _update_system_state(self, results: Dict):
        state = self.db.query(AutoTradingState).first()

        if not state:
            state = AutoTradingState(
                is_enabled=self.auto_trading_enabled,
                portfolio_value=0.0,
                cash_balance=0.0,
                buying_power=0.0,
            )
            self.db.add(state)

        state.is_enabled = self.auto_trading_enabled
        state.last_run_at = datetime.utcnow()
        state.daily_trades = results.get('trades_executed', 0)
        state.total_signals_generated += results.get('signals_generated', 0)
        state.total_trades_executed += results.get('trades_executed', 0)
        state.total_trades_rejected += results.get('trades_rejected', 0)
        state.recent_activity = {
            'last_rl_allocations': self.last_rl_allocations,
            'risk_report': self.risk_manager.get_risk_report(),
        }

        account = self.alpaca.get_account()
        if account['success']:
            acc = account['account']
            state.portfolio_value = float(acc['equity'])
            state.cash_balance = float(acc['cash'])
            state.buying_power = float(acc['buying_power'])

        self.db.commit()


if __name__ == "__main__":
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
