"""
Autonomous Trading Engine - Makes trading decisions and executes without human input

This is the brain of the autonomous system that:
1. Scans market for signals
2. Uses ML models (XGBoost) to validate signals
3. Detects market regime (HMM) to adjust strategy selection
4. Evaluates signals based on strategy performance
5. Calculates position sizes
6. Executes trades via Alpaca
7. Monitors positions
8. Learns from results
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from alpaca_client import AlpacaClient
from market_scanner import MarketScanner
from live_signal_generator import LiveSignalGenerator
from backtesting_engine import TechnicalIndicators
from database import (
    SessionLocal, Strategy, StrategyPerformance, LivePosition,
    TradeExecution, AutoTradingState
)

# ML Models for intelligent trading
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

    def __init__(self, max_correlation: float = 0.7):
        self.alpaca = AlpacaClient()
        self.db = SessionLocal()

        # Configuration from environment
        self.auto_trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
        self.max_position_size_pct = float(os.getenv('MAX_POSITION_SIZE_PCT', 20))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', 5))
        self.max_portfolio_positions = int(os.getenv('MAX_PORTFOLIO_POSITIONS', 10))
        self.min_signal_confidence = os.getenv('MIN_SIGNAL_CONFIDENCE', 'HIGH')
        self.max_correlation = float(os.getenv('MAX_CORRELATION', max_correlation))
        self._returns_cache: Dict[str, pd.Series] = {}

        # ML Configuration
        self.use_ml_validation = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
        self.use_regime_detection = os.getenv('USE_REGIME_DETECTION', 'true').lower() == 'true'
        self.ml_min_confidence = float(os.getenv('ML_MIN_CONFIDENCE', 0.6))  # 60% confidence minimum

        # Initialize ML models
        self.ml_predictor = None
        self.regime_detector = None
        self.current_regime = None

        if ML_AVAILABLE and self.use_ml_validation:
            self.ml_predictor = MLPricePredictor()
            print("   ✅ XGBoost ML Predictor loaded")

        if HMM_AVAILABLE and self.use_regime_detection:
            self.regime_detector = HMMRegimeDetector()
            print("   ✅ HMM Regime Detector loaded")

        print(f"🤖 Autonomous Trading Engine initialized")
        print(f"   Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"   Max position size: {self.max_position_size_pct}%")
        print(f"   Max correlation: {self.max_correlation}")
        print(f"   Max daily loss: {self.max_daily_loss_pct}%")
        print(f"   ML Validation: {'ENABLED' if self.use_ml_validation and ML_AVAILABLE else 'DISABLED'}")
        print(f"   Regime Detection: {'ENABLED' if self.use_regime_detection and HMM_AVAILABLE else 'DISABLED'}")

    def _get_recent_returns(self, ticker: str, lookback_days: int = 60) -> Optional[pd.Series]:
        """Fetch recent daily returns for correlation checks (cached per run)."""
        if ticker in self._returns_cache:
            return self._returns_cache[ticker]

        try:
            history = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False
            )
            if history is None or history.empty or 'Close' not in history.columns:
                return None

            returns = history['Close'].pct_change().dropna().tail(lookback_days)
            if returns.empty:
                return None

            self._returns_cache[ticker] = returns
            return returns
        except Exception:
            return None

    def _evaluate_correlation_risk(self, candidate_ticker: str, lookback_days: int = 60) -> Dict[str, Any]:
        """Evaluate return correlation between candidate and current open positions."""
        open_positions = self.db.query(LivePosition).filter(
            LivePosition.is_open == True
        ).all()

        open_tickers = [p.ticker for p in open_positions if p.ticker != candidate_ticker]
        if not open_tickers:
            return {'can_trade': True, 'size_multiplier': 1.0}

        candidate_returns = self._get_recent_returns(candidate_ticker, lookback_days=lookback_days)
        if candidate_returns is None:
            return {'can_trade': True, 'size_multiplier': 1.0}

        max_corr = -1.0
        max_corr_ticker = None

        for open_ticker in open_tickers:
            open_returns = self._get_recent_returns(open_ticker, lookback_days=lookback_days)
            if open_returns is None:
                continue

            aligned = pd.concat([candidate_returns, open_returns], axis=1, join='inner').dropna()
            if len(aligned) < 20:
                continue

            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if corr is None or np.isnan(corr):
                continue

            if corr > max_corr:
                max_corr = corr
                max_corr_ticker = open_ticker

        if max_corr <= self.max_correlation:
            return {'can_trade': True, 'size_multiplier': 1.0}

        # Risk response: skip for very high correlation, reduce size for moderate excess.
        if max_corr >= 0.9:
            return {
                'can_trade': False,
                'size_multiplier': 0.0,
                'max_correlation': max_corr,
                'against_ticker': max_corr_ticker
            }

        return {
            'can_trade': True,
            'size_multiplier': 0.5,
            'max_correlation': max_corr,
            'against_ticker': max_corr_ticker
        }

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

            # Step 3.5: Detect market regime (NEW - ML Integration)
            if self.regime_detector and self.use_regime_detection:
                print("\n🎯 Detecting market regime (HMM)...")
                self.current_regime = self._detect_market_regime()
                if self.current_regime:
                    regime_label = self.current_regime.get('label', 'UNKNOWN')
                    regime_confidence = self.current_regime.get('confidence', 0) * 100
                    results['market_regime'] = regime_label
                    results['regime_confidence'] = regime_confidence
                    print(f"   Market Regime: {regime_label} ({regime_confidence:.1f}% confidence)")

                    # Regime-based strategy recommendation
                    if regime_label == 'BULL':
                        print("   📈 Favoring: Momentum, Trend-Following strategies")
                    elif regime_label == 'BEAR':
                        print("   📉 Favoring: Mean-Reversion, Defensive strategies")
                    else:  # CONSOLIDATION
                        print("   📊 Favoring: Mean-Reversion, Range-bound strategies")

            # Step 4: Generate signals
            print("\n🔍 Scanning market for trading signals...")
            signals = self._generate_signals()
            results['signals_generated'] = len(signals)
            print(f"   Found {len(signals)} potential signals")

            # Step 4.5: Validate signals with ML (NEW - XGBoost Integration)
            if self.ml_predictor and self.use_ml_validation:
                print("\n🤖 Validating signals with XGBoost ML...")
                signals = self._validate_signals_with_ml(signals)
                results['ml_validated_signals'] = len(signals)
                print(f"   {len(signals)} signals passed ML validation")

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

    def _detect_market_regime(self, reference_ticker: str = "SPY") -> Optional[Dict]:
        """
        Detect current market regime using HMM on SPY (market proxy)

        Returns:
            Dict with regime label, confidence, and characteristics
        """
        if not self.regime_detector:
            return None

        try:
            # Train/update HMM model on recent SPY data
            result = self.regime_detector.train(reference_ticker, period="2y")

            if not result.get('success'):
                print(f"   ⚠️  Regime detection failed: {result.get('error')}")
                return None

            # Get current regime prediction
            current_regime = result.get('current_regime', {})

            return {
                'label': current_regime.get('label', 'UNKNOWN'),
                'confidence': current_regime.get('confidence', 0),
                'state': current_regime.get('state', -1),
                'probabilities': result.get('regime_probabilities', {}),
                'characteristics': result.get('regime_characteristics', {})
            }

        except Exception as e:
            print(f"   ⚠️  Regime detection error: {str(e)}")
            return None

    def _validate_signals_with_ml(self, signals: List[Dict]) -> List[Dict]:
        """
        Validate trading signals using XGBoost ML predictions

        Only keeps signals where:
        - BUY signal AND ML predicts UP with sufficient confidence
        - SELL signal AND ML predicts DOWN with sufficient confidence

        Returns:
            Filtered list of ML-validated signals
        """
        if not self.ml_predictor or not signals:
            return signals

        validated_signals = []

        for signal in signals:
            ticker = signal.get('ticker')
            signal_type = signal.get('signal')  # BUY or SELL

            try:
                # Get ML prediction for this ticker
                # First check if we have a trained model, if not train one
                model_path = self.ml_predictor.model_dir / f"{ticker}_model.pkl"

                if not model_path.exists():
                    # Train model on the fly (quick training)
                    print(f"      Training ML model for {ticker}...")
                    train_result = self.ml_predictor.train(ticker, period="1y")
                    if not train_result.get('success'):
                        # Can't train, skip ML validation for this signal
                        validated_signals.append(signal)
                        continue

                # Get prediction
                prediction = self.ml_predictor.predict(ticker)

                if not prediction.get('success'):
                    # Prediction failed, include signal anyway
                    validated_signals.append(signal)
                    continue

                ml_direction = prediction.get('prediction')  # 'UP' or 'DOWN'
                ml_confidence = prediction.get('confidence', 0)

                # Validate signal against ML prediction
                signal_agrees = False

                if signal_type == 'BUY' and ml_direction == 'UP':
                    signal_agrees = True
                elif signal_type == 'SELL' and ml_direction == 'DOWN':
                    signal_agrees = True

                # Add ML data to signal
                signal['ml_prediction'] = ml_direction
                signal['ml_confidence'] = ml_confidence
                signal['ml_agrees'] = signal_agrees

                # Only include if ML agrees with sufficient confidence
                if signal_agrees and ml_confidence >= self.ml_min_confidence:
                    # Boost quality score based on ML confidence
                    original_score = signal.get('quality_score', 50)
                    ml_bonus = (ml_confidence - 0.5) * 40  # Up to 20 point bonus
                    signal['quality_score'] = min(original_score + ml_bonus, 100)
                    signal['ml_validated'] = True

                    print(f"      ✅ {ticker}: {signal_type} agrees with ML ({ml_direction}, {ml_confidence:.1%})")
                    validated_signals.append(signal)
                else:
                    print(f"      ❌ {ticker}: {signal_type} rejected by ML ({ml_direction}, {ml_confidence:.1%})")

            except Exception as e:
                # On error, include signal without ML validation
                print(f"      ⚠️  ML validation error for {ticker}: {str(e)[:50]}")
                validated_signals.append(signal)

        return validated_signals

    def _get_regime_strategy_boost(self, strategy_type: str) -> float:
        """
        Get position size multiplier based on regime-strategy alignment

        Returns:
            Multiplier (0.5 to 1.5) for position sizing
        """
        if not self.current_regime:
            return 1.0

        regime = self.current_regime.get('label', 'UNKNOWN')
        strategy_type = strategy_type.lower()

        # Define regime-strategy alignment
        alignments = {
            'BULL': {
                'momentum': 1.5,      # Momentum excels in bull markets
                'trend_following': 1.4,
                'breakout': 1.3,
                'mean_reversion': 0.7,  # Less effective
            },
            'BEAR': {
                'momentum': 0.6,      # Risky in bear markets
                'trend_following': 0.7,
                'breakout': 0.5,
                'mean_reversion': 1.3,  # Better in volatile markets
            },
            'CONSOLIDATION': {
                'momentum': 0.8,
                'trend_following': 0.7,
                'breakout': 0.6,      # False breakouts common
                'mean_reversion': 1.4,  # Range trading works well
            }
        }

        regime_boosts = alignments.get(regime, {})
        return regime_boosts.get(strategy_type, 1.0)

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
        scan_results = MarketScanner.multi_timeframe_scan(
            strategies=strategy_configs,
            universe=None,  # Use default universe
            max_workers=10,
            min_confidence=self.min_signal_confidence,
            require_alignment=True
        )

        all_signals = scan_results.get('all_signals', [])

        # Filter for BUY/SELL only (no HOLD)
        return [s for s in all_signals if s['signal'] in ['BUY', 'SELL']]

    def _calculate_live_atr_stops(
        self,
        ticker: str,
        signal_type: str,
        multiplier: float = 2.0
    ) -> Dict[str, Optional[float]]:
        """
        Calculate live ATR-based stop/take-profit levels.

        Longs:
            stop_loss = entry - ATR_14 * multiplier
            take_profit = entry + ATR_14 * multiplier * 2
        """
        if signal_type != "BUY":
            return {"atr_14": None, "entry_price": None, "stop_loss": None, "take_profit": None}

        try:
            with LiveSignalGenerator._yf_lock:
                data = yf.download(
                    ticker,
                    period="3mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )

            if data is None or data.empty:
                return {"atr_14": None, "entry_price": None, "stop_loss": None, "take_profit": None}

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not {"High", "Low", "Close"}.issubset(set(data.columns)):
                return {"atr_14": None, "entry_price": None, "stop_loss": None, "take_profit": None}

            atr_series = TechnicalIndicators.atr(data["High"], data["Low"], data["Close"], period=14)
            atr_value = atr_series.iloc[-1]
            entry_price = float(data["Close"].iloc[-1])

            if pd.isna(atr_value) or atr_value <= 0:
                return {"atr_14": None, "entry_price": entry_price, "stop_loss": None, "take_profit": None}

            atr_value = float(atr_value)
            risk_distance = atr_value * float(multiplier)

            return {
                "atr_14": atr_value,
                "entry_price": entry_price,
                "stop_loss": max(0.01, entry_price - risk_distance),
                "take_profit": entry_price + (risk_distance * 2.0)
            }
        except Exception:
            return {"atr_14": None, "entry_price": None, "stop_loss": None, "take_profit": None}

    def _evaluate_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Evaluate signals and decide which to execute

        Factors:
        - Signal confidence
        - ML validation (XGBoost prediction agreement)
        - Market regime alignment (HMM)
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

            # NEW: Apply regime-based position sizing adjustment
            strategy_type = signal.get('strategy_type', 'unknown')
            regime_multiplier = self._get_regime_strategy_boost(strategy_type)
            signal['adjusted_position_size_pct'] *= regime_multiplier
            signal['regime_multiplier'] = regime_multiplier

            if regime_multiplier != 1.0:
                regime_label = self.current_regime.get('label', 'UNKNOWN') if self.current_regime else 'UNKNOWN'
                print(f"      📊 {signal['ticker']}: {strategy_type} in {regime_label} → {regime_multiplier:.1f}x sizing")

            # Cap at max position size
            signal['adjusted_position_size_pct'] = min(
                signal['adjusted_position_size_pct'],
                self.max_position_size_pct
            )

            # Correlation risk check against existing open positions (60-day returns)
            corr_check = self._evaluate_correlation_risk(signal['ticker'], lookback_days=60)
            if not corr_check.get('can_trade', True):
                print(
                    f"      ⚠️  {signal['ticker']}: skipped due to high correlation "
                    f"({corr_check.get('max_correlation', 0):.2f}) with "
                    f"{corr_check.get('against_ticker', 'existing position')}"
                )
                continue

            corr_multiplier = corr_check.get('size_multiplier', 1.0)
            if corr_multiplier < 1.0:
                signal['adjusted_position_size_pct'] *= corr_multiplier
                signal['correlation_multiplier'] = corr_multiplier
                signal['max_observed_correlation'] = corr_check.get('max_correlation')
                signal['correlated_with'] = corr_check.get('against_ticker')
                print(
                    f"      📉 {signal['ticker']}: reducing size {corr_multiplier:.2f}x "
                    f"due to correlation {corr_check.get('max_correlation', 0):.2f} with "
                    f"{corr_check.get('against_ticker', 'existing position')}"
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

            atr_multiplier = float(signal.get('atr_stop_multiplier', 2.0))
            atr_stop_data = self._calculate_live_atr_stops(
                ticker=signal['ticker'],
                signal_type=signal['signal'],
                multiplier=atr_multiplier
            )
            dynamic_stop_loss = atr_stop_data.get('stop_loss') or signal.get('stop_loss')
            dynamic_take_profit = atr_stop_data.get('take_profit') or signal.get('take_profit')

            # Place order
            if signal['signal'] == 'BUY':
                order = self.alpaca.place_order(
                    symbol=signal['ticker'],
                    notional=position_size_usd,
                    side='buy',
                    order_type='market',
                    time_in_force='day',
                    take_profit=dynamic_take_profit,
                    stop_loss=dynamic_stop_loss
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
                    'adjusted_position_size_pct': signal.get('adjusted_position_size_pct'),
                    'atr_14': atr_stop_data.get('atr_14'),
                    'atr_stop_multiplier': atr_multiplier,
                    'atr_entry_price': atr_stop_data.get('entry_price'),
                    'dynamic_stop_loss': dynamic_stop_loss,
                    'dynamic_take_profit': dynamic_take_profit,
                    'higher_timeframe_multiplier': signal.get('higher_timeframe_multiplier'),
                    'higher_timeframe_alignment': signal.get('higher_timeframe_alignment')
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
