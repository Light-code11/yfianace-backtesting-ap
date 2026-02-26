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

from alpaca_client import AlpacaClient, is_crypto_ticker, to_yfinance_ticker
from market_scanner import MarketScanner
from live_signal_generator import LiveSignalGenerator
from backtesting_engine import TechnicalIndicators
from database import (
    SessionLocal, Strategy, StrategyPerformance, LivePosition,
    TradeExecution, AutoTradingState, BacktestResult
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

try:
    from auto_retrain import run_weekly_auto_retraining
    AUTO_RETRAIN_AVAILABLE = True
except ImportError:
    AUTO_RETRAIN_AVAILABLE = False
    run_weekly_auto_retraining = None

# Insider signal amplifier (EDGAR Form 4)
try:
    import insider_amplifier
    INSIDER_AMPLIFIER_AVAILABLE = True
except ImportError:
    INSIDER_AMPLIFIER_AVAILABLE = False
    print("Warning: insider_amplifier not available — insider boosting disabled")

# Asymmetric exit rules (ATR-based: initial/trailing/breakeven/time stops)
try:
    from asymmetric_exits import compute_initial_stop, evaluate_exit, get_exit_levels, describe_stops
    ASYMMETRIC_EXITS_AVAILABLE = True
except ImportError:
    ASYMMETRIC_EXITS_AVAILABLE = False
    print("Warning: asymmetric_exits not available — using fixed stops")

# Conviction-based position sizer (Kelly-inspired)
try:
    from conviction_sizer import (
        compute_conviction, size_position, format_conviction_notes,
        apply_volatility_scaling,
    )
    CONVICTION_SIZER_AVAILABLE = True
except ImportError:
    CONVICTION_SIZER_AVAILABLE = False
    apply_volatility_scaling = None
    print("Warning: conviction_sizer not available — using fixed sizing")

# Regime filter master switch (VIX + yield curve + SPY trend + breadth)
try:
    from regime_filter import (
        get_regime as _get_regime,
        apply_regime_to_position_size,
        apply_regime_to_stop,
        should_allow_direction,
    )
    REGIME_FILTER_AVAILABLE = True
except ImportError:
    REGIME_FILTER_AVAILABLE = False
    _get_regime = None
    apply_regime_to_position_size = None
    apply_regime_to_stop = None
    should_allow_direction = None
    print("Warning: regime_filter not available — no macro regime gating")

# Earnings Catalyst strategy (PEAP + EDGAR insider + PEAD logic)
try:
    from earnings_catalyst import (
        generate_earnings_signals as _gen_earnings_signals,
        check_post_earnings_pead as _check_pead,
        get_earnings_signal_count_for_portfolio as _earnings_pos_count,
        earnings_regime_boost as _earnings_regime_boost,
        scan_earnings_calendar as _scan_earnings_calendar,
    )
    EARNINGS_CATALYST_AVAILABLE = True
except ImportError:
    EARNINGS_CATALYST_AVAILABLE = False
    _scan_earnings_calendar = None
    print("Warning: earnings_catalyst not available — earnings signals disabled")

# Crypto regime filter (BTC SMA + Fear & Greed Index)
try:
    from crypto_regime_filter import (
        get_crypto_regime as _get_crypto_regime,
        crypto_regime_conviction_pts as _crypto_conviction_pts,
        should_allow_crypto_direction as _should_allow_crypto_direction,
        print_crypto_regime as _print_crypto_regime,
    )
    CRYPTO_REGIME_AVAILABLE = True
except ImportError:
    CRYPTO_REGIME_AVAILABLE = False
    _get_crypto_regime = None
    print("Warning: crypto_regime_filter not available — crypto regime gating disabled")

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
        self._last_scan_results: Dict[str, Any] = {}

        # Configuration from environment
        self.auto_trading_enabled = os.getenv('AUTO_TRADING_ENABLED', 'false').lower() == 'true'
        self.max_position_size_pct = float(os.getenv('MAX_POSITION_SIZE_PCT', 30))
        self.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', 5))
        self.max_portfolio_positions = int(os.getenv('MAX_PORTFOLIO_POSITIONS', 10))
        self.min_signal_confidence = os.getenv('MIN_SIGNAL_CONFIDENCE', 'HIGH')
        self.min_conviction_score = float(os.getenv('MIN_CONVICTION_SCORE', 55))
        self.max_correlation = float(os.getenv('MAX_CORRELATION', max_correlation))
        self._returns_cache: Dict[str, pd.Series] = {}
        self._kelly_cache: Dict[str, Dict[str, float]] = {}
        self._earnings_days_cache: Dict[str, Optional[int]] = {}

        # ML Configuration
        self.use_ml_validation = os.getenv('USE_ML_VALIDATION', 'true').lower() == 'true'
        self.use_regime_detection = os.getenv('USE_REGIME_DETECTION', 'true').lower() == 'true'
        self.ml_min_confidence = float(os.getenv('ML_MIN_CONFIDENCE', 0.6))  # 60% confidence minimum

        # Initialize ML models
        self.ml_predictor = None
        self.regime_detector = None
        self.current_regime = None    # HMM regime (legacy, may be None)
        self.market_regime  = None    # Macro regime from regime_filter (5-tier)
        self.crypto_regime  = None    # Crypto regime from crypto_regime_filter

        if ML_AVAILABLE and self.use_ml_validation:
            self.ml_predictor = MLPricePredictor()
            print("   ✅ Ensemble ML Predictor loaded (XGBoost + RF + MLP)")

        if HMM_AVAILABLE and self.use_regime_detection:
            self.regime_detector = HMMRegimeDetector()
            print("   ✅ HMM Regime Detector loaded")

        if REGIME_FILTER_AVAILABLE:
            print("   ✅ Regime Filter loaded (VIX + yield curve + SPY + breadth)")

        if CRYPTO_REGIME_AVAILABLE:
            print("   ✅ Crypto Regime Filter loaded (BTC SMA + Fear & Greed Index)")

        print(f"🤖 Autonomous Trading Engine initialized")
        print(f"   Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        print(f"   Max position size: {self.max_position_size_pct}%")
        print(f"   Max correlation: {self.max_correlation}")
        print(f"   Max daily loss: {self.max_daily_loss_pct}%")
        print(f"   ML Validation: {'ENABLED' if self.use_ml_validation and ML_AVAILABLE else 'DISABLED'}")
        print(f"   Regime Detection: {'ENABLED' if self.use_regime_detection and HMM_AVAILABLE else 'DISABLED'}")
        print(f"   Regime Filter:    {'ENABLED' if REGIME_FILTER_AVAILABLE else 'DISABLED'}")

    def _get_recent_returns(self, ticker: str, lookback_days: int = 60) -> Optional[pd.Series]:
        """Fetch recent daily returns for correlation checks (cached per run)."""
        if ticker in self._returns_cache:
            return self._returns_cache[ticker]

        try:
            # Convert crypto tickers for yfinance (BTC/USD → BTC-USD)
            yf_ticker = to_yfinance_ticker(ticker)
            # Use yf_utils if available (retry + caching); fall back to raw yf
            try:
                from yf_utils import yf_history
                history = yf_history(yf_ticker, period="6mo", interval="1d", auto_adjust=True)
            except ImportError:
                history = yf.Ticker(yf_ticker).history(period="6mo", interval="1d", auto_adjust=True)

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
            "stop_checks_triggered": 0,
            "errors": [],
            "executed_tickers": [],
            "rejected_tickers": []
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

            # Step 3.2: Software stop-loss backup check
            stop_results = self.check_stops()
            results["stop_checks_triggered"] = stop_results.get("triggered", 0)

            # Step 3.5: Detect market regime (HMM — legacy ML)
            if self.regime_detector and self.use_regime_detection:
                print("\n🎯 Detecting market regime (HMM)...")
                self.current_regime = self._detect_market_regime()
                if self.current_regime:
                    regime_label = self.current_regime.get('label', 'UNKNOWN')
                    regime_confidence = self.current_regime.get('confidence', 0) * 100
                    print(f"   HMM Regime: {regime_label} ({regime_confidence:.1f}% confidence)")

                    # Regime-based strategy recommendation
                    if regime_label == 'BULL':
                        print("   📈 Favoring: Momentum, Trend-Following strategies")
                    elif regime_label == 'BEAR':
                        print("   📉 Favoring: Mean-Reversion, Defensive strategies")
                    else:  # CONSOLIDATION
                        print("   📊 Favoring: Mean-Reversion, Range-bound strategies")

            # Step 3.6: Macro regime filter (VIX + yield curve + SPY + breadth)
            if REGIME_FILTER_AVAILABLE and _get_regime is not None:
                try:
                    self.market_regime = _get_regime()
                    _r = self.market_regime
                    results['market_regime']        = _r['label']
                    results['regime_exposure_mult']  = _r['exposure_mult']
                    results['regime_stop_mult']      = _r['stop_mult']
                    results['regime_direction_bias'] = _r['direction_bias']
                    results['regime_vix']            = _r.get('vix')
                    results['regime_yield_spread']   = _r.get('yield_spread')
                except Exception as _re:
                    print(f"   ⚠️  Regime filter error (non-fatal): {_re}")
                    self.market_regime = None
            else:
                self.market_regime = None

            # Step 3.7: Crypto regime filter (BTC SMA + Fear & Greed)
            if CRYPTO_REGIME_AVAILABLE and _get_crypto_regime is not None:
                try:
                    self.crypto_regime = _get_crypto_regime()
                    _cr = self.crypto_regime
                    results['crypto_regime']       = _cr['label']
                    results['crypto_fg_score']     = _cr.get('fg_score')
                    results['crypto_exposure_mult'] = _cr.get('exposure_mult')
                    print(f"\n₿ Crypto regime: {_cr['label']} "
                          f"(F&G={_cr.get('fg_score')}, "
                          f"GoldenCross={_cr.get('golden_cross')}, "
                          f"exposure={_cr.get('exposure_mult', 1.0):.0%})")
                except Exception as _cre:
                    print(f"   ⚠️  Crypto regime error (non-fatal): {_cre}")
                    self.crypto_regime = None
            else:
                self.crypto_regime = None

            # Step 4: Generate signals
            print("\n🔍 Scanning market for trading signals...")
            signals = self._generate_signals()
            results['signals_generated'] = len(signals)
            results['stocks_scanned'] = self._last_scan_results.get('stocks_scanned', 0)
            results['strategies_used'] = self._last_scan_results.get('strategies_used', 0)
            print(f"   Found {len(signals)} potential signals")

            # Step 4.5: Validate signals with ML (NEW - XGBoost Integration)
            if self.ml_predictor and self.use_ml_validation:
                print("\n🤖 Validating signals with Ensemble ML...")
                signals = self._validate_signals_with_ml(signals)
                results['ml_validated_signals'] = len(signals)
                print(f"   {len(signals)} signals passed ML validation")

            # Step 5: Evaluate and filter signals
            print("\n🧠 Evaluating signals...")
            actionable_signals = self._evaluate_signals(signals)
            results['actionable_signals'] = len(actionable_signals)
            print(f"   {len(actionable_signals)} signals passed evaluation")

            # Step 5.5: Apply insider signal amplifier (EDGAR Form 4)
            if INSIDER_AMPLIFIER_AVAILABLE:
                print("\n📋 Applying insider signal amplifier (EDGAR Form 4)…")
                actionable_signals, insider_swing_signals = self._apply_insider_amplifier(actionable_signals)
                results['insider_amplified'] = len(actionable_signals)
                results['insider_swing_signals'] = len(insider_swing_signals)
                if insider_swing_signals:
                    print(f"   ⭐ {len(insider_swing_signals)} cluster-buy swing trade(s) created")
                # Append swing signals to the execution queue
                actionable_signals = actionable_signals + insider_swing_signals

            # Step 6: Execute trades
            print("\n💰 Executing trades...")
            for signal in actionable_signals:
                execution_result = self._execute_signal(signal)
                if execution_result['success']:
                    results['trades_executed'] += 1
                    results['executed_tickers'].append(signal.get('ticker'))
                else:
                    results['trades_rejected'] += 1
                    results['rejected_tickers'].append(signal.get('ticker'))
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

    def _check_circuit_breakers(self, crypto_only: bool = False) -> Dict[str, Any]:
        """
        Check if trading should be halted

        Reasons to halt:
        - Daily loss limit exceeded
        - Market is closed (skipped for crypto-only trades since they're 24/7)
        - System errors
        """
        # Check market hours (crypto trades 24/7, so skip this check for crypto)
        if not crypto_only and not self.alpaca.is_market_open():
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
        Validate trading signals using ensemble ML predictions

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

                # Get ensemble prediction
                prediction = self.ml_predictor.ensemble_predict(ticker)

                if not prediction.get('success'):
                    # Prediction failed, include signal anyway
                    validated_signals.append(signal)
                    continue

                ml_direction = prediction.get('prediction')  # 'UP' or 'DOWN'
                ml_confidence_raw = prediction.get('confidence', 0)
                if isinstance(ml_confidence_raw, dict):
                    ml_confidence = float(ml_confidence_raw.get('confidence_score', 0))
                else:
                    ml_confidence = float(ml_confidence_raw or 0)

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

    def _generate_pair_signals(self) -> List[Dict]:
        """
        Generate pair trading signals for all configured pairs.

        Each pair produces up to 2 signals (one per leg: BUY leg_A + SELL leg_B or vice-versa).
        Uses PairTradingStrategy z-score logic from pair_trading_strategy.py.
        """
        try:
            from trading_config import PAIRS, PAIR_TRADING_PARAMS
            from pair_trading_strategy import PairTradingStrategy, PairTradingStatistics
        except ImportError as e:
            print(f"   ⚠️  Pair trading imports failed: {e}")
            return []

        params = PAIR_TRADING_PARAMS
        pair_signals: List[Dict] = []

        for ticker_a, ticker_b in PAIRS:
            try:
                # Download 6 months for robust cointegration testing
                # Convert crypto tickers for yfinance (BTC/USD → BTC-USD)
                yf_a = to_yfinance_ticker(ticker_a)
                yf_b = to_yfinance_ticker(ticker_b)
                raw_a = yf.Ticker(yf_a).history(period="6mo", auto_adjust=True)
                raw_b = yf.Ticker(yf_b).history(period="6mo", auto_adjust=True)

                if raw_a.empty or raw_b.empty:
                    continue

                # Flatten MultiIndex columns if needed
                if isinstance(raw_a.columns, pd.MultiIndex):
                    raw_a.columns = raw_a.columns.get_level_values(0)
                if isinstance(raw_b.columns, pd.MultiIndex):
                    raw_b.columns = raw_b.columns.get_level_values(0)

                prices_a = raw_a['Close'].squeeze()
                prices_b = raw_b['Close'].squeeze()

                # Align and limit to lookback window
                common_idx = prices_a.index.intersection(prices_b.index)
                lookback = params.get('lookback_days', 60)
                prices_a = prices_a.loc[common_idx].tail(lookback)
                prices_b = prices_b.loc[common_idx].tail(lookback)

                if len(prices_a) < 30:
                    continue

                # Correlation pre-filter
                corr = float(prices_a.corr(prices_b))
                min_corr = params.get('min_correlation', 0.4)
                if abs(corr) < min_corr:
                    print(f"   ⏭️  Pair {ticker_a}/{ticker_b}: correlation={corr:.2f} < {min_corr} — skipped (note: cointegration may still hold)")
                    # Don't skip pre-configured pairs — they were validated via cointegration
                    # Correlation can temporarily diverge; cointegration is the stronger test
                    pass  # continue through to z-score check

                # Cointegration test
                stats = PairTradingStatistics()
                coint_result = stats.engle_granger_test(prices_a, prices_b)

                # Skip only clearly non-cointegrated pairs
                p_val_limit = params.get('cointegration_pvalue', 0.05) * 3
                if not coint_result.is_cointegrated and coint_result.p_value > p_val_limit:
                    continue

                # Get current z-score and signal
                strategy = PairTradingStrategy(
                    entry_threshold=params.get('zscore_entry', 2.0),
                    exit_threshold=params.get('zscore_exit', 0.5),
                    stop_loss_threshold=params.get('zscore_stop', 3.5),
                )
                current = strategy.get_current_signal(prices_a, prices_b)
                pair_sig = current.get('signal', 'HOLD')
                zscore = float(current.get('zscore', 0.0))

                if pair_sig not in ('LONG_SPREAD', 'SHORT_SPREAD'):
                    continue

                # Translate spread signal to individual leg signals
                # LONG_SPREAD  → BUY ticker_a, SELL ticker_b  (spread too low)
                # SHORT_SPREAD → SELL ticker_a, BUY ticker_b  (spread too high)
                leg_a_dir = 'BUY' if pair_sig == 'LONG_SPREAD' else 'SELL'
                leg_b_dir = 'SELL' if pair_sig == 'LONG_SPREAD' else 'BUY'

                price_a = float(prices_a.iloc[-1])
                price_b = float(prices_b.iloc[-1])
                confidence = 'HIGH' if abs(zscore) >= 2.5 else 'MEDIUM'
                quality = round(50 + min(abs(zscore) * 10, 30), 1)

                pair_meta = {
                    'strategy_name': 'cfg_pair_trading',
                    'strategy_type': 'pair_trading',
                    'confidence': confidence,
                    'quality_score': quality,
                    'pair_ticker_a': ticker_a,
                    'pair_ticker_b': ticker_b,
                    'pair_zscore': zscore,
                    'pair_signal': pair_sig,
                    'hedge_ratio': coint_result.hedge_ratio,
                    'correlation': corr,
                    'cointegration_pvalue': coint_result.p_value,
                    'position_size_pct': params.get('max_position_pct', 0.10) * 100,
                    'is_pair_trade': True,
                }

                pair_signals.append({
                    **pair_meta,
                    'ticker': ticker_a,
                    'signal': leg_a_dir,
                    'current_price': price_a,
                    'entry_price': price_a,
                    'pair_leg': 'A',
                    'reasoning': (
                        f"Pair {ticker_a}/{ticker_b}: {pair_sig} "
                        f"(z={zscore:.2f}, corr={corr:.2f}, coint_p={coint_result.p_value:.3f}) "
                        f"→ {leg_a_dir} {ticker_a}"
                    ),
                })
                pair_signals.append({
                    **pair_meta,
                    'ticker': ticker_b,
                    'signal': leg_b_dir,
                    'current_price': price_b,
                    'entry_price': price_b,
                    'pair_leg': 'B',
                    'reasoning': (
                        f"Pair {ticker_a}/{ticker_b}: {pair_sig} "
                        f"(z={zscore:.2f}) → {leg_b_dir} {ticker_b}"
                    ),
                })
                print(
                    f"   📊 Pair {ticker_a}/{ticker_b}: {pair_sig} "
                    f"z={zscore:.2f} corr={corr:.2f} coint_p={coint_result.p_value:.3f}"
                )

            except Exception as exc:
                print(f"   ⚠️  Pair {ticker_a}/{ticker_b} error: {str(exc)[:80]}")
                continue

        return pair_signals

    # ── Crypto Signal Generation ──────────────────────────────────────────────

    def _generate_crypto_pair_signals(self) -> List[Dict]:
        """
        Generate BTC/ETH spread pair signals.

        Uses CRYPTO_PAIRS from trading_config with shorter lookback (30d)
        and wider z-score thresholds tuned for crypto volatility.
        Skips cointegration test — BTC/ETH are structurally co-integrated.
        """
        try:
            from trading_config import CRYPTO_PAIRS, CRYPTO_PAIR_PARAMS
            from pair_trading_strategy import PairTradingStrategy
        except ImportError as exc:
            print(f"   ⚠️  Crypto pair imports failed: {exc}")
            return []

        crypto_pair_signals: List[Dict] = []

        for ticker_a, ticker_b in CRYPTO_PAIRS:
            try:
                yf_a = to_yfinance_ticker(ticker_a)   # BTC/USD → BTC-USD
                yf_b = to_yfinance_ticker(ticker_b)

                raw_a = yf.Ticker(yf_a).history(period="3mo", auto_adjust=True)
                raw_b = yf.Ticker(yf_b).history(period="3mo", auto_adjust=True)

                if raw_a.empty or raw_b.empty:
                    print(f"   ⚠️  Crypto pair {ticker_a}/{ticker_b}: no data")
                    continue

                for raw in (raw_a, raw_b):
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)

                pair_cfg = CRYPTO_PAIR_PARAMS.get(
                    (ticker_a, ticker_b),
                    {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 30}
                )
                lookback = pair_cfg.get("lookback_days", 30)

                prices_a = raw_a["Close"].squeeze()
                prices_b = raw_b["Close"].squeeze()
                common = prices_a.index.intersection(prices_b.index)
                prices_a = prices_a.loc[common].tail(lookback)
                prices_b = prices_b.loc[common].tail(lookback)

                if len(prices_a) < 20:
                    continue

                corr = float(prices_a.corr(prices_b))
                strategy = PairTradingStrategy(
                    entry_threshold=pair_cfg["zscore_entry"],
                    exit_threshold=pair_cfg["zscore_exit"],
                    stop_loss_threshold=pair_cfg["zscore_stop"],
                )
                current = strategy.get_current_signal(prices_a, prices_b)
                pair_sig = current.get("signal", "HOLD")
                zscore   = float(current.get("zscore", 0.0))

                if pair_sig not in ("LONG_SPREAD", "SHORT_SPREAD"):
                    continue

                leg_a_dir = "BUY"  if pair_sig == "LONG_SPREAD" else "SELL"
                leg_b_dir = "SELL" if pair_sig == "LONG_SPREAD" else "BUY"
                price_a = float(prices_a.iloc[-1])
                price_b = float(prices_b.iloc[-1])
                confidence = "HIGH" if abs(zscore) >= 2.0 else "MEDIUM"
                quality    = round(50 + min(abs(zscore) * 10, 30), 1)

                # Apply crypto regime filter for pairs
                if self.crypto_regime:
                    direction_ok = _should_allow_crypto_direction(
                        self.crypto_regime, leg_a_dir, "crypto_pair_trading"
                    ) if CRYPTO_REGIME_AVAILABLE else True
                    exposure = self.crypto_regime.get("exposure_mult", 1.0)
                else:
                    direction_ok = True
                    exposure = 1.0

                pair_meta = {
                    "strategy_name":        "crypto_pair_trading",
                    "strategy_type":        "crypto_pair_trading",
                    "is_pair_trade":        True,
                    "is_crypto":            True,
                    "confidence":           confidence,
                    "quality_score":        quality,
                    "pair_ticker_a":        ticker_a,
                    "pair_ticker_b":        ticker_b,
                    "pair_zscore":          zscore,
                    "pair_signal":          pair_sig,
                    "correlation":          corr,
                    "position_size_pct":    5.0 * exposure,
                    "atr_stop_multiplier":  3.0,
                    "crypto_regime":        self.crypto_regime.get("label") if self.crypto_regime else None,
                }

                crypto_pair_signals.append({**pair_meta, "ticker": ticker_a, "signal": leg_a_dir,
                    "current_price": price_a, "entry_price": price_a, "pair_leg": "A",
                    "reasoning": f"Crypto pair BTC/ETH {pair_sig} z={zscore:.2f} corr={corr:.2f} → {leg_a_dir} {ticker_a}"})
                crypto_pair_signals.append({**pair_meta, "ticker": ticker_b, "signal": leg_b_dir,
                    "current_price": price_b, "entry_price": price_b, "pair_leg": "B",
                    "reasoning": f"Crypto pair BTC/ETH {pair_sig} z={zscore:.2f} → {leg_b_dir} {ticker_b}"})

                print(f"   ₿ Crypto pair BTC/ETH: {pair_sig} z={zscore:.2f} corr={corr:.2f}")

            except Exception as exc:
                print(f"   ⚠️  Crypto pair {ticker_a}/{ticker_b} error: {str(exc)[:80]}")

        return crypto_pair_signals

    def _generate_crypto_signals(self) -> List[Dict]:
        """
        Generate directional signals for CRYPTO_UNIVERSE (BTC/USD, ETH/USD).

        Uses crypto-specific strategies with wider parameters (3× ATR stop,
        RSI thresholds 20/80). Skips EDGAR insider and earnings catalyst.
        Applies crypto regime filter to gate direction and size positions.

        Returns list of signal dicts tagged with is_crypto=True.
        """
        from trading_config import CRYPTO_UNIVERSE, CRYPTO_STRATEGIES, CRYPTO_RISK_PARAMS

        enabled_strategies = [s for s in CRYPTO_STRATEGIES if s.get("enabled", True)]
        if not enabled_strategies:
            return []

        crypto_signals: List[Dict] = []
        max_size_pct = CRYPTO_RISK_PARAMS.get("max_position_size_pct", 10.0)
        atr_mult     = CRYPTO_RISK_PARAMS.get("atr_stop_multiplier", 3.0)

        # Apply crypto regime exposure multiplier
        crypto_exposure = 1.0
        crypto_label    = "UNKNOWN"
        if self.crypto_regime:
            crypto_exposure = self.crypto_regime.get("exposure_mult", 1.0)
            crypto_label    = self.crypto_regime.get("label", "UNKNOWN")

        for ticker in CRYPTO_UNIVERSE:
            yf_ticker = to_yfinance_ticker(ticker)   # BTC/USD → BTC-USD

            try:
                hist = yf.Ticker(yf_ticker).history(period="6mo", interval="1d", auto_adjust=True)
                if hist is None or hist.empty:
                    continue
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if not {"Close", "High", "Low"}.issubset(set(hist.columns)):
                    continue

                close = hist["Close"]
                high  = hist["High"]
                low   = hist["Low"]
                price = float(close.iloc[-1])

                # ── Indicators ────────────────────────────────────────────────
                ema20  = close.ewm(span=20).mean()
                ema50  = close.ewm(span=50).mean()
                ema200 = close.ewm(span=200).mean()

                # ATR (14-day)
                tr     = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr14  = tr.rolling(14).mean().iloc[-1]

                # RSI (14-day) — crypto uses thresholds 20/80 (not 30/70)
                delta  = close.diff()
                gain   = delta.clip(lower=0).rolling(14).mean()
                loss   = (-delta.clip(upper=0)).rolling(14).mean()
                rs     = gain / loss.replace(0, float("inf"))
                rsi14  = (100 - 100 / (1 + rs)).iloc[-1]

                # Bollinger Bands (20-day)
                bb_mid = close.rolling(20).mean()
                bb_std = close.rolling(20).std()
                bb_pct = ((close - (bb_mid - 2 * bb_std)) / (4 * bb_std)).iloc[-1]

                # Current values
                e20 = float(ema20.iloc[-1])
                e50 = float(ema50.iloc[-1])
                e200 = float(ema200.iloc[-1])

                for strat in enabled_strategies:
                    if strat.get("strategy_type") in ("crypto_pair_trading",):
                        continue   # handled separately

                    stype = strat.get("strategy_type", "")
                    signal_dir = None
                    quality    = 50.0
                    reasoning  = ""

                    if stype == "crypto_trend_momentum":
                        # BUY: EMA20 > EMA50 > EMA200 + price above EMA20
                        if (e20 > e50 > e200) and price > e20:
                            signal_dir = "BUY"
                            quality    = 65 + min((price - e20) / e20 * 100, 15)
                            reasoning  = (f"{ticker}: EMA20({e20:.0f})>EMA50({e50:.0f})>EMA200({e200:.0f}), "
                                         f"price={price:.0f} above EMA20")
                        # SELL: EMA20 < EMA50 (death cross) + price below EMA20
                        elif (e20 < e50) and price < e20:
                            signal_dir = "SELL"
                            quality    = 60 + min((e20 - price) / e20 * 100, 15)
                            reasoning  = (f"{ticker}: EMA20({e20:.0f})<EMA50({e50:.0f}), "
                                         f"price={price:.0f} below EMA20 — downtrend")

                    elif stype == "crypto_mean_reversion":
                        # BUY: RSI<20 (extreme oversold) + price < lower BB
                        if rsi14 < 20 and bb_pct < 0.05:
                            signal_dir = "BUY"
                            quality    = 60 + (20 - rsi14)  # more oversold = higher quality
                            reasoning  = (f"{ticker}: RSI(14)={rsi14:.1f}<20 (extreme oversold), "
                                         f"BB%={bb_pct:.2f} (below lower band)")
                        # SELL: RSI>80 (extreme overbought) + price > upper BB
                        elif rsi14 > 80 and bb_pct > 0.95:
                            signal_dir = "SELL"
                            quality    = 60 + (rsi14 - 80)
                            reasoning  = (f"{ticker}: RSI(14)={rsi14:.1f}>80 (extreme overbought), "
                                         f"BB%={bb_pct:.2f} (above upper band)")

                    if signal_dir is None:
                        continue

                    # Apply crypto regime direction gate
                    if self.crypto_regime and CRYPTO_REGIME_AVAILABLE:
                        if not _should_allow_crypto_direction(self.crypto_regime, signal_dir, stype):
                            print(f"   🚫 {ticker}: {signal_dir} blocked by crypto regime ({crypto_label})")
                            continue

                    # Size with regime exposure multiplier
                    position_pct = min(
                        strat["risk_management"]["position_size_pct"] * crypto_exposure,
                        max_size_pct
                    )

                    crypto_signals.append({
                        "ticker":            ticker,
                        "signal":            signal_dir,
                        "confidence":        "HIGH" if quality >= 65 else "MEDIUM",
                        "quality_score":     round(quality, 1),
                        "strategy_name":     strat["name"],
                        "strategy_type":     stype,
                        "is_crypto":         True,
                        "current_price":     price,
                        "entry_price":       price,
                        "position_size_pct": position_pct,
                        "atr_stop_multiplier": atr_mult,
                        "stop_loss_pct":     strat["risk_management"]["stop_loss_pct"],
                        "take_profit_pct":   strat["risk_management"]["take_profit_pct"],
                        "crypto_regime":     crypto_label,
                        "crypto_exposure":   crypto_exposure,
                        "rsi14":             round(float(rsi14), 1),
                        "atr14":             round(float(atr14), 2),
                        "reasoning":         reasoning,
                    })
                    print(f"   ₿ {ticker}: {signal_dir} [{stype}] Q={quality:.0f} RSI={rsi14:.1f} regime={crypto_label}")

            except Exception as exc:
                print(f"   ⚠️  Crypto signal error for {ticker}: {str(exc)[:80]}")

        # Add crypto pair signals
        crypto_pair_sigs = self._generate_crypto_pair_signals()
        crypto_signals.extend(crypto_pair_sigs)

        return crypto_signals

    @staticmethod
    def _tag_confluence(signals: List[Dict]) -> List[Dict]:
        """
        Tag each signal with confluence_count: the number of DIFFERENT strategy
        names that generated a signal for the same ticker + direction.

        This fixes the bug where conviction_sizer always sees confluence=0
        because the market scanner deduplicates signals per ticker.

        Signals with 2+ strategies agreeing get confluence_count ≥ 1.
        Signals with 3+ strategies agreeing are flagged as HIGH_CONFLUENCE
        and are eligible for 15–20% position sizing.

        Mutates signals in-place, returns the same list.
        """
        from collections import defaultdict

        # Build index: (ticker, direction) → set of strategy names
        strategy_votes: dict = defaultdict(set)
        for s in signals:
            key = (s.get("ticker", ""), s.get("signal", ""))
            strat = s.get("strategy_name", "")
            if strat:
                strategy_votes[key].add(strat)

        for s in signals:
            key = (s.get("ticker", ""), s.get("signal", ""))
            all_strategies = strategy_votes[key]
            current_strat  = s.get("strategy_name", "")
            confirming = all_strategies - {current_strat}
            n = len(confirming)

            s["confluence_count"]        = n
            s["confirming_strategies"]   = list(confirming)
            s["high_confluence"]         = n >= 2   # 3+ total (current + 2 confirming)

            if n >= 1:
                print(
                    f"   🔀 Confluence: {s['ticker']} {s['signal']} — "
                    f"{n+1} strategies agree: {list(all_strategies)}"
                )

        return signals

    def _generate_signals(self) -> List[Dict]:
        """Generate trading signals from all active strategies"""
        # Get active strategies
        strategies = self.db.query(Strategy).filter(
            Strategy.is_active == True
        ).all()

        if not strategies:
            print("   ⚠️  No active strategies found")
            return []

        # Prepare strategy configs — EXCLUDE crypto tickers from equity universe
        from trading_config import CRYPTO_UNIVERSE as _CRYPTO_UNIV
        _crypto_set = set(_CRYPTO_UNIV)

        strategy_configs = []
        universe = set()
        for strat in strategies:
            tickers = strat.tickers if isinstance(strat.tickers, list) else []
            # Filter out crypto tickers from equity scanner universe
            equity_tickers = [t for t in tickers if isinstance(t, str) and t and t not in _crypto_set]
            universe.update(equity_tickers)
            strategy_configs.append({
                'id': strat.id,
                'name': strat.name,
                'strategy_type': strat.strategy_type,
                'indicators': strat.indicators,
                'atr_stop_multiplier': (
                    (strat.entry_conditions or {}).get('atr_stop_multiplier', 2.0)
                    if isinstance(strat.entry_conditions, dict) else 2.0
                ),
                'risk_management': {
                    'stop_loss_pct': strat.stop_loss_pct,
                    'take_profit_pct': strat.take_profit_pct,
                    'position_size_pct': strat.position_size_pct
                }
            })

        # Pre-warm sector ETF / SPY cache to avoid download contention during parallel scan
        try:
            print("   📡 Pre-warming sector ETF cache (SPY + sector ETFs)...")
            LiveSignalGenerator.prewarm_sector_cache(period="3mo")
            print("   ✅ Sector cache ready")
        except Exception as exc:
            print(f"   ⚠️  Sector cache pre-warm failed (non-fatal): {str(exc)[:80]}")

        # Run market scanner (equity only — crypto handled separately)
        scan_results = MarketScanner.multi_timeframe_scan(
            strategies=strategy_configs,
            universe=sorted(universe) if universe else None,
            max_workers=10,
            min_confidence=self.min_signal_confidence,
            require_alignment=True
        )
        self._last_scan_results = scan_results

        all_signals = scan_results.get('all_signals', [])

        # ── Equity pair trading signals ──────────────────────────────────────
        try:
            print("\n   🔗 Generating equity pair trading signals...")
            pair_sigs = self._generate_pair_signals()
            if pair_sigs:
                all_signals.extend(pair_sigs)
                print(f"   Added {len(pair_sigs)} pair trading signals ({len(pair_sigs)//2} pairs)")
            else:
                print("   No equity pair trading signals at current z-score thresholds")
        except Exception as exc:
            print(f"   ⚠️  Equity pair signal generation failed: {str(exc)[:120]}")

        # ── Earnings catalyst signals (equity only — not crypto) ─────────────
        if EARNINGS_CATALYST_AVAILABLE:
            try:
                print("\n   📅 Scanning earnings calendar for catalyst signals...")
                from database import LivePosition as _LP
                open_earnings_pos = (
                    self.db.query(_LP)
                    .filter(_LP.is_open == True)
                    .all()
                )
                open_pos_list = [
                    {"strategy_type": p.strategy_name or ""}
                    for p in open_earnings_pos
                ]
                n_earnings = _earnings_pos_count(open_pos_list)

                portfolio_val = 100_000.0
                try:
                    acct = self.alpaca.get_account()
                    portfolio_val = float(acct.get("portfolio_value", 100_000))
                except Exception:
                    pass

                # Pass TICKER_UNIVERSE (equity only — no crypto tickers)
                from trading_config import TICKER_UNIVERSE as _UNIV
                earn_sigs = _gen_earnings_signals(
                    _UNIV,   # TICKER_UNIVERSE no longer contains BTC/USD or ETH/USD
                    portfolio_value=portfolio_val,
                    current_earnings_positions=n_earnings,
                    dry_run=False,
                    quiet=False,
                )
                if earn_sigs:
                    all_signals.extend(earn_sigs)
                    print(f"   📅 Added {len(earn_sigs)} earnings catalyst signal(s)")
                else:
                    print("   📅 No earnings catalyst signals in current window")
            except Exception as exc:
                print(f"   ⚠️  Earnings catalyst scan failed (non-fatal): {str(exc)[:120]}")

        # ── Confluence detection across ALL equity signals ────────────────────
        # Tag each equity signal with how many other strategies agree.
        # This feeds conviction_sizer which awards up to 20pts for confluence.
        equity_signals = [s for s in all_signals if s.get("signal") in ("BUY", "SELL")]
        if equity_signals:
            print(f"\n   🔀 Running confluence detection across {len(equity_signals)} equity signals...")
            equity_signals = self._tag_confluence(equity_signals)
            # Boost quality score for confirmed confluence
            n_confluence = sum(1 for s in equity_signals if s.get("confluence_count", 0) >= 1)
            n_high       = sum(1 for s in equity_signals if s.get("high_confluence", False))
            if n_confluence:
                print(f"   🔀 Confluence: {n_confluence} signals have 2+ strategies agreeing "
                      f"({n_high} HIGH_CONFLUENCE with 3+)")

        # ── Crypto signals (BTC/USD, ETH/USD — separate pipeline) ────────────
        # No EDGAR insider, no earnings catalyst, no equity regime filter.
        # Uses CRYPTO_STRATEGIES with wider ATR/RSI parameters.
        crypto_sigs: List[Dict] = []
        try:
            print("\n   ₿ Generating crypto signals (BTC/USD, ETH/USD)...")
            crypto_sigs = self._generate_crypto_signals()
            if crypto_sigs:
                directional = [s for s in crypto_sigs if not s.get("is_pair_trade")]
                pair_legs   = [s for s in crypto_sigs if s.get("is_pair_trade")]
                print(f"   ₿ {len(directional)} directional + {len(pair_legs)} pair-leg crypto signals")
            else:
                print("   ₿ No crypto signals at current thresholds")
        except Exception as exc:
            print(f"   ⚠️  Crypto signal generation failed (non-fatal): {str(exc)[:120]}")

        # Combine: equity signals first (sorted by quality), then crypto
        # Crypto signals are tagged is_crypto=True so downstream code can
        # skip EDGAR, earnings, and equity-specific regime filters.
        combined = equity_signals + crypto_sigs
        return [s for s in combined if s.get("signal") in ("BUY", "SELL")]

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
            # Convert crypto tickers for yfinance (BTC/USD → BTC-USD)
            yf_ticker = to_yfinance_ticker(ticker)
            data = yf.Ticker(yf_ticker).history(period="3mo", interval="1d", auto_adjust=True)

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

    def _get_signal_conviction_score(self, signal: Dict[str, Any]) -> float:
        """Return conviction score, falling back to quality score if conviction is missing."""
        raw_conviction = signal.get("conviction_score")
        if raw_conviction is not None:
            try:
                return float(raw_conviction)
            except Exception:
                pass
        return float(signal.get("quality_score", 0) or 0)

    def _calculate_kelly_edge(self, ticker: str) -> Dict[str, float]:
        """
        Compute Kelly fraction from the latest 20 daily returns.

        Kelly formula:
            f* = p - (1 - p) / b
        where p is win rate, b is avg_win / avg_loss.
        """
        if ticker in self._kelly_cache:
            return self._kelly_cache[ticker]

        default_result = {
            "kelly_fraction": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "sample_size": 0,
        }

        try:
            yf_ticker = to_yfinance_ticker(ticker)
            history = yf.Ticker(yf_ticker).history(period="3mo", interval="1d", auto_adjust=True)
            if history is None or history.empty or "Close" not in history.columns:
                self._kelly_cache[ticker] = default_result
                return default_result

            closes = history["Close"].dropna()
            returns = closes.pct_change().dropna().tail(20)
            if returns.empty:
                self._kelly_cache[ticker] = default_result
                return default_result

            wins = returns[returns > 0]
            losses = returns[returns < 0]
            p = float(len(wins) / len(returns))
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(abs(losses.mean())) if len(losses) else 0.0

            if avg_loss <= 0:
                kelly = p if avg_win > 0 else 0.0
            else:
                b = avg_win / avg_loss if avg_win > 0 else 0.0
                kelly = p - ((1 - p) / b) if b > 0 else -1.0

            result = {
                "kelly_fraction": float(kelly),
                "win_rate": p,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "sample_size": int(len(returns)),
            }
            self._kelly_cache[ticker] = result
            return result
        except Exception:
            self._kelly_cache[ticker] = default_result
            return default_result

    def _get_earnings_days_away(self, ticker: str) -> Optional[int]:
        """Return days until next earnings for ticker (None if unavailable/not relevant)."""
        if ticker in self._earnings_days_cache:
            return self._earnings_days_cache[ticker]

        days_away: Optional[int] = None
        try:
            if (
                EARNINGS_CATALYST_AVAILABLE
                and _scan_earnings_calendar is not None
                and not is_crypto_ticker(ticker)
            ):
                events = _scan_earnings_calendar([ticker], lookahead_days=14, quiet=True)
                if events:
                    ev = next((e for e in events if e.get("ticker") == ticker), events[0])
                    raw_days = ev.get("days_until")
                    if raw_days is not None:
                        days_away = int(raw_days)
        except Exception:
            days_away = None

        self._earnings_days_cache[ticker] = days_away
        return days_away

    def _check_trend_confirmation(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Confirm trend with SMA50/RSI rules:
          BUY: price > SMA50 OR RSI < 30
          SELL: price < SMA50 OR RSI > 70
        """
        default = {"ok": True, "sma50": None, "rsi14": None, "price": None}
        ticker = signal.get("ticker")
        direction = signal.get("signal")
        if not ticker or is_crypto_ticker(ticker):
            return default

        try:
            yf_ticker = to_yfinance_ticker(ticker)
            history = yf.Ticker(yf_ticker).history(period="6mo", interval="1d", auto_adjust=True)
            if history is None or history.empty or "Close" not in history.columns:
                return default

            closes = history["Close"].dropna()
            if len(closes) < 50:
                return default

            sma50 = float(closes.tail(50).mean())
            price = float(closes.iloc[-1])
            rsi_series = TechnicalIndicators.rsi(closes, period=14)
            rsi14 = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else None

            if direction == "BUY":
                ok = (price > sma50) or (rsi14 is not None and rsi14 < 30)
            else:
                ok = (price < sma50) or (rsi14 is not None and rsi14 > 70)

            return {"ok": bool(ok), "sma50": sma50, "rsi14": rsi14, "price": price}
        except Exception:
            return default

    def _entry_gate_checks(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Run all required entry-gate checks before order placement."""
        ticker = signal.get("ticker", "")
        strategy_type = signal.get("strategy_type") or signal.get("strategy_name") or ""
        conviction_score = self._get_signal_conviction_score(signal)
        position_size_pct = float(signal.get("adjusted_position_size_pct", signal.get("position_size_pct", 0)) or 0)
        confluence_count = int(signal.get("confluence_count", 0) or 0)

        kelly = self._calculate_kelly_edge(ticker)
        kelly_fraction = float(kelly.get("kelly_fraction", 0.0))

        if kelly_fraction < 0:
            return {
                "allowed": False,
                "reason": "negative Kelly edge",
                "kelly": kelly,
                "earnings_days_away": self._get_earnings_days_away(ticker),
                "trend": {"ok": None, "sma50": None, "rsi14": None, "price": None},
                "conviction_score": conviction_score,
                "confluence_count": confluence_count,
                "position_size_pct": position_size_pct,
            }

        earnings_days_away = self._get_earnings_days_away(ticker)
        if earnings_days_away is not None and earnings_days_away <= 3:
            if strategy_type != "earnings_catalyst":
                return {
                    "allowed": False,
                    "reason": "earnings too close",
                    "kelly": kelly,
                    "earnings_days_away": earnings_days_away,
                    "trend": {"ok": None, "sma50": None, "rsi14": None, "price": None},
                    "conviction_score": conviction_score,
                    "confluence_count": confluence_count,
                    "position_size_pct": position_size_pct,
                }
            if conviction_score < 60:
                return {
                    "allowed": False,
                    "reason": f"earnings_catalyst conviction too low ({conviction_score:.1f} < 60)",
                    "kelly": kelly,
                    "earnings_days_away": earnings_days_away,
                    "trend": {"ok": None, "sma50": None, "rsi14": None, "price": None},
                    "conviction_score": conviction_score,
                    "confluence_count": confluence_count,
                    "position_size_pct": position_size_pct,
                }

        trend = self._check_trend_confirmation(signal)
        if trend.get("ok") is False:
            return {
                "allowed": False,
                "reason": "no trend confirmation",
                "kelly": kelly,
                "earnings_days_away": earnings_days_away,
                "trend": trend,
                "conviction_score": conviction_score,
                "confluence_count": confluence_count,
                "position_size_pct": position_size_pct,
            }

        if conviction_score < self.min_conviction_score:
            return {
                "allowed": False,
                "reason": f"conviction below minimum ({conviction_score:.1f} < {self.min_conviction_score:.1f})",
                "kelly": kelly,
                "earnings_days_away": earnings_days_away,
                "trend": trend,
                "conviction_score": conviction_score,
                "confluence_count": confluence_count,
                "position_size_pct": position_size_pct,
            }

        if position_size_pct > 5.0 and confluence_count < 1:
            return {
                "allowed": False,
                "reason": f"insufficient confluence ({confluence_count}) for >5% position",
                "kelly": kelly,
                "earnings_days_away": earnings_days_away,
                "trend": trend,
                "conviction_score": conviction_score,
                "confluence_count": confluence_count,
                "position_size_pct": position_size_pct,
            }

        if position_size_pct <= 5.0 and confluence_count < 1 and conviction_score < 60:
            return {
                "allowed": False,
                "reason": f"single-strategy position requires conviction >=60 ({conviction_score:.1f})",
                "kelly": kelly,
                "earnings_days_away": earnings_days_away,
                "trend": trend,
                "conviction_score": conviction_score,
                "confluence_count": confluence_count,
                "position_size_pct": position_size_pct,
            }

        return {
            "allowed": True,
            "reason": None,
            "kelly": kelly,
            "earnings_days_away": earnings_days_away,
            "trend": trend,
            "conviction_score": conviction_score,
            "confluence_count": confluence_count,
            "position_size_pct": position_size_pct,
        }

    def _upsert_position_stop_price(self, ticker: str, stop_price: float) -> None:
        """Persist actual stop price on live position record."""
        pos = self.db.query(LivePosition).filter(
            LivePosition.ticker == ticker,
            LivePosition.is_open == True
        ).first()

        if not pos:
            self._sync_positions()
            pos = self.db.query(LivePosition).filter(
                LivePosition.ticker == ticker,
                LivePosition.is_open == True
            ).first()

        if pos:
            pos.stop_loss_price = float(stop_price)
            self.db.commit()

    def _enforce_standalone_stop(self, ticker: str, stop_price: Optional[float]) -> Dict[str, Any]:
        """
        Place a standalone stop-loss order on Alpaca for long positions.
        Fractional shares are rounded down to whole shares for stop placement.
        """
        if stop_price is None:
            return {"success": False, "error": "missing stop price"}

        try:
            pos_resp = self.alpaca.get_position(ticker)
            if not pos_resp.get("success") or not pos_resp.get("position"):
                return {"success": False, "error": "position not found for stop enforcement"}

            pos = pos_resp["position"]
            qty = float(pos.get("qty", 0) or 0)
            whole_qty = int(np.floor(abs(qty)))
            if whole_qty <= 0:
                return {"success": False, "error": "fractional-only position; no whole shares to protect"}

            stop_order = self.alpaca.place_order(
                symbol=ticker,
                qty=whole_qty,
                side="sell",
                order_type="stop",
                time_in_force="gtc",
                stop_price=float(stop_price),
            )
            if stop_order.get("success"):
                self._upsert_position_stop_price(ticker, float(stop_price))
                return {
                    "success": True,
                    "protected_qty": whole_qty,
                    "unprotected_fractional_qty": round(abs(qty) - whole_qty, 8),
                    "stop_order": stop_order.get("order"),
                }
            return {"success": False, "error": stop_order.get("error", "stop order failed")}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

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

            # ── Macro regime direction filter ────────────────────────────────
            # Blocks short trades in bull regimes and long trades in crisis.
            if REGIME_FILTER_AVAILABLE and should_allow_direction and self.market_regime:
                direction = signal.get('signal', 'BUY')
                if not should_allow_direction(self.market_regime, direction):
                    regime_label = self.market_regime.get('label', 'UNKNOWN')
                    bias = self.market_regime.get('direction_bias', '')
                    if not signal.get('is_pair_trade'):
                        print(
                            f"      🚫 {signal['ticker']}: {direction} blocked "
                            f"(regime={regime_label}, bias={bias})"
                        )
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

            # Optional Kelly cap from latest backtest if available.
            strategy_id = signal.get('strategy_id')
            if strategy_id:
                latest_backtest = self.db.query(BacktestResult).filter(
                    BacktestResult.strategy_id == strategy_id
                ).order_by(BacktestResult.created_at.desc()).first()

                if latest_backtest and latest_backtest.kelly_position_pct is not None:
                    kelly_pct = float(latest_backtest.kelly_position_pct)
                    if kelly_pct > 0:
                        signal['kelly_position_pct'] = kelly_pct
                        signal['adjusted_position_size_pct'] = min(signal['adjusted_position_size_pct'], kelly_pct)

            # Apply HMM regime-based position sizing adjustment (strategy-level)
            strategy_type = signal.get('strategy_type', 'unknown')
            regime_multiplier = self._get_regime_strategy_boost(strategy_type)
            signal['adjusted_position_size_pct'] *= regime_multiplier
            signal['regime_multiplier'] = regime_multiplier

            if regime_multiplier != 1.0:
                regime_label = self.current_regime.get('label', 'UNKNOWN') if self.current_regime else 'UNKNOWN'
                print(f"      📊 {signal['ticker']}: {strategy_type} in {regime_label} → {regime_multiplier:.1f}x sizing (HMM)")

            # ── Macro regime exposure multiplier (regime_filter master switch) ──
            # Applied AFTER HMM strategy boost — scales entire position by macro regime
            if REGIME_FILTER_AVAILABLE and apply_regime_to_position_size and self.market_regime:
                pre_regime_size = signal['adjusted_position_size_pct']
                signal['adjusted_position_size_pct'] = apply_regime_to_position_size(
                    pre_regime_size,
                    regime=self.market_regime,
                    max_position_pct=self.max_position_size_pct,
                )
                signal['macro_regime_label']   = self.market_regime.get('label', 'UNKNOWN')
                signal['macro_exposure_mult']  = self.market_regime.get('exposure_mult', 1.0)
                if self.market_regime.get('exposure_mult', 1.0) < 1.0:
                    print(
                        f"      🌍 {signal['ticker']}: macro regime {self.market_regime['label']} "
                        f"→ {self.market_regime['exposure_mult']:.0%} exposure "
                        f"({pre_regime_size:.1f}% → {signal['adjusted_position_size_pct']:.1f}%)"
                    )

            # Cap at max position size
            signal['adjusted_position_size_pct'] = min(
                signal['adjusted_position_size_pct'],
                self.max_position_size_pct
            )

            # Correlation risk check against existing open positions (60-day returns)
            # Skip correlation gating for pair trades — both legs are intentionally correlated.
            if not signal.get('is_pair_trade'):
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

            # ── Crypto signals: apply crypto-specific sizing, skip equity conviction ──
            if signal.get("is_crypto") and not signal.get("is_pair_trade"):
                # Crypto directional signals use their pre-computed position_size_pct
                # (already adjusted by crypto_exposure multiplier in _generate_crypto_signals)
                signal["adjusted_position_size_pct"] = min(
                    float(signal.get("position_size_pct", 5.0)),
                    float(os.getenv("MAX_POSITION_SIZE_PCT", 10.0))
                )

                # HIGH_CONFLUENCE crypto: bump size slightly (up to 10%)
                if signal.get("high_confluence"):
                    boosted = signal["adjusted_position_size_pct"] * 1.25
                    signal["adjusted_position_size_pct"] = min(boosted, 10.0)
                    print(f"      ₿🔀 {signal['ticker']}: HIGH_CONFLUENCE crypto boost → "
                          f"{signal['adjusted_position_size_pct']:.1f}%")
                actionable.append(signal)
                continue

            # ── Pre-populate insider_bias for conviction scoring ───────────────
            # The insider cache is pre-warmed at startup; this is a fast dict
            # lookup (no API call). Without this, conviction scoring sees 0.0
            # for regular equity signals (only earnings_catalyst sets it up-front).
            if (
                INSIDER_AMPLIFIER_AVAILABLE
                and not signal.get('is_crypto')
                and not signal.get('is_pair_trade')
                and signal.get('insider_bias') is None
            ):
                try:
                    _ib = insider_amplifier.get_insider_bias(signal['ticker'])
                    signal['insider_bias'] = round(_ib, 4)
                except Exception:
                    signal['insider_bias'] = 0.0

            # ── Conviction-based position sizing (overrides fixed size — equity only) ─────
            if CONVICTION_SIZER_AVAILABLE and not signal.get('is_pair_trade'):
                try:
                    # Prefer macro regime (5-tier) over HMM regime (3-tier) for
                    # conviction scoring — richer directional info
                    _conviction_regime = self.market_regime if self.market_regime else self.current_regime
                    conviction = compute_conviction(
                        signal=signal,
                        all_signals=signals,
                        regime=_conviction_regime,
                    )
                    conviction_size = size_position(
                        conviction, max_position_pct=self.max_position_size_pct
                    )

                    # Skip low-conviction trades entirely
                    if conviction['tier'] == 'skip':
                        print(
                            f"      ❌ {signal['ticker']}: skipped — "
                            f"conviction={conviction['score']:.0f}/100 (below threshold)"
                        )
                        continue

                    # HIGH_CONFLUENCE (3+ strategies agree): override to 15-20% tier
                    if signal.get("high_confluence") and conviction['score'] >= 60:
                        conviction_size = min(
                            15.0 + (conviction['score'] - 60) * 0.25,  # 15–20% range
                            self.max_position_size_pct
                        )
                        print(
                            f"      🔀 {signal['ticker']}: HIGH_CONFLUENCE boost "
                            f"({len(signal.get('confirming_strategies', []))+1} strategies) → "
                            f"{conviction_size:.1f}%"
                        )

                    # Override position size with conviction-based sizing
                    signal['conviction_score']     = conviction['score']
                    signal['conviction_tier']      = conviction['tier']
                    signal['conviction_breakdown'] = conviction
                    signal['adjusted_position_size_pct'] = conviction_size

                    print(
                        f"      💡 {signal['ticker']}: "
                        f"{format_conviction_notes(conviction)} → {conviction_size:.1f}%"
                    )
                except Exception as _ce:
                    print(f"      [WARN] Conviction sizing error for {signal['ticker']}: {str(_ce)[:60]}")

            # ── Volatility-tier position scaling (equity non-pair only) ───────
            # Apply AFTER conviction sizing, BEFORE execution.
            # Mega-caps (low vol) → 1.0x; SOFI/COIN (high vol) → 0.5x, etc.
            if (
                CONVICTION_SIZER_AVAILABLE
                and apply_volatility_scaling is not None
                and not signal.get('is_pair_trade')
                and not signal.get('is_crypto')
            ):
                try:
                    _pre_vol_size = signal['adjusted_position_size_pct']
                    _vol_result   = apply_volatility_scaling(
                        _pre_vol_size, signal['ticker'], lookback_days=30
                    )
                    signal['adjusted_position_size_pct'] = _vol_result['scaled_size_pct']
                    signal['vol_tier']        = _vol_result['vol_tier']
                    signal['vol_multiplier']  = _vol_result['vol_multiplier']
                    signal['annualized_vol']  = _vol_result['annualized_vol']
                    if _vol_result['vol_multiplier'] != 1.0:
                        print(
                            f"      [VOL]  {signal['ticker']}: "
                            f"{_vol_result['vol_notes']}"
                        )
                except Exception as _ve:
                    print(f"      [WARN] Volatility scaling error for {signal['ticker']}: {str(_ve)[:60]}")

            actionable.append(signal)

        # Sort by quality score (highest first)
        actionable.sort(key=lambda x: x.get('quality_score', 0), reverse=True)

        return actionable

    def _execute_signal(self, signal: Dict) -> Dict[str, Any]:
        """Execute a trading signal via Alpaca"""
        try:
            is_pair = signal.get('is_pair_trade', False)
            is_entry_order = signal.get('signal') == 'BUY' or is_pair
            entry_checks = {
                "allowed": True,
                "kelly": {"kelly_fraction": None},
                "earnings_days_away": None,
                "trend": {"price": None, "sma50": None, "rsi14": None},
            }
            if is_entry_order:
                entry_checks = self._entry_gate_checks(signal)
                if not entry_checks.get("allowed"):
                    reason = entry_checks.get("reason", "entry gate rejection")
                    print(f"   🚫 REJECT {signal.get('ticker')}: {reason}")
                    return {"success": False, "error": reason}

                kelly_fraction = float(entry_checks["kelly"].get("kelly_fraction", 0.0))
                if kelly_fraction < 0.05:
                    old_size = float(signal.get('adjusted_position_size_pct', signal.get('position_size_pct', 0)) or 0)
                    signal['adjusted_position_size_pct'] = old_size * 0.5
                    print(
                        f"      📉 {signal.get('ticker')}: Kelly={kelly_fraction*100:.2f}% < 5% "
                        f"→ reducing size {old_size:.2f}% → {signal['adjusted_position_size_pct']:.2f}%"
                    )

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

            # ── Asymmetric exit rules ────────────────────────────────────────
            # Use ATR-based initial stop (1.5 ATR) with NO fixed take-profit.
            # Trailing / breakeven / time stops are managed by the close bot.
            # The macro regime stop_mult tightens/widens the initial stop ATR mult.
            if ASYMMETRIC_EXITS_AVAILABLE and atr_stop_data.get('atr_14'):
                _atr_val  = float(atr_stop_data['atr_14'])
                _entry_px = float(atr_stop_data.get('entry_price') or signal.get('current_price', 0))

                # Apply regime stop multiplier to initial stop ATR factor
                from asymmetric_exits import INITIAL_STOP_MULTIPLIER as _BASE_STOP_MULT
                if REGIME_FILTER_AVAILABLE and apply_regime_to_stop and self.market_regime:
                    _stop_mult = apply_regime_to_stop(_BASE_STOP_MULT, self.market_regime)
                else:
                    _stop_mult = _BASE_STOP_MULT

                dynamic_stop_loss   = compute_initial_stop(_entry_px, _atr_val, multiplier=_stop_mult)
                dynamic_take_profit = None   # No fixed target — let winners run
                # Log exit levels for transparency
                _levels = get_exit_levels(_entry_px, _atr_val)
                _regime_tag = (
                    f" [regime={self.market_regime['label']}, stopMult={_stop_mult:.2f}x]"
                    if self.market_regime else ""
                )
                print(
                    f"      📐 {signal['ticker']}: "
                    f"InitStop=${dynamic_stop_loss:.2f} "
                    f"TrailTrigger=${_levels['trailing_trigger']:.2f} "
                    f"BreakevenTrigger=${_levels['breakeven_trigger']:.2f} "
                    f"(ATR={_atr_val:.2f}){_regime_tag}"
                )
            else:
                # Fallback: use symmetric ATR-based stops from existing logic
                dynamic_stop_loss   = atr_stop_data.get('stop_loss')  or signal.get('stop_loss')
                dynamic_take_profit = atr_stop_data.get('take_profit') or signal.get('take_profit')

            # ── Place order via smart limit execution ─────────────────────────
            # BUY: limit at bid+$0.01, wait up to 5 min, then market fallback
            # SELL (pair): IOC limit at ask-$0.01 to avoid leg risk
            # SELL (close existing): use close_position (market)
            is_pair  = signal.get('is_pair_trade', False)
            is_crypto = signal.get('is_crypto', False)

            if signal['signal'] == 'BUY':
                order = self.alpaca.place_smart_limit_order(
                    symbol=signal['ticker'],
                    side='buy',
                    notional=position_size_usd,
                    is_pair_trade=is_pair,
                    # Disable bracket orders for now (fractional shares not supported)
                    # take_profit=dynamic_take_profit,
                    # stop_loss=dynamic_stop_loss,
                    # Crypto: no timeout wait needed (24/7, use short timeout)
                    timeout_sec=60 if is_crypto else None,
                )
            else:  # SELL
                if is_pair:
                    # Pair trade short-sell leg — IOC limit to avoid leg risk
                    order = self.alpaca.place_smart_limit_order(
                        symbol=signal['ticker'],
                        side='sell',
                        notional=position_size_usd,
                        is_pair_trade=True,   # forces IOC
                    )
                else:
                    # Normal close of existing long position (use market for reliability)
                    pos = self.alpaca.get_position(signal['ticker'])
                    if not pos['success'] or not pos['position']:
                        return {"success": False, "error": "No position to sell"}
                    order = self.alpaca.close_position(signal['ticker'])

            if not order['success']:
                return {"success": False, "error": order['error']}

            # Log execution — capture actual order type and fill price
            actual_order_type = order.get('order_type', 'market')
            actual_fill_price = order.get('fill_price')

            # Log slippage if available
            slippage = order.get('slippage_pct')
            if slippage is not None:
                direction_str = "▲" if signal['signal'] == 'BUY' else "▼"
                print(f"      📊 Slippage: {slippage:+.3f}% vs {'ask' if signal['signal'] == 'BUY' else 'bid'} {direction_str}")

            stop_enforcement = {"success": False, "error": None}
            if signal['signal'] == 'BUY':
                stop_enforcement = self._enforce_standalone_stop(
                    ticker=signal['ticker'],
                    stop_price=dynamic_stop_loss
                )
                if stop_enforcement.get("success"):
                    print(
                        f"      🛡️ {signal['ticker']}: standalone stop set @ ${float(dynamic_stop_loss):.2f} "
                        f"for {stop_enforcement.get('protected_qty')} share(s)"
                    )
                else:
                    print(
                        f"      ⚠️ {signal['ticker']}: stop enforcement skipped/failed "
                        f"({stop_enforcement.get('error', 'unknown')})"
                    )

            execution = TradeExecution(
                ticker=signal['ticker'],
                strategy_id=signal.get('strategy_id'),
                strategy_name=signal.get('strategy_name'),
                signal_type=signal['signal'],
                signal_confidence=signal.get('confidence'),
                signal_price=signal.get('current_price'),
                order_type=actual_order_type,
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
                    'asymmetric_exits': ASYMMETRIC_EXITS_AVAILABLE and atr_stop_data.get('atr_14') is not None,
                    'higher_timeframe_multiplier': signal.get('higher_timeframe_multiplier'),
                    'higher_timeframe_alignment': signal.get('higher_timeframe_alignment'),
                    # Conviction-based sizing
                    'conviction_score': signal.get('conviction_score'),
                    'conviction_tier': signal.get('conviction_tier'),
                    'conviction_breakdown': signal.get('conviction_breakdown'),
                    'min_conviction_score': self.min_conviction_score,
                    # Entry gate checks
                    'entry_gate_passed': True,
                    'kelly_fraction': entry_checks['kelly'].get('kelly_fraction'),
                    'kelly_win_rate': entry_checks['kelly'].get('win_rate'),
                    'kelly_avg_win': entry_checks['kelly'].get('avg_win'),
                    'kelly_avg_loss': entry_checks['kelly'].get('avg_loss'),
                    'kelly_sample_size': entry_checks['kelly'].get('sample_size'),
                    'earnings_days_away': entry_checks.get('earnings_days_away'),
                    'trend_price': entry_checks['trend'].get('price'),
                    'trend_sma50': entry_checks['trend'].get('sma50'),
                    'trend_rsi14': entry_checks['trend'].get('rsi14'),
                    # Insider amplifier metadata
                    'insider_bias': signal.get('insider_bias'),
                    'insider_stance': signal.get('insider_stance'),
                    'insider_notes': signal.get('insider_notes'),
                    'insider_swing': signal.get('insider_swing', False),
                    'holding_period_days': signal.get('holding_period_days'),
                    # Macro regime filter
                    'macro_regime': signal.get('macro_regime_label'),
                    'macro_exposure_mult': signal.get('macro_exposure_mult'),
                    'regime_stop_mult': self.market_regime.get('stop_mult') if self.market_regime else None,
                    'regime_direction_bias': self.market_regime.get('direction_bias') if self.market_regime else None,
                    'regime_vix': self.market_regime.get('vix') if self.market_regime else None,
                    'regime_yield_spread': self.market_regime.get('yield_spread') if self.market_regime else None,
                    # Limit order tracking
                    'actual_order_type': actual_order_type,
                    'actual_fill_price': actual_fill_price,
                    'slippage_pct': slippage,
                    # Crypto metadata
                    'is_crypto': signal.get('is_crypto', False),
                    'crypto_regime': signal.get('crypto_regime'),
                    'crypto_exposure': signal.get('crypto_exposure'),
                    # Confluence detection
                    'confluence_count': signal.get('confluence_count', 0),
                    'confirming_strategies': signal.get('confirming_strategies', []),
                    'high_confluence': signal.get('high_confluence', False),
                    # Volatility-tier scaling
                    'vol_tier': signal.get('vol_tier'),
                    'vol_multiplier': signal.get('vol_multiplier'),
                    'annualized_vol': signal.get('annualized_vol'),
                    # Stop-loss enforcement metadata
                    'standalone_stop_enforced': stop_enforcement.get('success', False),
                    'standalone_stop_error': stop_enforcement.get('error'),
                    'standalone_stop_protected_qty': stop_enforcement.get('protected_qty'),
                    'standalone_stop_unprotected_fractional_qty': stop_enforcement.get('unprotected_fractional_qty'),
                    'standalone_stop_order': stop_enforcement.get('stop_order'),
                    # Harmonised close-bot fields (atr_at_entry + initial_stop)
                    'atr_at_entry':  atr_stop_data.get('atr_14'),
                    'initial_stop':  dynamic_stop_loss,
                    'entry_price':   atr_stop_data.get('entry_price') or signal.get('current_price'),
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
        state.daily_trades = int(results.get('trades_executed', 0) or 0)
        state.total_signals_generated = int(state.total_signals_generated or 0) + int(results.get('signals_generated', 0) or 0)
        state.total_trades_executed = int(state.total_trades_executed or 0) + int(results.get('trades_executed', 0) or 0)
        state.total_trades_rejected = int(state.total_trades_rejected or 0) + int(results.get('trades_rejected', 0) or 0)

        # Update portfolio values
        account = self.alpaca.get_account()
        if account['success']:
            acc = account['account']
            state.portfolio_value = float(acc['equity'])
            state.cash_balance = float(acc['cash'])
            state.buying_power = float(acc['buying_power'])

        self.db.commit()

    def preview_daily_cycle(self) -> Dict[str, Any]:
        """
        Generate and evaluate signals without placing live orders.
        """
        preview = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "stocks_scanned": 0,
            "signals_generated": 0,
            "actionable_signals": 0,
            "would_trade": [],
            "errors": []
        }

        try:
            # ── Macro regime filter (compute once, cache for daily cycle) ──
            if REGIME_FILTER_AVAILABLE and _get_regime is not None:
                try:
                    self.market_regime = _get_regime()
                    _r = self.market_regime
                    preview['market_regime']        = _r['label']
                    preview['regime_exposure_mult']  = _r['exposure_mult']
                    preview['regime_stop_mult']      = _r['stop_mult']
                    preview['regime_direction_bias'] = _r['direction_bias']
                    preview['regime_vix']            = _r.get('vix')
                    preview['regime_yield_spread']   = _r.get('yield_spread')
                    preview['regime_breadth']        = _r.get('breadth')
                    preview['regime_description']    = _r.get('description')
                    preview['regime_signals']        = _r.get('signals', {})
                except Exception as _re:
                    print(f"   ⚠️  Regime filter error (non-fatal): {_re}")
                    self.market_regime = None
            else:
                self.market_regime = None

            # Crypto regime for preview
            if CRYPTO_REGIME_AVAILABLE and _get_crypto_regime is not None:
                try:
                    self.crypto_regime = _get_crypto_regime()
                    _cr = self.crypto_regime
                    preview['crypto_regime']       = _cr['label']
                    preview['crypto_fg_score']     = _cr.get('fg_score')
                    preview['crypto_fg_label']     = _cr.get('fg_label')
                    preview['crypto_exposure_mult'] = _cr.get('exposure_mult')
                    preview['crypto_golden_cross']  = _cr.get('golden_cross')
                    preview['crypto_btc_price']     = _cr.get('btc_price')
                except Exception as _cre:
                    print(f"   ⚠️  Crypto regime error (non-fatal): {_cre}")
                    self.crypto_regime = None
            else:
                self.crypto_regime = None

            self._sync_positions()
            signals = self._generate_signals()
            preview["stocks_scanned"] = self._last_scan_results.get("stocks_scanned", 0)
            preview["signals_generated"] = len(signals)

            if self.ml_predictor and self.use_ml_validation:
                signals = self._validate_signals_with_ml(signals)

            actionable = self._evaluate_signals(signals)

            # Apply insider amplifier in dry-run too
            if INSIDER_AMPLIFIER_AVAILABLE:
                actionable, swing_signals = self._apply_insider_amplifier(actionable)
                preview["insider_swing_signals"] = len(swing_signals)
                actionable = actionable + swing_signals

            preview["actionable_signals"] = len(actionable)
            preview["would_trade"] = [
                {
                    "ticker": s.get("ticker"),
                    "signal": s.get("signal"),
                    "confidence": s.get("confidence"),
                    "quality_score": s.get("quality_score"),
                    "position_size_pct": round(float(s.get("adjusted_position_size_pct", 0)), 2),
                    "current_price": s.get("current_price"),
                    "strategy_name": s.get("strategy_name"),
                    "insider_bias": s.get("insider_bias"),
                    "insider_notes": s.get("insider_notes"),
                    "insider_swing": s.get("insider_swing", False),
                    # Conviction-based sizing fields
                    "conviction_score": s.get("conviction_score"),
                    "conviction_tier": s.get("conviction_tier"),
                    "conviction_breakdown": s.get("conviction_breakdown"),
                    # Macro regime fields
                    "macro_regime": s.get("macro_regime_label"),
                    "macro_exposure_mult": s.get("macro_exposure_mult"),
                    # Pair trade fields (passthrough)
                    "is_pair_trade": s.get("is_pair_trade", False),
                    "pair_ticker_a": s.get("pair_ticker_a"),
                    "pair_ticker_b": s.get("pair_ticker_b"),
                    "pair_zscore": s.get("pair_zscore"),
                }
                for s in actionable
            ]
            rejection_counts: Dict[str, int] = {}
            for s in actionable:
                if s.get("signal") != "BUY" and not s.get("is_pair_trade", False):
                    continue
                checks = self._entry_gate_checks(s)
                if not checks.get("allowed"):
                    reason = checks.get("reason", "entry gate rejection")
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            preview["entry_gate_rejections"] = rejection_counts
            preview["success"] = True
            return preview
        except Exception as e:
            preview["errors"].append(str(e))
            return preview

    def _apply_insider_amplifier(
        self,
        signals: List[Dict],
    ) -> tuple:
        """
        Apply EDGAR Form 4 insider signal amplifier to each actionable signal.

        Called AFTER _evaluate_signals() and BEFORE _execute_signal().

        For each signal:
          1. Fetch insider bias from in-memory cache (pre-warmed at startup).
          2. Adjust the signal's quality_score using amplify_signal() rules.
          3. Store insider_bias, insider_notes on the signal dict for logging.
          4. If cluster buying detected (score > 0.7) on a BUY signal, create
             a separate "insider swing" trade with extended holding period.

        Returns:
            (amplified_signals, swing_signals)
            amplified_signals — original signals with updated quality_score
            swing_signals     — additional cluster-buy swing trade signals
        """
        if not INSIDER_AMPLIFIER_AVAILABLE:
            return signals, []

        amplified   : List[Dict] = []
        swing_signals: List[Dict] = []

        # Track tickers that already have a swing signal to avoid duplicates
        swing_seen: set = set()

        for signal in signals:
            ticker    = signal.get("ticker", "")
            direction = signal.get("signal", "BUY")

            # Skip EDGAR insider lookup for crypto — no Form 4 filings for crypto assets
            if signal.get("is_crypto"):
                signal["insider_bias"]   = 0.0
                signal["insider_stance"] = "N/A (crypto)"
                signal["insider_notes"]  = "Crypto asset — no EDGAR insider data"
                amplified.append(signal)
                continue

            score     = insider_amplifier.get_insider_bias(ticker)

            # Annotate signal with raw insider bias
            signal["insider_bias"]  = round(score, 4)
            signal["insider_stance"] = insider_amplifier.describe_bias(ticker)

            # Adjust quality_score (0–100 scale)
            base_quality = float(signal.get("quality_score", 50))
            adj_quality, notes = insider_amplifier.amplify_signal(
                ticker          = ticker,
                base_confidence = base_quality,
                base_direction  = direction,
            )
            signal["quality_score"] = adj_quality
            signal["insider_notes"] = notes

            if abs(adj_quality - base_quality) > 0.01:
                direction_arrow = "↑" if adj_quality > base_quality else "↓"
                print(
                    f"      📋 {ticker}: insider {signal['insider_stance']} "
                    f"| quality {base_quality:.1f} → {adj_quality:.1f} {direction_arrow} "
                    f"| {notes[:70]}"
                )

            amplified.append(signal)

            # ── Cluster swing trade ───────────────────────────────────────────
            if (
                direction == "BUY"
                and score >= insider_amplifier.CLUSTER_THRESHOLD
                and ticker not in swing_seen
            ):
                swing = insider_amplifier.build_cluster_swing_signal(signal, score)
                swing_signals.append(swing)
                swing_seen.add(ticker)
                print(
                    f"      ⭐ {ticker}: CLUSTER BUY (score={score:+.3f}) — "
                    f"insider swing trade created "
                    f"(hold {insider_amplifier.CLUSTER_HOLD_MIN_DAYS}"
                    f"-{insider_amplifier.CLUSTER_HOLD_MAX_DAYS}d, "
                    f"quality={swing['quality_score']:.1f})"
                )

        return amplified, swing_signals

    def check_stops(self) -> Dict[str, Any]:
        """
        Software stop backup:
        close any open long position if live price is below stored stop_loss_price.
        """
        summary = {
            "checked": 0,
            "triggered": 0,
            "errors": [],
        }
        try:
            positions = self.db.query(LivePosition).filter(
                LivePosition.is_open == True,
                LivePosition.stop_loss_price.isnot(None),
            ).all()

            for pos in positions:
                summary["checked"] += 1
                try:
                    quote = self.alpaca.get_quote(pos.ticker)
                    if not quote.get("success"):
                        continue
                    current = float(quote.get("mid") or quote.get("bid") or quote.get("ask") or 0)
                    stop = float(pos.stop_loss_price or 0)
                    if current > 0 and stop > 0 and current <= stop:
                        close_res = self.alpaca.close_position(pos.ticker)
                        if close_res.get("success"):
                            pos.is_open = False
                            pos.exit_reason = "software_stop"
                            self.db.commit()
                            summary["triggered"] += 1
                            print(
                                f"   🚨 SOFTWARE STOP: {pos.ticker} closed at ~${current:.2f} "
                                f"(stop=${stop:.2f})"
                            )
                        else:
                            summary["errors"].append(
                                f"{pos.ticker}: close failed ({close_res.get('error', 'unknown')})"
                            )
                except Exception as pos_err:
                    summary["errors"].append(f"{pos.ticker}: {pos_err}")
        except Exception as exc:
            summary["errors"].append(str(exc))
        return summary

    def check_position_exits(self) -> Dict[str, Any]:
        """
        Check all open positions for asymmetric exit conditions.

        Called by the close bot (6 AM AEDT daily) to manage:
          - Trailing stop (2 ATR below HWM, once trade is up 1 ATR)
          - Breakeven stop (once trade is up 2 ATR, stop = entry)
          - Time stop (no 1-ATR profit within 5 days → exit dead money)
          - Initial stop (hard floor: 1.5 ATR below entry)

        Returns:
            dict with checked, closed, updated, exits, errors counts/lists.
        """
        results: Dict[str, Any] = {
            "checked": 0,
            "closed": 0,
            "updated": 0,
            "exits": [],
            "errors": [],
        }

        if not ASYMMETRIC_EXITS_AVAILABLE:
            results["errors"].append("asymmetric_exits module not available")
            return results

        try:
            open_positions = self.db.query(LivePosition).filter(
                LivePosition.is_open == True
            ).all()

            print(f"\n📊 Checking {len(open_positions)} open position(s) for exit conditions...")

            for pos in open_positions:
                results["checked"] += 1
                ticker = pos.ticker

                try:
                    # ── Retrieve stored ATR from the original trade execution ─
                    trade_exec = (
                        self.db.query(TradeExecution)
                        .filter(
                            TradeExecution.ticker == ticker,
                            TradeExecution.side == 'buy',
                        )
                        .order_by(TradeExecution.created_at.desc())
                        .first()
                    )

                    atr = None
                    if trade_exec and isinstance(trade_exec.decision_factors, dict):
                        df = trade_exec.decision_factors
                        # Support both old field name ('atr_14') and new harmonised
                        # name ('atr_at_entry') — whichever agent wrote it.
                        atr = df.get('atr_at_entry') or df.get('atr_14')
                    if atr is None:
                        print(f"   [SKIP] {ticker}: no ATR stored (atr_at_entry/atr_14) — skipping asymmetric exit check")
                        continue

                    atr         = float(atr)
                    entry_price = float(pos.entry_price or 0)
                    entry_date  = pos.created_at or (datetime.utcnow() - timedelta(days=10))

                    if entry_price <= 0 or atr <= 0:
                        print(f"   ⏭️  {ticker}: invalid entry_price or ATR — skipping")
                        continue

                    # ── Fetch price history since entry to compute HWM ───────
                    try:
                        # Convert crypto tickers for yfinance (BTC/USD → BTC-USD)
                        yf_ticker_exit = to_yfinance_ticker(ticker)
                        hist = yf.Ticker(yf_ticker_exit).history(
                            start=entry_date.strftime("%Y-%m-%d"),
                            interval="1d",
                            auto_adjust=True,
                        )
                        if hist is not None and not hist.empty:
                            if isinstance(hist.columns, pd.MultiIndex):
                                hist.columns = hist.columns.get_level_values(0)
                            if 'High' in hist.columns:
                                hwm = float(hist['High'].max())
                            else:
                                hwm = float(hist['Close'].max())
                        else:
                            hwm = entry_price
                    except Exception:
                        hwm = entry_price

                    current_price = float(pos.current_price or entry_price)
                    hwm           = max(hwm, current_price, entry_price)

                    # ── Evaluate exit ────────────────────────────────────────
                    exit_result = evaluate_exit(
                        entry_price     = entry_price,
                        current_price   = current_price,
                        high_water_mark = hwm,
                        atr             = atr,
                        entry_date      = entry_date,
                        now             = datetime.utcnow(),
                        current_stop    = pos.stop_loss_price,
                    )

                    new_stop    = exit_result['new_stop']
                    stop_type   = exit_result['stop_type']
                    pnl_pct     = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0.0

                    # Always update the stop_loss_price in the DB (ratchet)
                    if new_stop and (
                        pos.stop_loss_price is None
                        or abs(new_stop - float(pos.stop_loss_price or 0)) > 0.001
                    ):
                        pos.stop_loss_price = new_stop
                        self.db.commit()
                        results["updated"] += 1

                    if exit_result['exit_now']:
                        exit_reason = exit_result['exit_reason']
                        print(
                            f"   🚨 {ticker}: EXIT → {exit_reason} | "
                            f"Entry=${entry_price:.2f} Current=${current_price:.2f} "
                            f"Stop=${new_stop:.2f} HWM=${hwm:.2f} "
                            f"P&L={pnl_pct:+.1f}%"
                        )

                        # Close via Alpaca
                        close_result = self.alpaca.close_position(ticker)
                        if close_result.get('success'):
                            pos.is_open     = False
                            pos.exit_reason = exit_reason
                            self.db.commit()
                            results["closed"] += 1
                            results["exits"].append({
                                "ticker":      ticker,
                                "exit_reason": exit_reason,
                                "entry":       entry_price,
                                "current":     current_price,
                                "stop":        new_stop,
                                "hwm":         hwm,
                                "pnl_pct":     round(pnl_pct, 2),
                                "stop_type":   stop_type,
                            })
                            print(f"   ✅ {ticker}: position closed ({exit_reason})")
                        else:
                            err = f"{ticker}: close failed — {close_result.get('error', 'unknown')}"
                            results["errors"].append(err)
                            print(f"   ❌ {err}")
                    else:
                        print(
                            f"   ✅ {ticker}: Hold | "
                            f"Stop=${new_stop:.2f} ({stop_type}) | "
                            f"HWM=${hwm:.2f} | P&L={pnl_pct:+.1f}%"
                        )

                except Exception as pos_err:
                    err = f"{ticker}: {str(pos_err)}"
                    results["errors"].append(err)
                    print(f"   ❌ {ticker}: error during exit check — {str(pos_err)[:80]}")

        except Exception as outer_err:
            results["errors"].append(str(outer_err))
            print(f"❌ check_position_exits error: {str(outer_err)}")

        # ── PEAD check for earnings catalyst positions ────────────────────
        if EARNINGS_CATALYST_AVAILABLE:
            try:
                print("\n📅 Checking PEAD logic for earnings catalyst positions…")
                from database import LivePosition as _LPe
                earnings_open = (
                    self.db.query(_LPe)
                    .filter(
                        _LPe.is_open == True,
                        _LPe.strategy_type == "earnings_catalyst",
                    )
                    .all()
                )

                if earnings_open:
                    # Build position dicts for PEAD checker
                    pos_dicts = []
                    for ep in earnings_open:
                        ec = ep.entry_conditions or {} if isinstance(ep.entry_conditions, dict) else {}
                        pos_dicts.append({
                            "ticker":        ep.ticker,
                            "earnings_date": ec.get("earnings_date", ""),
                            "strategy_type": ep.strategy_type or "earnings_catalyst",
                            "insider_bias":  ec.get("insider_bias", 0.0),
                        })

                    pead_results = _check_pead(pos_dicts, quiet=False)

                    # Act on PEAD exit signals
                    for pr in pead_results:
                        if pr.get("pead_action") == "exit":
                            tick = pr["ticker"]
                            close_res = self.alpaca.close_position(tick)
                            # Find and mark position closed in DB
                            match = next((p for p in earnings_open if p.ticker == tick), None)
                            if match and close_res.get("success"):
                                match.is_open     = False
                                match.exit_reason = "pead_miss"
                                self.db.commit()
                                results["closed"] += 1
                                results["exits"].append({
                                    "ticker":      tick,
                                    "exit_reason": "pead_miss",
                                    "pnl_pct":     0.0,
                                    "stop_type":   "pead_exit",
                                })
                                print(f"   🚨 {tick}: PEAD EXIT (earnings miss)")
                else:
                    print("   No open earnings catalyst positions to check")
            except Exception as pead_err:
                results["errors"].append(f"pead_check: {pead_err}")
                print(f"   ⚠️  PEAD check failed (non-fatal): {str(pead_err)[:80]}")

        print(
            f"\n   Exit check complete: "
            f"checked={results['checked']} closed={results['closed']} "
            f"updated={results['updated']} errors={len(results['errors'])}"
        )
        return results

    def run_weekly_ml_retraining(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run weekly ML retraining cycle from the engine context.
        """
        if not AUTO_RETRAIN_AVAILABLE:
            return {
                "success": False,
                "error": "auto_retrain module is not available"
            }

        try:
            summary = run_weekly_auto_retraining(tickers=tickers)
            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


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

