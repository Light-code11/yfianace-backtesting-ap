"""
Live Signal Generator - Provides actionable buy/sell signals for validated strategies
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from backtesting_engine import TechnicalIndicators
import threading


class LiveSignalGenerator:
    """Generate live trading signals from backtested strategies"""

    # Thread lock for yfinance downloads (fixes parallel download data mixing bug)
    _yf_lock = threading.Lock()

    @staticmethod
    def _confidence_to_score(confidence: str) -> float:
        """Convert confidence label to a normalized score used for scaling."""
        return {
            "HIGH": 1.0,
            "MEDIUM": 0.66,
            "LOW": 0.33
        }.get((confidence or "LOW").upper(), 0.33)

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        """Convert scaled confidence score back to label."""
        if score >= 0.8:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _evaluate_higher_timeframe_alignment(ticker: str, signal_direction: str) -> Dict[str, Any]:
        """
        Evaluate weekly trend/momentum alignment against daily signal direction.

        Returns:
            Dict containing multiplier (1.0/0.5/0.0), alignment status, and diagnostics.
        """
        with LiveSignalGenerator._yf_lock:
            weekly = yf.download(
                ticker,
                period="1y",
                interval="1wk",
                progress=False,
                auto_adjust=True
            )

        if weekly is None or weekly.empty:
            return {
                "multiplier": 0.5,
                "alignment": "NEUTRAL",
                "weekly_bias": "UNKNOWN",
                "reason": "No weekly data available"
            }

        if isinstance(weekly.columns, pd.MultiIndex):
            weekly.columns = weekly.columns.get_level_values(0)

        required_cols = {"Close", "High", "Low"}
        if not required_cols.issubset(set(weekly.columns)):
            return {
                "multiplier": 0.5,
                "alignment": "NEUTRAL",
                "weekly_bias": "UNKNOWN",
                "reason": "Weekly OHLC data missing required columns"
            }

        close = weekly["Close"]
        high = weekly["High"]
        low = weekly["Low"]

        sma20 = TechnicalIndicators.sma(close, 20)
        sma50 = TechnicalIndicators.sma(close, 50)
        rsi14 = TechnicalIndicators.rsi(close, 14)
        macd_line, macd_signal, _ = TechnicalIndicators.macd(close)

        latest = {
            "sma20": float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else None,
            "sma50": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None,
            "rsi14": float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else None,
            "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
            "macd_signal": float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else None
        }

        bullish_checks = 0
        bearish_checks = 0
        total_checks = 0

        if latest["sma20"] is not None and latest["sma50"] is not None:
            total_checks += 1
            if latest["sma20"] > latest["sma50"]:
                bullish_checks += 1
            elif latest["sma20"] < latest["sma50"]:
                bearish_checks += 1

        if latest["rsi14"] is not None:
            total_checks += 1
            if latest["rsi14"] > 50:
                bullish_checks += 1
            elif latest["rsi14"] < 50:
                bearish_checks += 1

        if latest["macd"] is not None and latest["macd_signal"] is not None:
            total_checks += 1
            if latest["macd"] > latest["macd_signal"]:
                bullish_checks += 1
            elif latest["macd"] < latest["macd_signal"]:
                bearish_checks += 1

        if total_checks == 0:
            return {
                "multiplier": 0.5,
                "alignment": "NEUTRAL",
                "weekly_bias": "UNKNOWN",
                "reason": "Insufficient weekly indicator data",
                "weekly_indicators": latest
            }

        if bullish_checks > bearish_checks:
            weekly_bias = "BULLISH"
        elif bearish_checks > bullish_checks:
            weekly_bias = "BEARISH"
        else:
            weekly_bias = "NEUTRAL"

        direction = (signal_direction or "").upper()
        desired_bias = "BULLISH" if direction == "BUY" else "BEARISH" if direction == "SELL" else "NEUTRAL"

        if desired_bias == "NEUTRAL" or weekly_bias == "NEUTRAL":
            multiplier = 0.5
            alignment = "NEUTRAL"
        elif weekly_bias == desired_bias:
            multiplier = 1.0
            alignment = "ALIGNED"
        else:
            multiplier = 0.0
            alignment = "CONFLICT"

        return {
            "multiplier": multiplier,
            "alignment": alignment,
            "weekly_bias": weekly_bias,
            "reason": (
                f"Weekly bias {weekly_bias} vs daily {direction}"
                if direction in {"BUY", "SELL"} else "No directional daily signal"
            ),
            "weekly_indicators": latest
        }

    @staticmethod
    def confirm_with_higher_timeframe(ticker: str, signal_direction: str) -> float:
        """
        Confirm daily signal with weekly data and return confidence multiplier.

        Multipliers:
        - 1.0: weekly and daily agree
        - 0.5: weekly neutral / insufficient
        - 0.0: weekly conflicts with daily
        """
        try:
            result = LiveSignalGenerator._evaluate_higher_timeframe_alignment(ticker, signal_direction)
            return float(result.get("multiplier", 0.5))
        except Exception:
            # Degrade gracefully when weekly confirmation cannot be computed.
            return 0.5

    @staticmethod
    def generate_signals(strategy_config: Dict[str, Any], period: str = "3mo") -> Dict[str, Any]:
        """
        Generate live trading signals for a strategy

        Args:
            strategy_config: Strategy configuration with tickers, indicators, risk management
            period: How much historical data to fetch for calculations (default: 3mo)

        Returns:
            Dict with signals for each ticker:
            {
                "ticker": "AAPL",
                "signal": "BUY" | "SELL" | "HOLD",
                "current_price": 150.50,
                "entry_price": 150.50,
                "stop_loss": 142.98,
                "take_profit": 165.55,
                "position_size_pct": 25.0,
                "position_size_usd": 25000,
                "confidence": "HIGH" | "MEDIUM" | "LOW",
                "indicators": {...},
                "reasoning": "SMA(20) crossed above SMA(50), RSI at 45 (not overbought)..."
            }
        """
        tickers = strategy_config.get('tickers', [])
        strategy_type = strategy_config.get('strategy_type', 'momentum')
        indicators_config = strategy_config.get('indicators', [])
        risk_mgmt = strategy_config.get('risk_management', {})

        signals = []

        for ticker in tickers:
            try:
                signal = LiveSignalGenerator._generate_ticker_signal(
                    ticker=ticker,
                    strategy_type=strategy_type,
                    indicators_config=indicators_config,
                    risk_mgmt=risk_mgmt,
                    period=period
                )
                signals.append(signal)
            except Exception as e:
                signals.append({
                    "ticker": ticker,
                    "signal": "ERROR",
                    "error": str(e)
                })

        return {
            "strategy_name": strategy_config.get('name', 'Unknown'),
            "strategy_type": strategy_type,
            "generated_at": datetime.now().isoformat(),
            "signals": signals
        }

    @staticmethod
    def _generate_ticker_signal(
        ticker: str,
        strategy_type: str,
        indicators_config: List[Dict],
        risk_mgmt: Dict,
        period: str
    ) -> Dict[str, Any]:
        """Generate signal for a single ticker"""

        # Validate ticker parameter
        if not ticker or not isinstance(ticker, str):
            return {
                "ticker": str(ticker),
                "signal": "ERROR",
                "error": "Invalid ticker parameter"
            }

        # Fetch recent data for THIS SPECIFIC ticker only
        # Use thread lock to prevent yfinance data mixing in parallel downloads
        with LiveSignalGenerator._yf_lock:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

        if data.empty:
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "error": "No market data available"
            }

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        current_price = float(close.iloc[-1])

        # Calculate indicators
        indicators = LiveSignalGenerator._calculate_indicators(
            data, indicators_config
        )

        # Generate signal based on strategy type
        signal_info = LiveSignalGenerator._get_strategy_signal(
            strategy_type=strategy_type,
            data=data,
            indicators=indicators,
            current_price=current_price
        )

        htf_result = LiveSignalGenerator._evaluate_higher_timeframe_alignment(
            ticker=ticker,
            signal_direction=signal_info.get("signal", "HOLD")
        )
        htf_multiplier = float(htf_result.get("multiplier", 0.5))

        base_confidence = signal_info.get("confidence", "LOW")
        scaled_confidence_score = LiveSignalGenerator._confidence_to_score(base_confidence) * htf_multiplier
        scaled_confidence = LiveSignalGenerator._score_to_confidence(scaled_confidence_score)

        if signal_info.get("signal") in {"BUY", "SELL"}:
            signal_info["confidence"] = scaled_confidence
            signal_info["reasoning"] = (
                f"{signal_info.get('reasoning', '')}. "
                f"Higher timeframe: {htf_result.get('reason', 'N/A')} "
                f"(multiplier={htf_multiplier:.2f}, confidence {base_confidence}->{scaled_confidence})"
            ).strip()

        # Calculate risk management levels
        stop_loss_pct = risk_mgmt.get('stop_loss_pct', 5.0)
        take_profit_pct = risk_mgmt.get('take_profit_pct', 10.0)
        position_size_pct = risk_mgmt.get('position_size_pct', 25.0)

        if signal_info['signal'] == 'BUY':
            entry_price = current_price
            stop_loss = round(entry_price * (1 - stop_loss_pct / 100), 2)
            take_profit = round(entry_price * (1 + take_profit_pct / 100), 2)
        elif signal_info['signal'] == 'SELL':
            entry_price = current_price
            stop_loss = round(entry_price * (1 + stop_loss_pct / 100), 2)
            take_profit = round(entry_price * (1 - take_profit_pct / 100), 2)
        else:  # HOLD
            entry_price = current_price
            stop_loss = None
            take_profit = None

        return {
            "ticker": ticker,
            "signal": signal_info['signal'],
            "current_price": round(current_price, 2),
            "entry_price": round(entry_price, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_pct": position_size_pct,
            "confidence": signal_info['confidence'],
            "confidence_score": round(float(scaled_confidence_score), 4),
            "base_confidence": base_confidence,
            "indicators": indicators,
            "higher_timeframe_multiplier": htf_multiplier,
            "higher_timeframe_alignment": htf_result.get("alignment"),
            "higher_timeframe_bias": htf_result.get("weekly_bias"),
            "higher_timeframe_indicators": htf_result.get("weekly_indicators", {}),
            "reasoning": signal_info['reasoning'],
            "last_updated": datetime.now().isoformat()
        }

    @staticmethod
    def _calculate_indicators(data: pd.DataFrame, indicators_config: List[Dict]) -> Dict[str, float]:
        """Calculate current indicator values"""
        indicators = {}
        close = data['Close']
        high = data['High']
        low = data['Low']

        for ind in indicators_config:
            name = ind.get('name', '').upper()

            if name == 'SMA':
                period = ind.get('period', 20)
                sma = TechnicalIndicators.sma(close, period)
                indicators[f'SMA_{period}'] = round(float(sma.iloc[-1]), 2)

            elif name == 'EMA':
                period = ind.get('period', 20)
                ema = TechnicalIndicators.ema(close, period)
                indicators[f'EMA_{period}'] = round(float(ema.iloc[-1]), 2)

            elif name == 'RSI':
                period = ind.get('period', 14)
                rsi = TechnicalIndicators.rsi(close, period)
                indicators[f'RSI_{period}'] = round(float(rsi.iloc[-1]), 2)

            elif name == 'MACD':
                macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
                indicators['MACD'] = round(float(macd_line.iloc[-1]), 2)
                indicators['MACD_Signal'] = round(float(signal_line.iloc[-1]), 2)
                indicators['MACD_Histogram'] = round(float(histogram.iloc[-1]), 2)

            elif name == 'BB':
                upper, middle, lower = TechnicalIndicators.bollinger_bands(close)
                indicators['BB_Upper'] = round(float(upper.iloc[-1]), 2)
                indicators['BB_Middle'] = round(float(middle.iloc[-1]), 2)
                indicators['BB_Lower'] = round(float(lower.iloc[-1]), 2)

            elif name == 'ATR':
                period = ind.get('period', 14)
                atr = TechnicalIndicators.atr(high, low, close, period)
                indicators[f'ATR_{period}'] = round(float(atr.iloc[-1]), 2)

        # Always include current price
        indicators['Price'] = round(float(close.iloc[-1]), 2)

        return indicators

    @staticmethod
    def _get_strategy_signal(
        strategy_type: str,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        current_price: float
    ) -> Dict[str, Any]:
        """Determine BUY/SELL/HOLD signal based on strategy type"""

        close = data['Close']

        if strategy_type == 'momentum':
            return LiveSignalGenerator._momentum_signal(indicators, close)
        elif strategy_type == 'mean_reversion':
            return LiveSignalGenerator._mean_reversion_signal(indicators, close)
        elif strategy_type == 'breakout':
            return LiveSignalGenerator._breakout_signal(indicators, data, current_price)
        elif strategy_type == 'trend_following':
            return LiveSignalGenerator._trend_following_signal(indicators, close)
        else:
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'reasoning': f'Unknown strategy type: {strategy_type}'
            }

    @staticmethod
    def _momentum_signal(indicators: Dict, close: pd.Series) -> Dict[str, Any]:
        """Momentum strategy signal logic"""
        reasoning_parts = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        # Check SMA crossover
        sma_20 = indicators.get('SMA_20')
        sma_50 = indicators.get('SMA_50')
        price = indicators.get('Price')
        rsi = indicators.get('RSI_14', 50)

        if sma_20 and sma_50 and price:
            if sma_20 > sma_50 and price > sma_20:
                if rsi < 70:  # Not overbought
                    signal = 'BUY'
                    confidence = 'HIGH' if rsi < 60 else 'MEDIUM'
                    reasoning_parts.append(f"Strong uptrend: SMA(20)={sma_20:.2f} > SMA(50)={sma_50:.2f}, Price above SMA(20)")
                    reasoning_parts.append(f"RSI={rsi:.1f} (not overbought)")
                else:
                    signal = 'HOLD'
                    reasoning_parts.append(f"Uptrend but RSI={rsi:.1f} (overbought - wait for pullback)")

            elif sma_20 < sma_50 and price < sma_20:
                if rsi > 30:  # Not oversold
                    signal = 'SELL'
                    confidence = 'HIGH' if rsi > 40 else 'MEDIUM'
                    reasoning_parts.append(f"Strong downtrend: SMA(20)={sma_20:.2f} < SMA(50)={sma_50:.2f}, Price below SMA(20)")
                    reasoning_parts.append(f"RSI={rsi:.1f} (not oversold)")
                else:
                    signal = 'HOLD'
                    reasoning_parts.append(f"Downtrend but RSI={rsi:.1f} (oversold - may bounce)")
            else:
                signal = 'HOLD'
                reasoning_parts.append(f"No clear trend: SMA(20)={sma_20:.2f}, SMA(50)={sma_50:.2f}, Price={price:.2f}")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'Insufficient data for signal'
        }

    @staticmethod
    def _mean_reversion_signal(indicators: Dict, close: pd.Series) -> Dict[str, Any]:
        """Mean reversion strategy signal logic"""
        reasoning_parts = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        bb_upper = indicators.get('BB_Upper')
        bb_lower = indicators.get('BB_Lower')
        bb_middle = indicators.get('BB_Middle')
        price = indicators.get('Price')
        rsi = indicators.get('RSI_14', 50)

        if bb_upper and bb_lower and bb_middle and price:
            # Buy when price touches lower band (oversold)
            if price <= bb_lower * 1.02 and rsi < 35:
                signal = 'BUY'
                confidence = 'HIGH' if rsi < 30 else 'MEDIUM'
                reasoning_parts.append(f"Oversold: Price={price:.2f} near lower BB={bb_lower:.2f}")
                reasoning_parts.append(f"RSI={rsi:.1f} confirms oversold")

            # Sell when price touches upper band (overbought)
            elif price >= bb_upper * 0.98 and rsi > 65:
                signal = 'SELL'
                confidence = 'HIGH' if rsi > 70 else 'MEDIUM'
                reasoning_parts.append(f"Overbought: Price={price:.2f} near upper BB={bb_upper:.2f}")
                reasoning_parts.append(f"RSI={rsi:.1f} confirms overbought")
            else:
                signal = 'HOLD'
                distance_to_lower = ((price - bb_lower) / bb_lower) * 100
                distance_to_upper = ((bb_upper - price) / price) * 100
                reasoning_parts.append(f"Price in mid-range: {distance_to_lower:.1f}% from lower, {distance_to_upper:.1f}% from upper BB")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'Insufficient data for signal'
        }

    @staticmethod
    def _breakout_signal(indicators: Dict, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Breakout strategy signal logic"""
        reasoning_parts = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        # Calculate 20-day high/low
        close = data['Close']
        high_20 = close.iloc[-20:].max() if len(close) >= 20 else close.max()
        low_20 = close.iloc[-20:].min() if len(close) >= 20 else close.min()

        atr = indicators.get('ATR_14', 0)
        sma_20 = indicators.get('SMA_20')

        # Breakout above resistance
        if current_price >= high_20 * 0.995:  # Within 0.5% of high
            if atr > 0:  # Confirm with volume/volatility
                signal = 'BUY'
                confidence = 'HIGH'
                reasoning_parts.append(f"Breakout above 20-day high: Price={current_price:.2f}, High={high_20:.2f}")
                reasoning_parts.append(f"ATR={atr:.2f} confirms volatility expansion")
            else:
                signal = 'HOLD'
                reasoning_parts.append(f"Near 20-day high but low volatility (wait for confirmation)")

        # Breakdown below support
        elif current_price <= low_20 * 1.005:  # Within 0.5% of low
            signal = 'SELL'
            confidence = 'MEDIUM'
            reasoning_parts.append(f"Breakdown below 20-day low: Price={current_price:.2f}, Low={low_20:.2f}")
        else:
            signal = 'HOLD'
            distance_to_high = ((high_20 - current_price) / current_price) * 100
            distance_to_low = ((current_price - low_20) / low_20) * 100
            reasoning_parts.append(f"In range: {distance_to_high:.1f}% from high, {distance_to_low:.1f}% above low")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'Insufficient data for signal'
        }

    @staticmethod
    def _trend_following_signal(indicators: Dict, close: pd.Series) -> Dict[str, Any]:
        """Trend following strategy signal logic"""
        reasoning_parts = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        sma_50 = indicators.get('SMA_50')
        sma_200 = indicators.get('SMA_200')
        price = indicators.get('Price')
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_Signal')

        if sma_50 and sma_200 and price:
            # Golden cross + MACD confirmation
            if sma_50 > sma_200 and price > sma_50:
                if macd and macd_signal and macd > macd_signal:
                    signal = 'BUY'
                    confidence = 'HIGH'
                    reasoning_parts.append(f"Golden cross: SMA(50)={sma_50:.2f} > SMA(200)={sma_200:.2f}")
                    reasoning_parts.append(f"MACD bullish: {macd:.2f} > Signal={macd_signal:.2f}")
                else:
                    signal = 'HOLD'
                    reasoning_parts.append(f"Golden cross but MACD not confirmed")

            # Death cross
            elif sma_50 < sma_200 and price < sma_50:
                if macd and macd_signal and macd < macd_signal:
                    signal = 'SELL'
                    confidence = 'HIGH'
                    reasoning_parts.append(f"Death cross: SMA(50)={sma_50:.2f} < SMA(200)={sma_200:.2f}")
                    reasoning_parts.append(f"MACD bearish: {macd:.2f} < Signal={macd_signal:.2f}")
                else:
                    signal = 'HOLD'
                    reasoning_parts.append(f"Death cross but MACD not confirmed")
            else:
                signal = 'HOLD'
                reasoning_parts.append(f"No clear long-term trend")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'Insufficient data for signal'
        }
