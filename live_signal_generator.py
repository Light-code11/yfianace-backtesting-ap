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
            "indicators": indicators,
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
