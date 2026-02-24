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

    # ── Sector ETF caches (refreshed once per calendar day) ──────────────────
    _sector_etf_cache: Dict[str, "pd.Series"] = {}
    _sector_spy_cache: Optional["pd.Series"] = None
    _sector_cache_date: Optional[str] = None
    _sector_cache_lock = threading.Lock()  # separate lock for sector cache management

    # ── Weekly (HTF) data cache — avoids re-downloading per strategy ──────────
    _weekly_data_cache: Dict[str, "pd.DataFrame"] = {}
    _weekly_cache_date: Optional[str] = None
    _weekly_cache_lock = threading.Lock()

    # ── Daily ticker data cache (keyed by ticker:period) ─────────────────────
    _daily_data_cache: Dict[str, "pd.DataFrame"] = {}
    _daily_cache_date: Optional[str] = None
    _daily_cache_lock = threading.Lock()

    # Ticker → sector ETF mapping
    SECTOR_ETF_MAP: Dict[str, str] = {
        # Technology / Semiconductors → XLK
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
        "AVGO": "XLK", "MRVL": "XLK", "ARM": "XLK", "QCOM": "XLK", "LRCX": "XLK",
        "KLAC": "XLK", "ASML": "XLK", "CRM": "XLK", "SMCI": "XLK", "PLTR": "XLK",
        "CRWD": "XLK", "NET": "XLK", "DDOG": "XLK", "ZS": "XLK", "PANW": "XLK",
        "SHOP": "XLK",
        # Communication Services → XLC
        "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "MELI": "XLC",
        # Consumer Discretionary → XLY
        "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY", "LULU": "XLY",
        "DECK": "XLY",
        # Consumer Staples → XLP
        "WMT": "XLP", "PG": "XLP", "COST": "XLP", "TGT": "XLP", "SBUX": "XLP",
        "MCD": "XLP",
        # Financials → XLF
        "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF", "WFC": "XLF",
        "PYPL": "XLF", "XYZ": "XLF", "COIN": "XLF", "AFRM": "XLF", "SOFI": "XLF",
        "NU": "XLF",
        # Healthcare → XLV
        "UNH": "XLV", "JNJ": "XLV", "MRNA": "XLV", "REGN": "XLV", "ABBV": "XLV",
        "LLY": "XLV", "TMO": "XLV", "ISRG": "XLV", "DXCM": "XLV",
        # Energy → XLE
        "XOM": "XLE", "CVX": "XLE", "SLB": "XLE", "EOG": "XLE", "OXY": "XLE",
        "MPC": "XLE",
        # Industrials → XLI
        "CAT": "XLI", "DE": "XLI", "GE": "XLI", "HON": "XLI", "RTX": "XLI",
        "LMT": "XLI",
    }

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
        today = datetime.now().strftime("%Y-%m-%d")

        # Fast path: use cached weekly data if available
        if (LiveSignalGenerator._weekly_cache_date == today
                and ticker in LiveSignalGenerator._weekly_data_cache):
            weekly = LiveSignalGenerator._weekly_data_cache[ticker]
        else:
            with LiveSignalGenerator._weekly_cache_lock:
                if LiveSignalGenerator._weekly_cache_date != today:
                    LiveSignalGenerator._weekly_data_cache = {}
                    LiveSignalGenerator._weekly_cache_date = today

                if ticker not in LiveSignalGenerator._weekly_data_cache:
                    with LiveSignalGenerator._yf_lock:
                        downloaded = yf.download(
                            ticker,
                            period="1y",
                            interval="1wk",
                            progress=False,
                            auto_adjust=True
                        )
                    LiveSignalGenerator._weekly_data_cache[ticker] = downloaded

            weekly = LiveSignalGenerator._weekly_data_cache.get(ticker)

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

        # Fetch recent data — use cache if already downloaded today for this period
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{ticker}:{period}"
        if (LiveSignalGenerator._daily_cache_date == today
                and cache_key in LiveSignalGenerator._daily_data_cache):
            data = LiveSignalGenerator._daily_data_cache[cache_key]
        else:
            with LiveSignalGenerator._daily_cache_lock:
                if LiveSignalGenerator._daily_cache_date != today:
                    LiveSignalGenerator._daily_data_cache = {}
                    LiveSignalGenerator._daily_cache_date = today
                if cache_key not in LiveSignalGenerator._daily_data_cache:
                    with LiveSignalGenerator._yf_lock:
                        downloaded = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                    LiveSignalGenerator._daily_data_cache[cache_key] = downloaded
            data = LiveSignalGenerator._daily_data_cache.get(cache_key)

        if data is None or data.empty:
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
            current_price=current_price,
            ticker=ticker
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

            elif name == 'VWAP':
                period = ind.get('period', 20)
                if 'Volume' in data.columns:
                    typical_price = (high + low + close) / 3
                    vol = data['Volume']
                    vwap_num = (typical_price * vol).rolling(period).sum()
                    vwap_den = vol.rolling(period).sum()
                    vwap = (vwap_num / vwap_den).replace([np.inf, -np.inf], np.nan)
                    vwap_val = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else float(close.iloc[-1])
                    indicators['VWAP'] = round(vwap_val, 2)

                    # Price deviation std vs rolling VWAP
                    dev = typical_price - vwap
                    dev_std = dev.rolling(period).std()
                    dev_std_val = float(dev_std.iloc[-1]) if not pd.isna(dev_std.iloc[-1]) else 0.0
                    indicators['VWAP_Std'] = round(dev_std_val, 4)

                    cur_tp = float(typical_price.iloc[-1])
                    indicators['VWAP_ZScore'] = round(
                        (cur_tp - vwap_val) / dev_std_val if dev_std_val != 0 else 0.0, 3
                    )

        # Always include current price
        indicators['Price'] = round(float(close.iloc[-1]), 2)

        return indicators

    @staticmethod
    def _get_strategy_signal(
        strategy_type: str,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        current_price: float,
        ticker: str = ""
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
        elif strategy_type == 'bb_squeeze':
            return LiveSignalGenerator._bb_squeeze_signal(indicators, close, data)
        elif strategy_type == 'vwap_reversion':
            return LiveSignalGenerator._vwap_reversion_signal(indicators, close)
        elif strategy_type == 'gap_fill':
            return LiveSignalGenerator._gap_fill_signal(indicators, data, current_price)
        elif strategy_type == 'earnings_momentum':
            return LiveSignalGenerator._earnings_momentum_signal(indicators, close, data)
        elif strategy_type == 'sector_rotation':
            return LiveSignalGenerator._sector_rotation_signal(indicators, close, data, ticker)
        elif strategy_type == 'pair_trading':
            # Pair trading signals are generated separately in the engine
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'reasoning': 'Pair trading handled separately via _generate_pair_signals()'
            }
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
        volume = data['Volume']
        current_volume = float(volume.iloc[-1]) if len(volume) > 0 else 0.0
        avg_volume_20 = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        volume_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 0.0

        # Breakout above resistance
        if current_price >= high_20 * 0.995:  # Within 0.5% of high
            if atr > 0 and volume_ratio >= 1.2:  # Confirm with volume + volatility expansion
                signal = 'BUY'
                confidence = 'HIGH'
                reasoning_parts.append(f"Breakout above 20-day high: Price={current_price:.2f}, High={high_20:.2f}")
                reasoning_parts.append(f"ATR={atr:.2f}, Volume {volume_ratio:.2f}x confirms expansion")
            else:
                signal = 'HOLD'
                reasoning_parts.append(
                    f"Near 20-day high but confirmation weak (ATR={atr:.2f}, Volume {volume_ratio:.2f}x)"
                )

        # Breakdown below support
        elif current_price <= low_20 * 1.005:  # Within 0.5% of low
            if volume_ratio >= 1.2:
                signal = 'SELL'
                confidence = 'HIGH'
                reasoning_parts.append(f"Breakdown below 20-day low: Price={current_price:.2f}, Low={low_20:.2f}")
                reasoning_parts.append(f"Volume {volume_ratio:.2f}x confirms bearish breakout")
            else:
                signal = 'HOLD'
                reasoning_parts.append(f"Near 20-day low but volume only {volume_ratio:.2f}x (no confirmation)")
        else:
            signal = 'HOLD'
            distance_to_high = ((high_20 - current_price) / current_price) * 100
            distance_to_low = ((current_price - low_20) / low_20) * 100
            reasoning_parts.append(
                f"In range: {distance_to_high:.1f}% from high, {distance_to_low:.1f}% above low, "
                f"Volume {volume_ratio:.2f}x"
            )

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

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW STRATEGY SIGNAL HANDLERS (Strategies 5-9)
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _bb_squeeze_signal(indicators: Dict, close: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Bollinger Band Squeeze breakout strategy.

        Identifies low-volatility squeeze periods (bandwidth in lowest 20th percentile)
        and fires on the breakout direction.
        """
        bb_upper = indicators.get('BB_Upper')
        bb_lower = indicators.get('BB_Lower')
        bb_middle = indicators.get('BB_Middle')
        price = indicators.get('Price')
        rsi = indicators.get('RSI_14', 50)

        reasoning_parts: List[str] = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        if not all([bb_upper, bb_lower, bb_middle, price]):
            return {'signal': 'HOLD', 'confidence': 'LOW', 'reasoning': 'Insufficient BB data for squeeze analysis'}

        # Current bandwidth
        bandwidth = (bb_upper - bb_lower) / bb_middle

        # Historical bandwidths to define "squeeze"
        try:
            upper_ser, middle_ser, lower_ser = TechnicalIndicators.bollinger_bands(close)
            bw_series = ((upper_ser - lower_ser) / middle_ser).dropna()
            lookback = min(50, len(bw_series))
            squeeze_threshold = float(bw_series.iloc[-lookback:].quantile(0.20))
        except Exception:
            squeeze_threshold = bandwidth * 1.0   # no history → never squeeze

        in_squeeze = bandwidth <= squeeze_threshold

        if in_squeeze:
            if price > bb_upper and rsi > 50:
                signal = 'BUY'
                confidence = 'HIGH' if rsi < 70 else 'MEDIUM'
                reasoning_parts.append(
                    f"BB Squeeze breakout UP: Price={price:.2f} > BB_Upper={bb_upper:.2f}"
                )
                reasoning_parts.append(
                    f"Bandwidth={bandwidth:.4f} ≤ squeeze threshold={squeeze_threshold:.4f}, RSI={rsi:.1f}"
                )
            elif price < bb_lower and rsi < 50:
                signal = 'SELL'
                confidence = 'HIGH' if rsi > 30 else 'MEDIUM'
                reasoning_parts.append(
                    f"BB Squeeze breakout DOWN: Price={price:.2f} < BB_Lower={bb_lower:.2f}"
                )
                reasoning_parts.append(
                    f"Bandwidth={bandwidth:.4f} ≤ squeeze threshold={squeeze_threshold:.4f}, RSI={rsi:.1f}"
                )
            else:
                reasoning_parts.append(
                    f"In BB squeeze (BW={bandwidth:.4f} ≤ {squeeze_threshold:.4f}) — awaiting breakout"
                )
        else:
            reasoning_parts.append(
                f"No squeeze: BW={bandwidth:.4f} > threshold={squeeze_threshold:.4f}"
            )

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'No BB squeeze signal'
        }

    @staticmethod
    def _vwap_reversion_signal(indicators: Dict, close: pd.Series) -> Dict[str, Any]:
        """
        VWAP mean reversion signal.

        Fades price when it is >2 standard deviations from the rolling VWAP.
        """
        vwap = indicators.get('VWAP')
        vwap_zscore = indicators.get('VWAP_ZScore')
        price = indicators.get('Price')
        rsi = indicators.get('RSI_14', 50)

        reasoning_parts: List[str] = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        if vwap is None or vwap_zscore is None:
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'reasoning': 'VWAP indicator unavailable (Volume data may be missing)'
            }

        if vwap_zscore < -2.0 and rsi < 40:
            signal = 'BUY'
            confidence = 'HIGH' if vwap_zscore < -2.5 else 'MEDIUM'
            reasoning_parts.append(f"Price well below rolling VWAP (z={vwap_zscore:.2f})")
            reasoning_parts.append(f"RSI={rsi:.1f} confirms oversold vs VWAP={vwap:.2f}")
        elif vwap_zscore > 2.0 and rsi > 60:
            signal = 'SELL'
            confidence = 'HIGH' if vwap_zscore > 2.5 else 'MEDIUM'
            reasoning_parts.append(f"Price well above rolling VWAP (z={vwap_zscore:.2f})")
            reasoning_parts.append(f"RSI={rsi:.1f} confirms overbought vs VWAP={vwap:.2f}")
        else:
            reasoning_parts.append(
                f"VWAP z={vwap_zscore:.2f} (no extreme deviation), RSI={rsi:.1f}, VWAP={vwap:.2f}"
            )

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'No VWAP reversion signal'
        }

    @staticmethod
    def _gap_fill_signal(indicators: Dict, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Overnight gap fade strategy.

        Detects gaps >2% from previous close and fades them expecting a 50% fill.
        """
        reasoning_parts: List[str] = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        if 'Open' not in data.columns or len(data) < 2:
            return {'signal': 'HOLD', 'confidence': 'LOW', 'reasoning': 'Open price data unavailable for gap detection'}

        today_open = float(data['Open'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])

        if prev_close == 0:
            return {'signal': 'HOLD', 'confidence': 'LOW', 'reasoning': 'Invalid previous close price'}

        gap_pct = (today_open - prev_close) / prev_close * 100
        rsi = indicators.get('RSI_14', 50)
        gap_threshold = 2.0

        if gap_pct >= gap_threshold:
            # Gap up → fade down toward previous close
            if current_price > prev_close:
                signal = 'SELL'
                confidence = 'HIGH' if gap_pct >= 4.0 else 'MEDIUM'
                fill_target = prev_close + (today_open - prev_close) * 0.5
                reasoning_parts.append(
                    f"Gap UP {gap_pct:.1f}%: Open={today_open:.2f} vs Prev Close={prev_close:.2f}"
                )
                reasoning_parts.append(f"Fading gap — 50% fill target ≈ {fill_target:.2f}")
        elif gap_pct <= -gap_threshold:
            # Gap down → fade up toward previous close
            if current_price < prev_close:
                signal = 'BUY'
                confidence = 'HIGH' if gap_pct <= -4.0 else 'MEDIUM'
                fill_target = prev_close + (today_open - prev_close) * 0.5
                reasoning_parts.append(
                    f"Gap DOWN {gap_pct:.1f}%: Open={today_open:.2f} vs Prev Close={prev_close:.2f}"
                )
                reasoning_parts.append(f"Fading gap — 50% fill target ≈ {fill_target:.2f}")
        else:
            reasoning_parts.append(f"No significant gap: {gap_pct:.2f}% (threshold ±{gap_threshold}%)")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'No gap detected'
        }

    @staticmethod
    def _earnings_momentum_signal(indicators: Dict, close: pd.Series, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Post-Earnings Announcement Drift (PEAD) momentum strategy.

        After a ≥5% single-day move in the last 20 trading days (earnings proxy),
        rides the momentum in the direction of the surprise.
        """
        reasoning_parts: List[str] = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        if len(close) < 25:
            return {'signal': 'HOLD', 'confidence': 'LOW', 'reasoning': 'Insufficient history for PEAD analysis'}

        # Detect earnings-like moves in recent history (last 20 days, excluding today)
        recent_returns = close.pct_change().iloc[-21:-1]
        earnings_threshold = 0.05

        large_moves = recent_returns[abs(recent_returns) >= earnings_threshold]

        if large_moves.empty:
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'reasoning': 'No earnings-like move (≥5%) detected in last 20 days'
            }

        # Focus on the most recent large move
        latest_move = float(large_moves.iloc[-1])
        move_idx = list(recent_returns.index).index(large_moves.index[-1])
        days_since_move = len(recent_returns) - move_idx

        rsi = indicators.get('RSI_14', 50)
        sma_20 = indicators.get('SMA_20')
        price = indicators.get('Price')

        if latest_move > earnings_threshold:
            # Positive surprise — PEAD long
            if rsi < 75:
                signal = 'BUY'
                confidence = 'HIGH' if latest_move > 0.10 else 'MEDIUM'
                reasoning_parts.append(
                    f"PEAD: +{latest_move*100:.1f}% earnings move {days_since_move} days ago"
                )
                reasoning_parts.append(f"Positive drift expected, RSI={rsi:.1f}")
        elif latest_move < -earnings_threshold:
            # Negative surprise — PEAD short
            if rsi > 25:
                signal = 'SELL'
                confidence = 'HIGH' if latest_move < -0.10 else 'MEDIUM'
                reasoning_parts.append(
                    f"PEAD: {latest_move*100:.1f}% earnings move {days_since_move} days ago"
                )
                reasoning_parts.append(f"Negative drift expected, RSI={rsi:.1f}")

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'PEAD analysis complete — no strong signal'
        }

    @staticmethod
    def prewarm_sector_cache(period: str = "3mo") -> None:
        """
        Pre-download SPY and all unique sector ETFs into the cache.
        Call this once before launching parallel scans to avoid download contention.
        """
        all_etfs = list(set(LiveSignalGenerator.SECTOR_ETF_MAP.values()))
        symbols = ["SPY"] + all_etfs
        today = datetime.now().strftime("%Y-%m-%d")

        with LiveSignalGenerator._sector_cache_lock:
            if LiveSignalGenerator._sector_cache_date != today:
                LiveSignalGenerator._sector_etf_cache = {}
                LiveSignalGenerator._sector_spy_cache = None
                LiveSignalGenerator._sector_cache_date = today

            for sym in symbols:
                if sym == "SPY" and LiveSignalGenerator._sector_spy_cache is not None:
                    continue
                if sym != "SPY" and sym in LiveSignalGenerator._sector_etf_cache:
                    continue
                try:
                    with LiveSignalGenerator._yf_lock:
                        raw = yf.download(sym, period=period, progress=False, auto_adjust=True)
                    if not raw.empty:
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = raw.columns.get_level_values(0)
                        series = raw['Close'].squeeze()
                        if sym == "SPY":
                            LiveSignalGenerator._sector_spy_cache = series
                        else:
                            LiveSignalGenerator._sector_etf_cache[sym] = series
                except Exception:
                    pass

    @staticmethod
    def _get_sector_etf_performance(sector_etf: str, period: str = "3mo") -> Optional[float]:
        """
        Return rolling 20-day return of sector_etf relative to SPY (cached per day).
        Positive = sector outperforming; Negative = underperforming.
        Uses _sector_cache_lock (separate from _yf_lock) to avoid scan deadlocks.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Fast path: return cached value without locking
        if (LiveSignalGenerator._sector_cache_date == today
                and sector_etf in LiveSignalGenerator._sector_etf_cache
                and LiveSignalGenerator._sector_spy_cache is not None):
            etf_close = LiveSignalGenerator._sector_etf_cache[sector_etf]
            spy_close = LiveSignalGenerator._sector_spy_cache
        else:
            # Slow path: acquire cache lock and ensure downloads
            with LiveSignalGenerator._sector_cache_lock:
                if LiveSignalGenerator._sector_cache_date != today:
                    LiveSignalGenerator._sector_etf_cache = {}
                    LiveSignalGenerator._sector_spy_cache = None
                    LiveSignalGenerator._sector_cache_date = today

                if sector_etf not in LiveSignalGenerator._sector_etf_cache:
                    try:
                        with LiveSignalGenerator._yf_lock:
                            etf_raw = yf.download(sector_etf, period=period, progress=False, auto_adjust=True)
                        if not etf_raw.empty:
                            if isinstance(etf_raw.columns, pd.MultiIndex):
                                etf_raw.columns = etf_raw.columns.get_level_values(0)
                            LiveSignalGenerator._sector_etf_cache[sector_etf] = etf_raw['Close'].squeeze()
                    except Exception:
                        return None

                if LiveSignalGenerator._sector_spy_cache is None:
                    try:
                        with LiveSignalGenerator._yf_lock:
                            spy_raw = yf.download("SPY", period=period, progress=False, auto_adjust=True)
                        if not spy_raw.empty:
                            if isinstance(spy_raw.columns, pd.MultiIndex):
                                spy_raw.columns = spy_raw.columns.get_level_values(0)
                            LiveSignalGenerator._sector_spy_cache = spy_raw['Close'].squeeze()
                    except Exception:
                        return None

            etf_close = LiveSignalGenerator._sector_etf_cache.get(sector_etf)
            spy_close = LiveSignalGenerator._sector_spy_cache

        if etf_close is None or spy_close is None:
            return None

        lookback = 20
        if len(etf_close) < lookback + 1 or len(spy_close) < lookback + 1:
            return None

        etf_ret = float(etf_close.iloc[-1] / etf_close.iloc[-lookback - 1]) - 1.0
        spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-lookback - 1]) - 1.0
        return etf_ret - spy_ret

    @staticmethod
    def _sector_rotation_signal(
        indicators: Dict,
        close: pd.Series,
        data: pd.DataFrame,
        ticker: str = ""
    ) -> Dict[str, Any]:
        """
        Sector rotation signal using sector ETF relative strength vs SPY.

        Strong sector (>+2% vs SPY 20d) + not overbought → BUY
        Weak sector (<-2% vs SPY 20d) + not oversold → SELL
        """
        reasoning_parts: List[str] = []
        signal = 'HOLD'
        confidence = 'MEDIUM'

        rsi = indicators.get('RSI_14', 50)
        sma_20 = indicators.get('SMA_20')
        price = indicators.get('Price')

        sector_etf = LiveSignalGenerator.SECTOR_ETF_MAP.get(ticker)

        if sector_etf is None:
            # Fallback: simple momentum vs own SMA(20)
            if sma_20 and price:
                pct_from_sma = (price - sma_20) / sma_20 * 100
                if pct_from_sma > 5 and rsi < 70:
                    signal = 'BUY'
                    confidence = 'MEDIUM'
                    reasoning_parts.append(
                        f"Relative strength: Price {pct_from_sma:.1f}% above SMA(20), RSI={rsi:.1f}"
                    )
                elif pct_from_sma < -5 and rsi > 30:
                    signal = 'SELL'
                    confidence = 'MEDIUM'
                    reasoning_parts.append(
                        f"Relative weakness: Price {pct_from_sma:.1f}% below SMA(20), RSI={rsi:.1f}"
                    )
                else:
                    reasoning_parts.append(f"Sector unknown, no clear SMA momentum signal")
            return {'signal': signal, 'confidence': confidence, 'reasoning': '. '.join(reasoning_parts)}

        rel_perf = LiveSignalGenerator._get_sector_etf_performance(sector_etf)

        if rel_perf is None:
            return {
                'signal': 'HOLD',
                'confidence': 'LOW',
                'reasoning': f'Could not fetch sector ETF ({sector_etf}) data'
            }

        price_above_sma = (price is None or sma_20 is None or price > sma_20)
        price_below_sma = (price is None or sma_20 is None or price < sma_20)

        if rel_perf > 0.02 and rsi < 70 and price_above_sma:
            signal = 'BUY'
            confidence = 'HIGH' if rel_perf > 0.04 else 'MEDIUM'
            reasoning_parts.append(
                f"Strong sector {sector_etf}: +{rel_perf*100:.1f}% vs SPY (20d)"
            )
            reasoning_parts.append(f"RSI={rsi:.1f}, price above SMA(20)")
        elif rel_perf < -0.02 and rsi > 30 and price_below_sma:
            signal = 'SELL'
            confidence = 'HIGH' if rel_perf < -0.04 else 'MEDIUM'
            reasoning_parts.append(
                f"Weak sector {sector_etf}: {rel_perf*100:.1f}% vs SPY (20d)"
            )
            reasoning_parts.append(f"RSI={rsi:.1f}, price below SMA(20)")
        else:
            reasoning_parts.append(
                f"Sector {sector_etf}: {rel_perf*100:.2f}% vs SPY — insufficient divergence for signal"
            )

        return {
            'signal': signal,
            'confidence': confidence,
            'reasoning': '. '.join(reasoning_parts) if reasoning_parts else 'No sector rotation signal'
        }
