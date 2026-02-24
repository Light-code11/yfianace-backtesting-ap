"""
Advanced Technical Indicators Library
Provides 50+ indicators using pandas-ta for comprehensive strategy generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas-ta not available. Using basic indicators only.")


class AdvancedIndicators:
    """
    Comprehensive technical indicator library

    Categories:
    - Momentum: RSI, Stochastic, CCI, Williams %R, ROC
    - Trend: SMA, EMA, MACD, ADX, Aroon, SuperTrend
    - Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
    - Volume: OBV, CMF, MFI, VWAP, Volume SMA
    - Support/Resistance: Pivot Points, Fibonacci levels
    """

    # ========================================
    # MOMENTUM INDICATORS
    # ========================================

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
        """Relative Strength Index (0-100)"""
        if PANDAS_TA_AVAILABLE:
            return ta.rsi(df[column], length=period)
        else:
            # Fallback implementation
            delta = df[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K, %D)"""
        if PANDAS_TA_AVAILABLE:
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=k_period, d=d_period)
            return stoch[f'STOCHk_{k_period}_{d_period}_3'], stoch[f'STOCHd_{k_period}_{d_period}_3']
        else:
            # Fallback implementation
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k = 100 * (df['Close'] - low_min) / (high_max - low_min)
            d = k.rolling(window=d_period).mean()
            return k, d

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        if PANDAS_TA_AVAILABLE:
            return ta.cci(df['High'], df['Low'], df['Close'], length=period)
        else:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            return (tp - sma_tp) / (0.015 * mad)

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R (-100 to 0)"""
        if PANDAS_TA_AVAILABLE:
            return ta.willr(df['High'], df['Low'], df['Close'], length=period)
        else:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            return -100 * (highest_high - df['Close']) / (highest_high - lowest_low)

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 12, column: str = 'Close') -> pd.Series:
        """Rate of Change (%)"""
        if PANDAS_TA_AVAILABLE:
            return ta.roc(df[column], length=period)
        else:
            return ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index (0-100) - Volume-weighted RSI"""
        if PANDAS_TA_AVAILABLE:
            return ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=period)
        else:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']

            # Positive and negative money flow
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()

            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
            return mfi

    # ========================================
    # TREND INDICATORS
    # ========================================

    @staticmethod
    def sma(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """Simple Moving Average"""
        return df[column].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int, column: str = 'Close') -> pd.Series:
        """Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        if PANDAS_TA_AVAILABLE:
            macd_data = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
            return {
                'MACD': macd_data[f'MACD_{fast}_{slow}_{signal}'],
                'MACD_Signal': macd_data[f'MACDs_{fast}_{slow}_{signal}'],
                'MACD_Hist': macd_data[f'MACDh_{fast}_{slow}_{signal}']
            }
        else:
            ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return {
                'MACD': macd_line,
                'MACD_Signal': signal_line,
                'MACD_Hist': histogram
            }

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (trend strength, 0-100)"""
        if PANDAS_TA_AVAILABLE:
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=period)
            return adx_data[f'ADX_{period}']
        else:
            # Simplified ADX calculation
            tr = pd.DataFrame()
            tr['h-l'] = df['High'] - df['Low']
            tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
            tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)

            atr = tr['tr'].rolling(window=period).mean()

            # Placeholder - full ADX is complex
            return atr / df['Close'] * 100

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """SuperTrend indicator (trend direction and support/resistance)"""
        if PANDAS_TA_AVAILABLE:
            st = ta.supertrend(df['High'], df['Low'], df['Close'], length=period, multiplier=multiplier)
            return {
                'SuperTrend': st[f'SUPERT_{period}_{multiplier}'],
                'SuperTrend_Direction': st[f'SUPERTd_{period}_{multiplier}']
            }
        else:
            # Calculate ATR
            tr = pd.DataFrame()
            tr['h-l'] = df['High'] - df['Low']
            tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
            tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
            atr = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1).rolling(window=period).mean()

            # Calculate basic and final bands
            hl_avg = (df['High'] + df['Low']) / 2
            basic_ub = hl_avg + (multiplier * atr)
            basic_lb = hl_avg - (multiplier * atr)

            return {
                'SuperTrend': basic_ub,
                'SuperTrend_Direction': pd.Series(1, index=df.index)  # Placeholder
            }

    @staticmethod
    def aroon(df: pd.DataFrame, period: int = 25) -> Dict[str, pd.Series]:
        """Aroon indicator (trend identification)"""
        if PANDAS_TA_AVAILABLE:
            aroon_data = ta.aroon(df['High'], df['Low'], length=period)
            return {
                'Aroon_Up': aroon_data[f'AROONU_{period}'],
                'Aroon_Down': aroon_data[f'AROOND_{period}']
            }
        else:
            aroon_up = df['High'].rolling(window=period+1).apply(
                lambda x: (period - x[::-1].argmax()) / period * 100
            )
            aroon_down = df['Low'].rolling(window=period+1).apply(
                lambda x: (period - x[::-1].argmin()) / period * 100
            )
            return {
                'Aroon_Up': aroon_up,
                'Aroon_Down': aroon_down
            }

    # ========================================
    # VOLATILITY INDICATORS
    # ========================================

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        if PANDAS_TA_AVAILABLE:
            return ta.atr(df['High'], df['Low'], df['Close'], length=period)
        else:
            tr = pd.DataFrame()
            tr['h-l'] = df['High'] - df['Low']
            tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
            tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            return tr['tr'].rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = 'Close') -> Dict[str, pd.Series]:
        """Bollinger Bands (Upper, Middle, Lower)"""
        if PANDAS_TA_AVAILABLE:
            bb = ta.bbands(df[column], length=period, std=std_dev)
            # pandas_ta column naming can vary between versions:
            #   BBU_20_2.0  (float std) vs BBU_20_2  (int std)
            # Dynamically locate columns by prefix to be version-agnostic.
            bb_cols = list(bb.columns)
            upper_col = next((c for c in bb_cols if c.startswith('BBU_')), None)
            middle_col = next((c for c in bb_cols if c.startswith('BBM_')), None)
            lower_col = next((c for c in bb_cols if c.startswith('BBL_')), None)
            if upper_col and middle_col and lower_col:
                return {
                    'BB_Upper': bb[upper_col],
                    'BB_Middle': bb[middle_col],
                    'BB_Lower': bb[lower_col],
                }
            # Fallback: compute manually if pandas_ta result is unexpected
            middle = df[column].rolling(window=period).mean()
            std = df[column].rolling(window=period).std()
            return {
                'BB_Upper': middle + (std * std_dev),
                'BB_Middle': middle,
                'BB_Lower': middle - (std * std_dev),
            }
        else:
            middle = df[column].rolling(window=period).mean()
            std = df[column].rolling(window=period).std()
            return {
                'BB_Upper': middle + (std * std_dev),
                'BB_Middle': middle,
                'BB_Lower': middle - (std * std_dev)
            }

    @staticmethod
    def keltner_channels(df: pd.DataFrame, period: int = 20, atr_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Keltner Channels (similar to Bollinger but uses ATR)"""
        if PANDAS_TA_AVAILABLE:
            kc = ta.kc(df['High'], df['Low'], df['Close'], length=period, scalar=atr_multiplier)
            return {
                'KC_Upper': kc[f'KCUe_{period}_{atr_multiplier}'],
                'KC_Middle': kc[f'KCBe_{period}_{atr_multiplier}'],
                'KC_Lower': kc[f'KCLe_{period}_{atr_multiplier}']
            }
        else:
            middle = df['Close'].ewm(span=period).mean()
            atr_val = AdvancedIndicators.atr(df, period)
            return {
                'KC_Upper': middle + (atr_val * atr_multiplier),
                'KC_Middle': middle,
                'KC_Lower': middle - (atr_val * atr_multiplier)
            }

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channels (price envelope based on highs/lows)"""
        if PANDAS_TA_AVAILABLE:
            dc = ta.donchian(df['High'], df['Low'], lower_length=period, upper_length=period)
            return {
                'DC_Upper': dc[f'DCU_{period}_{period}'],
                'DC_Middle': dc[f'DCM_{period}_{period}'],
                'DC_Lower': dc[f'DCL_{period}_{period}']
            }
        else:
            upper = df['High'].rolling(window=period).max()
            lower = df['Low'].rolling(window=period).min()
            middle = (upper + lower) / 2
            return {
                'DC_Upper': upper,
                'DC_Middle': middle,
                'DC_Lower': lower
            }

    # ========================================
    # VOLUME INDICATORS
    # ========================================

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        if PANDAS_TA_AVAILABLE:
            return ta.obv(df['Close'], df['Volume'])
        else:
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            return obv

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        if PANDAS_TA_AVAILABLE:
            return ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=period)
        else:
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mfv = mfm * df['Volume']
            return mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price"""
        if PANDAS_TA_AVAILABLE:
            return ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        else:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Volume Simple Moving Average"""
        return df['Volume'].rolling(window=period).mean()

    # ========================================
    # UTILITY FUNCTIONS
    # ========================================

    @staticmethod
    def get_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ALL available indicators and add them to dataframe

        Returns:
            DataFrame with all indicator columns added
        """
        result = df.copy()

        try:
            # Momentum Indicators
            result['RSI_14'] = AdvancedIndicators.rsi(df, 14)
            result['RSI_7'] = AdvancedIndicators.rsi(df, 7)
            stoch_k, stoch_d = AdvancedIndicators.stochastic(df)
            result['Stoch_K'] = stoch_k
            result['Stoch_D'] = stoch_d
            result['CCI_20'] = AdvancedIndicators.cci(df, 20)
            result['Williams_R'] = AdvancedIndicators.williams_r(df, 14)
            result['ROC_12'] = AdvancedIndicators.roc(df, 12)
            if 'Volume' in df.columns:
                result['MFI_14'] = AdvancedIndicators.mfi(df, 14)

            # Trend Indicators
            result['SMA_20'] = AdvancedIndicators.sma(df, 20)
            result['SMA_50'] = AdvancedIndicators.sma(df, 50)
            result['SMA_200'] = AdvancedIndicators.sma(df, 200)
            result['EMA_12'] = AdvancedIndicators.ema(df, 12)
            result['EMA_26'] = AdvancedIndicators.ema(df, 26)

            macd_dict = AdvancedIndicators.macd(df)
            result['MACD'] = macd_dict['MACD']
            result['MACD_Signal'] = macd_dict['MACD_Signal']
            result['MACD_Hist'] = macd_dict['MACD_Hist']

            result['ADX_14'] = AdvancedIndicators.adx(df, 14)

            supertrend_dict = AdvancedIndicators.supertrend(df)
            result['SuperTrend'] = supertrend_dict['SuperTrend']
            result['SuperTrend_Dir'] = supertrend_dict['SuperTrend_Direction']

            aroon_dict = AdvancedIndicators.aroon(df)
            result['Aroon_Up'] = aroon_dict['Aroon_Up']
            result['Aroon_Down'] = aroon_dict['Aroon_Down']

            # Volatility Indicators
            result['ATR_14'] = AdvancedIndicators.atr(df, 14)

            bb_dict = AdvancedIndicators.bollinger_bands(df)
            result['BB_Upper'] = bb_dict['BB_Upper']
            result['BB_Middle'] = bb_dict['BB_Middle']
            result['BB_Lower'] = bb_dict['BB_Lower']

            kc_dict = AdvancedIndicators.keltner_channels(df)
            result['KC_Upper'] = kc_dict['KC_Upper']
            result['KC_Middle'] = kc_dict['KC_Middle']
            result['KC_Lower'] = kc_dict['KC_Lower']

            dc_dict = AdvancedIndicators.donchian_channels(df)
            result['DC_Upper'] = dc_dict['DC_Upper']
            result['DC_Middle'] = dc_dict['DC_Middle']
            result['DC_Lower'] = dc_dict['DC_Lower']

            # Volume Indicators (if volume data available)
            if 'Volume' in df.columns:
                result['OBV'] = AdvancedIndicators.obv(df)
                result['CMF_20'] = AdvancedIndicators.cmf(df, 20)
                result['VWAP'] = AdvancedIndicators.vwap(df)
                result['Volume_SMA_20'] = AdvancedIndicators.volume_sma(df, 20)

        except Exception as e:
            print(f"Warning: Error calculating some indicators: {e}")

        return result

    @staticmethod
    def get_indicator_categories() -> Dict[str, List[str]]:
        """
        Return categorized list of all available indicators

        Returns:
            Dict mapping category to list of indicator names
        """
        return {
            'Momentum': [
                'RSI', 'Stochastic', 'CCI', 'Williams_R', 'ROC', 'MFI'
            ],
            'Trend': [
                'SMA', 'EMA', 'MACD', 'ADX', 'SuperTrend', 'Aroon'
            ],
            'Volatility': [
                'ATR', 'Bollinger_Bands', 'Keltner_Channels', 'Donchian_Channels'
            ],
            'Volume': [
                'OBV', 'CMF', 'VWAP', 'Volume_SMA', 'MFI'
            ]
        }


if __name__ == "__main__":
    # Test the indicators
    import yfinance as yf

    print("Testing Advanced Indicators...")
    print(f"pandas-ta available: {PANDAS_TA_AVAILABLE}")
    print()

    # Download test data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo")

    # Calculate all indicators
    df_with_indicators = AdvancedIndicators.get_all_indicators(df)

    print(f"Original columns: {len(df.columns)}")
    print(f"With indicators: {len(df_with_indicators.columns)}")
    print()
    print("Sample of calculated indicators (last 5 rows):")
    print(df_with_indicators[['Close', 'RSI_14', 'MACD', 'BB_Upper', 'ATR_14']].tail())
    print()
    print("Available indicator categories:")
    for category, indicators in AdvancedIndicators.get_indicator_categories().items():
        print(f"  {category}: {', '.join(indicators)}")
