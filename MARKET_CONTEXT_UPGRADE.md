# Market Context & Relative Strength Upgrade

## Overview
Upgraded the AI trading strategy generator with professional-grade market analysis features used by hedge funds:
- **Market Regime Detection**: Identifies bull/bear/sideways market conditions
- **Relative Strength Analysis**: Compares stock performance vs sector and market benchmarks
- **Context-Aware AI**: Strategies now adapt to market conditions

## What Was Implemented

### 1. Benchmark Mapping (ai_strategy_generator.py)
Added BENCHMARK_MAP dictionary that maps stocks to their sector and market benchmarks:

**Semiconductors** → SOXX (sector) + SPY (market)
- NVDA, AMD, INTC, TSM, AVGO

**Big Tech** → QQQ (sector) + SPY (market)
- AAPL, MSFT, GOOGL, META, AMZN, NFLX, TSLA

**ETFs** → SPY only (already diversified)
- QQQ, SPY, DIA, IWM

### 2. Market Regime Detection
Analyzes SPY (S&P 500) to determine current market environment:
- **strong_bull**: Price > SMA(200), 20-day return > 2%
- **bull**: Price > SMA(50), positive momentum
- **sideways**: Choppy, range-bound conditions
- **bear**: Price < SMA(50), negative momentum
- **strong_bear**: Price < SMA(200), 20-day return < -2%

### 3. Relative Strength Analysis
For each stock, calculates:
- **Performance vs Sector**: How stock performs vs its sector ETF
- **Performance vs Market**: How stock performs vs broad market (SPY)
- **Outperformance %**: Exact difference in returns
- **Strength Status**: Outperforming or underperforming

**Example from test:**
```
NVDA:
  vs SOXX: underperforming (-8.4%)
  vs SPY: underperforming (-1.9%)

AAPL:
  vs QQQ: outperforming (+16.0%)
  vs SPY: outperforming (+16.9%)
```

### 4. Enhanced AI Prompts
AI now receives:
- **Market Regime Header**: Emphasized at top of prompt
- **Relative Strength Data**: For each ticker
- **Strategy Guidelines**: Specific rules for each market regime
  - Bull markets: Favor momentum and breakouts
  - Bear markets: Defensive, tight stops
  - Sideways: Mean reversion, range-bound
- **Relative Strength Rules**:
  - Outperforming stocks: Favor long momentum
  - Underperforming stocks: More conservative, tighter stops

### 5. Automatic Benchmark Fetching
Modified data fetching in:
- `trading_platform_api.py`: Automatically adds SPY, QQQ, SOXX to all data requests
- `autonomous_learning.py`: Same for autonomous learning cycles

## Files Modified

1. **ai_strategy_generator.py**
   - Added BENCHMARK_MAP (28 stocks/ETFs mapped)
   - Added `_detect_market_regime()` method
   - Added `_calculate_relative_strength()` method
   - Enhanced `_analyze_market_data()` to include regime + relative strength
   - Updated `_build_strategy_prompt()` with market context

2. **trading_platform_api.py**
   - Updated `fetch_market_data()` to include benchmarks

3. **autonomous_learning.py**
   - Updated `fetch_market_data()` to include benchmarks

4. **test_market_context.py** (NEW)
   - Comprehensive test script
   - Validates all market context features

## How It Helps

### Before:
AI generated strategies blindly, without understanding:
- Is the market in a bull or bear phase?
- Is NVDA doing better or worse than the semiconductor sector?
- Should we be aggressive or defensive?

### After:
AI now knows:
- **Market Regime**: "We're in a bear market, use defensive strategies"
- **Relative Strength**: "AAPL is outperforming QQQ by 16%, favor momentum trades"
- **Sector Context**: "NVDA is underperforming semiconductors, be cautious"

This leads to:
- **Better entry timing**: Only go long on strong stocks in bull markets
- **Better risk management**: Tighter stops in bear markets
- **Better stock selection**: Focus on stocks with positive relative strength

## Test Results

```
✅ Market Regime Detection: bear
✅ Relative Strength Analysis:
   - NVDA: underperforming SOXX (-8.4%), underperforming SPY (-1.9%)
   - AAPL: outperforming QQQ (+16.0%), outperforming SPY (+16.9%)
✅ Full Market Analysis: Includes regime + relative strength
```

## Next Steps

1. **Deploy to Railway**: Push these changes
2. **Generate New Strategies**: With market context
3. **Monitor Performance**: See if strategies improve

The AI should now generate strategies like:
- "RSI Mean Reversion on AAPL (outperforming QQQ)" - more aggressive
- "Defensive Momentum on NVDA (underperforming SOXX)" - tighter stops

## Technical Notes

- Fixed SOX → SOXX (SOX doesn't exist, SOXX is iShares Semiconductor ETF)
- Automatically fetches 3 benchmark ETFs: SPY, QQQ, SOXX
- Graceful fallback if benchmarks fail to fetch
- All calculations handle missing data safely
