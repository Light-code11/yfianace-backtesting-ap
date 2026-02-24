# Trading Strategy Backtest Report
Generated: 2026-02-24 22:44

## Methodology
- **Data**: 3 years daily OHLCV via yfinance
- **Train window**: 252 days (1 year) — in-sample
- **Test window**: 63 days (3 months) — out-of-sample
- **Position size**: $10,000 per trade, fixed
- **Commission**: $1 per trade (Alpaca)
- **Slippage**: NOT modelled (limitation — results may be optimistic)
- **Signals**: Technical indicators computed on close prices

## Summary

| Strategy | Trades | Win Rate | Sharpe | Max DD | Profit Factor | Overfit Risk |
|----------|--------|----------|--------|--------|---------------|--------------|
| RSI Mean Reversion | 1690 | 42% | -0.08 | -100.5% | 0.99 | HIGH |
| MACD Trend Following | 986 | 31% | -0.82 | -100.1% | 0.90 | HIGH |
| SMA Momentum | 1558 | 35% | -0.18 | -100.0% | 0.98 | HIGH |
| Volume Breakout | 1291 | 29% | -1.22 | -100.1% | 0.82 | HIGH |
| Bollinger Band Squeeze | 345 | 26% | -2.38 | -99.8% | 0.73 | HIGH |
| VWAP Reversion | 1597 | 37% | -0.35 | -100.0% | 0.95 | HIGH |
| Gap Fill | 2316 | 40% | -1.31 | -100.0% | 0.82 | HIGH |
| Earnings Momentum | 1392 | 37% | -0.28 | -101.4% | 0.97 | HIGH |
| Sector Rotation | 1750 | 35% | -1.89 | -100.0% | 0.83 | HIGH |

## Overfitting Analysis

Strategies where out-of-sample Sharpe < 50% of in-sample Sharpe are flagged **HIGH** risk.

| Strategy | IS Sharpe | OOS Sharpe | OOS/IS Ratio | Overfit Risk |
|----------|-----------|------------|--------------|--------------|
| RSI Mean Reversion | -0.11 | -0.08 | 0.71 | HIGH |
| MACD Trend Following | 0.13 | -0.82 | -6.40 | HIGH |
| SMA Momentum | 0.19 | -0.18 | -0.93 | HIGH |
| Volume Breakout | -0.16 | -1.22 | 7.59 | HIGH |
| Bollinger Band Squeeze | -1.19 | -2.38 | 2.00 | HIGH |
| VWAP Reversion | -0.25 | -0.35 | 1.43 | HIGH |
| Gap Fill | -1.17 | -1.31 | 1.12 | HIGH |
| Earnings Momentum | 1.07 | -0.28 | -0.26 | HIGH |
| Sector Rotation | -1.01 | -1.89 | 1.87 | HIGH |

## Per-Strategy Detail

### RSI Mean Reversion

- **Total Trades**: 1690
- **Win Rate**: 42.5%
- **Avg Return/Trade**: -0.02%
- **Sharpe (OOS)**: -0.08
- **Max Drawdown**: -100.5%
- **Profit Factor**: 0.99
- **Avg Trade Duration**: 16 days
- **Total P&L**: $-6,372
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 203 | 34% | -4.63 | -99.4% |
| W2 2024-04-29â†’2024-07-29 | 210 | 45% | -0.30 | -88.8% |
| W3 2024-07-30â†’2024-10-25 | 191 | 54% | 6.40 | -60.8% |
| W4 2024-10-28â†’2025-01-29 | 219 | 43% | -2.15 | -97.8% |
| W5 2025-01-30â†’2025-04-30 | 246 | 45% | 1.29 | -142.3% |
| W6 2025-05-01â†’2025-07-31 | 202 | 30% | -5.02 | -99.6% |
| W7 2025-08-01â†’2025-10-29 | 200 | 45% | -0.25 | -90.4% |
| W8 2025-10-30â†’2026-01-30 | 219 | 44% | 0.84 | -75.5% |

- **Best period**: W3 2024-07-30â†’2024-10-25 (Sharpe 6.40)
- **Worst period**: W6 2025-05-01â†’2025-07-31 (Sharpe -5.02)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### MACD Trend Following

- **Total Trades**: 986
- **Win Rate**: 30.9%
- **Avg Return/Trade**: -0.47%
- **Sharpe (OOS)**: -0.82
- **Max Drawdown**: -100.1%
- **Profit Factor**: 0.90
- **Avg Trade Duration**: 19 days
- **Total P&L**: $-47,979
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 127 | 34% | 0.23 | -91.4% |
| W2 2024-04-29â†’2024-07-29 | 113 | 32% | -0.93 | -89.1% |
| W3 2024-07-30â†’2024-10-25 | 89 | 29% | -1.57 | -88.3% |
| W4 2024-10-28â†’2025-01-29 | 133 | 38% | 2.36 | -81.0% |
| W5 2025-01-30â†’2025-04-30 | 170 | 21% | -4.25 | -114.9% |
| W6 2025-05-01â†’2025-07-31 | 97 | 34% | 1.18 | -61.9% |
| W7 2025-08-01â†’2025-10-29 | 128 | 35% | 1.20 | -84.7% |
| W8 2025-10-30â†’2026-01-30 | 129 | 27% | -2.31 | -97.4% |

- **Best period**: W4 2024-10-28â†’2025-01-29 (Sharpe 2.36)
- **Worst period**: W5 2025-01-30â†’2025-04-30 (Sharpe -4.25)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### SMA Momentum

- **Total Trades**: 1558
- **Win Rate**: 34.5%
- **Avg Return/Trade**: -0.07%
- **Sharpe (OOS)**: -0.18
- **Max Drawdown**: -100.0%
- **Profit Factor**: 0.98
- **Avg Trade Duration**: 18 days
- **Total P&L**: $-14,403
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 181 | 39% | 1.35 | -77.1% |
| W2 2024-04-29â†’2024-07-29 | 183 | 30% | -2.71 | -97.7% |
| W3 2024-07-30â†’2024-10-25 | 195 | 33% | -1.67 | -98.4% |
| W4 2024-10-28â†’2025-01-29 | 214 | 36% | 0.75 | -99.2% |
| W5 2025-01-30â†’2025-04-30 | 266 | 34% | -0.85 | -181.2% |
| W6 2025-05-01â†’2025-07-31 | 157 | 42% | 3.44 | -81.2% |
| W7 2025-08-01â†’2025-10-29 | 170 | 34% | 0.75 | -85.4% |
| W8 2025-10-30â†’2026-01-30 | 192 | 31% | -1.49 | -93.8% |

- **Best period**: W6 2025-05-01â†’2025-07-31 (Sharpe 3.44)
- **Worst period**: W2 2024-04-29â†’2024-07-29 (Sharpe -2.71)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### Volume Breakout

- **Total Trades**: 1291
- **Win Rate**: 29.0%
- **Avg Return/Trade**: -0.73%
- **Sharpe (OOS)**: -1.22
- **Max Drawdown**: -100.1%
- **Profit Factor**: 0.82
- **Avg Trade Duration**: 16 days
- **Total P&L**: $-96,311
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 125 | 36% | 1.53 | -65.8% |
| W2 2024-04-29â†’2024-07-29 | 149 | 28% | -1.37 | -88.8% |
| W3 2024-07-30â†’2024-10-25 | 156 | 20% | -6.48 | -98.6% |
| W4 2024-10-28â†’2025-01-29 | 184 | 26% | -3.47 | -96.0% |
| W5 2025-01-30â†’2025-04-30 | 205 | 34% | -0.67 | -449.4% |
| W6 2025-05-01â†’2025-07-31 | 152 | 36% | 0.33 | -72.1% |
| W7 2025-08-01â†’2025-10-29 | 147 | 26% | -1.62 | -84.2% |
| W8 2025-10-30â†’2026-01-30 | 173 | 28% | -2.82 | -93.7% |

- **Best period**: W1 2024-01-29â†’2024-04-26 (Sharpe 1.53)
- **Worst period**: W3 2024-07-30â†’2024-10-25 (Sharpe -6.48)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### Bollinger Band Squeeze

- **Total Trades**: 345
- **Win Rate**: 26.1%
- **Avg Return/Trade**: -1.18%
- **Sharpe (OOS)**: -2.38
- **Max Drawdown**: -99.8%
- **Profit Factor**: 0.73
- **Avg Trade Duration**: 16 days
- **Total P&L**: $-41,404
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 46 | 26% | -4.69 | -56.0% |
| W2 2024-04-29â†’2024-07-29 | 42 | 21% | -2.13 | -70.0% |
| W3 2024-07-30â†’2024-10-25 | 40 | 12% | -7.41 | -81.4% |
| W4 2024-10-28â†’2025-01-29 | 51 | 31% | -2.56 | -64.7% |
| W5 2025-01-30â†’2025-04-30 | 30 | 37% | 0.96 | -54.8% |
| W6 2025-05-01â†’2025-07-31 | 46 | 41% | 4.15 | -52.0% |
| W7 2025-08-01â†’2025-10-29 | 48 | 15% | -8.34 | -76.1% |
| W8 2025-10-30â†’2026-01-30 | 42 | 26% | -3.36 | -69.1% |

- **Best period**: W6 2025-05-01â†’2025-07-31 (Sharpe 4.15)
- **Worst period**: W7 2025-08-01â†’2025-10-29 (Sharpe -8.34)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### VWAP Reversion

- **Total Trades**: 1597
- **Win Rate**: 36.6%
- **Avg Return/Trade**: -0.15%
- **Sharpe (OOS)**: -0.35
- **Max Drawdown**: -100.0%
- **Profit Factor**: 0.95
- **Avg Trade Duration**: 13 days
- **Total P&L**: $-27,518
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 234 | 32% | -3.10 | -99.4% |
| W2 2024-04-29â†’2024-07-29 | 171 | 39% | -0.33 | -87.1% |
| W3 2024-07-30â†’2024-10-25 | 147 | 37% | 0.78 | -62.8% |
| W4 2024-10-28â†’2025-01-29 | 212 | 40% | -0.18 | -98.6% |
| W5 2025-01-30â†’2025-04-30 | 247 | 44% | 1.26 | -100.3% |
| W6 2025-05-01â†’2025-07-31 | 195 | 24% | -6.08 | -99.3% |
| W7 2025-08-01â†’2025-10-29 | 186 | 40% | 0.95 | -78.7% |
| W8 2025-10-30â†’2026-01-30 | 205 | 36% | -0.33 | -85.3% |

- **Best period**: W5 2025-01-30â†’2025-04-30 (Sharpe 1.26)
- **Worst period**: W6 2025-05-01â†’2025-07-31 (Sharpe -6.08)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### Gap Fill

- **Total Trades**: 2316
- **Win Rate**: 40.3%
- **Avg Return/Trade**: -0.54%
- **Sharpe (OOS)**: -1.31
- **Max Drawdown**: -100.0%
- **Profit Factor**: 0.82
- **Avg Trade Duration**: 6 days
- **Total P&L**: $-129,247
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 276 | 49% | 2.67 | -82.3% |
| W2 2024-04-29â†’2024-07-29 | 203 | 37% | -4.11 | -95.4% |
| W3 2024-07-30â†’2024-10-25 | 266 | 37% | -2.71 | -98.6% |
| W4 2024-10-28â†’2025-01-29 | 289 | 39% | -1.86 | -97.6% |
| W5 2025-01-30â†’2025-04-30 | 555 | 36% | -3.18 | -137.9% |
| W6 2025-05-01â†’2025-07-31 | 269 | 50% | 2.86 | -83.6% |
| W7 2025-08-01â†’2025-10-29 | 191 | 38% | -2.14 | -80.4% |
| W8 2025-10-30â†’2026-01-30 | 267 | 39% | -1.20 | -97.4% |

- **Best period**: W6 2025-05-01â†’2025-07-31 (Sharpe 2.86)
- **Worst period**: W2 2024-04-29â†’2024-07-29 (Sharpe -4.11)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### Earnings Momentum

- **Total Trades**: 1392
- **Win Rate**: 37.1%
- **Avg Return/Trade**: -0.12%
- **Sharpe (OOS)**: -0.28
- **Max Drawdown**: -101.4%
- **Profit Factor**: 0.97
- **Avg Trade Duration**: 14 days
- **Total P&L**: $-19,554
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 132 | 33% | -2.28 | -88.4% |
| W2 2024-04-29â†’2024-07-29 | 135 | 34% | -0.91 | -93.4% |
| W3 2024-07-30â†’2024-10-25 | 170 | 36% | -0.11 | -80.6% |
| W4 2024-10-28â†’2025-01-29 | 166 | 32% | -0.83 | -93.8% |
| W5 2025-01-30â†’2025-04-30 | 325 | 46% | 1.56 | -307.2% |
| W6 2025-05-01â†’2025-07-31 | 163 | 37% | -0.12 | -99.9% |
| W7 2025-08-01â†’2025-10-29 | 137 | 36% | -2.24 | -91.0% |
| W8 2025-10-30â†’2026-01-30 | 164 | 32% | -1.34 | -96.4% |

- **Best period**: W5 2025-01-30â†’2025-04-30 (Sharpe 1.56)
- **Worst period**: W1 2024-01-29â†’2024-04-26 (Sharpe -2.28)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

### Sector Rotation

- **Total Trades**: 1750
- **Win Rate**: 34.6%
- **Avg Return/Trade**: -0.69%
- **Sharpe (OOS)**: -1.89
- **Max Drawdown**: -100.0%
- **Profit Factor**: 0.83
- **Avg Trade Duration**: 17 days
- **Total P&L**: $-124,240
- **Overfitting Risk**: HIGH

**Walk-Forward Windows:**

| Window | Trades | Win Rate | Sharpe | Max DD |
|--------|--------|----------|--------|--------|
| W1 2024-01-29â†’2024-04-26 | 197 | 36% | -0.15 | -98.4% |
| W2 2024-04-29â†’2024-07-29 | 223 | 33% | -4.08 | -99.1% |
| W3 2024-07-30â†’2024-10-25 | 225 | 40% | -0.22 | -81.5% |
| W4 2024-10-28â†’2025-01-29 | 230 | 38% | 0.04 | -96.0% |
| W5 2025-01-30â†’2025-04-30 | 284 | 29% | -5.56 | -100.0% |
| W6 2025-05-01â†’2025-07-31 | 168 | 39% | 0.91 | -86.1% |
| W7 2025-08-01â†’2025-10-29 | 167 | 27% | -4.05 | -98.7% |
| W8 2025-10-30â†’2026-01-30 | 256 | 36% | -4.11 | -97.2% |

- **Best period**: W6 2025-05-01â†’2025-07-31 (Sharpe 0.91)
- **Worst period**: W5 2025-01-30â†’2025-04-30 (Sharpe -5.56)

**Recommendation**: Consider looser parameters or eliminating this strategy. Significant performance degradation out-of-sample detected.

## Pair Trading Results

| Pair | Trades | Win Rate | Avg Return | Sharpe | Cointegration Stable? | Overfit Risk |
|------|--------|----------|------------|--------|-----------------------|--------------|
| V/MA | 12 | 25% | -1.69% | -4.46 | No | HIGH |
| GOOGL/META | 30 | 30% | -1.45% | -2.71 | No | HIGH |
| XOM/CVX | 13 | 23% | -1.69% | -3.98 | Yes | HIGH |
| COST/WMT | 16 | 31% | -0.66% | -1.48 | No | HIGH |
| JPM/BAC | 14 | 29% | -1.18% | -2.60 | No | HIGH |
| AMD/INTC | 27 | 37% | 0.21% | 0.28 | Yes | LOW |
| AAPL/MSFT | 18 | 33% | -1.06% | -2.34 | Yes | HIGH |
| LMT/RTX | 17 | 29% | -0.90% | -2.07 | Yes | HIGH |
| CRWD/PANW | 27 | 41% | 0.80% | 1.26 | No | LOW |
| DDOG/NET | 30 | 37% | 0.90% | 1.53 | Yes | HIGH |

## Recommendations

### Kill (negative Sharpe or HIGH overfit risk)
- RSI Mean Reversion
- MACD Trend Following
- SMA Momentum
- Volume Breakout
- Bollinger Band Squeeze
- VWAP Reversion
- Gap Fill
- Earnings Momentum
- Sector Rotation

### Active Pairs (positive Sharpe)
- AMD/INTC
- CRWD/PANW
- DDOG/NET

### Inactive Pairs (non-positive Sharpe)
- V/MA
- GOOGL/META
- XOM/CVX
- COST/WMT
- JPM/BAC
- AAPL/MSFT
- LMT/RTX

---
*Note: Results are approximations. Slippage not modelled. Past performance does not guarantee future results.*
