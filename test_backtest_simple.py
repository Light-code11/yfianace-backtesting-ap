"""
Simple test to verify backtesting engine works correctly
"""
import yfinance as yf
import pandas as pd
from backtesting_engine import BacktestingEngine

# Download SPY data for 1 year
print("Downloading SPY data...")
spy_data = yf.download("SPY", period="1y", progress=False)

# Check if MultiIndex
print(f"\nSPY data shape: {spy_data.shape}")
print(f"Columns: {spy_data.columns.tolist()}")
print(f"Is MultiIndex: {isinstance(spy_data.columns, pd.MultiIndex)}")

# If not MultiIndex, convert to MultiIndex format expected by backtesting engine
if not isinstance(spy_data.columns, pd.MultiIndex):
    print("\nConverting to MultiIndex...")
    spy_data.columns = pd.MultiIndex.from_product([spy_data.columns, ['SPY']])
    print(f"New columns: {spy_data.columns.tolist()[:5]}")

# Create a simple momentum strategy
strategy_config = {
    "name": "Test Momentum",
    "tickers": ["SPY"],
    "strategy_type": "momentum",
    "indicators": [
        {"name": "SMA", "period": 20},
        {"name": "SMA", "period": 50},
        {"name": "RSI", "period": 14}
    ],
    "risk_management": {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "position_size_pct": 100.0,  # Use all capital
        "max_positions": 1
    }
}

# Run backtest
print("\nRunning backtest...")
engine = BacktestingEngine(initial_capital=100000)
results = engine.backtest_strategy(strategy_config, spy_data)

# Print results
print(f"\nBacktest Results:")
print(f"Strategy: {results['strategy_name']}")
print(f"Total Trades: {len(results['trades'])}")
print(f"Metrics:")
for key, value in results['metrics'].items():
    print(f"  {key}: {value}")

# Show first few trades
if results['trades']:
    print(f"\nFirst 5 Trades:")
    for i, trade in enumerate(results['trades'][:5]):
        print(f"\n  Trade {i+1}:")
        for k, v in trade.items():
            print(f"    {k}: {v}")

# Buy and hold comparison
spy_start = spy_data['Close']['SPY'].iloc[0]
spy_end = spy_data['Close']['SPY'].iloc[-1]
buy_hold_return = ((spy_end - spy_start) / spy_start) * 100

print(f"\n\nBuy & Hold Comparison:")
print(f"SPY Buy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy Return: {results['metrics']['total_return_pct']:.2f}%")
print(f"Difference: {results['metrics']['total_return_pct'] - buy_hold_return:.2f}%")

if results['metrics']['total_return_pct'] < buy_hold_return - 10:
    print("\n⚠️ WARNING: Strategy underperforming buy-and-hold by >10%!")
    print("This suggests signals might be inverted or logic is flawed.")
