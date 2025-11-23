import yfinance as yf
import pandas as pd

# Check the tickers from your Breakout strategies
tickers = ['AMD', 'CLS', 'COIN']

for ticker in tickers:
    data = yf.download(ticker, period='3mo', progress=False)

    # Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    current_price = float(data['Close'].iloc[-1])
    print(f"{ticker}: ${current_price:.2f}")

print("\n" + "=" * 50)
print("Is $340.20 close to AMD or CLS?")
print("=" * 50)
