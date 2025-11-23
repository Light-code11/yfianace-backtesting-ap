import yfinance as yf
import pandas as pd

# Check COIN price
data = yf.download('COIN', period='1d', progress=False)

# Handle MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

current_price = float(data['Close'].iloc[-1])
print(f"COIN current price: ${current_price:.2f}")

# Also check with 3mo period (what the scanner uses)
data_3mo = yf.download('COIN', period='3mo', progress=False)
if isinstance(data_3mo.columns, pd.MultiIndex):
    data_3mo.columns = data_3mo.columns.get_level_values(0)

latest_3mo = float(data_3mo['Close'].iloc[-1])
print(f"COIN latest (3mo period): ${latest_3mo:.2f}")
