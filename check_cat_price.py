import yfinance as yf
import pandas as pd

# Check CAT price
data = yf.download('CAT', period='1d', progress=False)

# Handle MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

current_price = float(data['Close'].iloc[-1])
print(f"CAT current price: ${current_price:.2f}")
