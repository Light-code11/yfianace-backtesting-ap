import yfinance as yf
import pandas as pd

tickers = ['GOOGL', 'MCD']

for ticker in tickers:
    data = yf.download(ticker, period='1d', progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    current_price = float(data['Close'].iloc[-1])
    print(f"{ticker}: ${current_price:.2f}")
