import yfinance as yf
import pandas as pd

tickers = ['CAT', 'AMD', 'CLS', 'HD', 'WFC', 'VZ']

for ticker in tickers:
    try:
        data = yf.download(ticker, period='1d', progress=False)

        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if not data.empty:
            current_price = float(data['Close'].iloc[-1])
            print(f"{ticker}: ${current_price:.2f}")
    except Exception as e:
        print(f"{ticker}: Error - {str(e)}")
