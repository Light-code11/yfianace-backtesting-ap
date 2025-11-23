import yfinance as yf
import pandas as pd

print("=" * 70)
print("TEST 1: Download single ticker (COIN)")
print("=" * 70)
data1 = yf.download('COIN', period='3mo', progress=False)
print(f"Columns type: {type(data1.columns)}")
print(f"Columns: {data1.columns}")
print(f"Is MultiIndex: {isinstance(data1.columns, pd.MultiIndex)}")

if isinstance(data1.columns, pd.MultiIndex):
    print(f"MultiIndex levels: {data1.columns.levels}")
    print(f"MultiIndex names: {data1.columns.names}")
    data1.columns = data1.columns.get_level_values(0)
    print(f"After flattening: {data1.columns}")

print(f"Latest Close: ${float(data1['Close'].iloc[-1]):.2f}")

print("\n" + "=" * 70)
print("TEST 2: Download multiple tickers (AMD, CLS)")
print("=" * 70)
data2 = yf.download(['AMD', 'CLS'], period='3mo', progress=False)
print(f"Columns type: {type(data2.columns)}")
print(f"Columns: {data2.columns}")
print(f"Is MultiIndex: {isinstance(data2.columns, pd.MultiIndex)}")

if isinstance(data2.columns, pd.MultiIndex):
    print(f"MultiIndex levels: {data2.columns.levels}")
    print(f"MultiIndex names: {data2.columns.names}")
