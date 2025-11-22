"""
Test script for market context and relative strength features
"""
import yfinance as yf
import pandas as pd
from ai_strategy_generator import AIStrategyGenerator, BENCHMARK_MAP

def test_market_context():
    print("Testing Market Context Features")
    print("="*60)

    # Test data
    tickers = ['NVDA', 'AAPL']
    benchmarks = {'SPY', 'QQQ', 'SOXX'}  # SOXX = iShares Semiconductor ETF
    all_tickers = list(set(tickers) | benchmarks)

    print(f"\n1. Testing Benchmark Mapping:")
    print(f"   User tickers: {tickers}")
    print(f"   Benchmarks added: {benchmarks}")
    print(f"   All tickers to fetch: {all_tickers}")

    for ticker in tickers:
        if ticker in BENCHMARK_MAP:
            sector, market = BENCHMARK_MAP[ticker]
            print(f"   {ticker} -> Sector: {sector}, Market: {market}")

    # Fetch test data
    print(f"\n2. Fetching market data...")
    ticker_string = " ".join(all_tickers)
    market_data = yf.download(ticker_string, period="3mo", progress=False)
    print(f"   Data shape: {market_data.shape}")
    print(f"   Available tickers: {market_data['Close'].columns.tolist()}")

    # Test AI generator (skip if no API key for local test)
    print(f"\n3. Testing Market Analysis (without OpenAI)...")
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("   Note: Skipping OpenAI client initialization (no API key)")
        # Manually create a test instance without initializing OpenAI
        class TestGenerator:
            def _detect_market_regime(self, market_data):
                from ai_strategy_generator import AIStrategyGenerator
                temp = object.__new__(AIStrategyGenerator)
                return temp._detect_market_regime(market_data)

            def _calculate_relative_strength(self, ticker, ticker_prices, market_data):
                from ai_strategy_generator import AIStrategyGenerator
                temp = object.__new__(AIStrategyGenerator)
                return temp._calculate_relative_strength(ticker, ticker_prices, market_data)

            def _analyze_market_data(self, market_data, tickers):
                from ai_strategy_generator import AIStrategyGenerator
                temp = object.__new__(AIStrategyGenerator)
                return temp._analyze_market_data(market_data, tickers)

        ai_gen = TestGenerator()
    else:
        ai_gen = AIStrategyGenerator()

    # Test market regime detection
    print(f"\n   a) Market Regime Detection:")
    regime = ai_gen._detect_market_regime(market_data)
    print(f"      Current market regime: {regime}")

    # Test relative strength analysis
    print(f"\n   b) Relative Strength Analysis:")
    for ticker in tickers:
        if ticker in market_data['Close'].columns:
            ticker_prices = market_data['Close'][ticker]
            rs = ai_gen._calculate_relative_strength(ticker, ticker_prices, market_data)

            if rs:
                print(f"\n      {ticker}:")
                if 'vs_sector' in rs:
                    vs_sector = rs['vs_sector']
                    print(f"        vs {vs_sector['benchmark']}: {vs_sector['strength']} ({vs_sector['outperformance']:+.1f}%)")
                if 'vs_market' in rs:
                    vs_market = rs['vs_market']
                    print(f"        vs {vs_market['benchmark']}: {vs_market['strength']} ({vs_market['outperformance']:+.1f}%)")
            else:
                print(f"      {ticker}: No relative strength data")

    # Test market analysis
    print(f"\n   c) Full Market Analysis:")
    analysis = ai_gen._analyze_market_data(market_data, tickers)
    print(f"      Period: {analysis['period']['start']} to {analysis['period']['end']}")
    print(f"      Market Regime: {analysis['market_regime']}")

    for ticker, stats in analysis['market_stats'].items():
        print(f"\n      {ticker}:")
        print(f"        Return: {stats['total_return_pct']:.2f}%")
        print(f"        Volatility: {stats['volatility_pct']:.2f}%")
        print(f"        Trend: {stats['trend']}")

        if 'relative_strength' in stats:
            print(f"        Relative Strength: ✅ Included")
        else:
            print(f"        Relative Strength: ❌ Not available")

    print(f"\n{'='*60}")
    print("✅ Market Context Test Complete!")
    print("\nNext steps:")
    print("1. Deploy these changes to Railway")
    print("2. Generate new strategies with market context")
    print("3. Monitor if strategies are market-aware")

if __name__ == "__main__":
    test_market_context()
