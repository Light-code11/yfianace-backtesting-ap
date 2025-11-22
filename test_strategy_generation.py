"""
Test script to debug strategy generation error
"""
import os
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key loaded: {api_key[:20] if api_key else 'NOT FOUND'}...")

# Test imports
try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except Exception as e:
    print(f"✗ Error importing yfinance: {e}")
    sys.exit(1)

try:
    from ai_strategy_generator import AIStrategyGenerator
    print("✓ AIStrategyGenerator imported successfully")
except Exception as e:
    print(f"✗ Error importing AIStrategyGenerator: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test fetching market data
try:
    print("\nFetching market data for AAPL...")
    data = yf.download("AAPL", period="1mo", progress=False)
    print(f"✓ Market data fetched: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
except Exception as e:
    print(f"✗ Error fetching market data: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test AI strategy generation
try:
    print("\nInitializing AI Strategy Generator...")
    generator = AIStrategyGenerator(api_key=api_key)
    print("✓ Generator initialized")

    print("\nGenerating strategies...")
    strategies = generator.generate_strategies(
        market_data=data,
        tickers=["AAPL"],
        num_strategies=1,
        past_performance=None,
        learning_insights=None
    )

    print(f"✓ Strategies generated: {len(strategies)}")
    if strategies:
        print(f"\nFirst strategy: {strategies[0].get('name', 'Unknown')}")

except Exception as e:
    print(f"\n✗ Error generating strategies: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!")
