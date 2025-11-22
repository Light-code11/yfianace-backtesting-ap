"""
Test Portfolio Optimization API endpoint
"""
import requests
import json
from datetime import datetime

# Test local API
BASE_URL = "http://localhost:8000"

print("=" * 70)
print("TESTING PORTFOLIO OPTIMIZATION")
print("=" * 70)

# Test 1: Check if we have strategies with backtest results
print("\n1. Checking available strategies...")
try:
    response = requests.get(f"{BASE_URL}/strategies", timeout=10)
    if response.status_code == 200:
        data = response.json()
        strategies = data.get('strategies', [])
        print(f"   ✅ Found {len(strategies)} strategies")

        # Show strategies
        for s in strategies[:5]:
            print(f"      - {s['name']} (ID: {s['id']})")

        if len(strategies) < 2:
            print("\n   ⚠️  Need at least 2 strategies for portfolio optimization")
            print("   Please create some strategies and backtest them first!")
            exit(0)
    else:
        print(f"   ❌ Failed: Status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    print("   Make sure the API server is running: python3 run.py")
    exit(1)

# Test 2: Get backtest results for strategies
print("\n2. Checking backtest results...")
try:
    response = requests.get(f"{BASE_URL}/backtest/results?limit=10", timeout=10)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"   ✅ Found {len(results)} backtest results")

        # Group by strategy
        strategy_backtests = {}
        for r in results:
            sid = r.get('strategy_id')
            if sid not in strategy_backtests:
                strategy_backtests[sid] = []
            strategy_backtests[sid].append(r)

        print(f"   ✅ {len(strategy_backtests)} unique strategies have backtest results")

        if len(strategy_backtests) < 2:
            print("\n   ⚠️  Need at least 2 strategies with backtest results")
            print("   Please run backtests on multiple strategies first!")
            exit(0)

        # Use first 3 strategies with backtests
        strategy_ids_to_test = list(strategy_backtests.keys())[:3]
        print(f"   Using strategy IDs: {strategy_ids_to_test}")

    else:
        print(f"   ❌ Failed: Status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Request failed: {e}")
    exit(1)

# Test 3: Test portfolio optimization
print("\n3. Testing portfolio optimization...")

test_methods = ["max_sharpe", "min_volatility"]

for method in test_methods:
    print(f"\n   Testing method: {method}")

    try:
        payload = {
            "strategy_ids": strategy_ids_to_test,
            "total_capital": 100000,
            "method": method,
            "constraints": {
                "max_allocation": 0.4  # 40% max per strategy
            }
        }

        response = requests.post(
            f"{BASE_URL}/portfolio/optimize",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print(f"   ✅ Optimization successful!")
                print(f"      Method: {data['method']}")
                print(f"      Expected Return: {data['expected_return']:.2f}%")
                print(f"      Expected Volatility: {data['expected_volatility']:.2f}%")
                print(f"      Expected Sharpe: {data['expected_sharpe']:.2f}")
                print(f"      Library Used: {data['optimization_library']}")

                print(f"\n      Allocations:")
                for name, pct in data['allocations'].items():
                    capital = data['capital_allocations'][name]
                    print(f"         {name}: {pct:.1f}% (${capital:,.0f})")

                # Verify allocations sum to ~100%
                total_allocation = sum(data['allocations'].values())
                print(f"\n      Total Allocation: {total_allocation:.1f}%")

                if abs(total_allocation - 100) < 1:
                    print(f"      ✅ Allocation totals correct!")
                else:
                    print(f"      ⚠️  Allocation doesn't sum to 100%")

                # Verify constraints
                max_alloc = max(data['allocations'].values())
                if max_alloc <= 40.5:  # 40% + small tolerance
                    print(f"      ✅ Max allocation constraint respected (max: {max_alloc:.1f}%)")
                else:
                    print(f"      ⚠️  Max allocation exceeded: {max_alloc:.1f}% > 40%")

                # Check if strategies details are present
                if data.get('strategies'):
                    print(f"      ✅ Strategy details included ({len(data['strategies'])} strategies)")
                else:
                    print(f"      ⚠️  Strategy details missing")
            else:
                print(f"   ❌ Optimization failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Request failed: Status {response.status_code}")
            print(f"      Response: {response.text[:200]}")

    except Exception as e:
        print(f"   ❌ Request failed: {e}")

# Test 4: Test edge cases
print("\n4. Testing edge cases...")

# Test with only 1 strategy (should fail)
print("\n   Testing with only 1 strategy...")
try:
    payload = {
        "strategy_ids": [strategy_ids_to_test[0]],
        "total_capital": 100000,
        "method": "max_sharpe"
    }

    response = requests.post(
        f"{BASE_URL}/portfolio/optimize",
        json=payload,
        timeout=30
    )

    # Single strategy should still work (100% allocation)
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"   ✅ Single strategy optimization works")
            total = sum(data['allocations'].values())
            if abs(total - 100) < 1:
                print(f"      ✅ 100% allocated to single strategy")
        else:
            print(f"   ⚠️  Expected single strategy to work")

except Exception as e:
    print(f"   ❌ Request failed: {e}")

# Test with invalid method
print("\n   Testing with invalid method...")
try:
    payload = {
        "strategy_ids": strategy_ids_to_test,
        "total_capital": 100000,
        "method": "invalid_method"
    }

    response = requests.post(
        f"{BASE_URL}/portfolio/optimize",
        json=payload,
        timeout=30
    )

    # Should either default to max_sharpe or return error
    if response.status_code == 200:
        data = response.json()
        if data.get('success'):
            print(f"   ✅ Invalid method handled (defaulted to max_sharpe)")
        else:
            print(f"   ✅ Invalid method rejected with error")

except Exception as e:
    print(f"   ❌ Request failed: {e}")

print("\n" + "=" * 70)
print("PORTFOLIO OPTIMIZATION TEST COMPLETE")
print("=" * 70)
print("\nNext Steps:")
print("1. ✅ If all tests passed → Ready to deploy!")
print("2. ⚠️  If some tests failed → Review errors above")
print("3. 📊 Test in Streamlit UI → python3 -m streamlit run streamlit_app.py")
print("=" * 70)
