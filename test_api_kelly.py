"""
Test if Railway API is returning Kelly Criterion data
"""
import requests
import json

# Test both local and Railway
APIS = {
    "Railway": "https://your-railway-app-url.railway.app",  # Replace with your actual Railway URL
    "Local": "http://localhost:8000"
}

print("=" * 70)
print("TESTING KELLY CRITERION API RESPONSES")
print("=" * 70)

# Ask user for Railway URL
railway_url = input("\nEnter your Railway app URL (e.g., https://abc-production.up.railway.app): ").strip()
if railway_url:
    APIS["Railway"] = railway_url

for api_name, base_url in APIS.items():
    print(f"\n{'=' * 70}")
    print(f"Testing {api_name}: {base_url}")
    print("=" * 70)

    try:
        # Test 1: Check API version and Kelly enabled flag
        print(f"\n1. Checking API root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Version: {data.get('version', 'unknown')}")
            kelly_enabled = data.get('features', {}).get('kelly_criterion_enabled', False)
            print(f"   {'✅' if kelly_enabled else '❌'} Kelly Criterion Enabled: {kelly_enabled}")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")
            continue

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        continue

    try:
        # Test 2: Get latest backtest results
        print(f"\n2. Checking latest backtest results...")
        response = requests.get(f"{base_url}/backtest/results?limit=1", timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                print(f"   Strategy: {result['strategy_name']}")
                print(f"   Quality Score: {result.get('quality_score', 'N/A')}")

                # Check for Kelly fields
                has_kelly_criterion = 'kelly_criterion' in result
                has_kelly_position = 'kelly_position_pct' in result
                has_kelly_risk = 'kelly_risk_level' in result

                print(f"\n   Kelly Fields Present:")
                print(f"   {'✅' if has_kelly_criterion else '❌'} kelly_criterion: {result.get('kelly_criterion', 'MISSING')}")
                print(f"   {'✅' if has_kelly_position else '❌'} kelly_position_pct: {result.get('kelly_position_pct', 'MISSING')}")
                print(f"   {'✅' if has_kelly_risk else '❌'} kelly_risk_level: {result.get('kelly_risk_level', 'MISSING')}")

                if has_kelly_criterion and has_kelly_position and has_kelly_risk:
                    if result.get('kelly_position_pct') is not None:
                        print(f"\n   🎉 SUCCESS! Kelly data is being returned!")
                        print(f"   Recommendation: {result.get('kelly_position_pct', 0):.2f}% position size")
                    else:
                        print(f"\n   ⚠️  Kelly fields exist but are NULL (old backtest)")
                        print(f"   Solution: Run a NEW backtest to generate Kelly data")
                else:
                    print(f"\n   ❌ PROBLEM: API is not returning Kelly fields")
                    print(f"   This means Railway hasn't deployed the latest code yet")
            else:
                print(f"   ⚠️  No backtest results found in database")
        else:
            print(f"   ❌ Failed: Status {response.status_code}")

    except Exception as e:
        print(f"   ❌ Request failed: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\nWhat to do next:")
print("1. If Railway shows ❌ Kelly fields missing → Wait for Railway to deploy")
print("2. If Railway shows ⚠️ Kelly fields NULL → Run a NEW backtest")
print("3. If Railway shows ✅ SUCCESS → Kelly should appear in Streamlit UI")
print("=" * 70)
