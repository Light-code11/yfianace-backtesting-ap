import requests
import json

response = requests.post('http://localhost:8000/strategies/generate', json={
    "tickers": ["AAPL"],
    "period": "1mo",
    "num_strategies": 1,
    "use_past_performance": False
})

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("✅ SUCCESS!")
    result = response.json()
    print(f"Strategies generated: {result.get('strategies_generated')}")
    if result.get('strategies'):
        print(f"First strategy: {result['strategies'][0]['name']}")
else:
    print(f"❌ ERROR: {response.text}")
