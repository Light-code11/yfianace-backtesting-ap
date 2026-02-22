"""
Daily trading cycle runner - called by OpenClaw cron
Outputs a human-readable summary for notifications
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load env from project dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

def main():
    print(f"=== Daily Trading Cycle ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify Alpaca connection first
    import requests
    headers = {
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY'),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY')
    }
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    try:
        r = requests.get(f'{base_url}/v2/account', headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"ERROR: Alpaca auth failed ({r.status_code})")
            sys.exit(1)
        
        account = r.json()
        print(f"Account: ACTIVE | Cash: ${float(account['cash']):,.2f} | Portfolio: ${float(account['portfolio_value']):,.2f}")
        
        # Check if market is open
        clock = requests.get(f'{base_url}/v2/clock', headers=headers, timeout=10).json()
        if not clock.get('is_open'):
            print(f"Market is CLOSED. Next open: {clock.get('next_open', 'unknown')}")
            print("Skipping trading cycle.")
            return
        
        print("Market is OPEN - running trading cycle...")
        
    except Exception as e:
        print(f"ERROR connecting to Alpaca: {e}")
        sys.exit(1)
    
    # Run the autonomous engine
    try:
        from autonomous_trading_engine import AutonomousTradingEngine
        engine = AutonomousTradingEngine()
        results = engine.run_daily_cycle()
        
        print(f"\n=== Results ===")
        print(f"Success: {results.get('success', False)}")
        print(f"Signals: {results.get('signals_generated', 0)}")
        print(f"Trades executed: {results.get('trades_executed', 0)}")
        print(f"Trades rejected: {results.get('trades_rejected', 0)}")
        
        if results.get('error'):
            print(f"Error: {results['error']}")
        if results.get('errors'):
            for err in results['errors']:
                print(f"  - {err}")
                
        # Print P&L if available
        if results.get('portfolio_value'):
            print(f"Portfolio Value: ${results['portfolio_value']:,.2f}")
        if results.get('daily_pnl'):
            print(f"Daily P&L: ${results['daily_pnl']:,.2f}")
            
    except Exception as e:
        print(f"ERROR in trading cycle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
