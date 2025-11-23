"""
Alpaca Trading Client - Paper trading integration
"""
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()


class AlpacaClient:
    """
    Alpaca API client for paper trading

    Features:
    - Place market/limit orders
    - Check positions and account status
    - Cancel orders
    - Get market data
    """

    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not self.api_key or self.api_key == 'your_paper_api_key_here':
            raise ValueError("Alpaca API key not configured. Set ALPACA_API_KEY in .env file")

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers
            )
            response.raise_for_status()
            return {
                "success": True,
                "account": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_positions(self) -> Dict[str, Any]:
        """Get all open positions"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers
            )
            response.raise_for_status()
            return {
                "success": True,
                "positions": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for specific symbol"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self.headers
            )
            response.raise_for_status()
            return {
                "success": True,
                "position": response.json()
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {
                    "success": True,
                    "position": None  # No position
                }
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def place_order(
        self,
        symbol: str,
        qty: float = None,
        notional: float = None,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float = None,
        stop_price: float = None,
        take_profit: float = None,
        stop_loss: float = None
    ) -> Dict[str, Any]:
        """
        Place an order

        Args:
            symbol: Stock ticker
            qty: Number of shares (use qty OR notional, not both)
            notional: Dollar amount to trade (use qty OR notional, not both)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            take_profit: Take profit price (bracket order)
            stop_loss: Stop loss price (bracket order)
        """
        try:
            order_data = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }

            # Quantity vs notional
            if qty is not None:
                order_data["qty"] = qty
            elif notional is not None:
                order_data["notional"] = notional
            else:
                return {
                    "success": False,
                    "error": "Must specify either qty or notional"
                }

            # Prices
            if limit_price is not None:
                order_data["limit_price"] = limit_price
            if stop_price is not None:
                order_data["stop_price"] = stop_price

            # Bracket order (take profit + stop loss)
            if take_profit is not None or stop_loss is not None:
                order_data["order_class"] = "bracket"
                if take_profit is not None:
                    order_data["take_profit"] = {"limit_price": take_profit}
                if stop_loss is not None:
                    order_data["stop_loss"] = {"stop_price": stop_loss}

            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data
            )
            response.raise_for_status()

            return {
                "success": True,
                "order": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return {"success": True}
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def close_position(self, symbol: str, qty: float = None) -> Dict[str, Any]:
        """
        Close a position (or partial)

        Args:
            symbol: Stock ticker
            qty: Number of shares to close (None = close all)
        """
        try:
            params = {}
            if qty is not None:
                params["qty"] = qty

            response = requests.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self.headers,
                params=params
            )
            response.raise_for_status()

            return {
                "success": True,
                "order": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_orders(self, status: str = "open") -> Dict[str, Any]:
        """
        Get orders

        Args:
            status: "open", "closed", "all"
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                params={"status": status}
            )
            response.raise_for_status()
            return {
                "success": True,
                "orders": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_clock(self) -> Dict[str, Any]:
        """Get market clock (is market open?)"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/clock",
                headers=self.headers
            )
            response.raise_for_status()
            return {
                "success": True,
                "clock": response.json()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        result = self.get_clock()
        if result['success']:
            return result['clock'].get('is_open', False)
        return False


if __name__ == "__main__":
    # Test the client
    print("=" * 70)
    print("TESTING ALPACA CLIENT")
    print("=" * 70)

    try:
        client = AlpacaClient()

        # Test 1: Get account
        print("\n1. Account Info:")
        account = client.get_account()
        if account['success']:
            acc = account['account']
            print(f"   ✅ Portfolio Value: ${float(acc['portfolio_value']):,.2f}")
            print(f"   Cash: ${float(acc['cash']):,.2f}")
            print(f"   Buying Power: ${float(acc['buying_power']):,.2f}")
        else:
            print(f"   ❌ Error: {account['error']}")

        # Test 2: Market clock
        print("\n2. Market Status:")
        clock = client.get_clock()
        if clock['success']:
            c = clock['clock']
            print(f"   Market Open: {c['is_open']}")
            print(f"   Next Open: {c['next_open']}")
            print(f"   Next Close: {c['next_close']}")
        else:
            print(f"   ❌ Error: {clock['error']}")

        # Test 3: Positions
        print("\n3. Current Positions:")
        positions = client.get_positions()
        if positions['success']:
            if positions['positions']:
                for pos in positions['positions']:
                    pnl_pct = float(pos['unrealized_plpc']) * 100
                    print(f"   {pos['symbol']}: {pos['qty']} shares, P&L: {pnl_pct:+.2f}%")
            else:
                print("   No open positions")
        else:
            print(f"   ❌ Error: {positions['error']}")

    except ValueError as e:
        print(f"\n❌ {str(e)}")
        print("\nTo use Alpaca paper trading:")
        print("1. Go to: https://app.alpaca.markets/paper/dashboard/overview")
        print("2. Generate API keys")
        print("3. Add them to .env file:")
        print("   ALPACA_API_KEY=your_key")
        print("   ALPACA_SECRET_KEY=your_secret")
