"""
Alpaca Trading Client - Paper trading integration
"""
import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


def is_crypto_ticker(ticker: str) -> bool:
    """
    Detect if a ticker is a crypto pair (uses slash format like BTC/USD, ETH/USD).
    Alpaca accepts crypto as 'BTC/USD'; yfinance needs 'BTC-USD'.
    """
    return "/" in ticker and ticker.endswith("/USD")


def to_yfinance_ticker(ticker: str) -> str:
    """Convert Alpaca crypto ticker (BTC/USD) to yfinance format (BTC-USD)."""
    if is_crypto_ticker(ticker):
        return ticker.replace("/", "-")
    return ticker


def to_alpaca_ticker(ticker: str) -> str:
    """Convert yfinance crypto ticker (BTC-USD) to Alpaca format (BTC/USD)."""
    if "-" in ticker and ticker.endswith("-USD"):
        return ticker.replace("-", "/")
    return ticker


class AlpacaClient:
    """
    Alpaca API client for paper trading

    Features:
    - Place market/limit orders (with auto-convert to market if unfilled in 5min)
    - Smart limit order pricing: BUY at bid+0.01, SELL at ask-0.01
    - IOC limit orders for pair trades (avoids leg risk)
    - Check positions and account status
    - Cancel orders
    - Get market data / quotes (bid/ask)
    - Crypto ticker support (BTC/USD, ETH/USD via Alpaca crypto API)
    """

    LIMIT_ORDER_TIMEOUT_SEC = 300   # 5 minutes before converting to market
    LIMIT_POLL_INTERVAL_SEC = 15    # Check fill status every 15 seconds

    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        connect_timeout = float(os.getenv('ALPACA_CONNECT_TIMEOUT', '5'))
        read_timeout = float(os.getenv('ALPACA_READ_TIMEOUT', os.getenv('ALPACA_REQUEST_TIMEOUT', '20')))
        self.timeout = (connect_timeout, read_timeout)

        if not self.api_key or self.api_key == 'your_paper_api_key_here':
            raise ValueError("Alpaca API key not configured. Set ALPACA_API_KEY in .env file")

        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }

    # ── Quote / Market Data ───────────────────────────────────────────────────

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch latest bid/ask quote for a ticker.

        For crypto (BTC/USD, ETH/USD), uses the crypto quotes endpoint.
        For equities, uses the stocks latest quote endpoint.

        Returns:
            dict with keys: success, bid, ask, mid, spread, ticker
        """
        try:
            if is_crypto_ticker(ticker):
                # Crypto quotes via data API
                data_base = "https://data.alpaca.markets"
                response = requests.get(
                    f"{data_base}/v1beta3/crypto/us/latest/quotes",
                    headers=self.headers,
                    params={"symbols": ticker},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                quote_data = data.get("quotes", {}).get(ticker, {})
                bid = float(quote_data.get("bp", 0))
                ask = float(quote_data.get("ap", 0))
            else:
                # Equity quotes via data API
                data_base = "https://data.alpaca.markets"
                response = requests.get(
                    f"{data_base}/v2/stocks/{ticker}/quotes/latest",
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                quote_data = data.get("quote", {})
                bid = float(quote_data.get("bp", 0))
                ask = float(quote_data.get("ap", 0))

            if bid <= 0 or ask <= 0:
                return {"success": False, "error": f"Invalid quote for {ticker}: bid={bid}, ask={ask}"}

            mid = (bid + ask) / 2
            spread = ask - bid
            log.debug(f"Quote {ticker}: bid={bid:.4f} ask={ask:.4f} spread={spread:.4f}")

            return {
                "success": True,
                "ticker": ticker,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": spread,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "ticker": ticker,
            }

    def _get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current status of an order."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return {"success": True, "order": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Smart Limit Order Execution ────────────────────────────────────────────

    def place_smart_limit_order(
        self,
        symbol: str,
        side: str,               # "buy" or "sell"
        qty: float = None,
        notional: float = None,
        is_pair_trade: bool = False,
        take_profit: float = None,
        stop_loss: float = None,
        timeout_sec: int = None,
    ) -> Dict[str, Any]:
        """
        Place a smart limit order with automatic fallback to market.

        Strategy:
        - BUY: limit at bid + $0.01 (don't chase)
        - SELL: limit at ask - $0.01 (don't undercut)
        - Pair trades: IOC (Immediate or Cancel) to avoid leg risk
        - If not filled within timeout_sec (default 5 min): cancel and convert to market

        Logs order type, fill price, and slippage.

        Args:
            symbol:       Ticker (use Alpaca format: BTC/USD for crypto)
            side:         "buy" or "sell"
            qty:          Share/coin quantity
            notional:     Dollar amount (use qty OR notional, not both)
            is_pair_trade: Use IOC limit to avoid leg risk
            take_profit:  Optional take-profit price (for bracket orders)
            stop_loss:    Optional stop-loss price
            timeout_sec:  Seconds to wait before converting to market (default 300)

        Returns:
            dict with: success, order, fill_price, order_type, slippage_pct
        """
        if timeout_sec is None:
            timeout_sec = self.LIMIT_ORDER_TIMEOUT_SEC

        # Step 1: Get quote
        quote = self.get_quote(symbol)
        if not quote.get("success"):
            log.warning(f"  ⚠️  {symbol}: quote fetch failed ({quote.get('error')}) — placing market order")
            return self._place_market_fallback(symbol, side, qty, notional, take_profit, stop_loss)

        bid = quote["bid"]
        ask = quote["ask"]
        
        # Safety check: bid/ask should not be None
        if bid is None or ask is None:
            log.warning(f"  ⚠️  {symbol}: bid or ask is None (bid={bid}, ask={ask}) — falling back to market")
            return self._place_market_fallback(symbol, side, qty, notional, take_profit, stop_loss)

        # Step 2: Calculate limit price
        if side == "buy":
            # Don't chase: buy at bid + small premium (willing to pay slightly above bid)
            limit_price = round(bid + 0.01, 4 if is_crypto_ticker(symbol) else 2)
            reference_price = ask  # slippage vs ask
        else:
            # Don't undercut: sell at ask - small discount
            limit_price = round(ask - 0.01, 4 if is_crypto_ticker(symbol) else 2)
            reference_price = bid  # slippage vs bid

        # Step 3: For pair trades use IOC (fills immediately or cancels)
        tif = "ioc" if is_pair_trade else "day"

        log.info(
            f"  📋 {symbol}: placing {'IOC ' if is_pair_trade else ''}limit {side.upper()} "
            f"@ ${limit_price:.4f} (bid={bid:.4f}, ask={ask:.4f})"
        )
        # Format bid/ask safely for display (handle None)
        bid_disp = f"{bid:.4f}" if bid is not None else "None"
        ask_disp = f"{ask:.4f}" if ask is not None else "None"
        print(
            f"      📋 {symbol}: LIMIT {side.upper()} @ ${limit_price:.4f} "
            f"[bid={bid_disp} ask={ask_disp}] {'IOC' if is_pair_trade else 'GTC→Market fallback'}"
        )

        # Place the limit order
        limit_result = self.place_order(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            order_type="limit",
            time_in_force=tif,
            limit_price=limit_price,
            take_profit=take_profit if not is_pair_trade else None,
            stop_loss=stop_loss if not is_pair_trade else None,
        )

        if not limit_result.get("success"):
            log.warning(f"  ⚠️  {symbol}: limit order failed ({limit_result.get('error')}) — falling back to market")
            return self._place_market_fallback(symbol, side, qty, notional, take_profit, stop_loss)

        order_id = limit_result["order"]["id"]

        # IOC orders fill or die immediately — no need to wait
        if is_pair_trade:
            time.sleep(1)  # Brief pause for order to process
            status = self._get_order_status(order_id)
            if status.get("success"):
                o = status["order"]
                if o.get("status") in ("filled", "partially_filled"):
                    fill_price = float(o.get("filled_avg_price") or limit_price)
                    slippage = (fill_price - reference_price) / reference_price * 100 if reference_price else 0
                    print(f"      ✅ {symbol}: IOC limit FILLED @ ${fill_price:.4f} (slippage={slippage:+.3f}%)")
                    log.info(f"  {symbol} IOC filled @ {fill_price:.4f} slippage={slippage:+.4f}%")
                    return {
                        **limit_result,
                        "fill_price": fill_price,
                        "order_type": "ioc_limit",
                        "slippage_pct": round(slippage, 4),
                    }
                else:
                    # IOC not filled → place market immediately (pair trade needs both legs)
                    print(f"      ⚠️  {symbol}: IOC NOT filled (status={o.get('status')}) — market fallback")
                    return self._place_market_fallback(symbol, side, qty, notional, take_profit, stop_loss)

        # Step 4: Wait for fill with timeout
        deadline = time.time() + timeout_sec
        poll_interval = self.LIMIT_POLL_INTERVAL_SEC
        attempts = 0

        while time.time() < deadline:
            time.sleep(poll_interval)
            attempts += 1

            status = self._get_order_status(order_id)
            if not status.get("success"):
                continue

            o = status["order"]
            order_status = o.get("status", "")

            if order_status in ("filled", "partially_filled"):
                fill_price = float(o.get("filled_avg_price") or limit_price)
                filled_qty = float(o.get("filled_qty") or 0)
                slippage = (fill_price - reference_price) / reference_price * 100 if reference_price else 0
                elapsed = round(time.time() - (deadline - timeout_sec))
                print(
                    f"      ✅ {symbol}: limit FILLED @ ${fill_price:.4f} "
                    f"(qty={filled_qty:.4f}, slippage={slippage:+.3f}%, "
                    f"elapsed={elapsed}s)"
                )
                log.info(
                    f"  {symbol} limit filled @ {fill_price:.4f} "
                    f"slippage={slippage:+.4f}% elapsed={elapsed}s"
                )
                return {
                    **limit_result,
                    "order": o,
                    "fill_price": fill_price,
                    "order_type": "limit",
                    "slippage_pct": round(slippage, 4),
                }

            if order_status in ("cancelled", "expired", "rejected"):
                log.warning(f"  {symbol}: limit order {order_status} — falling back to market")
                break

            remaining = int(deadline - time.time())
            print(
                f"      ⏳ {symbol}: waiting for limit fill... "
                f"({remaining}s remaining, attempt {attempts})"
            )

        # Step 5: Timeout — cancel limit and place market
        print(f"      ⏰ {symbol}: limit order timed out after {timeout_sec}s — cancelling and placing market")
        log.info(f"  {symbol}: limit order timed out — cancelling {order_id}")
        self.cancel_order(order_id)
        time.sleep(1)

        market_result = self._place_market_fallback(symbol, side, qty, notional, take_profit, stop_loss)
        if market_result.get("success"):
            print(f"      ✅ {symbol}: market fallback order placed")
        return market_result

    def _place_market_fallback(
        self,
        symbol: str,
        side: str,
        qty: float = None,
        notional: float = None,
        take_profit: float = None,
        stop_loss: float = None,
    ) -> Dict[str, Any]:
        """Place a market order as fallback."""
        result = self.place_order(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            order_type="market",
            time_in_force="day",
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        return {
            **result,
            "fill_price": None,
            "order_type": "market_fallback",
            "slippage_pct": None,
        }

    # ── Standard Order Methods ─────────────────────────────────────────────────

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=self.timeout
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
                headers=self.headers,
                timeout=self.timeout
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
                headers=self.headers,
                timeout=self.timeout
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
            symbol: Stock ticker (use BTC/USD format for crypto)
            qty: Number of shares (use qty OR notional, not both)
            notional: Dollar amount to trade (use qty OR notional, not both)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            take_profit: Take profit price (bracket order)
            stop_loss: Stop loss price (bracket order)

        Note: Crypto tickers (BTC/USD, ETH/USD) use fractional qty only
              — notional not supported for crypto pairs.
        """
        try:
            # Crypto doesn't support notional orders — convert to qty
            if is_crypto_ticker(symbol) and notional is not None:
                # Get current price to convert notional → qty
                quote = self.get_quote(symbol)
                if quote.get("success"):
                    mid_price = quote["mid"]
                    if mid_price > 0:
                        qty = notional / mid_price
                        notional = None
                        log.info(f"  Crypto {symbol}: converted ${notional:.2f} notional → {qty:.6f} qty @ ${mid_price:.2f}")

            order_data = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }

            # Quantity vs notional
            if qty is not None:
                order_data["qty"] = str(round(qty, 8))  # str for fractional
            elif notional is not None:
                order_data["notional"] = str(round(notional, 2))
            else:
                return {
                    "success": False,
                    "error": "Must specify either qty or notional"
                }

            # Prices
            if limit_price is not None:
                order_data["limit_price"] = str(round(limit_price, 4))
            if stop_price is not None:
                order_data["stop_price"] = str(round(stop_price, 2))

            # Bracket / OTO order (take profit + stop loss)
            # Note: IOC/FOK orders don't support bracket orders
            if time_in_force not in ("ioc", "fok"):
                if take_profit is not None and stop_loss is not None:
                    order_data["order_class"] = "bracket"
                    order_data["take_profit"] = {"limit_price": str(round(take_profit, 2))}
                    order_data["stop_loss"]   = {"stop_price": str(round(stop_loss, 2))}
                elif stop_loss is not None:
                    order_data["order_class"] = "oto"
                    order_data["stop_loss"]   = {"stop_price": str(round(stop_loss, 2))}
                elif take_profit is not None:
                    order_data["order_class"] = "oto"
                    order_data["take_profit"] = {"limit_price": str(round(take_profit, 2))}

            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=self.timeout
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
                headers=self.headers,
                timeout=self.timeout
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
            symbol: Stock ticker (BTC/USD format for crypto)
            qty: Number of shares to close (None = close all)
        """
        try:
            params = {}
            if qty is not None:
                params["qty"] = qty

            response = requests.delete(
                f"{self.base_url}/v2/positions/{symbol}",
                headers=self.headers,
                params=params,
                timeout=self.timeout
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
                params={"status": status},
                timeout=self.timeout
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
                headers=self.headers,
                timeout=self.timeout
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
        """
        Check if market is currently open.
        Crypto (BTC/USD, ETH/USD) trades 24/7 — always returns True for crypto-only checks.
        """
        result = self.get_clock()
        if result['success']:
            return result['clock'].get('is_open', False)
        return False

    def is_tradeable(self, ticker: str) -> bool:
        """
        Check if a ticker can be traded right now.
        Crypto tickers (BTC/USD, ETH/USD) are always tradeable (24/7).
        Equities require market to be open.
        """
        if is_crypto_ticker(ticker):
            return True   # Crypto trades 24/7
        return self.is_market_open()


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

        # Test 4: Quote fetch
        print("\n4. Quote Test (AAPL):")
        quote = client.get_quote("AAPL")
        if quote['success']:
            print(f"   Bid: ${quote['bid']:.2f}  Ask: ${quote['ask']:.2f}  Mid: ${quote['mid']:.2f}")
        else:
            print(f"   ❌ {quote.get('error')}")

        # Test 5: Crypto ticker helpers
        print("\n5. Crypto Ticker Helpers:")
        print(f"   is_crypto_ticker('BTC/USD') = {is_crypto_ticker('BTC/USD')}")
        print(f"   is_crypto_ticker('AAPL')    = {is_crypto_ticker('AAPL')}")
        print(f"   to_yfinance_ticker('BTC/USD')= {to_yfinance_ticker('BTC/USD')}")
        print(f"   to_alpaca_ticker('BTC-USD')  = {to_alpaca_ticker('BTC-USD')}")

    except ValueError as e:
        print(f"\n❌ {str(e)}")
        print("\nTo use Alpaca paper trading:")
        print("1. Go to: https://app.alpaca.markets/paper/dashboard/overview")
        print("2. Generate API keys")
        print("3. Add them to .env file:")
        print("   ALPACA_API_KEY=your_key")
        print("   ALPACA_SECRET_KEY=your_secret")
