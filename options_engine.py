"""
Options Trading Engine

Fetches options chains, calculates Greeks / IV rank, generates strategy
signals, and submits orders via the Alpaca paper-trading API.

All positions must be defined-risk (spreads, cash-secured, or covered).
No naked calls. Ever.
"""
from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from scipy.stats import norm

load_dotenv()

try:
    from ml_price_predictor import MLPricePredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from options_config import (
    ALPACA_CONFIG,
    OPTIONS_STRATEGIES,
    OPTIONS_UNIVERSE,
    RISK_PARAMS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bs_delta(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Black-Scholes delta. T in years."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type.lower() == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)   # put delta is negative


def _bs_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Black-Scholes option price."""
    if T <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(market_price: float, S: float, K: float, T: float,
                 r: float, option_type: str = "call") -> float:
    """Brent bisection to find implied volatility."""
    if market_price <= 0 or T <= 0:
        return 0.30   # default 30 %
    lo, hi = 0.001, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        price = _bs_price(S, K, T, r, mid, option_type)
        if price > market_price:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 1e-5:
            break
    return (lo + hi) / 2


def _option_symbol(underlying: str, expiry: date, strike: float,
                   option_type: str) -> str:
    """Build OCC option symbol: AAPL240119C00190000."""
    type_char = "C" if option_type.lower() == "call" else "P"
    strike_str = f"{int(round(strike * 1000)):08d}"
    return f"{underlying}{expiry.strftime('%y%m%d')}{type_char}{strike_str}"


def _parse_dte(expiry_str: str) -> int:
    """Days to expiration from ISO date string."""
    try:
        exp = date.fromisoformat(expiry_str)
        return (exp - date.today()).days
    except Exception:
        return -1


# ── Main Engine ───────────────────────────────────────────────────────────────

class OptionsEngine:
    """
    Options paper-trading engine for Alpaca.

    Usage:
        engine = OptionsEngine()
        candidates = engine.scan_all()
        engine.manage_positions(dry_run=True)
    """

    RISK_FREE_RATE = 0.05   # approx 5 % T-bill yield

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = ALPACA_CONFIG["base_url"]
        self.timeout = (
            ALPACA_CONFIG["connect_timeout_sec"],
            ALPACA_CONFIG["read_timeout_sec"],
        )
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }
        self._ml: Optional[Any] = None
        self._price_cache: Dict[str, float] = {}
        self._iv_history_cache: Dict[str, List[float]] = {}

    # ── Alpaca helpers ────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        try:
            url = f"{self.base_url}{path}"
            resp = requests.get(url, headers=self.headers, params=params,
                                timeout=self.timeout)
            resp.raise_for_status()
            return {"success": True, "data": resp.json()}
        except Exception as exc:
            return {"success": False, "error": str(exc), "data": {}}

    def _post(self, path: str, payload: Dict) -> Dict:
        try:
            url = f"{self.base_url}{path}"
            resp = requests.post(url, headers=self.headers, json=payload,
                                 timeout=self.timeout)
            resp.raise_for_status()
            return {"success": True, "data": resp.json()}
        except requests.exceptions.HTTPError as exc:
            body = {}
            try:
                body = exc.response.json()
            except Exception:
                pass
            return {"success": False, "error": str(exc), "data": body}
        except Exception as exc:
            return {"success": False, "error": str(exc), "data": {}}

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        """Return Alpaca account dict or empty dict on failure."""
        result = self._get("/v2/account")
        return result.get("data", {}) if result["success"] else {}

    def get_portfolio_value(self) -> float:
        account = self.get_account()
        try:
            return float(account.get("portfolio_value", 0))
        except Exception:
            return 0.0

    def get_buying_power(self) -> float:
        account = self.get_account()
        try:
            return float(account.get("buying_power", 0))
        except Exception:
            return 0.0

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_all_positions(self) -> List[Dict]:
        result = self._get("/v2/positions")
        if result["success"]:
            return result["data"] if isinstance(result["data"], list) else []
        return []

    def get_options_positions(self) -> List[Dict]:
        """Filter positions to options only (asset_class == 'us_option')."""
        return [
            p for p in self.get_all_positions()
            if p.get("asset_class") == "us_option"
        ]

    def get_equity_positions(self) -> List[Dict]:
        """Filter positions to equities only."""
        return [
            p for p in self.get_all_positions()
            if p.get("asset_class") == "us_equity"
        ]

    # ── Options chain ─────────────────────────────────────────────────────────

    def get_options_chain(self, symbol: str,
                          dte_range: Tuple[int, int] = (21, 45)) -> List[Dict]:
        """
        Fetch options chain from Alpaca for *symbol* within *dte_range* days.
        Returns a list of contract dicts (may be empty on error or no data).
        """
        min_dte, max_dte = dte_range
        today = date.today()
        exp_gte = (today + timedelta(days=min_dte)).isoformat()
        exp_lte = (today + timedelta(days=max_dte)).isoformat()

        contracts: List[Dict] = []
        page_token: Optional[str] = None

        for _ in range(10):   # guard against runaway pagination
            params: Dict[str, Any] = {
                "underlying_symbols": symbol,
                "expiration_date_gte": exp_gte,
                "expiration_date_lte": exp_lte,
                "limit": 200,
            }
            if page_token:
                params["page_token"] = page_token

            result = self._get("/v2/options/contracts", params=params)
            if not result["success"]:
                break

            data = result["data"]
            batch = data.get("option_contracts", data if isinstance(data, list) else [])
            contracts.extend(batch)

            page_token = data.get("page_token") if isinstance(data, dict) else None
            if not page_token:
                break

        # Apply liquidity filters
        min_oi = RISK_PARAMS["min_open_interest"]
        min_vol = RISK_PARAMS["min_volume"]
        filtered = []
        for c in contracts:
            try:
                oi = int(c.get("open_interest") or 0)
                # Volume isn't always in the contract object; skip if missing
                if oi < min_oi:
                    continue
                filtered.append(c)
            except Exception:
                filtered.append(c)   # keep if we can't parse

        return filtered

    # ── Pricing helper ────────────────────────────────────────────────────────

    def _get_stock_price(self, symbol: str) -> float:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        except Exception:
            price = 0.0
        self._price_cache[symbol] = price
        return price

    # ── IV calculation ────────────────────────────────────────────────────────

    def calculate_iv_rank(self, symbol: str, lookback: int = 252) -> float:
        """
        IV rank = (current IV - 52w low) / (52w high - 52w low) × 100.
        Uses the historical close-to-close realised vol as a proxy for IV when
        Alpaca doesn't expose Greeks directly.
        Returns a value in [0, 100].
        """
        if symbol in self._iv_history_cache:
            hist_ivs = self._iv_history_cache[symbol]
        else:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if hist.empty or len(hist) < 20:
                    return 50.0  # neutral default

                # 20-day rolling realised vol as IV proxy
                log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                rolling_vol = log_ret.rolling(20).std() * math.sqrt(252)
                hist_ivs = rolling_vol.dropna().tolist()
                self._iv_history_cache[symbol] = hist_ivs
            except Exception:
                return 50.0

        if len(hist_ivs) < 2:
            return 50.0

        current_iv = hist_ivs[-1]
        lo = min(hist_ivs)
        hi = max(hist_ivs)
        if hi == lo:
            return 50.0
        rank = (current_iv - lo) / (hi - lo) * 100
        return float(max(0.0, min(100.0, rank)))

    # ── Greeks ────────────────────────────────────────────────────────────────

    def calculate_greeks(self, contract: Dict,
                         underlying_price: Optional[float] = None) -> Dict:
        """
        Calculate Black-Scholes Greeks for a contract dict.
        Falls back to Alpaca-provided greeks if present (key 'greeks').
        """
        # If Alpaca already provides greeks, use them
        if contract.get("greeks"):
            g = contract["greeks"]
            return {
                "delta": float(g.get("delta", 0)),
                "gamma": float(g.get("gamma", 0)),
                "theta": float(g.get("theta", 0)),
                "vega": float(g.get("vega", 0)),
                "iv": float(g.get("implied_volatility", 0.30)),
            }

        symbol = contract.get("underlying_symbol", contract.get("root_symbol", ""))
        S = underlying_price or self._get_stock_price(symbol)
        if S == 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0.30}

        K = float(contract.get("strike_price", 0))
        option_type = contract.get("type", "call")
        exp_str = contract.get("expiration_date", "")
        dte = _parse_dte(exp_str)
        T = max(dte, 0) / 365.0

        # Use close price as market price to calculate IV
        market_price = float(contract.get("close_price") or 0)
        if market_price <= 0:
            sigma = 0.30
        else:
            sigma = _implied_vol(market_price, S, K, T,
                                 self.RISK_FREE_RATE, option_type)

        delta = _bs_delta(S, K, T, self.RISK_FREE_RATE, sigma, option_type)

        # Gamma
        if T > 0 and sigma > 0:
            d1 = (math.log(S / K) + (self.RISK_FREE_RATE + 0.5 * sigma ** 2) * T) / (
                sigma * math.sqrt(T))
            gamma = float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))
            theta = float(-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) -
                          self.RISK_FREE_RATE * K * math.exp(
                              -self.RISK_FREE_RATE * T) * norm.cdf(
                                  d1 - sigma * math.sqrt(T))) / 365
            vega = float(S * norm.pdf(d1) * math.sqrt(T)) / 100
        else:
            gamma = theta = vega = 0.0

        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": sigma,
        }

    # ── Strike selection ──────────────────────────────────────────────────────

    def find_optimal_strike(self, chain: List[Dict], target_delta: float,
                            option_type: str,
                            underlying_price: float) -> Optional[Dict]:
        """
        Return the contract whose delta is closest to *target_delta*.
        """
        typed = [c for c in chain if c.get("type", "").lower() == option_type.lower()]
        if not typed:
            return None

        best: Optional[Dict] = None
        best_dist = float("inf")

        for c in typed:
            greeks = self.calculate_greeks(c, underlying_price)
            delta = greeks["delta"]
            dist = abs(delta - target_delta)
            if dist < best_dist:
                best_dist = dist
                best = {**c, "_greeks": greeks}

        return best

    # ── ML signal helper ──────────────────────────────────────────────────────

    def _ml_signal(self, symbol: str) -> Dict:
        """
        Return {"direction": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0-1}.
        Falls back to NEUTRAL/0.5 if ML is unavailable or fails.
        """
        if not ML_AVAILABLE:
            return {"direction": "NEUTRAL", "confidence": 0.5}
        if self._ml is None:
            try:
                self._ml = MLPricePredictor()
            except Exception:
                return {"direction": "NEUTRAL", "confidence": 0.5}
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            if hist.empty or len(hist) < 30:
                return {"direction": "NEUTRAL", "confidence": 0.5}
            result = self._ml.predict(hist)
            # result is expected to have keys: prediction (1=up, 0=down), confidence
            prediction = result.get("prediction", 0.5)
            confidence = float(result.get("confidence", 0.5))
            if prediction >= 0.5:
                direction = "BULLISH" if confidence >= 0.55 else "NEUTRAL"
            else:
                direction = "BEARISH" if confidence >= 0.55 else "NEUTRAL"
            return {"direction": direction, "confidence": confidence}
        except Exception:
            return {"direction": "NEUTRAL", "confidence": 0.5}

    # ── Earnings calendar ─────────────────────────────────────────────────────

    def _days_to_earnings(self, symbol: str) -> Optional[int]:
        """Return days until next earnings, or None if unknown."""
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None or cal.empty:
                return None
            # calendar index contains "Earnings Date"
            if "Earnings Date" in cal.index:
                earn_dates = cal.loc["Earnings Date"]
                if isinstance(earn_dates, pd.Series):
                    # Take first future date
                    for val in earn_dates:
                        try:
                            earn_date = pd.Timestamp(val).date()
                            days = (earn_date - date.today()).days
                            if days >= 0:
                                return days
                        except Exception:
                            pass
            return None
        except Exception:
            return None

    # ── Scanning ──────────────────────────────────────────────────────────────

    def scan_wheel_candidates(self) -> List[Dict]:
        """
        Wheel strategy: sell cash-secured puts.
        Criteria:
          - ML neutral or bullish
          - IV rank > 20 %
          - Stock price × 100 < max_allocation per underlying
          - Wheel strategy is enabled
        """
        if not OPTIONS_STRATEGIES["wheel"]["enabled"]:
            return []

        cfg = OPTIONS_STRATEGIES["wheel"]
        max_alloc = self.get_portfolio_value() * RISK_PARAMS["max_single_underlying_pct"]
        candidates = []

        for symbol in OPTIONS_UNIVERSE:
            try:
                price = self._get_stock_price(symbol)
                if price <= 0:
                    continue

                # Affording assignment of 1 contract = 100 shares
                if price * 100 > max_alloc:
                    continue

                ml = self._ml_signal(symbol)
                if ml["direction"] == "BEARISH":
                    continue   # Don't sell puts on bearish stocks

                iv_rank = self.calculate_iv_rank(symbol)
                if iv_rank < 20:
                    continue   # Need decent premium

                chain = self.get_options_chain(symbol, cfg["dte_range"])
                if not chain:
                    continue

                optimal = self.find_optimal_strike(
                    chain, cfg["target_delta"], "put", price
                )
                if optimal is None:
                    continue

                greeks = optimal["_greeks"]
                strike = float(optimal.get("strike_price", 0))
                expiry = optimal.get("expiration_date", "")
                premium = float(optimal.get("close_price") or 0)
                premium_pct = premium / strike if strike > 0 else 0

                if premium_pct < cfg["min_premium_pct"]:
                    continue

                candidates.append({
                    "strategy": "wheel",
                    "symbol": symbol,
                    "action": "sell_csp",
                    "option_type": "put",
                    "expiry": expiry,
                    "strike": strike,
                    "contract_symbol": optimal.get("symbol", ""),
                    "premium": premium,
                    "premium_pct": premium_pct,
                    "delta": greeks["delta"],
                    "iv_rank": iv_rank,
                    "ml_direction": ml["direction"],
                    "ml_confidence": ml["confidence"],
                    "underlying_price": price,
                    "dte": _parse_dte(expiry),
                })
            except Exception as exc:
                print(f"  [wheel scan] Error on {symbol}: {exc}")

        return candidates

    def scan_iron_condor_candidates(self) -> List[Dict]:
        """
        Iron condor on range-bound / high-IV stocks.
        Criteria:
          - IV rank > 30 %
          - ML confidence < 60 % (no strong directional view)
          - No earnings within DTE range
        """
        if not OPTIONS_STRATEGIES["iron_condor"]["enabled"]:
            return []

        cfg = OPTIONS_STRATEGIES["iron_condor"]
        candidates = []

        for symbol in OPTIONS_UNIVERSE:
            try:
                price = self._get_stock_price(symbol)
                if price <= 0:
                    continue

                iv_rank = self.calculate_iv_rank(symbol)
                if iv_rank < cfg["min_iv_rank"]:
                    continue

                ml = self._ml_signal(symbol)
                if ml["confidence"] >= 0.60:
                    continue   # Strong directional bias — skip iron condor

                max_dte = cfg["dte_range"][1]
                earn_days = self._days_to_earnings(symbol)
                if earn_days is not None and earn_days <= max_dte:
                    continue   # Earnings within window — skip

                chain = self.get_options_chain(symbol, cfg["dte_range"])
                if not chain:
                    continue

                # Short put leg
                short_put = self.find_optimal_strike(
                    chain, -cfg["short_delta"], "put", price
                )
                # Short call leg
                short_call = self.find_optimal_strike(
                    chain, cfg["short_delta"], "call", price
                )
                if not short_put or not short_call:
                    continue

                width = cfg["wing_width"]
                short_put_strike = float(short_put.get("strike_price", 0))
                short_call_strike = float(short_call.get("strike_price", 0))

                # Find same-expiry contracts for wings (OTM by wing_width)
                expiry = short_put.get("expiration_date", "")
                long_put_strike = short_put_strike - width
                long_call_strike = short_call_strike + width

                put_premium = float(short_put.get("close_price") or 0)
                call_premium = float(short_call.get("close_price") or 0)
                net_credit = put_premium + call_premium   # approximate, ignoring wing cost

                if net_credit <= 0:
                    continue

                candidates.append({
                    "strategy": "iron_condor",
                    "symbol": symbol,
                    "action": "open_iron_condor",
                    "expiry": expiry,
                    "short_put_strike": short_put_strike,
                    "long_put_strike": long_put_strike,
                    "short_call_strike": short_call_strike,
                    "long_call_strike": long_call_strike,
                    "net_credit": net_credit,
                    "max_risk": width * 100 - net_credit * 100,
                    "iv_rank": iv_rank,
                    "ml_confidence": ml["confidence"],
                    "underlying_price": price,
                    "dte": _parse_dte(expiry),
                })
            except Exception as exc:
                print(f"  [condor scan] Error on {symbol}: {exc}")

        return candidates

    def scan_credit_spread_candidates(self) -> List[Dict]:
        """
        Directional credit spreads driven by ML signals (>65 % confidence).
        Sell put spread if bullish, call spread if bearish.
        """
        if not OPTIONS_STRATEGIES["credit_spread"]["enabled"]:
            return []

        cfg = OPTIONS_STRATEGIES["credit_spread"]
        portfolio_value = self.get_portfolio_value()
        max_risk = portfolio_value * cfg["max_risk_per_trade_pct"]
        candidates = []

        for symbol in OPTIONS_UNIVERSE:
            try:
                ml = self._ml_signal(symbol)
                if ml["confidence"] < 0.65:
                    continue   # Need strong directional view

                price = self._get_stock_price(symbol)
                if price <= 0:
                    continue

                chain = self.get_options_chain(symbol, cfg["dte_range"])
                if not chain:
                    continue

                direction = ml["direction"]
                if direction == "BULLISH":
                    # Bull put spread: sell put, buy lower put
                    short = self.find_optimal_strike(
                        chain, -cfg["short_delta"], "put", price
                    )
                    if not short:
                        continue
                    short_strike = float(short.get("strike_price", 0))
                    long_strike = short_strike - cfg["spread_width"]
                    option_type = "put"
                    spread_type = "bull_put"
                elif direction == "BEARISH":
                    # Bear call spread: sell call, buy higher call
                    short = self.find_optimal_strike(
                        chain, cfg["short_delta"], "call", price
                    )
                    if not short:
                        continue
                    short_strike = float(short.get("strike_price", 0))
                    long_strike = short_strike + cfg["spread_width"]
                    option_type = "call"
                    spread_type = "bear_call"
                else:
                    continue

                expiry = short.get("expiration_date", "")
                short_premium = float(short.get("close_price") or 0)
                # Approximate: long premium ≈ 40 % of short (further OTM)
                long_premium_est = short_premium * 0.40
                net_credit = short_premium - long_premium_est
                max_risk_trade = (cfg["spread_width"] - net_credit) * 100

                if max_risk_trade > max_risk:
                    continue

                candidates.append({
                    "strategy": "credit_spread",
                    "symbol": symbol,
                    "action": f"open_{spread_type}",
                    "option_type": option_type,
                    "spread_type": spread_type,
                    "expiry": expiry,
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "net_credit_est": net_credit,
                    "max_risk": max_risk_trade,
                    "ml_direction": direction,
                    "ml_confidence": ml["confidence"],
                    "underlying_price": price,
                    "dte": _parse_dte(expiry),
                })
            except Exception as exc:
                print(f"  [spread scan] Error on {symbol}: {exc}")

        return candidates

    def scan_earnings_plays(self) -> List[Dict]:
        """
        Sell iron condors ahead of earnings to capture IV crush.
        Criteria: earnings within 7 days.
        """
        if not OPTIONS_STRATEGIES["earnings_play"]["enabled"]:
            return []

        cfg = OPTIONS_STRATEGIES["earnings_play"]
        candidates = []
        max_dte_min, max_dte_max = cfg["dte_range"]

        for symbol in OPTIONS_UNIVERSE:
            try:
                earn_days = self._days_to_earnings(symbol)
                if earn_days is None:
                    continue
                if not (max_dte_min <= earn_days <= max_dte_max):
                    continue

                price = self._get_stock_price(symbol)
                if price <= 0:
                    continue

                iv_rank = self.calculate_iv_rank(symbol)

                # Use a tight DTE range bracketing earnings
                dte_range = (max(1, earn_days - 1), earn_days + 2)
                chain = self.get_options_chain(symbol, dte_range)
                if not chain:
                    continue

                short_put = self.find_optimal_strike(
                    chain, -cfg["short_delta"], "put", price
                )
                short_call = self.find_optimal_strike(
                    chain, cfg["short_delta"], "call", price
                )
                if not short_put or not short_call:
                    continue

                width = cfg["wing_width"]
                expiry = short_put.get("expiration_date", "")
                sp_strike = float(short_put.get("strike_price", 0))
                sc_strike = float(short_call.get("strike_price", 0))
                put_p = float(short_put.get("close_price") or 0)
                call_p = float(short_call.get("close_price") or 0)
                net_credit = put_p + call_p

                candidates.append({
                    "strategy": "earnings_play",
                    "symbol": symbol,
                    "action": "open_earnings_condor",
                    "expiry": expiry,
                    "short_put_strike": sp_strike,
                    "long_put_strike": sp_strike - width,
                    "short_call_strike": sc_strike,
                    "long_call_strike": sc_strike + width,
                    "net_credit": net_credit,
                    "days_to_earnings": earn_days,
                    "iv_rank": iv_rank,
                    "underlying_price": price,
                    "dte": _parse_dte(expiry),
                })
            except Exception as exc:
                print(f"  [earnings scan] Error on {symbol}: {exc}")

        return candidates

    def scan_hedge_opportunities(self) -> List[Dict]:
        """
        Buy protective puts on existing equity positions (hedge 50 % of exposure).
        """
        if not OPTIONS_STRATEGIES["protective_puts"]["enabled"]:
            return []

        cfg = OPTIONS_STRATEGIES["protective_puts"]
        equity_positions = self.get_equity_positions()
        candidates = []

        # Check which equity positions already have put hedges
        options_positions = self.get_options_positions()
        hedged_symbols = set()
        for op in options_positions:
            sym = op.get("symbol", "")
            # OCC symbol: first chars are the underlying
            for eq in equity_positions:
                eq_sym = eq.get("symbol", "")
                if sym.startswith(eq_sym):
                    side = op.get("side", "")
                    # A long put hedge would be a long put position
                    if "P" in sym and side == "long":
                        hedged_symbols.add(eq_sym)

        for pos in equity_positions:
            symbol = pos.get("symbol", "")
            if symbol not in OPTIONS_UNIVERSE:
                continue
            if symbol in hedged_symbols:
                continue   # Already hedged

            try:
                price = self._get_stock_price(symbol)
                if price <= 0:
                    continue

                qty = int(float(pos.get("qty", 0)))
                # Number of contracts to hedge hedge_pct of position
                contracts_needed = max(1, int(qty * cfg["hedge_pct"] / 100))

                chain = self.get_options_chain(symbol, cfg["dte_range"])
                if not chain:
                    continue

                optimal_put = self.find_optimal_strike(
                    chain, cfg["target_delta"], "put", price
                )
                if not optimal_put:
                    continue

                greeks = optimal_put["_greeks"]
                strike = float(optimal_put.get("strike_price", 0))
                expiry = optimal_put.get("expiration_date", "")
                premium = float(optimal_put.get("close_price") or 0)

                candidates.append({
                    "strategy": "protective_puts",
                    "symbol": symbol,
                    "action": "buy_put",
                    "option_type": "put",
                    "expiry": expiry,
                    "strike": strike,
                    "contracts": contracts_needed,
                    "premium_per_contract": premium * 100,
                    "delta": greeks["delta"],
                    "equity_qty": qty,
                    "underlying_price": price,
                    "dte": _parse_dte(expiry),
                })
            except Exception as exc:
                print(f"  [hedge scan] Error on {symbol}: {exc}")

        return candidates

    # ── Scan all strategies ───────────────────────────────────────────────────

    def scan_all(self) -> Dict[str, List[Dict]]:
        """Run all enabled strategy scanners. Returns dict keyed by strategy."""
        print("\n[OPTIONS] Scanning strategies...")
        results: Dict[str, List[Dict]] = {}

        print("  → Wheel (CSP) candidates...")
        results["wheel"] = self.scan_wheel_candidates()
        print(f"     Found {len(results['wheel'])} candidates")

        print("  → Iron condor candidates...")
        results["iron_condor"] = self.scan_iron_condor_candidates()
        print(f"     Found {len(results['iron_condor'])} candidates")

        print("  → Credit spread candidates...")
        results["credit_spread"] = self.scan_credit_spread_candidates()
        print(f"     Found {len(results['credit_spread'])} candidates")

        print("  → Earnings play candidates...")
        results["earnings_play"] = self.scan_earnings_plays()
        print(f"     Found {len(results['earnings_play'])} candidates")

        print("  → Hedge opportunities...")
        results["protective_puts"] = self.scan_hedge_opportunities()
        print(f"     Found {len(results['protective_puts'])} candidates")

        return results

    # ── Order submission ──────────────────────────────────────────────────────

    def submit_option_order(self, contract_symbol: str, side: str, qty: int,
                            limit_price: Optional[float] = None) -> Dict:
        """
        Submit a single-leg option order via Alpaca.
        side: "buy" | "sell"
        """
        payload: Dict[str, Any] = {
            "symbol": contract_symbol,
            "qty": str(qty),
            "side": side,
            "type": "limit" if limit_price else "market",
            "time_in_force": "day",
        }
        if limit_price:
            payload["limit_price"] = str(round(limit_price, 2))

        if self.dry_run:
            print(f"  [DRY RUN] Would submit: {side.upper()} {qty}× {contract_symbol}"
                  + (f" @ ${limit_price:.2f}" if limit_price else " @ MKT"))
            return {"success": True, "dry_run": True, "payload": payload}

        return self._post("/v2/orders", payload)

    def submit_spread(self, legs: List[Dict]) -> Dict:
        """
        Submit a multi-leg spread order.
        Each leg: {"symbol": OCC_symbol, "side": "buy"|"sell", "ratio_qty": 1}
        Alpaca multi-leg orders use the same /v2/orders endpoint with order_class=mleg.
        """
        payload: Dict[str, Any] = {
            "type": "limit",
            "time_in_force": "day",
            "order_class": "mleg",
            "legs": legs,
        }
        # Compute net debit/credit limit from mid prices (caller should provide)
        # We'll use market for paper trading simplicity
        payload["type"] = "market"

        if self.dry_run:
            leg_str = ", ".join(
                f"{l['side'].upper()} {l['symbol']}" for l in legs
            )
            print(f"  [DRY RUN] Would submit SPREAD: {leg_str}")
            return {"success": True, "dry_run": True, "payload": payload}

        return self._post("/v2/orders", payload)

    def close_position(self, contract_symbol: str, qty: int,
                       limit_price: Optional[float] = None) -> Dict:
        """Close (buy back) a short option or sell a long option."""
        # To close a short, we buy; to close a long, we sell.
        # We infer direction from existing positions.
        positions = {p["symbol"]: p for p in self.get_options_positions()}
        if contract_symbol not in positions:
            return {"success": False, "error": f"No position found for {contract_symbol}"}

        pos = positions[contract_symbol]
        current_side = pos.get("side", "long")
        close_side = "buy" if current_side == "short" else "sell"

        return self.submit_option_order(contract_symbol, close_side, qty, limit_price)

    # ── Position management ───────────────────────────────────────────────────

    def manage_positions(self) -> List[Dict]:
        """
        Review existing options positions:
          - Close winners at 50 % of max profit
          - Stop losers at 200 % of premium received
          - Roll / close positions at ≤ 7 DTE
        Returns a list of actions taken / recommended.
        """
        actions: List[Dict] = []
        positions = self.get_options_positions()
        portfolio_value = self.get_portfolio_value()

        for pos in positions:
            try:
                symbol = pos.get("symbol", "")
                qty = abs(int(float(pos.get("qty", 0))))
                side = pos.get("side", "long")   # "long" or "short"

                cost_basis = float(pos.get("avg_entry_price", 0))
                current_price = float(pos.get("current_price", cost_basis))
                unrealized_pl = float(pos.get("unrealized_pl", 0))

                # Parse expiry from OCC symbol (positions don't always include dte)
                # OCC symbol format: AAPL240119C00190000
                # Expiry chars: index 4-9 (YYMMDD) for a 4-char ticker
                # Let's try to parse it from symbol
                dte = self._dte_from_occ(symbol)

                action: Optional[Dict] = None

                # 1. Roll / close at ≤ 7 DTE
                if dte is not None and 0 <= dte <= RISK_PARAMS["roll_dte_threshold"]:
                    action = {
                        "action": "close_expiry",
                        "symbol": symbol,
                        "reason": f"Approaching expiry ({dte} DTE)",
                        "qty": qty,
                    }

                # 2. Close winner at 50 % profit (for short positions, premium received × 0.5)
                elif side == "short" and cost_basis > 0:
                    # For a short, P&L > 0 means we profited (option lost value)
                    profit_pct = unrealized_pl / (cost_basis * qty * 100)
                    if profit_pct >= RISK_PARAMS["winner_close_pct"]:
                        action = {
                            "action": "close_winner",
                            "symbol": symbol,
                            "reason": f"50 % profit reached ({profit_pct:.0%})",
                            "qty": qty,
                        }

                # 3. Stop at 200 % of premium received
                elif side == "short" and cost_basis > 0:
                    loss_pct = -unrealized_pl / (cost_basis * qty * 100)
                    if loss_pct >= RISK_PARAMS["loser_close_pct"]:
                        action = {
                            "action": "close_loser",
                            "symbol": symbol,
                            "reason": f"2× stop hit ({loss_pct:.0%} loss)",
                            "qty": qty,
                        }

                if action:
                    if not self.dry_run:
                        result = self.close_position(symbol, qty)
                        action["order_result"] = result
                    else:
                        print(f"  [DRY RUN] Would close: {symbol} — {action['reason']}")
                    actions.append(action)

            except Exception as exc:
                print(f"  [manage] Error processing position {pos.get('symbol')}: {exc}")

        return actions

    def _dte_from_occ(self, occ_symbol: str) -> Optional[int]:
        """Extract DTE from OCC-format symbol like AAPL240119C00190000."""
        try:
            # Strip leading letters (ticker)
            i = 0
            while i < len(occ_symbol) and occ_symbol[i].isalpha():
                i += 1
            date_str = occ_symbol[i:i + 6]   # YYMMDD
            exp = datetime.strptime(date_str, "%y%m%d").date()
            return (exp - date.today()).days
        except Exception:
            return None

    # ── Risk check ────────────────────────────────────────────────────────────

    def check_risk(self, candidates: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """
        Filter candidates by portfolio-level risk constraints:
          - Total options allocation ≤ 40 % of portfolio
          - Per-underlying ≤ 10 % of portfolio
        Returns filtered candidates dict.
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            print("  [risk] Cannot compute risk — portfolio value is 0")
            return candidates

        max_total = portfolio_value * RISK_PARAMS["max_total_options_allocation_pct"]
        max_per_symbol = portfolio_value * RISK_PARAMS["max_single_underlying_pct"]

        # Existing options exposure
        existing_exposure = sum(
            abs(float(p.get("market_value", 0)))
            for p in self.get_options_positions()
        )

        # Track per-symbol exposure (existing + new)
        symbol_exposure: Dict[str, float] = {}
        for p in self.get_options_positions():
            sym = p.get("symbol", "")[:4].rstrip("0123456789")
            val = abs(float(p.get("market_value", 0)))
            symbol_exposure[sym] = symbol_exposure.get(sym, 0) + val

        remaining_budget = max_total - existing_exposure
        filtered: Dict[str, List[Dict]] = {}

        for strategy, cands in candidates.items():
            filtered[strategy] = []
            for c in cands:
                symbol = c.get("symbol", "")
                # Estimate capital at risk
                if strategy == "wheel":
                    risk = c.get("strike", 0) * 100
                elif strategy in ("iron_condor", "earnings_play"):
                    risk = c.get("max_risk", 1000)
                elif strategy == "credit_spread":
                    risk = c.get("max_risk", 1000)
                elif strategy == "protective_puts":
                    risk = c.get("premium_per_contract", 500) * c.get("contracts", 1)
                else:
                    risk = 1000

                if risk > remaining_budget:
                    continue
                if symbol_exposure.get(symbol, 0) + risk > max_per_symbol:
                    continue

                remaining_budget -= risk
                symbol_exposure[symbol] = symbol_exposure.get(symbol, 0) + risk
                filtered[strategy].append(c)

        return filtered

    # ── Execute candidates ────────────────────────────────────────────────────

    def execute_candidates(self, candidates: Dict[str, List[Dict]]) -> List[Dict]:
        """Execute (or dry-run) the top candidates for each strategy."""
        executed: List[Dict] = []

        for strategy, cands in candidates.items():
            cfg = OPTIONS_STRATEGIES.get(strategy, {})
            max_pos = cfg.get("max_positions", 3)

            for c in cands[:max_pos]:
                try:
                    result = self._execute_candidate(strategy, c)
                    executed.append({"strategy": strategy, "candidate": c, "result": result})
                except Exception as exc:
                    print(f"  [execute] Error executing {strategy} on {c.get('symbol')}: {exc}")

        return executed

    def _execute_candidate(self, strategy: str, c: Dict) -> Dict:
        symbol = c.get("symbol", "")
        action = c.get("action", "")

        if strategy == "wheel":
            # Sell 1 cash-secured put
            contract_sym = c.get("contract_symbol", "")
            if not contract_sym:
                contract_sym = _option_symbol(
                    symbol,
                    date.fromisoformat(c["expiry"]),
                    c["strike"],
                    "put",
                )
            premium = c.get("premium", 0)
            # Limit at mid (premium) — use bid × 0.95 for fills
            limit = round(premium * 0.95, 2) if premium > 0 else None
            return self.submit_option_order(contract_sym, "sell", 1, limit)

        elif strategy in ("iron_condor", "earnings_play"):
            expiry_date = date.fromisoformat(c["expiry"])
            legs = [
                {"symbol": _option_symbol(symbol, expiry_date, c["short_put_strike"], "put"),
                 "side": "sell", "ratio_qty": 1},
                {"symbol": _option_symbol(symbol, expiry_date, c["long_put_strike"], "put"),
                 "side": "buy", "ratio_qty": 1},
                {"symbol": _option_symbol(symbol, expiry_date, c["short_call_strike"], "call"),
                 "side": "sell", "ratio_qty": 1},
                {"symbol": _option_symbol(symbol, expiry_date, c["long_call_strike"], "call"),
                 "side": "buy", "ratio_qty": 1},
            ]
            return self.submit_spread(legs)

        elif strategy == "credit_spread":
            expiry_date = date.fromisoformat(c["expiry"])
            option_type = c.get("option_type", "put")
            if c.get("spread_type") in ("bull_put",):
                legs = [
                    {"symbol": _option_symbol(symbol, expiry_date, c["short_strike"], option_type),
                     "side": "sell", "ratio_qty": 1},
                    {"symbol": _option_symbol(symbol, expiry_date, c["long_strike"], option_type),
                     "side": "buy", "ratio_qty": 1},
                ]
            else:   # bear_call
                legs = [
                    {"symbol": _option_symbol(symbol, expiry_date, c["short_strike"], option_type),
                     "side": "sell", "ratio_qty": 1},
                    {"symbol": _option_symbol(symbol, expiry_date, c["long_strike"], option_type),
                     "side": "buy", "ratio_qty": 1},
                ]
            return self.submit_spread(legs)

        elif strategy == "protective_puts":
            contract_sym = _option_symbol(
                symbol,
                date.fromisoformat(c["expiry"]),
                c["strike"],
                "put",
            )
            qty = c.get("contracts", 1)
            premium = c.get("premium_per_contract", 0) / 100
            limit = round(premium * 1.05, 2) if premium > 0 else None
            return self.submit_option_order(contract_sym, "buy", qty, limit)

        return {"success": False, "error": f"Unknown strategy: {strategy}"}
