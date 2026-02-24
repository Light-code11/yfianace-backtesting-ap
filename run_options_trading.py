"""
Options Trading Daily Runner

Runs alongside the equity bot (5 min after).
Schedule:
  01:50 AEDT — new trade scan + execution
  06:05 AEDT — position management (close winners, stop losers, roll near-expiry)

Usage:
  python run_options_trading.py              # live paper trading
  python run_options_trading.py --dry-run   # preview only (no orders)
  python run_options_trading.py --manage    # position management pass only
"""
import argparse
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from options_config import (
    ALPACA_CONFIG,
    OPTIONS_STRATEGIES,
    OPTIONS_UNIVERSE,
    RISK_PARAMS,
)
from options_engine import OptionsEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }


def verify_alpaca() -> dict:
    """Return account info + market clock, or error."""
    base_url = ALPACA_CONFIG["base_url"]
    timeout = (
        ALPACA_CONFIG["connect_timeout_sec"],
        ALPACA_CONFIG["read_timeout_sec"],
    )
    headers = _alpaca_headers()
    try:
        account_resp = requests.get(f"{base_url}/v2/account",
                                    headers=headers, timeout=timeout)
        account_resp.raise_for_status()
        account = account_resp.json()

        clock_resp = requests.get(f"{base_url}/v2/clock",
                                  headers=headers, timeout=timeout)
        clock_resp.raise_for_status()
        clock = clock_resp.json()

        return {
            "success": True,
            "account": account,
            "clock": clock,
            "is_open": bool(clock.get("is_open")),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _print_banner(dry_run: bool, mode: str = "FULL") -> None:
    print("=" * 60)
    print("   OPTIONS TRADING ENGINE")
    print("=" * 60)
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode:    {'DRY RUN — ' if dry_run else 'LIVE PAPER — '}{mode}")
    print(f"  Universe:{len(OPTIONS_UNIVERSE)} tickers")
    enabled = [s for s, c in OPTIONS_STRATEGIES.items() if c.get("enabled")]
    print(f"  Strats:  {', '.join(enabled)}")
    print("=" * 60)


def _print_account(account: dict) -> None:
    cash = float(account.get("cash", 0))
    portfolio = float(account.get("portfolio_value", 0))
    buying_power = float(account.get("buying_power", 0))
    print(f"\n  Account: ACTIVE")
    print(f"  Cash:          ${cash:>12,.2f}")
    print(f"  Portfolio:     ${portfolio:>12,.2f}")
    print(f"  Buying Power:  ${buying_power:>12,.2f}")


def _print_candidates(candidates: dict, filtered: dict) -> None:
    total_raw = sum(len(v) for v in candidates.values())
    total_filtered = sum(len(v) for v in filtered.values())
    print(f"\n── Scan Results ──────────────────────────────────────")
    print(f"  Total candidates found:   {total_raw}")
    print(f"  After risk filter:        {total_filtered}")
    for strategy, cands in filtered.items():
        if not cands:
            continue
        print(f"\n  [{strategy.upper()}] {len(cands)} candidates:")
        for c in cands:
            sym = c.get("symbol", "")
            act = c.get("action", "")
            if strategy == "wheel":
                print(
                    f"    {sym:6s} | {act:12s} | "
                    f"Strike=${c.get('strike', 0):.1f} | "
                    f"Premium=${c.get('premium', 0):.2f} ({c.get('premium_pct', 0):.1%}) | "
                    f"IV Rank={c.get('iv_rank', 0):.0f}% | "
                    f"DTE={c.get('dte', 0)} | "
                    f"ML={c.get('ml_direction')}"
                )
            elif strategy in ("iron_condor", "earnings_play"):
                print(
                    f"    {sym:6s} | {act:20s} | "
                    f"Strikes={c.get('short_put_strike', 0):.0f}P/"
                    f"{c.get('short_call_strike', 0):.0f}C | "
                    f"Credit=${c.get('net_credit', 0):.2f} | "
                    f"IV Rank={c.get('iv_rank', 0):.0f}% | "
                    f"DTE={c.get('dte', 0)}"
                )
            elif strategy == "credit_spread":
                print(
                    f"    {sym:6s} | {act:12s} | "
                    f"Short={c.get('short_strike', 0):.0f} / Long={c.get('long_strike', 0):.0f} | "
                    f"Net credit~${c.get('net_credit_est', 0):.2f} | "
                    f"ML={c.get('ml_direction')} ({c.get('ml_confidence', 0):.0%})"
                )
            elif strategy == "protective_puts":
                print(
                    f"    {sym:6s} | {act:10s} | "
                    f"Strike=${c.get('strike', 0):.1f} | "
                    f"Contracts={c.get('contracts', 1)} | "
                    f"DTE={c.get('dte', 0)}"
                )
            else:
                print(f"    {sym:6s} | {act}")


def _print_manage_actions(actions: list) -> None:
    if not actions:
        print("\n── Position Management ───────────────────────────────")
        print("  No positions requiring action.")
        return
    print(f"\n── Position Management ({len(actions)} actions) ─────────────────")
    for a in actions:
        print(
            f"  [{a.get('action', '?').upper():15s}] "
            f"{a.get('symbol', '?'):25s} — {a.get('reason', '')}"
        )


def _print_executed(executed: list, dry_run: bool) -> None:
    label = "DRY RUN — would execute" if dry_run else "Executed"
    print(f"\n── Trades ({label}) ────────────────────────────────")
    if not executed:
        print("  None.")
        return
    for e in executed:
        strategy = e.get("strategy", "?")
        c = e.get("candidate", {})
        r = e.get("result", {})
        status = "OK" if r.get("success") else f"FAIL: {r.get('error', '?')}"
        print(f"  [{strategy:14s}] {c.get('symbol', '?'):6s} | {c.get('action', '?'):20s} | {status}")


def _print_summary(candidates: dict, filtered: dict, executed: list,
                   manage_actions: list, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Candidates scanned:  {sum(len(v) for v in candidates.values())}")
    print(f"  After risk filter:   {sum(len(v) for v in filtered.values())}")
    print(f"  Trades executed:     {len(executed)}")
    print(f"  Positions managed:   {len(manage_actions)}")
    print(f"  Mode: {'DRY RUN (no orders placed)' if dry_run else 'LIVE PAPER TRADING'}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Options paper trading daily runner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview trades without submitting any orders"
    )
    parser.add_argument(
        "--manage", action="store_true",
        help="Only run position management pass (no new trades)"
    )
    args = parser.parse_args()

    mode = "MANAGE ONLY" if args.manage else "FULL CYCLE"
    _print_banner(args.dry_run, mode)

    # ── 1. Verify Alpaca connection ───────────────────────────────────────────
    print("\n[1] Checking Alpaca connection...")
    broker = verify_alpaca()
    if not broker["success"]:
        if args.dry_run:
            print(f"  WARNING: Alpaca unreachable — {broker.get('error')}")
            print("  Continuing in dry-run mode with zero portfolio value.")
        else:
            print(f"  ERROR: Cannot connect to Alpaca — {broker.get('error')}")
            return 1
    else:
        _print_account(broker["account"])

    # ── 2. Initialise engine ──────────────────────────────────────────────────
    engine = OptionsEngine(dry_run=args.dry_run)

    # ── 3. Position management ────────────────────────────────────────────────
    print("\n[2] Managing existing options positions...")
    manage_actions = engine.manage_positions()
    _print_manage_actions(manage_actions)

    if args.manage:
        # Management-only pass (06:05 AEDT run)
        _print_summary({}, {}, [], manage_actions, args.dry_run)
        return 0

    # ── 4. Scan for new opportunities ─────────────────────────────────────────
    print("\n[3] Scanning for new opportunities...")
    candidates = engine.scan_all()

    # ── 5. Risk check ─────────────────────────────────────────────────────────
    print("\n[4] Applying risk filters...")
    filtered = engine.check_risk(candidates)
    _print_candidates(candidates, filtered)

    # ── 6. Execute ────────────────────────────────────────────────────────────
    print(f"\n[5] {'(DRY RUN) ' if args.dry_run else ''}Executing trades...")
    executed = engine.execute_candidates(filtered)
    _print_executed(executed, args.dry_run)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    _print_summary(candidates, filtered, executed, manage_actions, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
