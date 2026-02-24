"""
Daily trading cycle runner.
Called by cron with: python run_daily_trading.py
"""
import argparse
import os
import sys
from datetime import datetime

import requests
import yfinance as yf
from dotenv import load_dotenv

from database import SessionLocal, Strategy, init_db
from trading_config import STRATEGIES, TICKER_UNIVERSE, RISK_PARAMS, ALPACA_CONFIG, PAIRS, PAIR_TRADING_PARAMS


os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def apply_runtime_config() -> None:
    """Set runtime environment defaults from trading_config."""
    os.environ.setdefault("AUTO_TRADING_ENABLED", str(RISK_PARAMS.get("auto_trading_enabled", True)).lower())
    os.environ.setdefault("MAX_POSITION_SIZE_PCT", str(RISK_PARAMS.get("max_position_size_pct", 20.0)))
    os.environ.setdefault("MAX_DAILY_LOSS_PCT", str(RISK_PARAMS.get("max_daily_loss_pct", 5.0)))
    os.environ.setdefault("MAX_PORTFOLIO_POSITIONS", str(RISK_PARAMS.get("max_positions", 8)))
    os.environ.setdefault("MAX_CORRELATION", str(RISK_PARAMS.get("correlation_limit", 0.70)))
    os.environ.setdefault("MIN_SIGNAL_CONFIDENCE", str(RISK_PARAMS.get("min_signal_confidence", "MEDIUM")))
    os.environ.setdefault("ALPACA_BASE_URL", str(ALPACA_CONFIG.get("base_url", "https://paper-api.alpaca.markets")))
    os.environ.setdefault("ALPACA_REQUEST_TIMEOUT", str(ALPACA_CONFIG.get("request_timeout_sec", 15)))
    os.environ.setdefault("ALPACA_CONNECT_TIMEOUT", str(ALPACA_CONFIG.get("connect_timeout_sec", 5)))
    os.environ.setdefault("ALPACA_READ_TIMEOUT", str(ALPACA_CONFIG.get("read_timeout_sec", 20)))


def configure_yfinance_cache() -> None:
    """Force yfinance cache to a writable local directory."""
    cache_dir = os.path.join(os.getcwd(), ".yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        yf.set_tz_cache_location(cache_dir)
    except Exception:
        pass


def _strategy_to_db_payload(strategy_cfg):
    risk = strategy_cfg.get("risk_management", {})
    strategy_type = strategy_cfg.get("strategy_type", "unknown")
    # Pair trading strategy manages its own tickers via PAIRS config; don't override with universe.
    raw_tickers = strategy_cfg.get("tickers")
    if raw_tickers is None:
        tickers = TICKER_UNIVERSE
    elif not raw_tickers and strategy_type == "pair_trading":
        # Pair trading uses PAIRS, not TICKER_UNIVERSE — keep empty
        tickers = []
    else:
        tickers = raw_tickers or TICKER_UNIVERSE
    return {
        "description": strategy_cfg.get("description", ""),
        "tickers": tickers,
        "strategy_type": strategy_type,
        "indicators": strategy_cfg.get("indicators", []),
        "entry_conditions": {
            "atr_stop_multiplier": float(strategy_cfg.get("atr_stop_multiplier", 2.0))
        },
        "exit_conditions": {},
        "stop_loss_pct": float(risk.get("stop_loss_pct", 5.0)),
        "take_profit_pct": float(risk.get("take_profit_pct", 10.0)),
        "position_size_pct": float(risk.get("position_size_pct", 10.0)),
        "holding_period_days": 5,
        "rationale": "Config-managed strategy for autonomous live paper trading",
        "market_analysis": "Daily + weekly multi-timeframe confirmation enabled",
        "risk_assessment": "ATR-based stops and position caps enforced",
    }


def sync_configured_strategies() -> dict:
    """Upsert configured strategies so the autonomous engine always has active strategies."""
    db = SessionLocal()
    created = 0
    updated = 0
    activated = 0

    try:
        configured_names = set()
        for strategy_cfg in STRATEGIES:
            if not strategy_cfg.get("enabled", True):
                continue

            name = strategy_cfg["name"]
            configured_names.add(name)
            payload = _strategy_to_db_payload(strategy_cfg)

            strategy = db.query(Strategy).filter(Strategy.name == name).first()
            if not strategy:
                strategy = Strategy(name=name, is_active=True, is_paper_trading=True, **payload)
                db.add(strategy)
                created += 1
            else:
                for key, value in payload.items():
                    setattr(strategy, key, value)
                if not strategy.is_active:
                    activated += 1
                strategy.is_active = True
                strategy.is_paper_trading = True
                updated += 1

        # Deactivate config-managed strategies removed from config.
        managed = db.query(Strategy).filter(Strategy.name.like("cfg_%")).all()
        for strategy in managed:
            if strategy.name not in configured_names and strategy.is_active:
                strategy.is_active = False

        db.commit()
        active_count = db.query(Strategy).filter(Strategy.is_active == True).count()

        return {
            "success": True,
            "created": created,
            "updated": updated,
            "reactivated": activated,
            "active_strategies": active_count,
        }
    except Exception as exc:
        db.rollback()
        return {"success": False, "error": str(exc)}
    finally:
        db.close()


def verify_alpaca_and_market_open() -> dict:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return {"success": False, "error": "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY"}

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    connect_timeout = float(os.getenv("ALPACA_CONNECT_TIMEOUT", "5"))
    read_timeout = float(os.getenv("ALPACA_READ_TIMEOUT", "20"))
    timeout = (connect_timeout, read_timeout)

    try:
        account_resp = requests.get(f"{base_url}/v2/account", headers=headers, timeout=timeout)
        account_resp.raise_for_status()
        account = account_resp.json()

        clock_resp = requests.get(f"{base_url}/v2/clock", headers=headers, timeout=timeout)
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


def print_signal_preview(preview: dict) -> None:
    print("\n=== Dry Run Summary ===")
    print(f"Success:           {preview.get('success', False)}")
    print(f"Ticker universe:   {len(TICKER_UNIVERSE)} tickers")
    print(f"Pair configs:      {len(PAIRS)} pairs")
    print(f"Strategy count:    {len(STRATEGIES)} strategies configured")
    print(f"Stocks scanned:    {preview.get('stocks_scanned', 0)}")
    print(f"Signals generated: {preview.get('signals_generated', 0)}")
    print(f"Actionable signals:{preview.get('actionable_signals', 0)}")

    candidates = preview.get("would_trade", [])
    regular = [c for c in candidates if not c.get("is_pair_trade")]
    pair_cands = [c for c in candidates if c.get("is_pair_trade")]

    if regular:
        print("\nWould trade (directional):")
        for item in regular[:20]:
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):6s} | "
                f"Conf={item.get('confidence'):6s} | "
                f"Size={item.get('position_size_pct', 0):.1f}% | "
                f"Price=${item.get('current_price')} | "
                f"Strategy={item.get('strategy_name')}"
            )
    else:
        print("Would trade (directional): none")

    if pair_cands:
        print("\nWould trade (pair legs):")
        for item in pair_cands[:20]:
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):6s} | "
                f"Pair={item.get('pair_ticker_a')}/{item.get('pair_ticker_b')} | "
                f"Z={item.get('pair_zscore', 0):.2f} | "
                f"Conf={item.get('confidence')}"
            )
    else:
        print("Would trade (pair legs): none (no pairs met z-score threshold)")

    for err in preview.get("errors", []):
        print(f"Error: {err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autonomous daily trading cycle")
    parser.add_argument("--dry-run", action="store_true", help="Generate/evaluate signals but do not execute orders")
    args = parser.parse_args()

    print("=== Daily Trading Cycle ===")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'LIVE PAPER TRADING'}")
    print(f"Tickers:    {len(TICKER_UNIVERSE)}")
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Pairs:      {len(PAIRS)}")

    init_db()
    apply_runtime_config()
    configure_yfinance_cache()

    strategy_sync = sync_configured_strategies()
    if not strategy_sync.get("success"):
        print(f"ERROR syncing strategies: {strategy_sync.get('error')}")
        return 1

    print(
        "Strategies synced: "
        f"created={strategy_sync['created']}, "
        f"updated={strategy_sync['updated']}, "
        f"reactivated={strategy_sync['reactivated']}, "
        f"active_total={strategy_sync['active_strategies']}"
    )

    broker = verify_alpaca_and_market_open()
    if not broker.get("success"):
        if args.dry_run:
            print(f"Warning: Alpaca check failed in dry run: {broker.get('error')}")
        else:
            print(f"ERROR connecting to Alpaca: {broker.get('error')}")
            return 1
    else:
        account = broker["account"]
        print(
            f"Account: ACTIVE | Cash: ${float(account['cash']):,.2f} | "
            f"Portfolio: ${float(account['portfolio_value']):,.2f}"
        )

        if not args.dry_run and not broker.get("is_open"):
            print(f"Market is CLOSED. Next open: {broker['clock'].get('next_open', 'unknown')}")
            print("Skipping live trading cycle.")
            return 0

    from autonomous_trading_engine import AutonomousTradingEngine

    engine = AutonomousTradingEngine(max_correlation=float(RISK_PARAMS.get("correlation_limit", 0.70)))

    try:
        if args.dry_run:
            preview = engine.preview_daily_cycle()
            print_signal_preview(preview)
            return 0 if preview.get("success") else 1

        results = engine.run_daily_cycle()
        print("\n=== Live Cycle Summary ===")
        print(f"Success: {results.get('success', False)}")
        print(f"Stocks scanned: {results.get('stocks_scanned', 0)}")
        print(f"Signals generated: {results.get('signals_generated', 0)}")
        print(f"Actionable signals: {results.get('actionable_signals', 0)}")
        print(f"Trades executed: {results.get('trades_executed', 0)}")
        print(f"Trades rejected: {results.get('trades_rejected', 0)}")

        executed = results.get("executed_tickers", [])
        rejected = results.get("rejected_tickers", [])
        print(f"Executed tickers: {', '.join(executed) if executed else 'none'}")
        print(f"Rejected tickers: {', '.join(rejected) if rejected else 'none'}")

        if results.get("error"):
            print(f"Error: {results['error']}")
        for err in results.get("errors", []):
            if err:
                print(f"  - {err}")

        return 0 if results.get("success") else 1
    finally:
        engine.db.close()


if __name__ == "__main__":
    sys.exit(main())
