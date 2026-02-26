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
from trading_config import (
    STRATEGIES, TICKER_UNIVERSE, RISK_PARAMS, ALPACA_CONFIG,
    PAIRS, PAIR_TRADING_PARAMS,
    CRYPTO_UNIVERSE, CRYPTO_STRATEGIES, CRYPTO_PAIRS, CRYPTO_RISK_PARAMS,
)

# Insider signal amplifier (optional — warm cache at startup for speed)
try:
    import insider_amplifier as _insider_amp
    _INSIDER_AMP_AVAILABLE = True
except ImportError:
    _INSIDER_AMP_AVAILABLE = False

# Crypto regime filter (BTC SMA + Fear & Greed Index)
try:
    from crypto_regime_filter import get_crypto_regime as _get_crypto_regime, print_crypto_regime as _print_crypto_regime
    _CRYPTO_REGIME_AVAILABLE = True
except ImportError:
    _CRYPTO_REGIME_AVAILABLE = False
    _get_crypto_regime = None
    _print_crypto_regime = None

# Earnings catalyst (pre-earnings + PEAD swing trades)
try:
    from earnings_catalyst import (
        print_upcoming_earnings_preview as _earnings_preview,
        scan_earnings_calendar as _scan_earnings,
    )
    _EARNINGS_CATALYST_AVAILABLE = True
except ImportError:
    _EARNINGS_CATALYST_AVAILABLE = False
    _earnings_preview = None
    _scan_earnings = None


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
    os.environ.setdefault("MIN_CONVICTION_SCORE", str(RISK_PARAMS.get("min_conviction_score", 55.0)))
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


def warm_insider_cache(quiet: bool = False) -> dict:
    """
    Pre-fetch SEC EDGAR Form 4 insider scores for all tickers in TICKER_UNIVERSE.

    Called once at daily cycle startup so the insider_amplifier's in-memory cache
    is populated before signal generation.  The EDGAR client caches results in
    SQLite (edgar_cache.db) so subsequent runs within the same TTL window are fast.

    Args:
        quiet: Suppress per-ticker output (set False to see each ticker's score).

    Returns:
        Dict of ticker → insider score, or empty dict if amplifier unavailable.
    """
    if not _INSIDER_AMP_AVAILABLE:
        print("⚠️  insider_amplifier not available — skipping insider cache warm")
        return {}

    try:
        # Equity tickers only — crypto assets have no EDGAR Form 4 filings
        equity_tickers = [t for t in TICKER_UNIVERSE if "/" not in t]
        scores = _insider_amp.warm_cache(
            tickers      = equity_tickers,
            lookback_days= 30,
            quiet        = quiet,
        )
        return scores
    except Exception as exc:
        print(f"⚠️  Insider cache warming failed (non-fatal): {exc}")
        return {}


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
    print(f"Equities:          {len(TICKER_UNIVERSE)} tickers | {len(PAIRS)} pairs | {len(STRATEGIES)} strategies")
    print(f"Crypto:            {len(CRYPTO_UNIVERSE)} tickers ({', '.join(CRYPTO_UNIVERSE)}) | {len(CRYPTO_PAIRS)} pair(s)")
    print(f"Stocks scanned:    {preview.get('stocks_scanned', 0)}")
    print(f"Signals generated: {preview.get('signals_generated', 0)}")
    print(f"Actionable signals:{preview.get('actionable_signals', 0)}")

    # ── Crypto Regime display ─────────────────────────────────────────────────
    if preview.get("crypto_regime"):
        _emoji_c = {
            "CRYPTO_BULL": "🚀", "CRYPTO_NEUTRAL": "📊",
            "CRYPTO_BEAR": "🐻", "CRYPTO_FEAR": "😱", "CRYPTO_GREED": "🤑",
        }.get(preview["crypto_regime"], "₿")
        _fg    = preview.get("crypto_fg_score", "?")
        _fglbl = preview.get("crypto_fg_label", "")
        _gc    = preview.get("crypto_golden_cross")
        _btc   = preview.get("crypto_btc_price")
        _exp   = preview.get("crypto_exposure_mult", 1.0)
        print(f"\n{_emoji_c} CRYPTO REGIME: {preview['crypto_regime']} | "
              f"F&G={_fg} ({_fglbl}) | GoldenCross={_gc} | "
              f"Exposure={_exp:.0%}" + (f" | BTC=${_btc:,.0f}" if _btc else ""))

    # ── Macro Regime Filter display ───────────────────────────────────────
    if preview.get('market_regime'):
        _emoji = {
            "STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "📊",
            "BEAR": "📉", "CRISIS": "🚨",
        }.get(preview['market_regime'], "❓")
        _exposure = preview.get('regime_exposure_mult', 1.0)
        _stop     = preview.get('regime_stop_mult', 1.5)
        _bias     = preview.get('regime_direction_bias', 'unknown')
        _vix      = preview.get('regime_vix')
        _yc       = preview.get('regime_yield_spread')
        _breadth  = preview.get('regime_breadth')
        print(f"\n{_emoji} MACRO REGIME: {preview['market_regime']}")
        print(f"   Exposure multiplier: {_exposure:.0%}  |  Stop multiplier: {_stop:.1f}×ATR")
        print(f"   Direction bias:      {_bias}")
        if _vix is not None:
            print(f"   VIX:                 {_vix:.1f}")
        if _yc is not None:
            print(f"   Yield curve (10Y-2Y):{_yc:+.2f}%")
        if _breadth is not None:
            print(f"   Market breadth:      {_breadth:.0%} above 50-SMA")
        _desc = preview.get('regime_description')
        if _desc:
            print(f"   → {_desc}")
        # Raw signals detail
        _raw = preview.get('regime_signals', {})
        if _raw:
            print("   Signals detail:")
            for k, v in _raw.items():
                print(f"     {k}: {v}")
    elif preview.get('regime_confidence') is not None:
        # Legacy HMM regime fallback display
        conf = preview.get('regime_confidence', 0)
        print(f"Market regime:     {preview.get('market_regime')} ({conf*100:.0f}% confidence)")

    candidates = preview.get("would_trade", [])
    # Split: crypto vs equity, directional vs pair
    crypto_regular = [c for c in candidates if c.get("is_crypto") and not c.get("is_pair_trade")]
    crypto_pairs   = [c for c in candidates if c.get("is_crypto") and c.get("is_pair_trade")]
    regular   = [c for c in candidates if not c.get("is_crypto") and not c.get("is_pair_trade")]
    pair_cands = [c for c in candidates if not c.get("is_crypto") and c.get("is_pair_trade")]

    if regular:
        print("\nWould trade (directional):")
        for item in regular[:20]:
            insider_tag = ""
            if item.get("insider_swing"):
                insider_tag = " ⭐INSIDER-SWING"
            elif item.get("insider_bias") is not None:
                b = float(item["insider_bias"])
                if b >= 0.7:
                    insider_tag = f" [CLUSTER_BUY {b:+.2f}]"
                elif b >= 0.3:
                    insider_tag = f" [INSIDER_BUY {b:+.2f}]"
                elif b <= -0.3:
                    insider_tag = f" [INSIDER_SELL {b:+.2f}]"

            # Conviction display
            conviction_tag = ""
            if item.get("conviction_score") is not None:
                cv = item["conviction_score"]
                tier = (item.get("conviction_tier") or "").upper()
                conviction_tag = f" | CV={cv:.0f}({tier})"
                # Show breakdown if available
                bd = item.get("conviction_breakdown")
                if bd:
                    conviction_tag += (
                        f"[q={bd.get('quality_pts', 0):.0f}"
                        f"+i={bd.get('insider_pts', 0):+.0f}"
                        f"+c={bd.get('confluence_pts', 0):.0f}"
                        f"+r={bd.get('regime_pts', 0):.0f}]"
                    )

            # Macro regime tag
            regime_tag = ""
            if item.get("macro_regime"):
                _mult = item.get("macro_exposure_mult", 1.0) or 1.0
                regime_tag = f" | Regime={item['macro_regime']}({_mult:.0%})"

            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):6s} | "
                f"Conf={item.get('confidence'):6s} | "
                f"Size={item.get('position_size_pct', 0):.1f}% | "
                f"Q={item.get('quality_score', 0):.0f} | "
                f"Price=${item.get('current_price')} | "
                f"Strategy={item.get('strategy_name')}"
                f"{conviction_tag}"
                f"{regime_tag}"
                f"{insider_tag}"
            )
    else:
        print("Would trade (directional): none")

    if pair_cands:
        print("\nWould trade (equity pair legs):")
        for item in pair_cands[:20]:
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):6s} | "
                f"Pair={item.get('pair_ticker_a')}/{item.get('pair_ticker_b')} | "
                f"Z={item.get('pair_zscore', 0):.2f} | "
                f"Conf={item.get('confidence')}"
            )
    else:
        print("Would trade (equity pair legs): none (no pairs met z-score threshold)")

    # ── Crypto Signals display ─────────────────────────────────────────────
    if crypto_regular:
        print("\nWould trade (crypto directional):")
        for item in crypto_regular[:10]:
            confluence_tag = ""
            if item.get("high_confluence"):
                confluence_tag = f" 🔀HIGH_CONFLUENCE({item.get('confluence_count',0)+1})"
            elif item.get("confluence_count", 0) >= 1:
                confluence_tag = f" 🔀x{item.get('confluence_count',0)+1}"
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):9s} | "
                f"Conf={item.get('confidence'):6s} | "
                f"Size={item.get('position_size_pct', 0):.1f}% | "
                f"Q={item.get('quality_score', 0):.0f} | "
                f"RSI={item.get('rsi14', '?')} | "
                f"Price=${item.get('current_price'):,.0f} | "
                f"Regime={item.get('crypto_regime','?')}"
                f"{confluence_tag}"
            )
    else:
        print("Would trade (crypto directional): none")

    if crypto_pairs:
        print("\nWould trade (crypto pair legs):")
        for item in crypto_pairs[:6]:
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):9s} | "
                f"Pair={item.get('pair_ticker_a')}/{item.get('pair_ticker_b')} | "
                f"Z={item.get('pair_zscore', 0):.2f} | "
                f"Conf={item.get('confidence')}"
            )
    else:
        print("Would trade (crypto pair legs): none")

    # ── Earnings Catalyst signals display ─────────────────────────────────
    earnings_cands = [c for c in candidates if c.get("strategy_type") == "earnings_catalyst"]
    if earnings_cands:
        print("\nWould trade (earnings catalyst):")
        for item in earnings_cands[:10]:
            surprise_str = ""
            if item.get("recent_surprises"):
                surps = item["recent_surprises"][:2]
                surprise_str = f" | Recent={surps}"
            print(
                f"  {item.get('signal'):4s} {item.get('ticker'):6s} | "
                f"T-{item.get('days_until_earnings', '?')} ({item.get('earnings_date', 'unknown')}) | "
                f"Size={item.get('position_size_pct', 0):.1f}% | "
                f"Insider={float(item.get('insider_bias', 0)):+.2f} | "
                f"BeatRate={item.get('beat_rate', 0) or 0:.0%} | "
                f"Stop={item.get('stop_loss_pct', 0):.1f}% | "
                f"Q={item.get('quality_score', 0):.0f}"
                f"{surprise_str}"
            )

    for err in preview.get("errors", []):
        print(f"Error: {err}")

    rejects = preview.get("entry_gate_rejections", {})
    if rejects:
        print("\nEntry gate rejections by reason:")
        for reason, count in sorted(rejects.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {count:3d}  {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autonomous daily trading cycle")
    parser.add_argument("--dry-run", action="store_true", help="Generate/evaluate signals but do not execute orders")
    parser.add_argument("--check-exits", action="store_true", help="Check open positions for trailing/time/breakeven stop exits (close bot mode)")
    args = parser.parse_args()

    _mode = "DRY RUN" if args.dry_run else ("CHECK EXITS" if args.check_exits else "LIVE PAPER TRADING")
    print("=== Daily Trading Cycle ===")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode:       {_mode}")
    print(f"Equities:   {len(TICKER_UNIVERSE)} tickers | {len(STRATEGIES)} strategies | {len(PAIRS)} pairs")
    print(f"Crypto:     {len(CRYPTO_UNIVERSE)} tickers ({', '.join(CRYPTO_UNIVERSE)}) | "
          f"{len([s for s in CRYPTO_STRATEGIES if s.get('enabled',True)])} strategies | "
          f"{len(CRYPTO_PAIRS)} pair(s)")

    init_db()
    apply_runtime_config()
    configure_yfinance_cache()

    # ── Insider cache warm (EDGAR Form 4 — equity tickers only, not crypto) ─────────
    print("\n📋 Pre-fetching insider signals (EDGAR Form 4 — equity only)…")
    warm_insider_cache(quiet=True)  # uses TICKER_UNIVERSE internally (no BTC/USD etc.)

    # ── Crypto regime check ──────────────────────────────────────────────────
    if _CRYPTO_REGIME_AVAILABLE and _get_crypto_regime is not None:
        try:
            print("\n₿ Fetching crypto regime (BTC SMA + Fear & Greed Index)…")
            _crypto_regime = _get_crypto_regime()
            _print_crypto_regime(_crypto_regime)
        except Exception as _cre:
            print(f"⚠️  Crypto regime fetch failed (non-fatal): {_cre}")

    # ── Earnings calendar scan (upcoming earnings in next 14 days) ──────────
    if _EARNINGS_CATALYST_AVAILABLE and _earnings_preview is not None:
        try:
            if args.dry_run:
                # Full preview in dry-run mode
                _portfolio_val = 100_000.0
                _earnings_preview(
                    TICKER_UNIVERSE,
                    lookahead_days=14,
                    portfolio_value=_portfolio_val,
                    quiet=False,
                )
            else:
                # Quiet calendar scan (just print upcoming dates)
                print("\n📅 Scanning earnings calendar (next 14 days)…")
                upcoming = _scan_earnings(TICKER_UNIVERSE, lookahead_days=14, quiet=True)
                if upcoming:
                    print(f"   {len(upcoming)} earnings events in next 14 days:")
                    for ev in upcoming[:10]:
                        print(f"     {ev['ticker']:<8} → {ev['earnings_date']} (T-{ev['days_until']})")
                    if len(upcoming) > 10:
                        print(f"     ... and {len(upcoming)-10} more")
                else:
                    print("   No earnings in next 14 days for tracked tickers")
        except Exception as _earn_exc:
            print(f"⚠️  Earnings calendar scan failed (non-fatal): {_earn_exc}")

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

        if args.check_exits:
            print("\n=== Position Exit Check (Close Bot) ===")
            exit_results = engine.check_position_exits()
            print("\n=== Exit Check Summary ===")
            print(f"Positions checked: {exit_results.get('checked', 0)}")
            print(f"Positions closed:  {exit_results.get('closed', 0)}")
            print(f"Stops updated:     {exit_results.get('updated', 0)}")
            exits = exit_results.get("exits", [])
            if exits:
                print("\nClosed positions:")
                for ex in exits:
                    print(
                        f"  {ex['ticker']:6s} | {ex['exit_reason']:18s} | "
                        f"Entry=${ex['entry']:.2f} → Current=${ex['current']:.2f} | "
                        f"P&L={ex['pnl_pct']:+.1f}%"
                    )
            for err in exit_results.get("errors", []):
                print(f"Error: {err}")
            return 0 if not exit_results.get("errors") else 1

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
    _exit_code = main()
    # Flush stdout/stderr before force-exiting so buffered output is not lost
    # when stdout is redirected to a file (file redirection uses block buffering).
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    # Force immediate process exit — bypasses any residual non-daemon threads
    # (e.g. from yfinance's multitasking library) that would otherwise block
    # Python's normal shutdown and make the script appear to "hang".
    os._exit(_exit_code)
