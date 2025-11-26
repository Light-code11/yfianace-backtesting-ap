"""
Market Scanner - Scan hundreds of stocks to find the best trading opportunities
With integrated backtesting to show historical performance of recommended strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from live_signal_generator import LiveSignalGenerator
from backtesting_engine import BacktestingEngine


class MarketScanner:
    """
    Scan large universe of stocks to find best trading opportunities

    Ranks signals by:
    - Sharpe ratio from backtests
    - Signal confidence (HIGH/MEDIUM/LOW)
    - Risk/Reward ratio
    - Multi-strategy confirmation
    """

    # Curated universe of top 50 liquid stocks across all major sectors
    # Optimized for fast scanning with thread-safe yfinance downloads
    DEFAULT_UNIVERSE = {
        # Technology (Top 10)
        "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA", "CRM", "NFLX",

        # Finance (Top 8)
        "JPM", "BAC", "WFC", "GS", "V", "MA", "AXP", "BLK",

        # Healthcare (Top 6)
        "JNJ", "UNH", "PFE", "ABBV", "LLY", "TMO",

        # Consumer (Top 6)
        "WMT", "HD", "MCD", "NKE", "COST", "DIS",

        # Energy (Top 4)
        "XOM", "CVX", "COP", "SLB",

        # Industrials (Top 4)
        "BA", "CAT", "GE", "HON",

        # Crypto/Alternative (4)
        "COIN", "MARA", "MSTR", "HOOD",

        # Major ETFs (8)
        "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "XLE", "XLK",
    }

    @staticmethod
    def scan_market(
        strategies: List[Dict[str, Any]],
        universe: Optional[List[str]] = None,
        max_workers: int = 10,
        min_confidence: str = "LOW"
    ) -> Dict[str, Any]:
        """
        Scan market for best opportunities

        Args:
            strategies: List of strategy configs to test
            universe: List of tickers to scan (default: curated 150+ stocks)
            max_workers: Number of parallel threads
            min_confidence: Filter signals by confidence (LOW/MEDIUM/HIGH)

        Returns:
            Dict with ranked opportunities
        """
        if universe is None:
            universe = sorted(list(MarketScanner.DEFAULT_UNIVERSE))

        print(f"🔍 Scanning {len(universe)} stocks with {len(strategies)} strategies...")

        all_signals = []
        scanned_count = 0
        error_count = 0

        # Scan each stock with all strategies in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    MarketScanner._scan_ticker,
                    ticker,
                    strategies
                ): ticker
                for ticker in universe
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    signals = future.result()
                    if signals:
                        all_signals.extend(signals)
                    scanned_count += 1

                    if scanned_count % 20 == 0:
                        print(f"  Scanned {scanned_count}/{len(universe)} stocks...")

                except Exception as e:
                    error_count += 1
                    print(f"  ⚠️ Error scanning {ticker}: {str(e)[:50]}")

        print(f"✅ Scan complete: {scanned_count} stocks, {len(all_signals)} signals, {error_count} errors")

        # Filter by confidence
        confidence_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_conf_value = confidence_order.get(min_confidence, 1)

        filtered_signals = [
            s for s in all_signals
            if confidence_order.get(s.get('confidence', 'LOW'), 1) >= min_conf_value
        ]

        # Rank signals
        ranked_signals = MarketScanner._rank_signals(filtered_signals)

        # Separate BUY and SELL signals
        buy_signals = [s for s in ranked_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in ranked_signals if s['signal'] == 'SELL']

        return {
            "scan_time": datetime.now().isoformat(),
            "stocks_scanned": scanned_count,
            "strategies_used": len(strategies),
            "total_signals": len(all_signals),
            "filtered_signals": len(filtered_signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "top_buys": buy_signals[:20],  # Top 20 BUY opportunities
            "top_sells": sell_signals[:20],  # Top 20 SELL opportunities
            "all_signals": ranked_signals
        }

    @staticmethod
    def _scan_ticker(ticker: str, strategies: List[Dict]) -> List[Dict]:
        """Scan a single ticker with all strategies"""
        signals = []

        for strategy in strategies:
            try:
                # Generate signal for this ticker with this strategy
                strategy_config = {
                    **strategy,
                    'tickers': [ticker]  # Override tickers
                }

                result = LiveSignalGenerator.generate_signals(strategy_config, period="3mo")

                if result and result.get('signals'):
                    for sig in result['signals']:
                        if sig.get('signal') in ['BUY', 'SELL']:
                            # Add strategy metadata
                            sig['strategy_name'] = strategy.get('name', 'Unknown')
                            sig['strategy_type'] = strategy.get('strategy_type', 'unknown')
                            sig['strategy_id'] = strategy.get('id')

                            # Calculate quality score
                            sig['quality_score'] = MarketScanner._calculate_quality_score(sig, strategy)

                            signals.append(sig)

            except Exception as e:
                # Silently skip errors for individual ticker+strategy combos
                continue

        return signals

    @staticmethod
    def _calculate_quality_score(signal: Dict, strategy: Dict) -> float:
        """
        Calculate quality score for ranking (0-100)

        Factors:
        - Confidence (HIGH=30, MEDIUM=20, LOW=10)
        - Risk/Reward ratio (higher is better, max 30)
        - Strategy Sharpe from backtest (max 20)
        - Stop loss tightness (tighter = better, max 10)
        - Multi-strategy confirmation bonus (max 10)
        """
        score = 0.0

        # Confidence score (30 points max)
        confidence_scores = {"HIGH": 30, "MEDIUM": 20, "LOW": 10}
        score += confidence_scores.get(signal.get('confidence', 'LOW'), 10)

        # Risk/Reward ratio (30 points max)
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        current_price = signal.get('current_price')

        if stop_loss and take_profit and current_price:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk > 0:
                rr_ratio = reward / risk
                # Cap at 5:1 for scoring
                score += min(rr_ratio * 6, 30)

        # Strategy Sharpe ratio (20 points max)
        # Would need to pass this from backtest results
        # For now, use a default medium value
        score += 10

        # Stop loss tightness (10 points max)
        # Tighter stops (smaller risk %) = better
        if stop_loss and current_price:
            risk_pct = abs((current_price - stop_loss) / current_price * 100)
            # Prefer 2-5% stops, penalize >10%
            if risk_pct <= 5:
                score += 10
            elif risk_pct <= 10:
                score += 5
            # else 0 points

        return round(score, 1)

    @staticmethod
    def _rank_signals(signals: List[Dict]) -> List[Dict]:
        """Rank signals by quality score (highest first)"""
        return sorted(signals, key=lambda x: x.get('quality_score', 0), reverse=True)

    @staticmethod
    def backtest_signal(
        signal: Dict,
        strategy: Dict,
        period: str = "1y",
        initial_capital: float = 10000
    ) -> Dict[str, Any]:
        """
        Backtest a strategy on a specific ticker to show historical performance

        Args:
            signal: The signal dict containing ticker info
            strategy: The strategy configuration
            period: Historical period to backtest (default: 1 year)
            initial_capital: Starting capital for backtest

        Returns:
            Dict with backtest metrics and equity curve
        """
        ticker = signal.get('ticker')
        if not ticker:
            return {'success': False, 'error': 'No ticker in signal'}

        try:
            # Download historical data
            data = yf.download(ticker, period=period, progress=False)

            if data.empty or len(data) < 60:
                return {
                    'success': False,
                    'error': f'Insufficient data for {ticker}'
                }

            # Prepare data for backtesting engine (multi-index format)
            data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])

            # Create strategy config for this ticker
            backtest_strategy = {
                **strategy,
                'tickers': [ticker]
            }

            # Run backtest
            engine = BacktestingEngine(initial_capital=initial_capital)
            results = engine.backtest_strategy(backtest_strategy, data)

            # Extract key metrics
            metrics = results.get('metrics', {})
            trades = results.get('trades', [])
            equity_curve = results.get('equity_curve', [])

            return {
                'success': True,
                'ticker': ticker,
                'strategy_name': strategy.get('name', 'Unknown'),
                'period': period,
                'metrics': {
                    'total_return_pct': metrics.get('total_return_pct', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'avg_win': metrics.get('avg_win', 0),
                    'avg_loss': metrics.get('avg_loss', 0),
                    'quality_score': metrics.get('quality_score', 0)
                },
                'trade_count': len(trades),
                'equity_curve': equity_curve[-30:] if len(equity_curve) > 30 else equity_curve,  # Last 30 days
                'last_trades': trades[-5:] if len(trades) > 5 else trades  # Last 5 trades
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }

    @staticmethod
    def scan_with_backtest(
        strategies: List[Dict[str, Any]],
        universe: Optional[List[str]] = None,
        max_workers: int = 5,
        min_confidence: str = "LOW",
        backtest_period: str = "6mo"
    ) -> Dict[str, Any]:
        """
        Scan market and include backtest results for each signal

        Args:
            strategies: List of strategy configs to test
            universe: List of tickers to scan
            max_workers: Number of parallel threads
            min_confidence: Filter signals by confidence
            backtest_period: Period to use for backtesting (3mo, 6mo, 1y)

        Returns:
            Dict with ranked opportunities including backtest performance
        """
        # First, run the regular scan
        scan_results = MarketScanner.scan_market(
            strategies=strategies,
            universe=universe,
            max_workers=max_workers,
            min_confidence=min_confidence
        )

        print(f"📊 Running backtests on top signals...")

        # Get top signals to backtest (limit to avoid too many API calls)
        top_buys = scan_results.get('top_buys', [])[:10]
        top_sells = scan_results.get('top_sells', [])[:10]

        # Create a map of strategy_id to strategy config
        strategy_map = {s.get('id'): s for s in strategies if s.get('id')}

        # Backtest top buy signals
        backtested_buys = []
        for signal in top_buys:
            strategy_id = signal.get('strategy_id')
            strategy = strategy_map.get(strategy_id, strategies[0] if strategies else {})

            backtest_result = MarketScanner.backtest_signal(
                signal=signal,
                strategy=strategy,
                period=backtest_period
            )

            # Merge backtest results into signal
            signal_with_backtest = {
                **signal,
                'backtest': backtest_result
            }

            # Update quality score based on backtest results
            if backtest_result.get('success'):
                bt_metrics = backtest_result.get('metrics', {})
                # Boost quality score if backtest shows positive results
                original_score = signal.get('quality_score', 0)
                bt_bonus = 0

                if bt_metrics.get('total_return_pct', 0) > 0:
                    bt_bonus += min(bt_metrics['total_return_pct'] / 5, 10)  # Up to 10 points for returns
                if bt_metrics.get('sharpe_ratio', 0) > 1:
                    bt_bonus += 5  # 5 points for Sharpe > 1
                if bt_metrics.get('win_rate', 0) > 50:
                    bt_bonus += 5  # 5 points for win rate > 50%

                signal_with_backtest['quality_score'] = min(original_score + bt_bonus, 100)
                signal_with_backtest['backtest_adjusted'] = True

            backtested_buys.append(signal_with_backtest)

        # Backtest top sell signals
        backtested_sells = []
        for signal in top_sells:
            strategy_id = signal.get('strategy_id')
            strategy = strategy_map.get(strategy_id, strategies[0] if strategies else {})

            backtest_result = MarketScanner.backtest_signal(
                signal=signal,
                strategy=strategy,
                period=backtest_period
            )

            signal_with_backtest = {
                **signal,
                'backtest': backtest_result
            }

            if backtest_result.get('success'):
                bt_metrics = backtest_result.get('metrics', {})
                original_score = signal.get('quality_score', 0)
                bt_bonus = 0

                if bt_metrics.get('total_return_pct', 0) > 0:
                    bt_bonus += min(bt_metrics['total_return_pct'] / 5, 10)
                if bt_metrics.get('sharpe_ratio', 0) > 1:
                    bt_bonus += 5
                if bt_metrics.get('win_rate', 0) > 50:
                    bt_bonus += 5

                signal_with_backtest['quality_score'] = min(original_score + bt_bonus, 100)
                signal_with_backtest['backtest_adjusted'] = True

            backtested_sells.append(signal_with_backtest)

        # Re-rank by updated quality scores
        backtested_buys = sorted(backtested_buys, key=lambda x: x.get('quality_score', 0), reverse=True)
        backtested_sells = sorted(backtested_sells, key=lambda x: x.get('quality_score', 0), reverse=True)

        # Update scan results
        scan_results['top_buys'] = backtested_buys
        scan_results['top_sells'] = backtested_sells
        scan_results['backtest_period'] = backtest_period
        scan_results['backtests_run'] = len(backtested_buys) + len(backtested_sells)

        print(f"✅ Backtests complete: {scan_results['backtests_run']} signals analyzed")

        return scan_results

    @staticmethod
    def get_multi_strategy_confirmations(signals: List[Dict]) -> List[Dict]:
        """
        Find tickers where multiple strategies agree (higher conviction)

        Returns signals grouped by ticker with confirmation count
        """
        ticker_signals = {}

        for sig in signals:
            ticker = sig['ticker']
            signal_type = sig['signal']  # BUY or SELL

            key = f"{ticker}_{signal_type}"

            if key not in ticker_signals:
                ticker_signals[key] = {
                    'ticker': ticker,
                    'signal': signal_type,
                    'strategies': [],
                    'confirmation_count': 0,
                    'avg_quality_score': 0,
                    'signals': []
                }

            ticker_signals[key]['strategies'].append(sig['strategy_name'])
            ticker_signals[key]['confirmation_count'] += 1
            ticker_signals[key]['signals'].append(sig)

        # Calculate average quality score for each ticker
        confirmations = []
        for key, data in ticker_signals.items():
            if data['confirmation_count'] >= 2:  # At least 2 strategies agree
                avg_score = sum(s.get('quality_score', 0) for s in data['signals']) / len(data['signals'])
                data['avg_quality_score'] = round(avg_score, 1)

                # Use best signal's details
                best_signal = max(data['signals'], key=lambda x: x.get('quality_score', 0))
                data.update({
                    'current_price': best_signal.get('current_price'),
                    'stop_loss': best_signal.get('stop_loss'),
                    'take_profit': best_signal.get('take_profit'),
                    'confidence': best_signal.get('confidence'),
                    'reasoning': f"Confirmed by {data['confirmation_count']} strategies: {', '.join(data['strategies'])}"
                })

                confirmations.append(data)

        # Sort by confirmation count, then quality score
        return sorted(
            confirmations,
            key=lambda x: (x['confirmation_count'], x['avg_quality_score']),
            reverse=True
        )
