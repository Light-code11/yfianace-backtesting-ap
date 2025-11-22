"""
Paper trading simulator for live testing strategies
"""
import yfinance as yf
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
from database import PaperTrade, SessionLocal
from backtesting_engine import TechnicalIndicators


class PaperTradingSimulator:
    """Simulate trading with live data without real money"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # {ticker: {qty, entry_price, entry_date, strategy_id}}
        self.indicators = TechnicalIndicators()
        self.db = SessionLocal()

    def get_current_price(self, ticker: str) -> float:
        """Get current market price for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return 0.0

    def get_recent_data(self, ticker: str, days: int = 100) -> pd.DataFrame:
        """Get recent historical data for indicator calculations"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=f"{days}d")
            return data
        except:
            return pd.DataFrame()

    def execute_strategy(
        self,
        strategy: Dict[str, Any],
        live_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Execute a strategy in paper trading mode

        Args:
            strategy: Strategy configuration
            live_data: Optional pre-fetched live data

        Returns:
            Execution results
        """
        results = {
            "strategy_id": strategy.get("id"),
            "strategy_name": strategy.get("name"),
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "positions": [],
            "portfolio_value": 0,
            "cash": self.capital
        }

        tickers = strategy.get("tickers", [])
        risk_mgmt = strategy.get("risk_management", {})
        stop_loss_pct = risk_mgmt.get("stop_loss_pct", 5.0)
        take_profit_pct = risk_mgmt.get("take_profit_pct", 10.0)
        position_size_pct = risk_mgmt.get("position_size_pct", 10.0)
        max_positions = risk_mgmt.get("max_positions", 3)

        # Check existing positions for exit signals
        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]
            current_price = self.get_current_price(ticker)

            if current_price == 0:
                continue

            # Calculate P&L
            entry_price = position['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            # Check stop loss
            if pnl_pct <= -stop_loss_pct:
                self._close_position(ticker, current_price, "stop_loss", strategy.get("id"))
                results["actions_taken"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "reason": "stop_loss",
                    "price": current_price,
                    "pnl_pct": pnl_pct
                })

            # Check take profit
            elif pnl_pct >= take_profit_pct:
                self._close_position(ticker, current_price, "take_profit", strategy.get("id"))
                results["actions_taken"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "reason": "take_profit",
                    "price": current_price,
                    "pnl_pct": pnl_pct
                })

            # Check exit signal
            else:
                recent_data = self.get_recent_data(ticker)
                if not recent_data.empty:
                    indicators_data = self._calculate_strategy_indicators(
                        recent_data, strategy.get("indicators", [])
                    )
                    if self._check_exit_signal(indicators_data, strategy):
                        self._close_position(ticker, current_price, "signal", strategy.get("id"))
                        results["actions_taken"].append({
                            "action": "SELL",
                            "ticker": ticker,
                            "reason": "exit_signal",
                            "price": current_price,
                            "pnl_pct": pnl_pct
                        })

        # Check for new entry signals
        if len(self.positions) < max_positions:
            for ticker in tickers:
                if ticker in self.positions:
                    continue

                current_price = self.get_current_price(ticker)
                if current_price == 0:
                    continue

                recent_data = self.get_recent_data(ticker)
                if recent_data.empty:
                    continue

                indicators_data = self._calculate_strategy_indicators(
                    recent_data, strategy.get("indicators", [])
                )

                if self._check_entry_signal(indicators_data, strategy):
                    # Calculate position size
                    position_value = self.capital * (position_size_pct / 100)
                    qty = position_value / current_price
                    cost = qty * current_price

                    if cost <= self.capital:
                        self._open_position(
                            ticker=ticker,
                            qty=qty,
                            entry_price=current_price,
                            strategy_id=strategy.get("id"),
                            strategy_name=strategy.get("name"),
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct
                        )
                        results["actions_taken"].append({
                            "action": "BUY",
                            "ticker": ticker,
                            "price": current_price,
                            "quantity": qty,
                            "cost": cost
                        })

        # Calculate portfolio value
        portfolio_value = self.capital
        for ticker, position in self.positions.items():
            current_price = self.get_current_price(ticker)
            position_value = position['qty'] * current_price
            portfolio_value += position_value

            results["positions"].append({
                "ticker": ticker,
                "quantity": position['qty'],
                "entry_price": position['entry_price'],
                "current_price": current_price,
                "position_value": position_value,
                "pnl_pct": ((current_price - position['entry_price']) / position['entry_price']) * 100
            })

        results["portfolio_value"] = portfolio_value
        results["cash"] = self.capital
        results["total_return_pct"] = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        return results

    def _open_position(
        self,
        ticker: str,
        qty: float,
        entry_price: float,
        strategy_id: int,
        strategy_name: str,
        stop_loss_pct: float,
        take_profit_pct: float
    ):
        """Open a new position"""
        cost = qty * entry_price
        self.capital -= cost

        self.positions[ticker] = {
            'qty': qty,
            'entry_price': entry_price,
            'entry_date': datetime.now(),
            'strategy_id': strategy_id,
            'cost': cost
        }

        # Save to database
        paper_trade = PaperTrade(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            ticker=ticker,
            action="BUY",
            quantity=qty,
            entry_price=entry_price,
            entry_date=datetime.now(),
            position_size_usd=cost,
            stop_loss_price=entry_price * (1 - stop_loss_pct / 100),
            take_profit_price=entry_price * (1 + take_profit_pct / 100),
            is_open=True
        )
        self.db.add(paper_trade)
        self.db.commit()

    def _close_position(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str,
        strategy_id: int
    ):
        """Close an existing position"""
        if ticker not in self.positions:
            return

        position = self.positions[ticker]
        qty = position['qty']
        entry_price = position['entry_price']

        exit_value = qty * exit_price
        profit_loss = exit_value - position['cost']
        profit_loss_pct = (profit_loss / position['cost']) * 100

        self.capital += exit_value
        del self.positions[ticker]

        # Update database
        trade = self.db.query(PaperTrade).filter(
            PaperTrade.ticker == ticker,
            PaperTrade.is_open == True,
            PaperTrade.strategy_id == strategy_id
        ).first()

        if trade:
            trade.exit_price = exit_price
            trade.exit_date = datetime.now()
            trade.profit_loss_usd = profit_loss
            trade.profit_loss_pct = profit_loss_pct
            trade.is_open = False
            trade.exit_reason = exit_reason
            self.db.commit()

    def _calculate_strategy_indicators(
        self,
        data: pd.DataFrame,
        indicators_config: List[Dict]
    ) -> pd.DataFrame:
        """Calculate indicators for a strategy"""
        df = data.copy()

        for indicator in indicators_config:
            name = indicator.get('name', '').upper()
            period = indicator.get('period', 14)

            if name == 'SMA':
                df[f'SMA_{period}'] = self.indicators.sma(df['Close'], period)
            elif name == 'EMA':
                df[f'EMA_{period}'] = self.indicators.ema(df['Close'], period)
            elif name == 'RSI':
                df['RSI'] = self.indicators.rsi(df['Close'], period)
            elif name == 'MACD':
                macd, signal, hist = self.indicators.macd(df['Close'])
                df['MACD'] = macd
                df['MACD_Signal'] = signal
                df['MACD_Hist'] = hist
            elif name == 'BB' or name == 'BOLLINGER':
                upper, middle, lower = self.indicators.bollinger_bands(df['Close'], period)
                df['BB_Upper'] = upper
                df['BB_Middle'] = middle
                df['BB_Lower'] = lower

        return df

    def _check_entry_signal(self, data: pd.DataFrame, strategy: Dict) -> bool:
        """Check if entry conditions are met"""
        if len(data) < 50:
            return False

        strategy_type = strategy.get('strategy_type', '').lower()
        row = data.iloc[-1]
        prev_row = data.iloc[-2]

        if strategy_type == 'momentum':
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                return row['SMA_20'] > row['SMA_50'] and prev_row['SMA_20'] <= prev_row['SMA_50']
            elif 'RSI' in data.columns:
                return row['RSI'] > 50 and prev_row['RSI'] <= 50

        elif strategy_type == 'mean_reversion':
            if 'BB_Lower' in data.columns:
                return row['Close'] <= row['BB_Lower']
            elif 'RSI' in data.columns:
                return row['RSI'] < 30

        elif strategy_type == 'breakout':
            if 'BB_Upper' in data.columns:
                return row['Close'] > row['BB_Upper']

        elif strategy_type == 'trend_following':
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                return row['MACD'] > row['MACD_Signal'] and prev_row['MACD'] <= prev_row['MACD_Signal']

        return False

    def _check_exit_signal(self, data: pd.DataFrame, strategy: Dict) -> bool:
        """Check if exit conditions are met"""
        if len(data) < 50:
            return False

        strategy_type = strategy.get('strategy_type', '').lower()
        row = data.iloc[-1]

        if strategy_type == 'momentum':
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                return row['SMA_20'] < row['SMA_50']

        elif strategy_type == 'mean_reversion':
            if 'BB_Middle' in data.columns:
                return row['Close'] >= row['BB_Middle']

        elif strategy_type == 'breakout':
            if 'BB_Middle' in data.columns:
                return row['Close'] < row['BB_Middle']

        elif strategy_type == 'trend_following':
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                return row['MACD'] < row['MACD_Signal']

        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get paper trading performance summary"""
        all_trades = self.db.query(PaperTrade).all()

        closed_trades = [t for t in all_trades if not t.is_open]
        open_positions = [t for t in all_trades if t.is_open]

        total_pnl = sum(t.profit_loss_usd for t in closed_trades if t.profit_loss_usd)
        winning_trades = [t for t in closed_trades if t.profit_loss_usd and t.profit_loss_usd > 0]
        losing_trades = [t for t in closed_trades if t.profit_loss_usd and t.profit_loss_usd <= 0]

        return {
            "initial_capital": self.initial_capital,
            "current_cash": self.capital,
            "total_trades": len(closed_trades),
            "open_positions": len(open_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
        }

    def __del__(self):
        """Cleanup database connection"""
        self.db.close()
