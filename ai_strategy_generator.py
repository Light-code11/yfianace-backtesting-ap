"""
AI-powered trading strategy generator using OpenAI GPT-4
"""
import os
import json
import time
from typing import List, Dict, Any
from openai import OpenAI
from datetime import datetime
import pandas as pd
import httpx
import yfinance as yf

# Benchmark mapping: Stock → (Sector ETF, Broad Market)
BENCHMARK_MAP = {
    # Semiconductors
    'NVDA': ('SOXX', 'SPY'),  # SOXX = iShares Semiconductor ETF
    'AMD': ('SOXX', 'SPY'),
    'INTC': ('SOXX', 'SPY'),
    'TSM': ('SOXX', 'SPY'),
    'AVGO': ('SOXX', 'SPY'),

    # Big Tech (FAANG+)
    'AAPL': ('QQQ', 'SPY'),
    'MSFT': ('QQQ', 'SPY'),
    'GOOGL': ('QQQ', 'SPY'),
    'GOOG': ('QQQ', 'SPY'),
    'META': ('QQQ', 'SPY'),
    'AMZN': ('QQQ', 'SPY'),
    'NFLX': ('QQQ', 'SPY'),
    'TSLA': ('QQQ', 'SPY'),

    # ETFs (no sector, just broad market)
    'QQQ': (None, 'SPY'),
    'SPY': (None, None),
    'DIA': (None, 'SPY'),
    'IWM': (None, 'SPY'),
}

class AIStrategyGenerator:
    def __init__(self, api_key: str = None):
        # Strip whitespace and newlines from API key (Railway adds these sometimes)
        raw_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_key = raw_key.strip() if raw_key else None

        # Create HTTP client with proper timeout settings
        http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),  # 60s total timeout, 10s connect timeout
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

        # Initialize OpenAI client with custom HTTP client
        self.client = OpenAI(
            api_key=self.api_key,
            http_client=http_client,
            max_retries=3  # Built-in retry mechanism
        )
        # Use gpt-4o for superior reasoning and strategy generation
        self.model = "gpt-4o"  # Upgraded for better strategy quality

    def generate_strategies(
        self,
        market_data: pd.DataFrame,
        tickers: List[str],
        num_strategies: int = 3,
        past_performance: List[Dict] = None,
        learning_insights: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate trading strategies using AI

        Args:
            market_data: Historical price data
            tickers: List of ticker symbols
            num_strategies: Number of strategies to generate
            past_performance: Historical strategy performance data
            learning_insights: AI learning insights from past runs

        Returns:
            List of strategy dictionaries
        """
        # Prepare market analysis
        market_summary = self._analyze_market_data(market_data, tickers)

        # Build prompt
        prompt = self._build_strategy_prompt(
            market_summary=market_summary,
            tickers=tickers,
            num_strategies=num_strategies,
            past_performance=past_performance,
            learning_insights=learning_insights
        )

        # Generate strategies with retry logic
        response = self._call_openai_with_retry(prompt, max_retries=3)

        # Parse response
        strategies_json = json.loads(response.choices[0].message.content)
        strategies = strategies_json.get("strategies", [])

        return strategies

    def _call_openai_with_retry(self, prompt: str, max_retries: int = 3):
        """
        Call OpenAI API with exponential backoff retry logic
        """
        models_to_try = [self.model, "gpt-4o-mini", "gpt-3.5-turbo"]  # Fallback models

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"Attempting OpenAI call with model={model}, attempt={attempt+1}/{max_retries}")

                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert quantitative trader and strategy developer. Generate data-driven, backtestable trading strategies with clear entry/exit rules. IMPORTANT: Return ONLY valid JSON, no additional text."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=3000
                    )

                    print(f"OpenAI call successful with model={model}")
                    return response

                except Exception as e:
                    error_msg = str(e)
                    print(f"Attempt {attempt+1} with {model} failed: {error_msg}")

                    # If this is the last attempt with this model, try the next model
                    if attempt == max_retries - 1:
                        print(f"All retries exhausted for {model}, trying next model...")
                        break

                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # If all models and retries failed, raise an exception
        raise Exception(f"Failed to generate strategies after trying all models and retries. Last error: Connection timeout or API unavailable.")

    def _analyze_market_data(self, market_data: pd.DataFrame, tickers: List[str]) -> Dict[str, Any]:
        """Analyze market data with relative strength and market regime detection"""
        analysis = {
            "period": {
                "start": market_data.index.min().strftime("%Y-%m-%d"),
                "end": market_data.index.max().strftime("%Y-%m-%d"),
                "days": len(market_data)
            },
            "tickers": tickers,
            "market_regime": self._detect_market_regime(market_data),
            "market_stats": {}
        }

        for ticker in tickers:
            if ticker in market_data.columns.levels[1]:
                close_prices = market_data['Close'][ticker]

                # Calculate key statistics
                total_return = ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100
                volatility = close_prices.pct_change().std() * (252 ** 0.5) * 100

                # Trend analysis
                ma_20 = close_prices.rolling(20).mean().iloc[-1]
                ma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else None
                current_price = close_prices.iloc[-1]

                trend = "bullish" if current_price > ma_20 else "bearish"
                if ma_50 and ma_20 > ma_50:
                    trend = "strong_bullish"
                elif ma_50 and ma_20 < ma_50:
                    trend = "strong_bearish"

                ticker_stats = {
                    "total_return_pct": round(total_return, 2),
                    "volatility_pct": round(volatility, 2),
                    "current_price": round(current_price, 2),
                    "ma_20": round(ma_20, 2),
                    "ma_50": round(ma_50, 2) if ma_50 else None,
                    "trend": trend
                }

                # Add relative strength analysis
                relative_strength = self._calculate_relative_strength(ticker, close_prices, market_data)
                if relative_strength:
                    ticker_stats["relative_strength"] = relative_strength

                analysis["market_stats"][ticker] = ticker_stats

        return analysis

    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect overall market regime (bull, bear, sideways)"""
        try:
            # Use SPY as market proxy
            if 'Close' in market_data.columns and 'SPY' in market_data.columns.levels[1]:
                spy_prices = market_data['Close']['SPY']

                # Calculate moving averages
                sma_20 = spy_prices.rolling(20).mean().iloc[-1]
                sma_50 = spy_prices.rolling(50).mean().iloc[-1] if len(spy_prices) >= 50 else None
                sma_200 = spy_prices.rolling(200).mean().iloc[-1] if len(spy_prices) >= 200 else None
                current_price = spy_prices.iloc[-1]

                # Calculate momentum
                return_20d = ((spy_prices.iloc[-1] / spy_prices.iloc[-20]) - 1) * 100 if len(spy_prices) >= 20 else 0

                # Regime detection logic
                if sma_200 and current_price > sma_200 and return_20d > 2:
                    return "strong_bull"
                elif sma_50 and current_price > sma_50 and return_20d > 0:
                    return "bull"
                elif sma_200 and current_price < sma_200 and return_20d < -2:
                    return "strong_bear"
                elif sma_50 and current_price < sma_50 and return_20d < 0:
                    return "bear"
                else:
                    return "sideways"

            return "unknown"
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return "unknown"

    def _calculate_relative_strength(self, ticker: str, ticker_prices: pd.Series, market_data: pd.DataFrame) -> Dict:
        """Calculate relative strength vs sector and market benchmarks"""
        try:
            if ticker not in BENCHMARK_MAP:
                return None

            sector_etf, market_etf = BENCHMARK_MAP[ticker]
            relative_strength = {}

            # Calculate ticker return
            ticker_return = ((ticker_prices.iloc[-1] / ticker_prices.iloc[0]) - 1) * 100

            # Compare to sector ETF
            if sector_etf and 'Close' in market_data.columns and sector_etf in market_data.columns.levels[1]:
                sector_prices = market_data['Close'][sector_etf]
                sector_return = ((sector_prices.iloc[-1] / sector_prices.iloc[0]) - 1) * 100
                relative_strength['vs_sector'] = {
                    'benchmark': sector_etf,
                    'ticker_return': round(ticker_return, 2),
                    'benchmark_return': round(sector_return, 2),
                    'outperformance': round(ticker_return - sector_return, 2),
                    'strength': 'outperforming' if ticker_return > sector_return else 'underperforming'
                }

            # Compare to broad market
            if market_etf and 'Close' in market_data.columns and market_etf in market_data.columns.levels[1]:
                market_prices = market_data['Close'][market_etf]
                market_return = ((market_prices.iloc[-1] / market_prices.iloc[0]) - 1) * 100
                relative_strength['vs_market'] = {
                    'benchmark': market_etf,
                    'ticker_return': round(ticker_return, 2),
                    'benchmark_return': round(market_return, 2),
                    'outperformance': round(ticker_return - market_return, 2),
                    'strength': 'outperforming' if ticker_return > market_return else 'underperforming'
                }

            return relative_strength if relative_strength else None

        except Exception as e:
            print(f"Error calculating relative strength for {ticker}: {e}")
            return None

    def _build_strategy_prompt(
        self,
        market_summary: Dict,
        tickers: List[str],
        num_strategies: int,
        past_performance: List[Dict],
        learning_insights: List[Dict]
    ) -> str:
        """Build the prompt for strategy generation"""

        # Extract market regime for emphasis
        market_regime = market_summary.get('market_regime', 'unknown')

        # Build relative strength summary
        rs_summary = []
        for ticker, stats in market_summary.get('market_stats', {}).items():
            if 'relative_strength' in stats:
                rs_data = stats['relative_strength']
                if 'vs_sector' in rs_data:
                    rs_summary.append(f"{ticker} vs {rs_data['vs_sector']['benchmark']}: {rs_data['vs_sector']['strength']} ({rs_data['vs_sector']['outperformance']:+.1f}%)")
                if 'vs_market' in rs_data:
                    rs_summary.append(f"{ticker} vs {rs_data['vs_market']['benchmark']}: {rs_data['vs_market']['strength']} ({rs_data['vs_market']['outperformance']:+.1f}%)")

        prompt = f"""Generate {num_strategies} distinct trading strategies based on the following market data and analysis.

🌍 CURRENT MARKET REGIME: {market_regime.upper()}
This is CRITICAL - strategies MUST be appropriate for {market_regime} conditions!

MARKET DATA ANALYSIS:
{json.dumps(market_summary, indent=2)}

TICKERS TO ANALYZE: {', '.join(tickers)}

📊 RELATIVE STRENGTH ANALYSIS:
{chr(10).join(rs_summary) if rs_summary else 'No relative strength data available'}

MARKET REGIME STRATEGY GUIDELINES:
- strong_bull: Favor momentum and breakout strategies, higher position sizes
- bull: Balanced approach, trend-following with moderate stops
- sideways: Mean reversion strategies, tighter stops, range-bound trading
- bear/strong_bear: Defensive strategies, tight stops, focus on high-quality setups only

RELATIVE STRENGTH GUIDELINES:
- If ticker is OUTPERFORMING sector/market: Favor long momentum strategies
- If ticker is UNDERPERFORMING sector/market: Be more conservative, tighter stops, or avoid

"""

        if past_performance:
            prompt += f"""
PAST STRATEGY PERFORMANCE (Learn from this):
{json.dumps(past_performance[-10:], indent=2)}

Key lessons:
- Strategies with Sharpe ratio > 1.0 performed well
- High win rates (>60%) are desirable but not essential if profit factor is good
- Avoid strategies that had max drawdown > 25%
"""

        if learning_insights:
            prompt += f"""
AI LEARNING INSIGHTS:
{json.dumps(learning_insights, indent=2)}
"""

        # Add requirements section with safety constraints
        prompt += f"""
CRITICAL SAFETY CONSTRAINTS (MUST FOLLOW):
1. Stop Loss: MAXIMUM 3% (never use 5% or higher - causes excessive losses)
2. Position Size: MAXIMUM 5% per trade (never use 10% or higher)
3. Win Rate Target: Aim for strategies with 55%+ win rate potential
4. Risk/Reward: Minimum 2:1 ratio (take_profit should be 2x stop_loss)
5. Max Trades Per Day: Limit to 2-3 to avoid overtrading

PROVEN STRATEGY TEMPLATES (Use these as base, then customize):
1. RSI Mean Reversion: Buy when RSI < 30, sell when RSI > 70, stop_loss=2%, take_profit=4%
2. Golden Cross: Buy when SMA(50) > SMA(200), sell when opposite, stop_loss=3%, take_profit=6%
3. Momentum Breakout: Buy on 20-day high, sell on 10-day low, stop_loss=2.5%, take_profit=5%
4. MACD Trend: Buy on MACD crossover + price > SMA(20), stop_loss=2%, take_profit=4%

REQUIREMENTS:
Generate diverse strategies using these proven approaches:
1. RSI mean reversion (oversold/overbought)
2. Moving average crossovers (trend following)
3. Momentum breakouts (new highs/lows)
4. MACD + trend confirmation

IMPORTANT: Each strategy MUST use the tickers: {', '.join(tickers)}
The "tickers" field in EVERY strategy response must be: {json.dumps(tickers)}

For EACH strategy, provide:
{{
  "strategies": [
    {{
      "name": "Strategy name (descriptive)",
      "description": "Brief description of the strategy",
      "strategy_type": "momentum|mean_reversion|breakout|trend_following",
      "tickers": {json.dumps(tickers)},
      "indicators": [
        {{"name": "SMA", "period": 20}},
        {{"name": "RSI", "period": 14}}
      ],
      "entry_conditions": {{
        "primary": "Specific entry condition",
        "secondary": "Confirmation signal",
        "filters": ["Additional filters"]
      }},
      "exit_conditions": {{
        "profit_target": "Take profit condition",
        "stop_loss": "Stop loss condition",
        "time_based": "Time-based exit (optional)"
      }},
      "risk_management": {{
        "stop_loss_pct": 2.5,
        "take_profit_pct": 5.0,
        "position_size_pct": 5.0,
        "max_positions": 3
      }},
      "holding_period_days": 5,
      "rationale": "Why this strategy should work in current market conditions",
      "market_analysis": "Current market environment analysis",
      "risk_assessment": "Potential risks and mitigation"
    }}
  ]
}}

Make strategies:
- SPECIFIC and ACTIONABLE with clear numeric thresholds
- DIVERSE in approach and methodology
- RISK-MANAGED with clear stop losses
- REALISTIC based on current market conditions
- BACKTESTABLE with quantifiable entry/exit rules

Return ONLY valid JSON, no additional text.
"""

        return prompt

    def generate_recommendations(
        self,
        backtest_results: List[Dict],
        strategy_details: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate actionable recommendations based on backtest results
        """
        prompt = f"""Analyze these backtested trading strategies and provide actionable recommendations.

BACKTEST RESULTS:
{json.dumps(backtest_results, indent=2)}

STRATEGY DETAILS:
{json.dumps(strategy_details, indent=2)}

Provide a comprehensive analysis including:
1. Best performing strategy and why
2. Worst performing strategy and what to avoid
3. Overall market insights
4. Specific action items
5. Risk warnings
6. Expected outcomes for top strategies

Return response as JSON:
{{
  "executive_summary": "Brief overview",
  "best_strategy": {{
    "name": "Strategy name",
    "reason": "Why it performed best",
    "action_items": ["Specific actions to take"]
  }},
  "market_insights": ["Key market observations"],
  "recommendations": ["Actionable recommendations"],
  "risk_warnings": ["Important risks to consider"],
  "next_steps": ["What to do next"]
}}
"""

        response = self._call_openai_with_retry(prompt, max_retries=3)
        recommendations = json.loads(response.choices[0].message.content)
        return recommendations

    def learn_from_results(
        self,
        strategies: List[Dict],
        backtest_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Extract learning insights from strategy performance
        """
        prompt = f"""Analyze these trading strategy results and extract key learnings for future strategy generation.

STRATEGIES:
{json.dumps(strategies, indent=2)}

BACKTEST RESULTS:
{json.dumps(backtest_results, indent=2)}

Identify patterns in:
1. What made successful strategies work
2. Common failures to avoid
3. Market conditions that favor certain approaches
4. Optimal parameter ranges
5. Risk management lessons

Return as JSON:
{{
  "success_patterns": [
    {{
      "pattern": "Description",
      "evidence": "Why this pattern works",
      "confidence": 0.85
    }}
  ],
  "failure_patterns": [
    {{
      "pattern": "Description",
      "evidence": "Why this failed",
      "confidence": 0.75
    }}
  ],
  "parameter_insights": {{
    "stop_loss": "Optimal range",
    "position_size": "Recommended size",
    "holding_period": "Ideal duration"
  }},
  "market_insights": ["Key market observations"],
  "recommendations_for_next_generation": ["How to improve future strategies"]
}}
"""

        response = self._call_openai_with_retry(prompt, max_retries=3)
        learning = json.loads(response.choices[0].message.content)
        return learning
