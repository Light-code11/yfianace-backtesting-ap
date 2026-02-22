"""
Portfolio Correlation Heatmap utilities.

# Streamlit usage:
# import plotly.figure_factory as ff
# data = requests.get('http://localhost:8000/portfolio/correlation/live').json()
# fig = ff.create_annotated_heatmap(data['correlation_matrix'], x=data['tickers'], y=data['tickers'])
# st.plotly_chart(fig)
"""
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yfinance as yf

from alpaca_client import AlpacaClient


class PortfolioHeatmap:
    """Generate correlation heatmaps and diversification analytics."""

    DEFAULT_WATCHLIST = [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
        "AMD", "GOOGL", "AMZN", "TSLA", "META"
    ]

    def _normalize_tickers(self, tickers: Optional[List[str]]) -> List[str]:
        if not tickers:
            return []
        normalized: List[str] = []
        for ticker in tickers:
            cleaned = str(ticker).strip().upper()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _fetch_close_prices(
        self,
        tickers: List[str],
        period: str = "6mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        raw = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )

        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close_prices = raw["Close"].copy()
            else:
                close_prices = raw.xs("Close", axis=1, level=-1, drop_level=False).copy()
        else:
            series = raw["Close"] if "Close" in raw.columns else raw.squeeze()
            close_prices = pd.DataFrame({tickers[0]: series})

        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=tickers[0])

        close_prices = close_prices.dropna(how="all")
        close_prices.columns = [str(col).upper() for col in close_prices.columns]
        return close_prices

    def generate_correlation_matrix(
        self,
        tickers: List[str],
        period: str = "6mo",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        symbols = self._normalize_tickers(tickers)
        if not symbols:
            return {
                "matrix": [],
                "tickers": [],
                "stats": {
                    "avg_correlation": 0.0,
                    "max_pair": None,
                    "min_pair": None
                }
            }

        close_prices = self._fetch_close_prices(symbols, period=period, interval=interval)
        available = [t for t in symbols if t in close_prices.columns]
        if not available:
            return {
                "matrix": [],
                "tickers": symbols,
                "stats": {
                    "avg_correlation": 0.0,
                    "max_pair": None,
                    "min_pair": None
                }
            }

        returns = close_prices[available].pct_change().dropna(how="all")
        if returns.empty:
            identity = np.eye(len(available), dtype=float).tolist()
            return {
                "matrix": identity,
                "tickers": available,
                "stats": {
                    "avg_correlation": 0.0,
                    "max_pair": None,
                    "min_pair": None
                }
            }

        corr_df = returns.corr(method="pearson").fillna(0.0)
        np.fill_diagonal(corr_df.values, 1.0)
        matrix = corr_df.values.astype(float).tolist()

        pairs: List[Dict[str, Any]] = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                pairs.append({
                    "pair": [available[i], available[j]],
                    "correlation": float(corr_df.iloc[i, j])
                })

        if pairs:
            avg_corr = float(np.mean([p["correlation"] for p in pairs]))
            max_pair = max(pairs, key=lambda p: p["correlation"])
            min_pair = min(pairs, key=lambda p: p["correlation"])
            max_pair_stats = {
                "tickers": max_pair["pair"],
                "correlation": round(max_pair["correlation"], 4)
            }
            min_pair_stats = {
                "tickers": min_pair["pair"],
                "correlation": round(min_pair["correlation"], 4)
            }
        else:
            avg_corr = 0.0
            max_pair_stats = None
            min_pair_stats = None

        return {
            "matrix": matrix,
            "tickers": available,
            "stats": {
                "avg_correlation": round(avg_corr, 4),
                "max_pair": max_pair_stats,
                "min_pair": min_pair_stats
            }
        }

    def get_risk_clusters(
        self,
        correlation_matrix: List[List[float]],
        tickers: List[str],
        threshold: float = 0.7
    ) -> List[List[str]]:
        if not correlation_matrix or not tickers:
            return []

        n = min(len(correlation_matrix), len(tickers))
        graph: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                try:
                    corr = float(correlation_matrix[i][j])
                except (IndexError, TypeError, ValueError):
                    continue
                if corr > threshold:
                    graph[i].add(j)
                    graph[j].add(i)

        visited: Set[int] = set()
        clusters: List[List[str]] = []
        for node in range(n):
            if node in visited or not graph[node]:
                continue
            stack = [node]
            component: List[int] = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                for nxt in graph[cur]:
                    if nxt not in visited:
                        stack.append(nxt)
            if len(component) > 1:
                clusters.append(sorted([tickers[idx] for idx in component]))

        return sorted(clusters, key=lambda c: (-len(c), c))

    def calculate_diversification_score(self, correlation_matrix: List[List[float]]) -> float:
        if not correlation_matrix:
            return 100.0

        n = len(correlation_matrix)
        if n <= 1:
            return 100.0

        off_diagonal_abs: List[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    off_diagonal_abs.append(abs(float(correlation_matrix[i][j])))
                except (IndexError, TypeError, ValueError):
                    continue

        if not off_diagonal_abs:
            return 100.0

        avg_abs_corr = float(np.mean(off_diagonal_abs))
        score = max(0.0, min(100.0, (1.0 - avg_abs_corr) * 100.0))
        return round(score, 2)

    def _get_position_tickers(self) -> List[str]:
        try:
            client = AlpacaClient()
            result = client.get_positions()
            if not result.get("success"):
                return []
            symbols = [p.get("symbol", "") for p in result.get("positions", [])]
            return self._normalize_tickers(symbols)
        except Exception:
            return []

    def get_warnings(self, matrix: List[List[float]], tickers: List[str]) -> List[str]:
        if not matrix or len(tickers) <= 1:
            return []

        high_corr_pairs = []
        highly_linked: Set[str] = set()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr = float(matrix[i][j])
                if corr > 0.8:
                    high_corr_pairs.append((tickers[i], tickers[j], corr))
                    highly_linked.add(tickers[i])
                    highly_linked.add(tickers[j])

        warnings: List[str] = []
        if highly_linked:
            warnings.append(f"{len(highly_linked)} positions have >0.8 correlation")
        if len(high_corr_pairs) >= 5:
            warnings.append("Portfolio concentration risk: many highly correlated pairs")
        return warnings

    def generate_heatmap_data(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        normalized = self._normalize_tickers(tickers)
        source = "request"

        if not normalized:
            normalized = self._get_position_tickers()
            source = "alpaca_positions"

        if not normalized:
            normalized = self.DEFAULT_WATCHLIST.copy()
            source = "default_watchlist"

        corr_result = self.generate_correlation_matrix(normalized)
        matrix = corr_result["matrix"]
        final_tickers = corr_result["tickers"]

        risk_clusters = self.get_risk_clusters(matrix, final_tickers, threshold=0.7)
        diversification_score = self.calculate_diversification_score(matrix)
        warnings = self.get_warnings(matrix, final_tickers)

        return {
            "correlation_matrix": matrix,
            "tickers": final_tickers,
            "stats": corr_result["stats"],
            "color_scale": {
                "min": -1.0,
                "max": 1.0,
                "center": 0.0,
                "palette": ["#2c7bb6", "#ffffbf", "#d7191c"]
            },
            "risk_clusters": risk_clusters,
            "diversification_score": diversification_score,
            "warnings": warnings,
            "metadata": {
                "source": source,
                "generated_at": pd.Timestamp.utcnow().isoformat()
            }
        }
