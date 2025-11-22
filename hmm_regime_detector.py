"""
Hidden Markov Model (HMM) Regime Detection
Auto-detects bull/bear/consolidation market regimes
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. Install with: pip install hmmlearn")


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection

    Detects 3 hidden states:
    - State 0: BULL (high returns, low volatility)
    - State 1: BEAR (negative returns, high volatility)
    - State 2: CONSOLIDATION (low returns, low volatility)

    Features:
    - Automatic regime detection from price data
    - Regime probability distributions
    - Regime transition predictions
    - Visual regime timeline
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Args:
            n_regimes: Number of hidden states (default 3: bull/bear/consolidation)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_labels = {}  # Will map state numbers to regime names
        self.feature_means = None  # For characterizing each regime

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training

        Uses:
        - Daily returns
        - Rolling volatility (20-day)
        - Volume changes

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Feature array for HMM
        """
        features = pd.DataFrame()

        # Returns
        features['returns'] = df['Close'].pct_change()

        # Volatility (rolling std of returns)
        features['volatility'] = features['returns'].rolling(window=20).std()

        # Volume change
        features['volume_change'] = df['Volume'].pct_change()

        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)

        return features.values

    def train(
        self,
        ticker: str,
        period: str = "2y",
        n_iter: int = 100
    ) -> Dict[str, Any]:
        """
        Train HMM model to detect market regimes

        Args:
            ticker: Stock ticker symbol
            period: Historical data period (default 2 years)
            n_iter: Number of training iterations

        Returns:
            Dict with training results and regime characteristics
        """
        if not HMM_AVAILABLE:
            return {
                "success": False,
                "error": "hmmlearn not available. Install with: pip install hmmlearn"
            }

        try:
            # Download data
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 100:
                return {
                    "success": False,
                    "error": f"Insufficient data for {ticker}. Need at least 100 days."
                }

            # Prepare features
            X = self.prepare_features(df)

            if len(X) < 50:
                return {
                    "success": False,
                    "error": "Not enough valid samples after feature preparation"
                }

            # Train Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=self.random_state
            )

            self.model.fit(X)

            # Predict regimes for historical data
            hidden_states = self.model.predict(X)

            # Characterize each regime by its mean features
            regime_characteristics = []
            for i in range(self.n_regimes):
                regime_mask = hidden_states == i
                regime_data = X[regime_mask]

                if len(regime_data) > 0:
                    mean_return = regime_data[:, 0].mean()
                    mean_vol = regime_data[:, 1].mean()
                    mean_volume_change = regime_data[:, 2].mean()

                    regime_characteristics.append({
                        "state": i,
                        "mean_return": float(mean_return),
                        "mean_volatility": float(mean_vol),
                        "mean_volume_change": float(mean_volume_change),
                        "occurrences": int(regime_mask.sum()),
                        "percentage": float(regime_mask.sum() / len(hidden_states) * 100)
                    })

            # Assign regime labels based on characteristics
            self.regime_labels = self._assign_regime_labels(regime_characteristics)

            # Update regime characteristics with labels
            for char in regime_characteristics:
                char['label'] = self.regime_labels.get(char['state'], 'UNKNOWN')

            # Calculate transition matrix
            transition_matrix = self.model.transmat_.tolist()

            # Get current regime
            current_regime_state = hidden_states[-1]
            current_regime_label = self.regime_labels.get(current_regime_state, 'UNKNOWN')

            # Predict regime probabilities for latest data
            latest_features = X[-1:, :]
            regime_probabilities = self.model.predict_proba(latest_features)[0]

            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "n_samples": len(X),
                "current_regime": {
                    "state": int(current_regime_state),
                    "label": current_regime_label,
                    "probabilities": {
                        self.regime_labels.get(i, f"State_{i}"): float(regime_probabilities[i])
                        for i in range(self.n_regimes)
                    }
                },
                "regime_characteristics": regime_characteristics,
                "transition_matrix": transition_matrix,
                "regime_labels": self.regime_labels,
                "convergence_score": float(self.model.score(X))
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}"
            }

    def _assign_regime_labels(self, characteristics: List[Dict]) -> Dict[int, str]:
        """
        Assign interpretable labels to regimes based on their characteristics

        Logic:
        - BULL: Highest mean return, lower volatility
        - BEAR: Negative or lowest returns, higher volatility
        - CONSOLIDATION: Low returns, low volatility
        """
        if len(characteristics) < 3:
            # If less than 3 regimes, use simple labels
            return {i: f"State_{i}" for i in range(len(characteristics))}

        # Sort by mean return
        sorted_by_return = sorted(characteristics, key=lambda x: x['mean_return'])

        labels = {}

        # Highest return = BULL
        bull_state = sorted_by_return[-1]['state']
        labels[bull_state] = "BULL"

        # Lowest return = BEAR
        bear_state = sorted_by_return[0]['state']
        labels[bear_state] = "BEAR"

        # Middle = CONSOLIDATION (or high volatility regime if vol is high)
        if len(sorted_by_return) > 2:
            consolidation_state = sorted_by_return[1]['state']

            # Check if it's actually high volatility
            consolidation_char = next(c for c in characteristics if c['state'] == consolidation_state)
            bear_char = next(c for c in characteristics if c['state'] == bear_state)

            if consolidation_char['mean_volatility'] > bear_char['mean_volatility'] * 1.2:
                labels[consolidation_state] = "HIGH_VOLATILITY"
            else:
                labels[consolidation_state] = "CONSOLIDATION"

        return labels

    def predict_regime(
        self,
        ticker: str,
        data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Predict current market regime

        Args:
            ticker: Stock ticker
            data: Optional DataFrame with current data. If None, fetches latest.

        Returns:
            Dict with current regime and probabilities
        """
        if not HMM_AVAILABLE:
            return {
                "success": False,
                "error": "hmmlearn not available"
            }

        if self.model is None:
            return {
                "success": False,
                "error": f"No trained model. Train first using train() method."
            }

        try:
            # Get data
            if data is None:
                data = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)

                # Flatten multi-index columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

            if data.empty:
                return {
                    "success": False,
                    "error": f"No data available for {ticker}"
                }

            # Prepare features
            X = self.prepare_features(data)

            if len(X) == 0:
                return {
                    "success": False,
                    "error": "Insufficient data for prediction"
                }

            # Predict regime
            latest_features = X[-1:, :]
            predicted_state = self.model.predict(latest_features)[0]
            regime_probabilities = self.model.predict_proba(latest_features)[0]

            regime_label = self.regime_labels.get(predicted_state, 'UNKNOWN')

            # Get transition probabilities (what's likely next)
            next_regime_probs = self.model.transmat_[predicted_state]

            return {
                "success": True,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "current_price": float(data['Close'].iloc[-1]),
                "current_regime": {
                    "state": int(predicted_state),
                    "label": regime_label,
                    "confidence": float(regime_probabilities[predicted_state])
                },
                "regime_probabilities": {
                    self.regime_labels.get(i, f"State_{i}"): float(regime_probabilities[i])
                    for i in range(self.n_regimes)
                },
                "next_regime_probabilities": {
                    self.regime_labels.get(i, f"State_{i}"): float(next_regime_probs[i])
                    for i in range(self.n_regimes)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }

    def get_regime_history(
        self,
        ticker: str,
        period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Get regime history over time for visualization

        Args:
            ticker: Stock ticker
            period: Historical period

        Returns:
            Dict with dates, prices, and regimes
        """
        if not HMM_AVAILABLE or self.model is None:
            return {
                "success": False,
                "error": "Model not available or not trained"
            }

        try:
            # Download data
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                return {
                    "success": False,
                    "error": f"No data for {ticker}"
                }

            # Prepare features and predict
            X = self.prepare_features(df)
            hidden_states = self.model.predict(X)

            # Build timeline
            timeline = []
            for i, (date, row) in enumerate(df.iterrows()):
                if i < len(hidden_states):
                    regime_state = int(hidden_states[i])
                    regime_label = self.regime_labels.get(regime_state, 'UNKNOWN')

                    timeline.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "price": float(row['Close']),
                        "regime_state": regime_state,
                        "regime_label": regime_label,
                        "returns": float(X[i, 0]) if i < len(X) else 0
                    })

            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "timeline": timeline,
                "regime_labels": self.regime_labels
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get history: {str(e)}"
            }


if __name__ == "__main__":
    # Test the regime detector
    print("=" * 70)
    print("TESTING HMM REGIME DETECTOR")
    print("=" * 70)
    print(f"HMM available: {HMM_AVAILABLE}")

    if HMM_AVAILABLE:
        detector = HMMRegimeDetector(n_regimes=3)

        # Train on NVDA
        print("\n1. Training HMM on NVDA...")
        result = detector.train("NVDA", period="2y")

        if result['success']:
            print("✅ Training successful!")
            print(f"\nDataset: {result['n_samples']} samples")
            print(f"Convergence Score: {result['convergence_score']:.2f}")

            print(f"\n📊 Current Regime:")
            current = result['current_regime']
            print(f"  State: {current['state']}")
            print(f"  Label: {current['label']}")
            print(f"  Probabilities:")
            for regime, prob in current['probabilities'].items():
                print(f"    {regime}: {prob*100:.1f}%")

            print(f"\n📈 Regime Characteristics:")
            for char in result['regime_characteristics']:
                print(f"\n  {char['label']} (State {char['state']}):")
                print(f"    Mean Return: {char['mean_return']*100:.2f}%")
                print(f"    Mean Volatility: {char['mean_volatility']*100:.2f}%")
                print(f"    Occurrences: {char['occurrences']} days ({char['percentage']:.1f}%)")

            print(f"\n🔄 Transition Matrix:")
            labels = list(result['regime_labels'].values())
            trans_matrix = result['transition_matrix']
            print(f"  From → To  {' | '.join(labels)}")
            for i, row in enumerate(trans_matrix):
                label = labels[i]
                probs = ' | '.join([f"{p*100:5.1f}%" for p in row])
                print(f"  {label:15} {probs}")

            # Test prediction
            print("\n2. Testing regime prediction...")
            pred = detector.predict_regime("NVDA")

            if pred['success']:
                print("✅ Prediction successful!")
                print(f"Current Price: ${pred['current_price']:.2f}")
                print(f"Current Regime: {pred['current_regime']['label']} ({pred['current_regime']['confidence']*100:.1f}% confidence)")

                print(f"\nRegime Probabilities:")
                for regime, prob in pred['regime_probabilities'].items():
                    print(f"  {regime}: {prob*100:.1f}%")

                print(f"\nNext Period Likely Regimes:")
                for regime, prob in pred['next_regime_probabilities'].items():
                    print(f"  {regime}: {prob*100:.1f}%")
            else:
                print(f"❌ Prediction failed: {pred['error']}")
        else:
            print(f"❌ Training failed: {result['error']}")
    else:
        print("\n⚠️  hmmlearn not installed. Install with:")
        print("   pip install hmmlearn")
