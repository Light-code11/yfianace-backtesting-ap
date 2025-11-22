"""
Machine Learning Price Predictor using XGBoost
Predicts next-day price movements based on technical indicators
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. ML predictions disabled.")

try:
    from advanced_indicators import AdvancedIndicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    print("Warning: Advanced indicators not available")


class MLPricePredictor:
    """
    XGBoost-based price movement predictor

    Features:
    - Binary classification: Predict UP (1) or DOWN (0) next day
    - Uses 50+ technical indicators as features
    - Time series cross-validation
    - Model persistence (save/load)
    - Feature importance analysis
    - Prediction confidence scores
    """

    def __init__(self, model_dir: str = "./ml_models"):
        """
        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.feature_columns = []
        self.scaler_params = {}  # For feature normalization

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from price data using technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicator features
        """
        if not INDICATORS_AVAILABLE:
            # Fallback: basic features
            df['returns'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            return df

        try:
            # Calculate all technical indicators
            features = df.copy()

            # Momentum indicators
            features['RSI'] = AdvancedIndicators.rsi(df, period=14)
            features['RSI_fast'] = AdvancedIndicators.rsi(df, period=7)
            features['Stoch_K'], features['Stoch_D'] = AdvancedIndicators.stochastic(df)
            features['CCI'] = AdvancedIndicators.cci(df, period=20)
            features['Williams_R'] = AdvancedIndicators.williams_r(df, period=14)
            features['ROC'] = AdvancedIndicators.roc(df, period=12)
            features['MFI'] = AdvancedIndicators.mfi(df, period=14)

            # Trend indicators
            features['SMA_20'] = AdvancedIndicators.sma(df, period=20)
            features['SMA_50'] = AdvancedIndicators.sma(df, period=50)
            features['EMA_12'] = AdvancedIndicators.ema(df, period=12)
            features['EMA_26'] = AdvancedIndicators.ema(df, period=26)

            macd_result = AdvancedIndicators.macd(df)
            features['MACD'] = macd_result['MACD']
            features['MACD_signal'] = macd_result['MACD_Signal']
            features['MACD_hist'] = macd_result['MACD_Hist']

            features['ADX'] = AdvancedIndicators.adx(df, period=14)

            aroon = AdvancedIndicators.aroon(df, period=25)
            features['Aroon_Up'] = aroon['Aroon_Up']
            features['Aroon_Down'] = aroon['Aroon_Down']

            # Volatility indicators
            features['ATR'] = AdvancedIndicators.atr(df, period=14)

            bb = AdvancedIndicators.bollinger_bands(df, period=20)
            features['BB_upper'] = bb['BB_Upper']
            features['BB_middle'] = bb['BB_Middle']
            features['BB_lower'] = bb['BB_Lower']
            features['BB_width'] = (bb['BB_Upper'] - bb['BB_Lower']) / bb['BB_Middle']

            # Volume indicators
            features['OBV'] = AdvancedIndicators.obv(df)
            features['CMF'] = AdvancedIndicators.cmf(df, period=20)
            features['VWAP'] = AdvancedIndicators.vwap(df)

            # Price-based features
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            features['high_low_ratio'] = df['High'] / df['Low']
            features['close_open_ratio'] = df['Close'] / df['Open']
            features['volume_change'] = df['Volume'].pct_change()

            # Lagged features (past N days)
            for lag in [1, 2, 3, 5, 10]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume_change'].shift(lag)

            # Rolling statistics
            for window in [5, 10, 20]:
                features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
                features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
                features[f'volume_mean_{window}'] = features['Volume'].rolling(window).mean()

            return features

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            # Return basic features on error
            features = df.copy()
            features['returns'] = df['Close'].pct_change()
            features['high_low_ratio'] = df['High'] / df['Low']
            features['close_open_ratio'] = df['Close'] / df['Open']
            features['volume_change'] = df['Volume'].pct_change()
            return features

    def prepare_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Prepare target variable: 1 if price goes up, 0 if down

        Args:
            df: DataFrame with price data
            horizon: Number of days ahead to predict (default 1)

        Returns:
            Series with binary target (1=up, 0=down)
        """
        # Calculate future return
        future_return = df['Close'].shift(-horizon) / df['Close'] - 1

        # Binary classification: 1 if up, 0 if down
        target = (future_return > 0).astype(int)

        return target

    def train_model(
        self,
        ticker: str,
        period: str = "2y",
        test_size: float = 0.2,
        horizon: int = 1,
        **xgb_params
    ) -> Dict[str, Any]:
        """
        Train XGBoost model to predict price movements

        Args:
            ticker: Stock ticker symbol
            period: Historical data period (default 2 years)
            test_size: Fraction of data for testing (default 0.2)
            horizon: Days ahead to predict (default 1)
            **xgb_params: Additional XGBoost parameters

        Returns:
            Dict with training results and metrics
        """
        if not XGBOOST_AVAILABLE:
            return {
                "success": False,
                "error": "XGBoost not available. Install with: pip install xgboost scikit-learn"
            }

        try:
            # Download data
            df = yf.download(ticker, period=period, progress=False)

            if df.empty or len(df) < 100:
                return {
                    "success": False,
                    "error": f"Insufficient data for {ticker}. Need at least 100 days."
                }

            # Prepare features
            features_df = self.prepare_features(df)

            # Prepare target
            target = self.prepare_target(df, horizon=horizon)

            # Align features and target (remove NaN rows)
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
            features_df = features_df[valid_idx]
            target = target[valid_idx]

            # Select feature columns (exclude price/volume columns)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]

            X = features_df[self.feature_columns].values
            y = target.values

            if len(X) < 50:
                return {
                    "success": False,
                    "error": "Not enough valid samples after feature preparation"
                }

            # Time series split (preserve temporal order)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Check class balance
            train_positive_pct = (y_train.sum() / len(y_train)) * 100
            test_positive_pct = (y_test.sum() / len(y_test)) * 100

            # Default XGBoost parameters
            default_params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            default_params.update(xgb_params)

            # Train model
            self.model = xgb.XGBClassifier(**default_params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)

            # Prediction probabilities
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_test_proba = self.model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)

            # Feature importance
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

            # Save model
            model_path = self.model_dir / f"{ticker}_model.pkl"
            self._save_model(model_path, ticker, horizon)

            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "horizon": horizon,
                "samples": {
                    "total": len(X),
                    "train": len(X_train),
                    "test": len(X_test)
                },
                "class_balance": {
                    "train_positive_pct": round(train_positive_pct, 2),
                    "test_positive_pct": round(test_positive_pct, 2)
                },
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "top_features": [{"feature": f, "importance": round(imp, 4)} for f, imp in top_features],
                "model_path": str(model_path),
                "features_count": len(self.feature_columns)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}"
            }

    def _calculate_metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4)
        }

        # ROC AUC only if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        else:
            metrics["roc_auc"] = 0.0

        return metrics

    def predict(
        self,
        ticker: str,
        data: Optional[pd.DataFrame] = None,
        return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        Predict next-day price movement

        Args:
            ticker: Stock ticker
            data: Optional DataFrame with current data. If None, fetches latest.
            return_proba: Return probability scores

        Returns:
            Dict with prediction and confidence
        """
        if not XGBOOST_AVAILABLE:
            return {
                "success": False,
                "error": "XGBoost not available"
            }

        try:
            # Load model if not in memory
            if self.model is None:
                model_path = self.model_dir / f"{ticker}_model.pkl"
                if not model_path.exists():
                    return {
                        "success": False,
                        "error": f"No trained model found for {ticker}. Train first."
                    }
                self._load_model(model_path)

            # Get data
            if data is None:
                data = yf.download(ticker, period="6mo", progress=False)

            if data.empty:
                return {
                    "success": False,
                    "error": f"No data available for {ticker}"
                }

            # Prepare features
            features_df = self.prepare_features(data)

            # Get latest row (most recent data)
            latest = features_df.iloc[-1]
            X = latest[self.feature_columns].values.reshape(1, -1)

            # Check for NaN
            if np.isnan(X).any():
                return {
                    "success": False,
                    "error": "Insufficient data for prediction (missing indicators)"
                }

            # Predict
            prediction = self.model.predict(X)[0]

            result = {
                "success": True,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "prediction": "UP" if prediction == 1 else "DOWN",
                "prediction_binary": int(prediction),
                "current_price": float(data['Close'].iloc[-1])
            }

            if return_proba:
                proba = self.model.predict_proba(X)[0]
                result["confidence"] = {
                    "down_probability": round(float(proba[0]), 4),
                    "up_probability": round(float(proba[1]), 4),
                    "confidence_score": round(float(max(proba)), 4)
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }

    def _save_model(self, path: Path, ticker: str, horizon: int):
        """Save trained model to disk"""
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "ticker": ticker,
            "horizon": horizon,
            "trained_at": datetime.now().isoformat()
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']

    def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models"""
        models = []
        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                models.append({
                    "ticker": model_data['ticker'],
                    "horizon": model_data['horizon'],
                    "trained_at": model_data['trained_at'],
                    "features_count": len(model_data['feature_columns']),
                    "model_file": model_file.name
                })
            except:
                continue

        return sorted(models, key=lambda x: x['trained_at'], reverse=True)


if __name__ == "__main__":
    # Test the predictor
    print("=" * 70)
    print("TESTING ML PRICE PREDICTOR")
    print("=" * 70)
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"Advanced Indicators available: {INDICATORS_AVAILABLE}")

    if XGBOOST_AVAILABLE:
        predictor = MLPricePredictor()

        # Train model on NVDA
        print("\n1. Training model on NVDA...")
        result = predictor.train_model("NVDA", period="1y")

        if result['success']:
            print("✅ Training successful!")
            print(f"\nDataset: {result['samples']['total']} samples")
            print(f"Train: {result['samples']['train']}, Test: {result['samples']['test']}")
            print(f"\nClass Balance:")
            print(f"  Train positive: {result['class_balance']['train_positive_pct']}%")
            print(f"  Test positive: {result['class_balance']['test_positive_pct']}%")
            print(f"\nTrain Metrics:")
            for metric, value in result['train_metrics'].items():
                print(f"  {metric}: {value}")
            print(f"\nTest Metrics:")
            for metric, value in result['test_metrics'].items():
                print(f"  {metric}: {value}")
            print(f"\nTop 5 Features:")
            for feat in result['top_features'][:5]:
                print(f"  {feat['feature']}: {feat['importance']}")

            # Test prediction
            print("\n2. Testing prediction...")
            pred = predictor.predict("NVDA")

            if pred['success']:
                print("✅ Prediction successful!")
                print(f"Ticker: {pred['ticker']}")
                print(f"Current Price: ${pred['current_price']:.2f}")
                print(f"Prediction: {pred['prediction']}")
                print(f"Confidence: {pred['confidence']['confidence_score'] * 100:.1f}%")
                print(f"Up Probability: {pred['confidence']['up_probability'] * 100:.1f}%")
                print(f"Down Probability: {pred['confidence']['down_probability'] * 100:.1f}%")
            else:
                print(f"❌ Prediction failed: {pred['error']}")
        else:
            print(f"❌ Training failed: {result['error']}")
    else:
        print("\n⚠️  XGBoost not installed. Install with:")
        print("   pip install xgboost scikit-learn")
