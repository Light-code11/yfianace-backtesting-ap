"""
Machine Learning Price Predictor using an ensemble of XGBoost, RandomForest, and MLP.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neural_network import MLPClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML predictions disabled.")

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost component disabled.")

try:
    from advanced_indicators import AdvancedIndicators

    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    print("Warning: Advanced indicators not available")


class MLPricePredictor:
    DEFAULT_WEIGHTS = {"xgboost": 0.4, "random_forest": 0.3, "mlp": 0.3}

    def __init__(self, model_dir: str = "./ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models: Dict[str, Any] = {"xgboost": None, "random_forest": None, "mlp": None}
        self.model = None  # compatibility alias
        self.feature_columns: List[str] = []
        self.baseline_feature_importances: Dict[str, float] = {}
        self.model_weights: Dict[str, float] = self.DEFAULT_WEIGHTS.copy()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not INDICATORS_AVAILABLE:
            features = df.copy()
            features["returns"] = df["Close"].pct_change()
            features["high_low_ratio"] = df["High"] / df["Low"]
            features["close_open_ratio"] = df["Close"] / df["Open"]
            features["volume_change"] = df["Volume"].pct_change()
            return features

        features = df.copy()
        try:
            features["RSI"] = AdvancedIndicators.rsi(df, period=14)
            features["RSI_fast"] = AdvancedIndicators.rsi(df, period=7)
            features["Stoch_K"], features["Stoch_D"] = AdvancedIndicators.stochastic(df)
            features["CCI"] = AdvancedIndicators.cci(df, period=20)
            features["Williams_R"] = AdvancedIndicators.williams_r(df, period=14)
            features["ROC"] = AdvancedIndicators.roc(df, period=12)
            features["MFI"] = AdvancedIndicators.mfi(df, period=14)
            features["SMA_20"] = AdvancedIndicators.sma(df, period=20)
            features["SMA_50"] = AdvancedIndicators.sma(df, period=50)
            features["EMA_12"] = AdvancedIndicators.ema(df, period=12)
            features["EMA_26"] = AdvancedIndicators.ema(df, period=26)
            macd = AdvancedIndicators.macd(df)
            features["MACD"] = macd["MACD"]
            features["MACD_signal"] = macd["MACD_Signal"]
            features["MACD_hist"] = macd["MACD_Hist"]
            features["ADX"] = AdvancedIndicators.adx(df, period=14)
            aroon = AdvancedIndicators.aroon(df, period=25)
            features["Aroon_Up"] = aroon["Aroon_Up"]
            features["Aroon_Down"] = aroon["Aroon_Down"]
            features["ATR"] = AdvancedIndicators.atr(df, period=14)
            bb = AdvancedIndicators.bollinger_bands(df, period=20)
            features["BB_upper"] = bb["BB_Upper"]
            features["BB_middle"] = bb["BB_Middle"]
            features["BB_lower"] = bb["BB_Lower"]
            features["BB_width"] = (bb["BB_Upper"] - bb["BB_Lower"]) / bb["BB_Middle"]
            features["OBV"] = AdvancedIndicators.obv(df)
            features["CMF"] = AdvancedIndicators.cmf(df, period=20)
            features["VWAP"] = AdvancedIndicators.vwap(df)
        except Exception as e:
            print(f"Indicator calculation error: {e}")

        features["returns"] = df["Close"].pct_change()
        features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
        features["high_low_ratio"] = df["High"] / df["Low"]
        features["close_open_ratio"] = df["Close"] / df["Open"]
        features["volume_change"] = df["Volume"].pct_change()
        for lag in [1, 2, 3, 5, 10]:
            features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
            features[f"volume_lag_{lag}"] = features["volume_change"].shift(lag)
        for window in [5, 10, 20]:
            features[f"returns_mean_{window}"] = features["returns"].rolling(window).mean()
            features[f"returns_std_{window}"] = features["returns"].rolling(window).std()
            features[f"volume_mean_{window}"] = features["Volume"].rolling(window).mean()
        return features

    def prepare_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        return (df["Close"].shift(-horizon) / df["Close"] - 1 > 0).astype(int)

    def _download_data(self, ticker: str, period: str) -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def _prepare_training_arrays(
        self, ticker: str, period: str, test_size: float, horizon: int, data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        df = self._download_data(ticker, period) if data is None else data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 100:
            return {"success": False, "error": f"Insufficient data for {ticker}. Need at least 100 days."}

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return {"success": False, "error": f"Missing required columns: {missing}"}

        feat = self.prepare_features(df)
        target = self.prepare_target(df, horizon=horizon)
        valid = ~(feat.isna().any(axis=1) | target.isna())
        feat, target = feat[valid], target[valid]

        exclude = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        self.feature_columns = [c for c in feat.columns if c not in exclude]
        X, y = feat[self.feature_columns].values, target.values
        if len(X) < 50:
            return {"success": False, "error": "Not enough valid samples after feature preparation"}

        split_idx = int(len(X) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(X):
            return {"success": False, "error": "Invalid test_size produced empty train/test split"}

        return {
            "success": True,
            "X": X,
            "y": y,
            "X_train": X[:split_idx],
            "X_test": X[split_idx:],
            "y_train": y[:split_idx],
            "y_test": y[split_idx:],
        }

    def _calculate_metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        metrics = {
            "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
            "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
            "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
            "f1_score": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        }
        metrics["roc_auc"] = (
            float(round(roc_auc_score(y_true, y_proba), 4)) if len(np.unique(y_true)) > 1 else 0.0
        )
        return metrics

    def _extract_feature_importances(self, models: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        models = models or self.models
        vectors = []
        for name in ["xgboost", "random_forest"]:
            model = models.get(name)
            if model is None or not hasattr(model, "feature_importances_"):
                continue
            vec = np.array(model.feature_importances_, dtype=float)
            vec = vec / vec.sum() if vec.sum() > 0 else vec
            vectors.append(vec)
        if not vectors:
            return {}
        avg = np.mean(np.vstack(vectors), axis=0)
        return {f: float(avg[i]) for i, f in enumerate(self.feature_columns)}

    def _active_weights(self) -> Dict[str, float]:
        active = {n: float(w) for n, w in self.model_weights.items() if self.models.get(n) is not None}
        total = sum(active.values())
        if total <= 0:
            return {}
        return {n: w / total for n, w in active.items()}

    def train_model(self, ticker: str, period: str = "2y", test_size: float = 0.2, horizon: int = 1, **xgb_params):
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "scikit-learn not available"}
        try:
            prep = self._prepare_training_arrays(ticker, period, test_size, horizon)
            if not prep["success"]:
                return prep

            X, X_train, X_test = prep["X"], prep["X_train"], prep["X_test"]
            y, y_train, y_test = prep["y"], prep["y_train"], prep["y_test"]
            self.models = {"xgboost": None, "random_forest": None, "mlp": None}
            per_model_metrics: Dict[str, Dict[str, float]] = {}
            test_probs: Dict[str, np.ndarray] = {}
            train_probs: Dict[str, np.ndarray] = {}

            if XGBOOST_AVAILABLE:
                params = {
                    "objective": "binary:logistic",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 120,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "eval_metric": "logloss",
                }
                params.update(xgb_params)
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                self.models["xgboost"] = model
                y_pred, y_proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
                per_model_metrics["xgboost"] = self._calculate_metrics(y_test, y_pred, y_proba)
                test_probs["xgboost"] = y_proba
                train_probs["xgboost"] = model.predict_proba(X_train)[:, 1]

            rf = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            self.models["random_forest"] = rf
            y_pred, y_proba = rf.predict(X_test), rf.predict_proba(X_test)[:, 1]
            per_model_metrics["random_forest"] = self._calculate_metrics(y_test, y_pred, y_proba)
            test_probs["random_forest"] = y_proba
            train_probs["random_forest"] = rf.predict_proba(X_train)[:, 1]

            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=400, random_state=42
            )
            mlp.fit(X_train, y_train)
            self.models["mlp"] = mlp
            y_pred, y_proba = mlp.predict(X_test), mlp.predict_proba(X_test)[:, 1]
            per_model_metrics["mlp"] = self._calculate_metrics(y_test, y_pred, y_proba)
            test_probs["mlp"] = y_proba
            train_probs["mlp"] = mlp.predict_proba(X_train)[:, 1]

            self.model = self.models.get("xgboost") or self.models.get("random_forest")
            weights = self._active_weights()
            weighted_test = sum(test_probs[m] * w for m, w in weights.items())
            weighted_train = sum(train_probs[m] * w for m, w in weights.items())
            y_test_pred = (weighted_test >= 0.5).astype(int)
            y_train_pred = (weighted_train >= 0.5).astype(int)
            train_metrics = self._calculate_metrics(y_train, y_train_pred, weighted_train)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, weighted_test)

            feature_importance = self._extract_feature_importances()
            self.baseline_feature_importances = feature_importance.copy()
            top = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

            model_path = self.model_dir / f"{ticker}_model.pkl"
            self._save_model(model_path, ticker, horizon)
            return {
                "success": True,
                "ticker": ticker,
                "period": period,
                "horizon": horizon,
                "samples": {"total": int(len(X)), "train": int(len(X_train)), "test": int(len(X_test))},
                "class_balance": {
                    "train_positive_pct": float(round((y_train.sum() / len(y_train)) * 100, 2)),
                    "test_positive_pct": float(round((y_test.sum() / len(y_test)) * 100, 2)),
                },
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "per_model_metrics": per_model_metrics,
                "ensemble_weights": weights,
                "top_features": [{"feature": f, "importance": float(round(v, 4))} for f, v in top],
                "model_path": str(model_path),
                "features_count": int(len(self.feature_columns)),
                "validation_accuracy": test_metrics.get("accuracy", 0.0),
            }
        except Exception as e:
            return {"success": False, "error": f"Training failed: {str(e)}"}

    def train(self, ticker: str, period: str = "2y", test_size: float = 0.2, horizon: int = 1, **xgb_params):
        return self.train_model(ticker=ticker, period=period, test_size=test_size, horizon=horizon, **xgb_params)

    def ensemble_predict(self, ticker: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            return {"success": False, "error": "scikit-learn not available"}
        try:
            if not any(self.models.values()):
                path = self.model_dir / f"{ticker}_model.pkl"
                if not path.exists():
                    return {"success": False, "error": f"No trained model found for {ticker}. Train first."}
                self._load_model(path)

            if data is None:
                data = self._download_data(ticker, period="6mo")
            elif isinstance(data.columns, pd.MultiIndex):
                data = data.copy()
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                return {"success": False, "error": f"No data available for {ticker}"}

            feat = self.prepare_features(data)
            X = feat.iloc[-1][self.feature_columns].values.reshape(1, -1)
            if np.isnan(X).any():
                return {"success": False, "error": "Insufficient data for prediction (missing indicators)"}

            breakdown: Dict[str, Any] = {}
            weights = self._active_weights()
            if not weights:
                return {"success": False, "error": "No active models available for prediction"}

            weighted_up = 0.0
            for name in ["xgboost", "random_forest", "mlp"]:
                model = self.models.get(name)
                if model is None:
                    continue
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [1.0, 0.0]
                down_prob, up_prob = float(proba[0]), float(proba[1])
                conf = max(up_prob, down_prob)
                weighted_up += up_prob * weights.get(name, 0.0)
                breakdown[name] = {
                    "prediction": "UP" if up_prob >= down_prob else "DOWN",
                    "up_probability": round(up_prob, 4),
                    "down_probability": round(down_prob, 4),
                    "confidence": round(conf, 4),
                    "weight": round(weights.get(name, 0.0), 4),
                }

            weighted_down = 1.0 - weighted_up
            direction = "UP" if weighted_up >= 0.5 else "DOWN"
            confidence = max(weighted_up, weighted_down)
            return {
                "success": True,
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "prediction": direction,
                "prediction_binary": int(1 if direction == "UP" else 0),
                "confidence": float(round(confidence, 4)),
                "up_probability": float(round(weighted_up, 4)),
                "down_probability": float(round(weighted_down, 4)),
                "per_model_breakdown": breakdown,
                "weights": {k: round(v, 4) for k, v in weights.items()},
                "current_price": float(data["Close"].iloc[-1]),
            }
        except Exception as e:
            return {"success": False, "error": f"Ensemble prediction failed: {str(e)}"}

    def predict(self, ticker: str, data: Optional[pd.DataFrame] = None, return_proba: bool = True):
        result = self.ensemble_predict(ticker=ticker, data=data)
        if result.get("success") and return_proba:
            result["confidence_scores"] = {
                "down_probability": result["down_probability"],
                "up_probability": result["up_probability"],
                "confidence_score": result["confidence"],
            }
        return result

    def detect_feature_drift(self, ticker: str, lookback_days: int = 60) -> Dict[str, Any]:
        path = self.model_dir / f"{ticker}_model.pkl"
        if not path.exists():
            return {"success": False, "error": f"No trained model found for {ticker}. Train first."}
        try:
            self._load_model(path)
            baseline = self.baseline_feature_importances or {}
            if not baseline:
                return {"success": False, "error": "No baseline feature importances saved for this model"}

            recent_df = self._download_data(ticker, period="6mo").tail(max(lookback_days + 60, 120))
            prep = self._prepare_training_arrays(
                ticker=ticker, period="6mo", test_size=0.2, horizon=1, data=recent_df
            )
            if not prep.get("success"):
                return prep

            X_train, y_train = prep["X_train"], prep["y_train"]
            temp_models: Dict[str, Any] = {"xgboost": None, "random_forest": None, "mlp": None}
            if XGBOOST_AVAILABLE:
                tx = xgb.XGBClassifier(
                    objective="binary:logistic",
                    max_depth=5,
                    learning_rate=0.1,
                    n_estimators=80,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric="logloss",
                )
                tx.fit(X_train, y_train, verbose=False)
                temp_models["xgboost"] = tx
            trf = RandomForestClassifier(
                n_estimators=200, max_depth=7, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            trf.fit(X_train, y_train)
            temp_models["random_forest"] = trf

            current = self._extract_feature_importances(models=temp_models)
            if not current:
                return {"success": False, "error": "Unable to derive current feature importances"}

            baseline_top10 = sorted(baseline.items(), key=lambda x: x[1], reverse=True)[:10]
            current_top10 = sorted(current.items(), key=lambda x: x[1], reverse=True)[:10]
            union_features = sorted({f for f, _ in baseline_top10} | {f for f, _ in current_top10})
            b_vec = np.array([baseline.get(f, 0.0) for f in union_features], dtype=float)
            c_vec = np.array([current.get(f, 0.0) for f in union_features], dtype=float)
            similarity = (
                float(cosine_similarity([b_vec], [c_vec])[0][0])
                if np.linalg.norm(b_vec) > 0 and np.linalg.norm(c_vec) > 0
                else 0.0
            )
            drift_score = float(max(0.0, min(1.0, 1.0 - similarity)))
            needs_retraining = drift_score > 0.3

            drifted_features = []
            for f in union_features:
                diff = abs(float(current.get(f, 0.0)) - float(baseline.get(f, 0.0)))
                if diff >= 0.03:
                    drifted_features.append(
                        {
                            "feature": f,
                            "baseline_importance": round(float(baseline.get(f, 0.0)), 4),
                            "current_importance": round(float(current.get(f, 0.0)), 4),
                            "absolute_change": round(diff, 4),
                        }
                    )
            drifted_features = sorted(drifted_features, key=lambda x: x["absolute_change"], reverse=True)
            print(
                f"[Feature Drift] {ticker} | drift_score={drift_score:.4f} "
                f"| needs_retraining={'YES' if needs_retraining else 'NO'}"
            )
            return {
                "success": True,
                "ticker": ticker,
                "lookback_days": lookback_days,
                "drift_score": round(drift_score, 4),
                "drifted_features": drifted_features,
                "needs_retraining": needs_retraining,
                "baseline_top10": [{"feature": f, "importance": round(float(v), 4)} for f, v in baseline_top10],
                "current_top10": [{"feature": f, "importance": round(float(v), 4)} for f, v in current_top10],
            }
        except Exception as e:
            return {"success": False, "error": f"Drift detection failed: {str(e)}"}

    def _save_model(self, path: Path, ticker: str, horizon: int):
        model_data = {
            "models": self.models,
            "model": self.models.get("xgboost") or self.models.get("random_forest"),
            "feature_columns": self.feature_columns,
            "ticker": ticker,
            "horizon": horizon,
            "trained_at": datetime.now().isoformat(),
            "baseline_feature_importances": self.baseline_feature_importances,
            "model_weights": self.model_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path):
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        if "models" in model_data:
            self.models = model_data["models"]
            self.model = self.models.get("xgboost") or self.models.get("random_forest")
        else:
            legacy = model_data.get("model")
            self.models = {"xgboost": legacy if XGBOOST_AVAILABLE else None, "random_forest": None, "mlp": None}
            self.model = legacy
        self.feature_columns = model_data.get("feature_columns", [])
        self.baseline_feature_importances = model_data.get("baseline_feature_importances", {})
        saved_weights = model_data.get("model_weights", {})
        self.model_weights = {
            "xgboost": float(saved_weights.get("xgboost", self.DEFAULT_WEIGHTS["xgboost"])),
            "random_forest": float(saved_weights.get("random_forest", self.DEFAULT_WEIGHTS["random_forest"])),
            "mlp": float(saved_weights.get("mlp", self.DEFAULT_WEIGHTS["mlp"])),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                with open(model_file, "rb") as f:
                    data = pickle.load(f)
                model_map = data.get("models")
                components = (
                    [name for name, obj in model_map.items() if obj is not None]
                    if isinstance(model_map, dict)
                    else ["legacy_single_model"]
                )
                models.append(
                    {
                        "ticker": data.get("ticker"),
                        "horizon": data.get("horizon"),
                        "trained_at": data.get("trained_at"),
                        "features_count": len(data.get("feature_columns", [])),
                        "model_components": components,
                        "model_file": model_file.name,
                    }
                )
            except Exception:
                continue
        return sorted(models, key=lambda x: x.get("trained_at", ""), reverse=True)


if __name__ == "__main__":
    predictor = MLPricePredictor()
    result = predictor.train_model("NVDA", period="1y")
    print(result)
