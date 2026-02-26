# ML Model Upgrade Spec

## Current State
- Ensemble: XGBoost (40%) + RandomForest (30%) + MLP (30%)
- Binary classification target (up/down next day)
- Only technical indicator features
- ML used as validator only, not signal generator

## Upgrades Required

### 1. Replace Models
**Remove:** RandomForest (redundant with XGBoost)
**Add:** LightGBM, CatBoost
**Keep:** XGBoost, MLP

New ensemble: **LightGBM (30%) + XGBoost (30%) + CatBoost (25%) + MLP (15%)**

LightGBM config:
```python
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)
```

CatBoost config:
```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0,
    auto_class_weights='Balanced',
)
```

### 2. Regression Target (in addition to classification)
Add a second prediction mode: **expected return regression**

```python
def prepare_target_regression(self, df, horizon=5):
    """Predict 5-day forward return (not just direction)"""
    return df["Close"].shift(-horizon) / df["Close"] - 1
```

Use for position sizing: if ML predicts +3% return with high confidence → larger position.
Classification still used for direction (buy/sell), regression for magnitude and sizing.

### 3. Cross-Asset Features
Add to `prepare_features()`:

```python
# Market regime features (download once, cache)
spy = yf.Ticker("SPY").history(period="2y")
vix = yf.Ticker("^VIX").history(period="2y")

features["spy_returns_5d"] = spy["Close"].pct_change(5)  # broad market momentum
features["spy_returns_20d"] = spy["Close"].pct_change(20)
features["vix_level"] = vix["Close"]
features["vix_change_5d"] = vix["Close"].pct_change(5)
features["yield_spread"] = # 10Y-2Y (use ^TNX - ^IRX proxy)
features["relative_strength_vs_spy"] = stock_returns_20d - spy_returns_20d
features["sector_momentum"] = # sector ETF 20d return
features["correlation_to_spy_20d"] = stock_returns.rolling(20).corr(spy_returns)
```

These features capture macro context that pure technical indicators miss.

### 4. ML as Signal Generator
Add new method `generate_ml_signals()` to the autonomous engine:

```python
def generate_ml_signals(self, tickers, top_n=5):
    """
    Use ML to rank all tickers by expected return.
    Top N with >70% confidence and >1% expected return become BUY signals.
    """
    predictions = []
    for ticker in tickers:
        prob = self.ml_predictor.predict(ticker)  # classification probability
        expected_return = self.ml_predictor.predict_return(ticker)  # regression
        predictions.append({
            "ticker": ticker,
            "buy_probability": prob,
            "expected_return": expected_return,
            "confidence": abs(prob - 0.5) * 2,  # 0-1 scale
        })
    
    # Rank by expected return * confidence
    ranked = sorted(predictions, key=lambda x: x["expected_return"] * x["confidence"], reverse=True)
    
    # Top N with minimum thresholds
    signals = []
    for p in ranked[:top_n]:
        if p["buy_probability"] > 0.65 and p["expected_return"] > 0.01 and p["confidence"] > 0.3:
            signals.append({
                "ticker": p["ticker"],
                "action": "BUY",
                "strategy_name": "ml_ensemble",
                "confidence": p["confidence"],
                "expected_return": p["expected_return"],
                "signal_quality": p["buy_probability"] * 100,
            })
    return signals
```

Wire this into the scan loop in `autonomous_trading_engine.py` as an additional strategy alongside the existing 5.

### 5. Walk-Forward Training
Currently trains on all data at once. Add walk-forward:

```python
def train_walk_forward(self, ticker, total_months=24, train_months=18, test_months=6):
    """
    Train on 18 months, test on 6 months.
    Only deploy if test performance meets thresholds.
    """
    # Split data
    # Train models on train set
    # Evaluate on test set
    # Return test metrics for deployment decision
```

### 6. Requirements
Add to `requirements-trading-platform.txt`:
```
lightgbm>=4.0.0
catboost>=1.2.0
```

Add to `requirements.txt` (Streamlit Cloud):
```
lightgbm>=4.0.0
catboost>=1.2.0
```

### File Changes
- `ml_price_predictor.py` — main changes (models, features, regression, walk-forward)
- `autonomous_trading_engine.py` — add `generate_ml_signals()` to scan loop
- `requirements.txt` — add lightgbm, catboost
- `requirements-trading-platform.txt` — add lightgbm, catboost
- `trading_config.py` — add ML strategy config

### Testing
```bash
python -c "from ml_price_predictor import MLPricePredictor; p = MLPricePredictor(); result = p.train_and_predict('AAPL', '2y'); print(result)"
```

### Important
- Graceful fallback if lightgbm/catboost not installed (try/except import)
- Cache cross-asset data (SPY, VIX) — don't re-download per ticker
- sys.stdout.reconfigure for Windows encoding
- Don't break existing classification API — add regression alongside it
- Keep model weights configurable in trading_config.py
