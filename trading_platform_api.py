"""
Trading Platform API - Extended FastAPI server with AI strategy generation,
backtesting, paper trading, and portfolio optimization
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Strip whitespace from API key (Railway sometimes adds trailing newlines)
raw_api_key = os.getenv('OPENAI_API_KEY')
if raw_api_key:
    clean_api_key = raw_api_key.strip()
    os.environ['OPENAI_API_KEY'] = clean_api_key
    print(f"DEBUG: OPENAI_API_KEY loaded and cleaned: {clean_api_key[:20]}...")
else:
    print("DEBUG: OPENAI_API_KEY NOT FOUND")

from database import (
    init_db, get_db, Strategy, BacktestResult, PaperTrade,
    PortfolioAllocation, AILearning, PerformanceLog,
    PairTradingPair, PairTradingPosition, PairScanResult
)
from ai_strategy_generator import AIStrategyGenerator
from backtesting_engine import BacktestingEngine
from paper_trading import PaperTradingSimulator
from portfolio_optimizer import PortfolioOptimizer
from advanced_portfolio_optimizer import AdvancedPortfolioOptimizer
from autonomous_learning import AutonomousLearningAgent
from strategy_visualizer import StrategyVisualizer
from ml_price_predictor import MLPricePredictor
from hmm_regime_detector import HMMRegimeDetector
from live_signal_generator import LiveSignalGenerator
from market_scanner import MarketScanner
from pair_trading_strategy import (
    PairTradingStatistics, PairTradingStrategy, PairBacktester, PairScanner,
    PairAnalysis, CointegrationResult
)
import threading
import time

# Helper function to convert numpy types to Python types for PostgreSQL
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    import numpy as np

    # Handle None/NaN
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None

    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]

    # Return as-is for Python native types
    return obj

# Initialize FastAPI app
app = FastAPI(
    title="AI Trading Platform API",
    description="AI-powered trading strategy generator with backtesting and portfolio optimization",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
print("Initializing database...")
init_db()
print("Database initialized successfully")

# Run Kelly Criterion migration
print("Checking Kelly Criterion database columns...")
try:
    from sqlalchemy import create_engine, text, inspect
    from database import DATABASE_URL

    # Fix Railway PostgreSQL URL format
    db_url = DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(db_url)
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('backtest_results')]

    kelly_columns = ['kelly_criterion', 'kelly_position_pct', 'kelly_risk_level']
    missing_columns = [col for col in kelly_columns if col not in columns]

    if missing_columns:
        print(f"  Adding missing Kelly columns: {missing_columns}")
        with engine.connect() as conn:
            if 'kelly_criterion' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_criterion FLOAT"))
                print("  ✅ Added kelly_criterion")
            if 'kelly_position_pct' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_position_pct FLOAT"))
                print("  ✅ Added kelly_position_pct")
            if 'kelly_risk_level' in missing_columns:
                conn.execute(text("ALTER TABLE backtest_results ADD COLUMN kelly_risk_level VARCHAR"))
                print("  ✅ Added kelly_risk_level")
            conn.commit()
        print("✅ Kelly Criterion migration complete!")
    else:
        print("  ✅ Kelly Criterion columns already exist")
except Exception as e:
    print(f"  ⚠️  Kelly migration warning: {e}")
    print("  Database will still work, but Kelly data may not save properly")

# Global autonomous learning agent
autonomous_agent = None
autonomous_agent_thread = None
autonomous_agent_config = None  # Store current agent configuration
autonomous_agent_enabled = os.getenv("ENABLE_AUTONOMOUS_LEARNING", "false").lower() == "true"

# Pydantic models
class StrategyGenerationRequest(BaseModel):
    tickers: List[str]
    period: str = "6mo"
    num_strategies: int = 3
    use_past_performance: bool = True


class BacktestRequest(BaseModel):
    strategy_id: Optional[int] = None
    strategy_config: Optional[Dict] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000
    period: str = "1y"  # Backtest period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)


class PaperTradeRequest(BaseModel):
    strategy_id: int
    auto_execute: bool = True


class PortfolioOptimizationRequest(BaseModel):
    strategy_ids: Optional[List[int]] = None  # For saved strategies from database
    strategies: Optional[List[Dict]] = None   # For on-the-fly strategies (Complete Trading System)
    total_capital: float = 100000
    method: str = "sharpe"  # sharpe, min_variance, max_return, risk_parity
    constraints: Optional[Dict] = None


# Helper function to fetch market data
def fetch_market_data(tickers: List[str], period: str = "6mo") -> pd.DataFrame:
    """Fetch market data for multiple tickers, including benchmark ETFs for relative strength analysis"""
    # Add benchmark tickers for relative strength analysis
    benchmarks = {'SPY', 'QQQ', 'SOXX'}  # Market and sector benchmarks
    all_tickers = list(set(tickers) | benchmarks)  # Combine and remove duplicates

    ticker_string = " ".join(all_tickers)
    data = yf.download(ticker_string, period=period, progress=False)
    return data


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Trading Platform API",
        "version": "2.1.0",  # Updated for Kelly Criterion
        "docs": "/docs",
        "features": {
            "ai_strategy_generation": "/strategies/generate",
            "kelly_criterion_enabled": True,  # NEW FEATURE FLAG
            "backtesting": "/backtest",
            "paper_trading": "/paper-trading",
            "portfolio_optimization": "/portfolio/optimize",
            "learning": "/ai/learning"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/debug/network")
async def debug_network():
    """Test network connectivity to OpenAI"""
    import socket
    import requests

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Test DNS
    try:
        ip = socket.gethostbyname("api.openai.com")
        results["tests"]["dns"] = {"status": "ok", "ip": ip}
    except Exception as e:
        results["tests"]["dns"] = {"status": "failed", "error": str(e)}

    # Test HTTPS
    try:
        response = requests.get("https://api.openai.com/v1/models", timeout=10)
        results["tests"]["https"] = {"status": "ok", "http_code": response.status_code}
    except Exception as e:
        results["tests"]["https"] = {"status": "failed", "error": str(e)}

    # Test OpenAI API
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            results["tests"]["openai_api"] = {
                "status": "ok" if response.status_code == 200 else "failed",
                "http_code": response.status_code
            }
        else:
            results["tests"]["openai_api"] = {"status": "skipped", "reason": "no api key"}
    except Exception as e:
        results["tests"]["openai_api"] = {"status": "failed", "error": str(e)}

    return results


# =======================
# STRATEGY ENDPOINTS
# =======================

@app.post("/strategies/generate")
async def generate_strategies(
    request: StrategyGenerationRequest,
    db=Depends(get_db)
):
    """Generate AI trading strategies"""
    import traceback
    try:
        print(f"DEBUG: Received request with tickers={request.tickers}, period={request.period}")
        # Fetch market data
        market_data = fetch_market_data(request.tickers, request.period)
        print(f"DEBUG: Market data fetched, shape={market_data.shape if not market_data.empty else 'EMPTY'}")

        if market_data.empty:
            raise HTTPException(status_code=400, detail="No market data available for tickers")

        # Get past performance if requested
        past_performance = []
        learning_insights = []

        if request.use_past_performance:
            past_results = db.query(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(20).all()
            past_performance = [
                {
                    "strategy_name": r.strategy_name,
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_return_pct": r.total_return_pct,
                    "win_rate": r.win_rate,
                    "max_drawdown_pct": r.max_drawdown_pct
                }
                for r in past_results
            ]

            learnings = db.query(AILearning).order_by(AILearning.created_at.desc()).limit(5).all()
            learning_insights = [
                {
                    "type": l.learning_type,
                    "description": l.description,
                    "insights": l.key_insights
                }
                for l in learnings
            ]

        # Generate strategies using AI
        print(f"DEBUG: Creating AI generator")
        ai_generator = AIStrategyGenerator()
        print(f"DEBUG: Calling generate_strategies with {request.num_strategies} strategies")
        strategies = ai_generator.generate_strategies(
            market_data=market_data,
            tickers=request.tickers,
            num_strategies=request.num_strategies,
            past_performance=past_performance,
            learning_insights=learning_insights
        )
        print(f"DEBUG: Strategies generated: {len(strategies)}")

        # Save strategies to database
        print(f"DEBUG: Saving {len(strategies)} strategies to database")
        saved_strategies = []
        for i, strategy in enumerate(strategies):
            print(f"DEBUG: Saving strategy {i+1}: {strategy.get('name', 'Unknown')}")

            # Ensure tickers are set - use input tickers if AI didn't provide them
            strategy_tickers = strategy.get('tickers', [])
            if not strategy_tickers:
                strategy_tickers = request.tickers
                print(f"DEBUG: No tickers in AI response, using input tickers: {strategy_tickers}")

            # Make strategy name unique by adding timestamp
            unique_name = f"{strategy['name']} [{datetime.now().strftime('%Y%m%d_%H%M%S')}]"

            db_strategy = Strategy(
                name=unique_name,
                description=strategy.get('description', ''),
                tickers=strategy_tickers,
                entry_conditions=strategy.get('entry_conditions', {}),
                exit_conditions=strategy.get('exit_conditions', {}),
                stop_loss_pct=strategy.get('risk_management', {}).get('stop_loss_pct', 5.0),
                take_profit_pct=strategy.get('risk_management', {}).get('take_profit_pct', 10.0),
                position_size_pct=strategy.get('risk_management', {}).get('position_size_pct', 10.0),
                holding_period_days=strategy.get('holding_period_days', 5),
                rationale=strategy.get('rationale', ''),
                market_analysis=strategy.get('market_analysis', ''),
                risk_assessment=strategy.get('risk_assessment', ''),
                strategy_type=strategy.get('strategy_type', 'unknown'),
                indicators=strategy.get('indicators', []),
                is_active=True
            )
            db.add(db_strategy)
            db.commit()
            db.refresh(db_strategy)

            saved_strategies.append({
                "id": db_strategy.id,
                "name": db_strategy.name,
                "description": db_strategy.description,
                "strategy_type": db_strategy.strategy_type,
                "tickers": db_strategy.tickers,
                "indicators": db_strategy.indicators,
                "risk_management": {
                    "stop_loss_pct": db_strategy.stop_loss_pct,
                    "take_profit_pct": db_strategy.take_profit_pct,
                    "position_size_pct": db_strategy.position_size_pct
                }
            })

        print(f"DEBUG: Successfully saved {len(saved_strategies)} strategies")
        return {
            "success": True,
            "strategies_generated": len(saved_strategies),
            "strategies": saved_strategies
        }

    except Exception as e:
        error_msg = f"ERROR in generate_strategies: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/strategies")
async def list_strategies(
    active_only: bool = True,
    limit: int = 50,
    db=Depends(get_db)
):
    """List all strategies"""
    query = db.query(Strategy)

    if active_only:
        query = query.filter(Strategy.is_active == True)

    strategies = query.order_by(Strategy.created_at.desc()).limit(limit).all()

    return {
        "strategies": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "strategy_type": s.strategy_type,
                "tickers": s.tickers,
                "created_at": s.created_at.isoformat(),
                "is_active": s.is_active,
                "backtest_count": db.query(BacktestResult).filter(BacktestResult.strategy_id == s.id).count()
            }
            for s in strategies
        ]
    }


@app.post("/strategies")
async def create_strategy(request: dict, db=Depends(get_db)):
    """Create a new strategy manually"""
    try:
        # Extract data from request
        name = request.get('name')
        description = request.get('description', '')
        tickers = request.get('tickers', [])
        strategy_type = request.get('strategy_type', '')
        indicators = request.get('indicators', [])
        risk_management = request.get('risk_management', {})

        # Validate required fields
        if not name:
            raise HTTPException(status_code=400, detail="Strategy name is required")
        if not tickers:
            raise HTTPException(status_code=400, detail="At least one ticker is required")
        if not strategy_type:
            raise HTTPException(status_code=400, detail="Strategy type is required")

        # Check if strategy with this name already exists
        existing = db.query(Strategy).filter(Strategy.name == name).first()
        if existing:
            # Make name unique by adding timestamp
            name = f"{name} [{datetime.now().strftime('%Y%m%d_%H%M%S')}]"

        # Create strategy entry/exit conditions based on strategy type and indicators
        entry_conditions = {}
        exit_conditions = {}

        # Create database record
        db_strategy = Strategy(
            name=name,
            description=description,
            tickers=tickers,
            strategy_type=strategy_type,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            stop_loss_pct=risk_management.get('stop_loss_pct', 5.0),
            take_profit_pct=risk_management.get('take_profit_pct', 10.0),
            position_size_pct=risk_management.get('position_size_pct', 10.0),
            holding_period_days=risk_management.get('holding_period_days', 5),
            rationale=f"Manual strategy creation: {strategy_type} strategy for {', '.join(tickers)}",
            market_analysis="",
            risk_assessment="",
            is_active=True
        )

        db.add(db_strategy)
        db.commit()
        db.refresh(db_strategy)

        return {
            "success": True,
            "strategy_id": db_strategy.id,
            "message": f"Strategy '{db_strategy.name}' created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create strategy: {str(e)}")


@app.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: int, db=Depends(get_db)):
    """Get strategy details"""
    strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return {
        "id": strategy.id,
        "name": strategy.name,
        "description": strategy.description,
        "strategy_type": strategy.strategy_type,
        "tickers": strategy.tickers,
        "indicators": strategy.indicators,
        "entry_conditions": strategy.entry_conditions,
        "exit_conditions": strategy.exit_conditions,
        "risk_management": {
            "stop_loss_pct": strategy.stop_loss_pct,
            "take_profit_pct": strategy.take_profit_pct,
            "position_size_pct": strategy.position_size_pct
        },
        "holding_period_days": strategy.holding_period_days,
        "rationale": strategy.rationale,
        "market_analysis": strategy.market_analysis,
        "risk_assessment": strategy.risk_assessment,
        "created_at": strategy.created_at.isoformat(),
        "is_active": strategy.is_active
    }


@app.post("/strategies/cleanup")
async def cleanup_zero_trade_strategies(db=Depends(get_db)):
    """Delete all strategies and backtest results with 0 trades"""
    try:
        # Find all backtest results with 0 trades
        zero_trade_results = db.query(BacktestResult).filter(
            BacktestResult.total_trades == 0
        ).all()

        strategy_ids_to_delete = set()
        deleted_backtest_count = 0

        # Delete backtest results and collect strategy IDs
        for result in zero_trade_results:
            strategy_ids_to_delete.add(result.strategy_id)
            db.delete(result)
            deleted_backtest_count += 1

        # Delete strategies that only have zero-trade results
        deleted_strategy_count = 0
        for strategy_id in strategy_ids_to_delete:
            # Check if this strategy has ANY backtest with trades > 0
            has_good_results = db.query(BacktestResult).filter(
                BacktestResult.strategy_id == strategy_id,
                BacktestResult.total_trades > 0
            ).first()

            # Only delete if ALL backtests had 0 trades
            if not has_good_results:
                strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
                if strategy:
                    db.delete(strategy)
                    deleted_strategy_count += 1

        db.commit()

        return {
            "success": True,
            "deleted_strategies": deleted_strategy_count,
            "deleted_backtests": deleted_backtest_count,
            "message": f"Deleted {deleted_strategy_count} strategies and {deleted_backtest_count} backtest results with 0 trades"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# BACKTESTING ENDPOINTS
# =======================

@app.post("/backtest")
async def backtest_strategy(request: BacktestRequest, db=Depends(get_db)):
    """Backtest a strategy"""
    try:
        # Get strategy configuration
        if request.strategy_id:
            strategy_db = db.query(Strategy).filter(Strategy.id == request.strategy_id).first()
            if not strategy_db:
                raise HTTPException(status_code=404, detail="Strategy not found")

            strategy_config = {
                "id": strategy_db.id,
                "name": strategy_db.name,
                "tickers": strategy_db.tickers,
                "indicators": strategy_db.indicators,
                "strategy_type": strategy_db.strategy_type,
                "risk_management": {
                    "stop_loss_pct": strategy_db.stop_loss_pct,
                    "take_profit_pct": strategy_db.take_profit_pct,
                    "position_size_pct": strategy_db.position_size_pct
                }
            }
        elif request.strategy_config:
            strategy_config = request.strategy_config
        else:
            raise HTTPException(status_code=400, detail="Either strategy_id or strategy_config required")

        # Fetch historical data
        tickers = strategy_config.get('tickers', [])
        period = request.period if request.period else "1y"
        market_data = fetch_market_data(tickers, period)

        if market_data.empty:
            raise HTTPException(status_code=400, detail="No market data available")

        # Run backtest
        engine = BacktestingEngine(initial_capital=request.initial_capital)
        results = engine.backtest_strategy(strategy_config, market_data)

        # Convert numpy types to Python types for PostgreSQL compatibility
        trades_clean = convert_numpy_types(results['trades'])
        equity_curve_clean = convert_numpy_types(results['equity_curve'])
        metrics_clean = convert_numpy_types(results['metrics'])

        # Calculate start date based on period
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180,
            "1y": 365, "2y": 730, "5y": 1825, "10y": 3650
        }
        days_back = period_days.get(period, 365)

        # Save backtest results
        backtest_result = BacktestResult(
            strategy_id=request.strategy_id,
            strategy_name=results['strategy_name'],
            start_date=datetime.now() - timedelta(days=days_back),
            end_date=datetime.now(),
            initial_capital=request.initial_capital,
            tickers_tested=results['tickers'],
            total_return_pct=metrics_clean['total_return_pct'],
            total_trades=metrics_clean['total_trades'],
            winning_trades=metrics_clean['winning_trades'],
            losing_trades=metrics_clean['losing_trades'],
            win_rate=metrics_clean['win_rate'],
            sharpe_ratio=metrics_clean['sharpe_ratio'],
            sortino_ratio=metrics_clean['sortino_ratio'],
            max_drawdown_pct=metrics_clean['max_drawdown_pct'],
            profit_factor=metrics_clean['profit_factor'],
            avg_win=metrics_clean['avg_win'],
            avg_loss=metrics_clean['avg_loss'],
            trades=trades_clean,
            equity_curve=equity_curve_clean,
            quality_score=metrics_clean['quality_score'],
            kelly_criterion=metrics_clean.get('kelly_criterion'),
            kelly_position_pct=metrics_clean.get('kelly_position_pct'),
            kelly_risk_level=metrics_clean.get('kelly_risk_level')
        )
        db.add(backtest_result)
        db.commit()
        db.refresh(backtest_result)

        # Build response and ensure ALL numpy types are converted
        response = {
            "success": True,
            "backtest_id": int(backtest_result.id),
            "strategy_name": str(results['strategy_name']),
            "metrics": metrics_clean,  # Use cleaned metrics (numpy types converted)
            "total_trades": int(len(trades_clean)),
            "equity_curve_points": int(len(equity_curve_clean))
        }

        # Final safety conversion to catch any remaining numpy types
        return convert_numpy_types(response)

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"BACKTEST ERROR: {error_detail}")  # Log to Railway console
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/results")
async def list_backtest_results(
    limit: int = 50,
    min_sharpe: float = None,
    db=Depends(get_db)
):
    """List backtest results"""
    query = db.query(BacktestResult)

    if min_sharpe is not None:
        query = query.filter(BacktestResult.sharpe_ratio >= min_sharpe)

    results = query.order_by(BacktestResult.created_at.desc()).limit(limit).all()

    return {
        "results": [
            {
                "id": r.id,
                "strategy_name": r.strategy_name,
                "total_return_pct": r.total_return_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "win_rate": r.win_rate,
                "max_drawdown_pct": r.max_drawdown_pct,
                "total_trades": r.total_trades,
                "quality_score": r.quality_score,
                "created_at": r.created_at.isoformat(),
                # Kelly Criterion fields
                "kelly_criterion": r.kelly_criterion,
                "kelly_position_pct": r.kelly_position_pct,
                "kelly_risk_level": r.kelly_risk_level
            }
            for r in results
        ]
    }


@app.get("/backtest/results/{backtest_id}")
async def get_backtest_result(backtest_id: int, db=Depends(get_db)):
    """Get detailed backtest result"""
    result = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()

    if not result:
        raise HTTPException(status_code=404, detail="Backtest result not found")

    return {
        "id": result.id,
        "strategy_id": result.strategy_id,
        "strategy_name": result.strategy_name,
        "created_at": result.created_at.isoformat(),
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "initial_capital": result.initial_capital,
        "tickers_tested": result.tickers_tested,
        "metrics": {
            "total_return_pct": result.total_return_pct,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "profit_factor": result.profit_factor,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "quality_score": result.quality_score,
            # Kelly Criterion optimal position sizing
            "kelly_criterion": result.kelly_criterion,
            "kelly_position_pct": result.kelly_position_pct,
            "kelly_risk_level": result.kelly_risk_level
        },
        "trades": result.trades,
        "equity_curve": result.equity_curve
    }


@app.get("/backtest/strategy-details/{backtest_id}")
async def get_strategy_details(backtest_id: int, db=Depends(get_db)):
    """Get detailed strategy explanation and logic"""
    result = db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()

    if not result:
        raise HTTPException(status_code=404, detail="Backtest result not found")

    # Get the strategy
    strategy = db.query(Strategy).filter(Strategy.id == result.strategy_id).first()

    if not strategy:
        return {
            "error": "Strategy not found",
            "strategy_name": result.strategy_name
        }

    # Build strategy config for visualization
    strategy_config = {
        'id': strategy.id,
        'name': strategy.name,
        'strategy_type': strategy.strategy_type,
        'tickers': strategy.tickers,
        'indicators': strategy.indicators,
        'risk_management': {
            'stop_loss_pct': strategy.stop_loss_pct,
            'take_profit_pct': strategy.take_profit_pct,
            'position_size_pct': strategy.position_size_pct,
            'max_positions': 3
        }
    }

    # Get human-readable description
    description = StrategyVisualizer.get_strategy_description(strategy_config)

    return {
        "backtest_id": backtest_id,
        "strategy_id": strategy.id,
        "strategy_config": strategy_config,
        "description": description,
        "tickers_tested": result.tickers_tested,
        "trades": result.trades,
        "start_date": result.start_date.isoformat() if result.start_date else None,
        "end_date": result.end_date.isoformat() if result.end_date else None
    }


# =======================
# LIVE SIGNALS ENDPOINT
# =======================

@app.get("/signals/live/{strategy_id}")
async def get_live_signals(strategy_id: int, capital: float = 100000, db=Depends(get_db)):
    """
    Get live trading signals for a strategy

    Returns current BUY/SELL/HOLD signals with entry/exit prices and risk management
    """
    try:
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Build strategy config
        strategy_config = {
            'id': strategy.id,
            'name': strategy.name,
            'tickers': strategy.tickers,
            'strategy_type': strategy.strategy_type,
            'indicators': strategy.indicators,
            'risk_management': {
                'stop_loss_pct': strategy.stop_loss_pct,
                'take_profit_pct': strategy.take_profit_pct,
                'position_size_pct': strategy.position_size_pct
            }
        }

        # Generate live signals
        signals_data = LiveSignalGenerator.generate_signals(strategy_config, period="3mo")

        # Add position sizing based on capital
        for signal in signals_data['signals']:
            if signal.get('signal') in ['BUY', 'SELL']:
                position_size_pct = signal.get('position_size_pct', 25.0)
                signal['position_size_usd'] = round(capital * (position_size_pct / 100), 2)
                signal['shares_to_trade'] = int(signal['position_size_usd'] / signal['current_price'])

        return {
            "success": True,
            "strategy_id": strategy_id,
            "capital": capital,
            **signals_data
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"LIVE SIGNALS ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# MARKET SCANNER ENDPOINT
# =======================

@app.post("/scanner/run")
async def run_market_scan(
    strategy_ids: List[int],
    universe: Optional[List[str]] = None,
    min_confidence: str = "MEDIUM",
    include_backtest: bool = False,
    backtest_period: str = "6mo",
    db=Depends(get_db)
):
    """
    Scan market for best trading opportunities across multiple stocks and strategies

    Args:
        strategy_ids: List of strategy IDs to use for scanning
        universe: Optional list of tickers to scan (default: curated 50+ stocks)
        min_confidence: Minimum signal confidence (LOW, MEDIUM, HIGH)
        include_backtest: Whether to run backtests on top signals (slower but shows historical performance)
        backtest_period: Period for backtesting (3mo, 6mo, 1y)

    Returns ranked list of BUY and SELL signals with quality scores and optional backtest results
    """
    try:
        # Get strategy configurations
        strategies = []
        for strategy_id in strategy_ids:
            strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()

            if not strategy:
                continue

            strategies.append({
                'id': strategy.id,
                'name': strategy.name,
                # DO NOT include 'tickers' here - scanner will override per stock
                'strategy_type': strategy.strategy_type,
                'indicators': strategy.indicators,
                'risk_management': {
                    'stop_loss_pct': strategy.stop_loss_pct,
                    'take_profit_pct': strategy.take_profit_pct,
                    'position_size_pct': strategy.position_size_pct
                }
            })

        if not strategies:
            raise HTTPException(status_code=400, detail="No valid strategies found")

        # Run scan - with or without backtesting
        if include_backtest:
            scan_results = MarketScanner.scan_with_backtest(
                strategies=strategies,
                universe=universe,
                max_workers=10,  # Fewer workers when backtesting to avoid rate limits
                min_confidence=min_confidence,
                backtest_period=backtest_period
            )
        else:
            scan_results = MarketScanner.scan_market(
                strategies=strategies,
                universe=universe,
                max_workers=20,  # Parallel processing
                min_confidence=min_confidence
            )

        # Get multi-strategy confirmations
        confirmations = MarketScanner.get_multi_strategy_confirmations(
            scan_results.get('all_signals', [])
        )

        return {
            "success": True,
            "scan_results": scan_results,
            "multi_strategy_confirmations": confirmations[:10],  # Top 10 confirmed signals
            "backtest_included": include_backtest
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"MARKET SCANNER ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# PAPER TRADING ENDPOINTS
# =======================

@app.post("/paper-trading/execute")
async def execute_paper_trading(request: PaperTradeRequest, db=Depends(get_db)):
    """Execute paper trading for a strategy"""
    try:
        strategy = db.query(Strategy).filter(Strategy.id == request.strategy_id).first()

        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy_config = {
            "id": strategy.id,
            "name": strategy.name,
            "tickers": strategy.tickers,
            "indicators": strategy.indicators,
            "strategy_type": strategy.strategy_type,
            "risk_management": {
                "stop_loss_pct": strategy.stop_loss_pct,
                "take_profit_pct": strategy.take_profit_pct,
                "position_size_pct": strategy.position_size_pct,
                "max_positions": 3
            }
        }

        simulator = PaperTradingSimulator()
        results = simulator.execute_strategy(strategy_config)

        return {
            "success": True,
            "strategy_name": results['strategy_name'],
            "timestamp": results['timestamp'],
            "actions_taken": results['actions_taken'],
            "open_positions": len(results['positions']),
            "portfolio_value": results['portfolio_value'],
            "cash": results['cash'],
            "total_return_pct": results['total_return_pct']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper-trading/performance")
async def get_paper_trading_performance(db=Depends(get_db)):
    """Get paper trading performance summary"""
    simulator = PaperTradingSimulator()
    summary = simulator.get_performance_summary()
    return summary


@app.get("/paper-trading/positions")
async def get_paper_trading_positions(db=Depends(get_db)):
    """Get current paper trading positions"""
    positions = db.query(PaperTrade).filter(PaperTrade.is_open == True).all()

    return {
        "open_positions": [
            {
                "id": p.id,
                "strategy_name": p.strategy_name,
                "ticker": p.ticker,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "entry_date": p.entry_date.isoformat(),
                "position_size_usd": p.position_size_usd,
                "stop_loss_price": p.stop_loss_price,
                "take_profit_price": p.take_profit_price
            }
            for p in positions
        ]
    }


# =======================
# PORTFOLIO OPTIMIZATION
# =======================

@app.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest, db=Depends(get_db)):
    """
    Optimize portfolio allocation across strategies using advanced optimization

    Supports:
    - max_sharpe: Maximize Sharpe ratio (best risk-adjusted returns)
    - min_volatility: Minimize portfolio risk
    - max_return: Maximize returns
    - risk_parity: Equal risk contribution from each strategy
    """
    try:
        # Get strategies and backtest results
        strategies = []
        backtest_results = []
        missing_backtests = []
        missing_strategies = []

        # Case 1: Using strategy_ids (from database)
        if request.strategy_ids:
            for strategy_id in request.strategy_ids:
                strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
                if not strategy:
                    missing_strategies.append(strategy_id)
                    continue

                # Get latest backtest result
                backtest = db.query(BacktestResult).filter(
                    BacktestResult.strategy_id == strategy_id
                ).order_by(BacktestResult.created_at.desc()).first()

                if not backtest:
                    missing_backtests.append(strategy.name)
                    continue

                strategies.append({
                    "id": strategy.id,
                    "name": strategy.name,
                    "strategy_type": strategy.strategy_type
                })

                backtest_results.append({
                    "strategy_name": strategy.name,
                    "equity_curve": backtest.equity_curve,
                    "total_return_pct": backtest.total_return_pct,
                    "sharpe_ratio": backtest.sharpe_ratio
                })

        # Case 2: Using strategies directly (from Complete Trading System)
        elif request.strategies:
            for idx, strat_data in enumerate(request.strategies):
                # Generate synthetic equity curve from expected return and volatility
                # This is a simplification - ideally we'd re-run backtest
                expected_return = strat_data.get('expected_return', 0)
                volatility = strat_data.get('volatility', 20)
                sharpe = strat_data.get('sharpe_ratio', 0)

                # Create synthetic equity curve (252 trading days)
                days = 252
                daily_return = expected_return / 100 / days
                daily_vol = volatility / 100 / np.sqrt(days)

                # Generate random walk with drift
                np.random.seed(idx)  # For reproducibility
                returns = np.random.normal(daily_return, daily_vol, days)
                equity = request.total_capital * (1 + returns).cumprod()

                strategies.append({
                    "id": strat_data.get('id', f"strategy_{idx}"),
                    "name": strat_data.get('name', f"Strategy {idx}"),
                    "strategy_type": "custom"
                })

                # Calculate max drawdown from synthetic equity curve
                peak = equity[0]
                max_dd = 0
                for value in equity:
                    if value > peak:
                        peak = value
                    dd = ((peak - value) / peak) * 100
                    max_dd = max(max_dd, dd)

                backtest_results.append({
                    "strategy_name": strat_data.get('name', f"Strategy {idx}"),
                    "equity_curve": [
                        {"date": f"2024-{i//21+1:02d}-{i%21+1:02d}", "equity": float(equity[i])}
                        for i in range(days)
                    ],
                    "total_return_pct": expected_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown_pct": max_dd,
                    "win_rate": 50.0  # Default assumption for synthetic data
                })
        else:
            return {"success": False, "error": "Must provide either strategy_ids or strategies"}

        # Build helpful error message if not enough strategies
        if len(strategies) < 2:
            error_parts = []

            if missing_backtests:
                error_parts.append(f"These strategies need backtest results: {', '.join(missing_backtests)}")
                error_parts.append("Go to the Backtest page and run backtests for these strategies first.")

            if missing_strategies:
                error_parts.append(f"Strategy IDs not found: {', '.join(map(str, missing_strategies))}")

            if not error_parts:
                error_parts.append("Need at least 2 strategies with backtest results for portfolio optimization")

            error_msg = " ".join(error_parts)
            return {"success": False, "error": error_msg}

        # Use advanced optimizer
        optimizer = AdvancedPortfolioOptimizer(risk_free_rate=0.02)

        # Run optimization
        optimization_result = optimizer.optimize_portfolio(
            strategies=strategies,
            backtest_results=backtest_results,
            total_capital=request.total_capital,
            method=request.method,
            constraints=request.constraints or {}
        )

        if not optimization_result.get('success'):
            # Log the error for debugging
            print(f"Portfolio optimization failed: {optimization_result.get('error')}")
            return optimization_result  # Return error from optimizer

        # Save optimization
        # Use strategy_ids if provided, otherwise use strategy identifiers from strategies
        strategy_list = request.strategy_ids if request.strategy_ids else [s.get('id') for s in request.strategies]

        portfolio = PortfolioAllocation(
            name=f"Portfolio {request.method} {datetime.now().strftime('%Y-%m-%d')}",
            total_capital=request.total_capital,
            strategies=strategy_list,
            optimization_method=request.method,
            constraints=request.constraints or {},
            expected_return=optimization_result['expected_return'],
            expected_volatility=optimization_result['expected_volatility'],
            expected_sharpe=optimization_result['expected_sharpe'],
            allocations=optimization_result['allocations'],
            is_active=True
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)

        response = {
            "success": True,
            "portfolio_id": portfolio.id,
            **optimization_result
        }

        # Convert numpy types to Python types for JSON serialization
        return convert_numpy_types(response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# AI LEARNING ENDPOINTS
# =======================

@app.post("/ai/learn")
async def ai_learn_from_results(db=Depends(get_db)):
    """Analyze results and extract learning insights"""
    try:
        # Get recent strategies and backtest results
        strategies = db.query(Strategy).order_by(Strategy.created_at.desc()).limit(10).all()
        backtest_results = db.query(BacktestResult).order_by(BacktestResult.created_at.desc()).limit(10).all()

        if not strategies or not backtest_results:
            raise HTTPException(status_code=400, detail="Not enough data to learn from")

        # Prepare data
        strategies_data = [
            {
                "name": s.name,
                "strategy_type": s.strategy_type,
                "indicators": s.indicators,
                "stop_loss_pct": s.stop_loss_pct,
                "take_profit_pct": s.take_profit_pct
            }
            for s in strategies
        ]

        results_data = [
            {
                "strategy_name": r.strategy_name,
                "sharpe_ratio": r.sharpe_ratio,
                "total_return_pct": r.total_return_pct,
                "win_rate": r.win_rate,
                "max_drawdown_pct": r.max_drawdown_pct,
                "quality_score": r.quality_score
            }
            for r in backtest_results
        ]

        # Use AI to learn
        ai_generator = AIStrategyGenerator()
        learning = ai_generator.learn_from_results(strategies_data, results_data)

        # Save learning
        ai_learning = AILearning(
            learning_type="performance_analysis",
            description="AI analyzed strategy performance and extracted insights",
            strategy_ids=[s.id for s in strategies],
            performance_data=results_data,
            key_insights=learning,
            recommendations=learning.get('recommendations_for_next_generation', []),
            confidence_score=0.8
        )
        db.add(ai_learning)
        db.commit()

        return {
            "success": True,
            "learning_id": ai_learning.id,
            "insights": learning
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/learning")
async def get_learning_insights(limit: int = 10, db=Depends(get_db)):
    """Get AI learning insights"""
    learnings = db.query(AILearning).order_by(AILearning.created_at.desc()).limit(limit).all()

    return {
        "insights": [
            {
                "id": l.id,
                "learning_type": l.learning_type,
                "description": l.description,
                "key_insights": l.key_insights,
                "recommendations": l.recommendations,
                "confidence_score": l.confidence_score,
                "created_at": l.created_at.isoformat()
            }
            for l in learnings
        ]
    }


# =======================
# ML PREDICTION ENDPOINTS
# =======================

class MLTrainingRequest(BaseModel):
    ticker: str
    period: str = "2y"  # Historical data period
    test_size: float = 0.2  # Fraction for testing
    horizon: int = 1  # Days ahead to predict


@app.post("/ml/train")
async def train_ml_model(request: MLTrainingRequest):
    """
    Train XGBoost model to predict price movements

    Args:
        ticker: Stock ticker symbol
        period: Historical data period (default 2 years)
        test_size: Fraction of data for testing (default 0.2)
        horizon: Days ahead to predict (default 1)

    Returns:
        Training results with model performance metrics
    """
    try:
        predictor = MLPricePredictor()

        # Train model
        result = predictor.train_model(
            ticker=request.ticker,
            period=request.period,
            test_size=request.test_size,
            horizon=request.horizon
        )

        # Convert numpy types to Python types for JSON serialization
        return convert_numpy_types(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict/{ticker}")
async def get_ml_prediction(ticker: str):
    """
    Get ML prediction for next-day price movement

    Args:
        ticker: Stock ticker symbol

    Returns:
        Prediction with confidence scores
    """
    try:
        predictor = MLPricePredictor()

        # Get prediction
        result = predictor.predict(ticker, return_proba=True)

        # Convert numpy types to Python types for JSON serialization
        return convert_numpy_types(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/models")
async def list_ml_models():
    """
    List all trained ML models

    Returns:
        List of trained models with metadata
    """
    try:
        predictor = MLPricePredictor()
        models = predictor.list_models()

        return {
            "success": True,
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ml/model/{ticker}")
async def delete_ml_model(ticker: str):
    """
    Delete a trained ML model

    Args:
        ticker: Stock ticker symbol

    Returns:
        Success status
    """
    try:
        predictor = MLPricePredictor()
        model_path = predictor.model_dir / f"{ticker}_model.pkl"

        if model_path.exists():
            model_path.unlink()
            return {
                "success": True,
                "message": f"Model for {ticker} deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": f"No model found for {ticker}"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# HMM REGIME DETECTION ENDPOINTS
# =======================

class RegimeTrainingRequest(BaseModel):
    ticker: str
    period: str = "2y"  # Historical data period
    n_regimes: int = 3  # Number of regimes (default 3: bull/bear/consolidation)
    n_iter: int = 100  # Training iterations


@app.post("/regime/train")
async def train_regime_detector(request: RegimeTrainingRequest):
    """
    Train HMM model to detect market regimes

    Args:
        ticker: Stock ticker symbol
        period: Historical data period (default 2 years)
        n_regimes: Number of hidden states (default 3)
        n_iter: Training iterations

    Returns:
        Training results with regime characteristics
    """
    try:
        detector = HMMRegimeDetector(
            n_regimes=request.n_regimes,
            random_state=42
        )

        # Train model
        result = detector.train(
            ticker=request.ticker,
            period=request.period,
            n_iter=request.n_iter
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regime/predict/{ticker}")
async def predict_regime(ticker: str):
    """
    Predict current market regime

    Args:
        ticker: Stock ticker symbol

    Returns:
        Current regime with probabilities and transitions
    """
    try:
        # Need to train first or load model
        # For now, train on-demand (in production, would cache models)
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)

        # Train on 2 years of data
        train_result = detector.train(ticker, period="2y")

        if not train_result['success']:
            return train_result

        # Get prediction
        pred_result = detector.predict_regime(ticker)

        return pred_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regime/history/{ticker}")
async def get_regime_history(ticker: str, period: str = "1y"):
    """
    Get regime history over time for visualization

    Args:
        ticker: Stock ticker symbol
        period: Historical period (default 1 year)

    Returns:
        Timeline with prices and regimes
    """
    try:
        # Train detector
        detector = HMMRegimeDetector(n_regimes=3, random_state=42)

        train_result = detector.train(ticker, period=period)

        if not train_result['success']:
            return train_result

        # Get history
        history_result = detector.get_regime_history(ticker, period=period)

        return history_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =======================
# VECTORIZED BACKTESTING ENDPOINTS
# =======================

from vectorized_backtester import VectorizedBacktester

vectorized_backtester = VectorizedBacktester()


class VectorizedOptimizeRequest(BaseModel):
    ticker: str
    strategy_type: str
    period: str = "1y"
    param_ranges: Optional[Dict] = None


class BatchOptimizeRequest(BaseModel):
    tickers: List[str]
    strategies: List[str]
    period: str = "1y"


@app.post("/vectorized/optimize")
async def vectorized_optimize(request: VectorizedOptimizeRequest):
    """
    Optimize strategy parameters using vectorized backtesting (100x faster)

    Tests hundreds/thousands of parameter combinations in seconds
    """
    try:
        result = vectorized_backtester.optimize_strategy(
            ticker=request.ticker,
            strategy_type=request.strategy_type,
            period=request.period,
            param_ranges=request.param_ranges
        )

        # Convert numpy types to Python types for JSON serialization
        return convert_numpy_types(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorized/batch-optimize")
async def batch_optimize(request: BatchOptimizeRequest):
    """
    Optimize multiple strategies on multiple tickers using vectorized backtesting

    Example: 5 tickers × 4 strategies = 20 optimizations in seconds
    """
    try:
        result = vectorized_backtester.batch_optimize(
            tickers=request.tickers,
            strategies=request.strategies,
            period=request.period
        )

        # Convert numpy types to Python types for JSON serialization
        return convert_numpy_types(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vectorized/status")
async def vectorized_status():
    """Check if vectorbt is available"""
    from vectorized_backtester import VECTORBT_AVAILABLE

    return {
        "success": True,
        "vectorbt_available": VECTORBT_AVAILABLE,
        "method": "vectorbt" if VECTORBT_AVAILABLE else "numpy_fallback",
        "speedup": "100-1000x" if VECTORBT_AVAILABLE else "1x (install vectorbt for speedup)"
    }


# =======================
# AUTONOMOUS LEARNING ENDPOINTS
# =======================

class AutonomousAgentConfig(BaseModel):
    tickers: List[str] = ['SPY', 'QQQ', 'AAPL']
    interval_hours: Optional[int] = None
    interval_minutes: Optional[int] = None
    strategies_per_cycle: int = 3


@app.post("/autonomous/start")
async def start_autonomous_learning(config: AutonomousAgentConfig):
    """Start the autonomous learning agent"""
    global autonomous_agent, autonomous_agent_thread, autonomous_agent_config

    if autonomous_agent_thread and autonomous_agent_thread.is_alive():
        return {
            "success": False,
            "message": "Autonomous learning agent is already running"
        }

    # Store configuration
    autonomous_agent_config = {
        "tickers": config.tickers,
        "interval_hours": config.interval_hours,
        "interval_minutes": config.interval_minutes,
        "strategies_per_cycle": config.strategies_per_cycle,
        "min_quality_score": 70.0
    }

    # Create agent with either minutes or hours
    autonomous_agent = AutonomousLearningAgent(
        tickers=config.tickers,
        learning_interval_minutes=config.interval_minutes,
        learning_interval_hours=config.interval_hours,
        strategies_per_cycle=config.strategies_per_cycle,
        min_quality_score=70.0
    )

    # Start in background thread
    autonomous_agent_thread = threading.Thread(
        target=autonomous_agent.run_forever,
        daemon=True
    )
    autonomous_agent_thread.start()

    return {
        "success": True,
        "message": f"Autonomous learning agent started successfully for tickers: {', '.join(config.tickers)}",
        "config": autonomous_agent_config
    }


@app.post("/autonomous/stop")
async def stop_autonomous_learning():
    """Stop the autonomous learning agent"""
    global autonomous_agent, autonomous_agent_thread

    if not autonomous_agent or not autonomous_agent_thread or not autonomous_agent_thread.is_alive():
        return {
            "success": False,
            "message": "Autonomous learning agent is not running"
        }

    # Signal the agent to stop
    autonomous_agent.stop()

    return {
        "success": True,
        "message": "Stop signal sent to autonomous learning agent. It will stop after the current cycle."
    }


@app.get("/autonomous/status")
async def get_autonomous_status(db=Depends(get_db)):
    """Get status of autonomous learning agent"""
    global autonomous_agent_thread, autonomous_agent_config

    is_running = autonomous_agent_thread and autonomous_agent_thread.is_alive()

    # Get recent autonomous learning cycles
    recent_learnings = db.query(AILearning).filter(
        AILearning.learning_type == "autonomous_cycle"
    ).order_by(AILearning.created_at.desc()).limit(5).all()

    # Get statistics
    total_autonomous_cycles = db.query(AILearning).filter(
        AILearning.learning_type == "autonomous_cycle"
    ).count()

    # Get strategies created by autonomous agent
    autonomous_strategies = []
    if recent_learnings:
        for learning in recent_learnings:
            for strategy_id in learning.strategy_ids:
                strategy = db.query(Strategy).get(strategy_id)
                if strategy:
                    autonomous_strategies.append({
                        "id": strategy.id,
                        "name": strategy.name,
                        "created_at": strategy.created_at.isoformat(),
                        "is_active": strategy.is_active
                    })

    return {
        "is_running": is_running,
        "enabled_by_env": autonomous_agent_enabled,
        "current_config": autonomous_agent_config if is_running else None,
        "statistics": {
            "total_cycles": total_autonomous_cycles,
            "last_cycle": recent_learnings[0].created_at.isoformat() if recent_learnings else None,
            "strategies_generated": len(autonomous_strategies)
        },
        "recent_cycles": [
            {
                "cycle_id": l.id,
                "completed_at": l.created_at.isoformat(),
                "strategies_tested": len(l.strategy_ids),
                "confidence_score": l.confidence_score
            }
            for l in recent_learnings
        ]
    }


@app.post("/autonomous/trigger")
async def trigger_learning_cycle():
    """Manually trigger one learning cycle"""
    global autonomous_agent

    if not autonomous_agent:
        # Create temporary agent
        agent = AutonomousLearningAgent(
            tickers=['SPY', 'QQQ', 'AAPL'],
            learning_interval_hours=6,
            strategies_per_cycle=3,
            min_quality_score=70.0
        )
    else:
        agent = autonomous_agent

    # Run one cycle in background
    def run_cycle():
        agent.learning_cycle()

    thread = threading.Thread(target=run_cycle, daemon=True)
    thread.start()

    return {
        "success": True,
        "message": "Learning cycle triggered. Check /autonomous/status for results."
    }


# =======================
# ANALYTICS ENDPOINTS
# =======================

@app.get("/analytics/dashboard")
async def get_dashboard_analytics(db=Depends(get_db)):
    """Get dashboard analytics"""
    total_strategies = db.query(Strategy).count()
    active_strategies = db.query(Strategy).filter(Strategy.is_active == True).count()
    total_backtests = db.query(BacktestResult).count()
    paper_trades = db.query(PaperTrade).count()

    # Best performing strategy (exclude 0-trade strategies)
    best_backtest = db.query(BacktestResult).filter(
        BacktestResult.total_trades > 0
    ).order_by(BacktestResult.quality_score.desc()).first()

    return {
        "summary": {
            "total_strategies": total_strategies,
            "active_strategies": active_strategies,
            "total_backtests": total_backtests,
            "total_paper_trades": paper_trades
        },
        "best_strategy": {
            "name": best_backtest.strategy_name if best_backtest else None,
            "quality_score": best_backtest.quality_score if best_backtest else None,
            "sharpe_ratio": best_backtest.sharpe_ratio if best_backtest else None,
            "total_return_pct": best_backtest.total_return_pct if best_backtest else None
        } if best_backtest else None
    }


# =======================
# PAIR TRADING ENDPOINTS
# =======================

class PairAnalyzeRequest(BaseModel):
    stock_a: str
    stock_b: str
    period: str = "1y"
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 4.0


class PairScanRequest(BaseModel):
    universe: Optional[str] = "tech"  # tech, finance, healthcare, energy, consumer, etf, gold
    custom_tickers: Optional[List[str]] = None
    period: str = "1y"
    min_quality_score: float = 50.0
    correlation_threshold: float = 0.7


class PairBacktestRequest(BaseModel):
    stock_a: str
    stock_b: str
    period: str = "1y"
    initial_capital: float = 100000
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 4.0


@app.post("/pairs/analyze")
async def analyze_pair(request: PairAnalyzeRequest, db=Depends(get_db)):
    """
    Analyze a specific pair for cointegration and trading suitability.

    Returns full statistical analysis including:
    - Engle-Granger cointegration test
    - ADF stationarity test
    - Hurst exponent (mean reversion tendency)
    - Half-life (time to mean reversion)
    - Current z-score and signal
    """
    try:
        # Download price data
        data_a = yf.download(request.stock_a, period=request.period, progress=False)['Close']
        data_b = yf.download(request.stock_b, period=request.period, progress=False)['Close']

        if data_a.empty or data_b.empty:
            raise HTTPException(status_code=400, detail="Failed to download price data")

        # Initialize analyzers
        stats = PairTradingStatistics()
        strategy = PairTradingStrategy(
            entry_threshold=request.entry_threshold,
            exit_threshold=request.exit_threshold,
            stop_loss_threshold=request.stop_loss_threshold
        )

        # Cointegration test
        coint_result = stats.engle_granger_test(data_a, data_b)

        # Calculate spread
        spread, hedge_ratio = strategy.calculate_spread(data_a, data_b, coint_result.hedge_ratio)

        # ADF test on spread
        adf_result = stats.adf_test(spread)

        # Hurst exponent
        hurst = stats.calculate_hurst_exponent(spread)

        # Half-life
        half_life = stats.calculate_half_life(spread)

        # Correlation
        common_idx = data_a.index.intersection(data_b.index)
        correlation = data_a.loc[common_idx].corr(data_b.loc[common_idx])

        # Current signal
        current_signal = strategy.get_current_signal(data_a, data_b)

        # Calculate quality score
        scanner = PairScanner()
        quality_score = scanner._calculate_quality_score(
            coint_p_value=coint_result.p_value,
            adf_p_value=adf_result['p_value'],
            hurst=hurst,
            half_life=half_life
        )

        # Determine recommendation
        recommendation = "NOT_SUITABLE"
        if coint_result.is_cointegrated and quality_score >= 50:
            if current_signal['zscore'] < -request.entry_threshold:
                recommendation = "LONG_SPREAD"
            elif current_signal['zscore'] > request.entry_threshold:
                recommendation = "SHORT_SPREAD"
            elif abs(current_signal['zscore']) < request.exit_threshold:
                recommendation = "SPREAD_AT_MEAN"
            else:
                recommendation = "WAIT"

        # Save to database if cointegrated
        if coint_result.is_cointegrated:
            existing_pair = db.query(PairTradingPair).filter(
                PairTradingPair.stock_a == request.stock_a,
                PairTradingPair.stock_b == request.stock_b
            ).first()

            if existing_pair:
                # Update existing
                existing_pair.cointegration_pvalue = coint_result.p_value
                existing_pair.hedge_ratio = coint_result.hedge_ratio
                existing_pair.correlation = float(correlation)
                existing_pair.hurst_exponent = hurst
                existing_pair.half_life_days = half_life
                existing_pair.adf_pvalue = adf_result['p_value']
                existing_pair.quality_score = quality_score
                existing_pair.spread_mean = float(spread.mean())
                existing_pair.spread_std = float(spread.std())
                existing_pair.current_zscore = current_signal['zscore']
                existing_pair.current_signal = current_signal['signal']
                existing_pair.last_tested = datetime.utcnow()
            else:
                # Create new
                new_pair = PairTradingPair(
                    stock_a=request.stock_a,
                    stock_b=request.stock_b,
                    is_cointegrated=True,
                    cointegration_pvalue=coint_result.p_value,
                    hedge_ratio=coint_result.hedge_ratio,
                    correlation=float(correlation),
                    hurst_exponent=hurst,
                    half_life_days=half_life,
                    adf_pvalue=adf_result['p_value'],
                    quality_score=quality_score,
                    spread_mean=float(spread.mean()),
                    spread_std=float(spread.std()),
                    current_zscore=current_signal['zscore'],
                    current_signal=current_signal['signal'],
                    entry_threshold=request.entry_threshold,
                    exit_threshold=request.exit_threshold,
                    stop_loss_threshold=request.stop_loss_threshold,
                    last_tested=datetime.utcnow()
                )
                db.add(new_pair)

            db.commit()

        return {
            "success": True,
            "pair": f"{request.stock_a}/{request.stock_b}",
            "analysis": {
                "cointegration": {
                    "is_cointegrated": coint_result.is_cointegrated,
                    "p_value": coint_result.p_value,
                    "hedge_ratio": coint_result.hedge_ratio,
                    "test_statistic": coint_result.test_statistic,
                    "critical_values": coint_result.critical_values
                },
                "adf_test": adf_result,
                "hurst_exponent": hurst,
                "hurst_interpretation": "Mean Reverting" if hurst < 0.5 else "Trending" if hurst > 0.5 else "Random Walk",
                "half_life_days": half_life,
                "correlation": float(correlation),
                "quality_score": quality_score
            },
            "current_state": {
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std()),
                "current_zscore": current_signal['zscore'],
                "current_signal": current_signal['signal'],
                "price_a": current_signal['price_a'],
                "price_b": current_signal['price_b']
            },
            "recommendation": recommendation,
            "thresholds": {
                "entry": request.entry_threshold,
                "exit": request.exit_threshold,
                "stop_loss": request.stop_loss_threshold
            }
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"PAIR ANALYZE ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pairs/scan")
async def scan_for_pairs(request: PairScanRequest, db=Depends(get_db)):
    """
    Auto-discover cointegrated pairs from a universe of stocks.

    Scans all possible pairs using:
    1. Correlation pre-filter
    2. Engle-Granger cointegration test
    3. Quality scoring

    Returns ranked list of suitable pairs.
    """
    try:
        start_time = time.time()

        scanner = PairScanner(
            correlation_threshold=request.correlation_threshold
        )

        # Run scan
        results = scanner.scan_for_pairs(
            universe=request.custom_tickers,
            universe_name=request.universe if not request.custom_tickers else None,
            period=request.period,
            min_quality_score=request.min_quality_score
        )

        scan_duration = time.time() - start_time

        # Convert results to dicts
        pairs_data = []
        for pair in results:
            pairs_data.append({
                "stock_a": pair.stock_a,
                "stock_b": pair.stock_b,
                "quality_score": pair.quality_score,
                "correlation": pair.correlation,
                "cointegration_pvalue": pair.cointegration.p_value,
                "hedge_ratio": pair.cointegration.hedge_ratio,
                "hurst_exponent": pair.hurst_exponent,
                "half_life_days": pair.half_life,
                "adf_pvalue": pair.adf_p_value,
                "current_zscore": pair.current_zscore,
                "recommendation": pair.recommendation
            })

        # Save scan result to database
        tickers_list = request.custom_tickers or PairScanner.UNIVERSES.get(
            request.universe.lower(), []
        )
        n_tickers = len(tickers_list)
        n_pairs_tested = n_tickers * (n_tickers - 1) // 2

        scan_record = PairScanResult(
            universe_name=request.universe if not request.custom_tickers else "custom",
            tickers_scanned=tickers_list,
            pairs_tested=n_pairs_tested,
            period=request.period,
            correlation_threshold=request.correlation_threshold,
            significance_level=0.05,
            min_quality_score=request.min_quality_score,
            pairs_found=len(results),
            avg_quality_score=np.mean([p.quality_score for p in results]) if results else None,
            best_pair=f"{results[0].stock_a}/{results[0].stock_b}" if results else None,
            best_quality_score=results[0].quality_score if results else None,
            results=pairs_data,
            scan_duration_seconds=scan_duration
        )
        db.add(scan_record)

        # Also save/update individual pairs
        for pair in results:
            existing = db.query(PairTradingPair).filter(
                PairTradingPair.stock_a == pair.stock_a,
                PairTradingPair.stock_b == pair.stock_b
            ).first()

            if existing:
                existing.quality_score = pair.quality_score
                existing.current_zscore = pair.current_zscore
                existing.current_signal = pair.recommendation
                existing.last_tested = datetime.utcnow()
            else:
                new_pair = PairTradingPair(
                    stock_a=pair.stock_a,
                    stock_b=pair.stock_b,
                    is_cointegrated=True,
                    cointegration_pvalue=pair.cointegration.p_value,
                    hedge_ratio=pair.cointegration.hedge_ratio,
                    correlation=pair.correlation,
                    hurst_exponent=pair.hurst_exponent,
                    half_life_days=pair.half_life,
                    adf_pvalue=pair.adf_p_value,
                    quality_score=pair.quality_score,
                    spread_mean=pair.spread_mean,
                    spread_std=pair.spread_std,
                    current_zscore=pair.current_zscore,
                    current_signal=pair.recommendation,
                    last_tested=datetime.utcnow()
                )
                db.add(new_pair)

        db.commit()

        return {
            "success": True,
            "scan_id": scan_record.id,
            "universe": request.universe if not request.custom_tickers else "custom",
            "tickers_scanned": n_tickers,
            "pairs_tested": n_pairs_tested,
            "pairs_found": len(results),
            "scan_duration_seconds": round(scan_duration, 2),
            "top_pairs": pairs_data[:20],  # Top 20
            "all_pairs": pairs_data
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"PAIR SCAN ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pairs/backtest")
async def backtest_pair(request: PairBacktestRequest, db=Depends(get_db)):
    """
    Backtest a pair trading strategy.

    Simulates dollar-neutral trading with:
    - Transaction costs (0.1%)
    - Borrowing costs (2% annual)
    - Slippage (0.05%)

    Returns performance metrics and trade history.
    """
    try:
        # Download price data
        data_a = yf.download(request.stock_a, period=request.period, progress=False)['Close']
        data_b = yf.download(request.stock_b, period=request.period, progress=False)['Close']

        if data_a.empty or data_b.empty:
            raise HTTPException(status_code=400, detail="Failed to download price data")

        # Initialize strategy and backtester
        strategy = PairTradingStrategy(
            entry_threshold=request.entry_threshold,
            exit_threshold=request.exit_threshold,
            stop_loss_threshold=request.stop_loss_threshold
        )

        backtester = PairBacktester(initial_capital=request.initial_capital)

        # Run backtest
        results = backtester.backtest_pair(data_a, data_b, strategy)

        if not results.get('success'):
            raise HTTPException(status_code=400, detail=results.get('error', 'Backtest failed'))

        # Convert numpy types
        results = convert_numpy_types(results)

        return {
            "success": True,
            "pair": f"{request.stock_a}/{request.stock_b}",
            "period": request.period,
            "initial_capital": request.initial_capital,
            "strategy_params": {
                "entry_threshold": request.entry_threshold,
                "exit_threshold": request.exit_threshold,
                "stop_loss_threshold": request.stop_loss_threshold
            },
            **results
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"PAIR BACKTEST ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pairs/signal/{stock_a}/{stock_b}")
async def get_pair_signal(
    stock_a: str,
    stock_b: str,
    period: str = "3mo",
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    db=Depends(get_db)
):
    """
    Get current trading signal for a pair.

    Returns real-time signal based on current z-score:
    - LONG_SPREAD: Buy stock A, Sell stock B
    - SHORT_SPREAD: Sell stock A, Buy stock B
    - EXIT: Close positions
    - HOLD: Maintain current position
    """
    try:
        # Download recent price data
        data_a = yf.download(stock_a, period=period, progress=False)['Close']
        data_b = yf.download(stock_b, period=period, progress=False)['Close']

        if data_a.empty or data_b.empty:
            raise HTTPException(status_code=400, detail="Failed to download price data")

        strategy = PairTradingStrategy(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )

        signal = strategy.get_current_signal(data_a, data_b)

        # Update database if pair exists
        existing = db.query(PairTradingPair).filter(
            PairTradingPair.stock_a == stock_a,
            PairTradingPair.stock_b == stock_b
        ).first()

        if existing:
            existing.current_zscore = signal['zscore']
            existing.current_signal = signal['signal']
            existing.last_signal_date = datetime.utcnow()
            db.commit()

        return {
            "success": True,
            "pair": f"{stock_a}/{stock_b}",
            **signal,
            "action_required": signal['signal'] in ['LONG_SPREAD', 'SHORT_SPREAD', 'EXIT']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pairs/active")
async def list_active_pairs(
    min_quality_score: float = 50.0,
    limit: int = 50,
    db=Depends(get_db)
):
    """
    List all active cointegrated pairs from the database.

    Returns pairs sorted by quality score with current signals.
    """
    try:
        pairs = db.query(PairTradingPair).filter(
            PairTradingPair.is_active == True,
            PairTradingPair.quality_score >= min_quality_score
        ).order_by(PairTradingPair.quality_score.desc()).limit(limit).all()

        return {
            "success": True,
            "count": len(pairs),
            "pairs": [
                {
                    "id": p.id,
                    "stock_a": p.stock_a,
                    "stock_b": p.stock_b,
                    "quality_score": p.quality_score,
                    "correlation": p.correlation,
                    "cointegration_pvalue": p.cointegration_pvalue,
                    "hedge_ratio": p.hedge_ratio,
                    "hurst_exponent": p.hurst_exponent,
                    "half_life_days": p.half_life_days,
                    "current_zscore": p.current_zscore,
                    "current_signal": p.current_signal,
                    "total_trades": p.total_trades,
                    "total_pnl": p.total_pnl,
                    "last_tested": p.last_tested.isoformat() if p.last_tested else None,
                    "thresholds": {
                        "entry": p.entry_threshold,
                        "exit": p.exit_threshold,
                        "stop_loss": p.stop_loss_threshold
                    }
                }
                for p in pairs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pairs/{pair_id}")
async def get_pair_details(pair_id: int, db=Depends(get_db)):
    """Get detailed information about a specific pair"""
    pair = db.query(PairTradingPair).filter(PairTradingPair.id == pair_id).first()

    if not pair:
        raise HTTPException(status_code=404, detail="Pair not found")

    # Get positions for this pair
    positions = db.query(PairTradingPosition).filter(
        PairTradingPosition.pair_id == pair_id
    ).order_by(PairTradingPosition.created_at.desc()).limit(20).all()

    return {
        "success": True,
        "pair": {
            "id": pair.id,
            "stock_a": pair.stock_a,
            "stock_b": pair.stock_b,
            "quality_score": pair.quality_score,
            "correlation": pair.correlation,
            "cointegration_pvalue": pair.cointegration_pvalue,
            "hedge_ratio": pair.hedge_ratio,
            "hurst_exponent": pair.hurst_exponent,
            "half_life_days": pair.half_life_days,
            "adf_pvalue": pair.adf_pvalue,
            "spread_mean": pair.spread_mean,
            "spread_std": pair.spread_std,
            "current_zscore": pair.current_zscore,
            "current_signal": pair.current_signal,
            "total_trades": pair.total_trades,
            "winning_trades": pair.winning_trades,
            "win_rate": (pair.winning_trades / pair.total_trades * 100) if pair.total_trades > 0 else 0,
            "total_pnl": pair.total_pnl,
            "thresholds": {
                "entry": pair.entry_threshold,
                "exit": pair.exit_threshold,
                "stop_loss": pair.stop_loss_threshold
            },
            "created_at": pair.created_at.isoformat(),
            "last_tested": pair.last_tested.isoformat() if pair.last_tested else None,
            "is_active": pair.is_active
        },
        "recent_positions": [
            {
                "id": pos.id,
                "position_type": pos.position_type,
                "entry_date": pos.entry_date.isoformat(),
                "exit_date": pos.exit_date.isoformat() if pos.exit_date else None,
                "entry_zscore": pos.entry_zscore,
                "exit_zscore": pos.exit_zscore,
                "total_pnl": pos.total_pnl,
                "return_pct": pos.return_pct,
                "exit_reason": pos.exit_reason,
                "is_open": pos.is_open
            }
            for pos in positions
        ]
    }


@app.delete("/pairs/{pair_id}")
async def deactivate_pair(pair_id: int, db=Depends(get_db)):
    """Deactivate a pair (soft delete)"""
    pair = db.query(PairTradingPair).filter(PairTradingPair.id == pair_id).first()

    if not pair:
        raise HTTPException(status_code=404, detail="Pair not found")

    pair.is_active = False
    db.commit()

    return {
        "success": True,
        "message": f"Pair {pair.stock_a}/{pair.stock_b} deactivated"
    }


@app.get("/pairs/scan/history")
async def get_scan_history(limit: int = 10, db=Depends(get_db)):
    """Get history of pair scans"""
    scans = db.query(PairScanResult).order_by(
        PairScanResult.created_at.desc()
    ).limit(limit).all()

    return {
        "success": True,
        "scans": [
            {
                "id": s.id,
                "universe": s.universe_name,
                "pairs_tested": s.pairs_tested,
                "pairs_found": s.pairs_found,
                "best_pair": s.best_pair,
                "best_quality_score": s.best_quality_score,
                "avg_quality_score": s.avg_quality_score,
                "scan_duration_seconds": s.scan_duration_seconds,
                "created_at": s.created_at.isoformat()
            }
            for s in scans
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
