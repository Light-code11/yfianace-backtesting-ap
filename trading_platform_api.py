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
    PortfolioAllocation, AILearning, PerformanceLog
)
from ai_strategy_generator import AIStrategyGenerator
from backtesting_engine import BacktestingEngine
from paper_trading import PaperTradingSimulator
from portfolio_optimizer import PortfolioOptimizer
from autonomous_learning import AutonomousLearningAgent
import threading

# Helper function to convert numpy types to Python types for PostgreSQL
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
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


class PaperTradeRequest(BaseModel):
    strategy_id: int
    auto_execute: bool = True


class PortfolioOptimizationRequest(BaseModel):
    strategy_ids: List[int]
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
                "is_active": s.is_active
            }
            for s in strategies
        ]
    }


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
        period = "1y"  # Use 1 year for backtesting
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

        # Save backtest results
        backtest_result = BacktestResult(
            strategy_id=request.strategy_id,
            strategy_name=results['strategy_name'],
            start_date=datetime.now() - timedelta(days=365),
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

        return {
            "success": True,
            "backtest_id": backtest_result.id,
            "strategy_name": results['strategy_name'],
            "metrics": results['metrics'],
            "total_trades": len(results['trades']),
            "equity_curve_points": len(results['equity_curve'])
        }

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
    """Optimize portfolio allocation across strategies"""
    try:
        # Get strategies
        strategies = []
        backtest_results = []

        for strategy_id in request.strategy_ids:
            strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
            if not strategy:
                continue

            # Get latest backtest result
            backtest = db.query(BacktestResult).filter(
                BacktestResult.strategy_id == strategy_id
            ).order_by(BacktestResult.created_at.desc()).first()

            if not backtest:
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

        if not strategies:
            raise HTTPException(status_code=400, detail="No valid strategies with backtest results found")

        # Optimize
        optimizer = PortfolioOptimizer()
        optimization_result = optimizer.optimize_allocations(
            strategies=strategies,
            backtest_results=backtest_results,
            total_capital=request.total_capital,
            method=request.method,
            constraints=request.constraints
        )

        # Save optimization
        portfolio = PortfolioAllocation(
            name=f"Portfolio {request.method} {datetime.now().strftime('%Y-%m-%d')}",
            total_capital=request.total_capital,
            strategies=request.strategy_ids,
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

        return {
            "success": True,
            "portfolio_id": portfolio.id,
            **optimization_result
        }

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
