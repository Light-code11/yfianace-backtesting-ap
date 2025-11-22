#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI server wrapper for yfinance
Exposes yfinance functionality via REST API endpoints
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import yfinance as yf
from datetime import datetime
import pandas as pd

app = FastAPI(
    title="YFinance API",
    description="REST API wrapper for Yahoo Finance data using yfinance",
    version="1.0.0"
)

# Helper function to convert pandas objects to JSON-serializable format
def convert_to_json(data):
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    elif isinstance(data, pd.Series):
        return data.to_dict()
    elif isinstance(data, dict):
        return {k: convert_to_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_json(item) for item in data]
    elif pd.isna(data):
        return None
    return data

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YFinance API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ticker_info": "/ticker/{symbol}",
            "history": "/ticker/{symbol}/history",
            "download": "/download",
            "search": "/search",
            "market": "/market/{country_code}",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/ticker/{symbol}")
async def get_ticker_info(symbol: str):
    """
    Get comprehensive information about a ticker
    
    Example: /ticker/AAPL
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return convert_to_json(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ticker/{symbol}/history")
async def get_ticker_history(
    symbol: str,
    period: str = Query("1mo", description="Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
):
    """
    Get historical price data for a ticker
    
    Example: /ticker/AAPL/history?period=1mo&interval=1d
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        # Convert DataFrame to JSON-friendly format
        result = hist.reset_index().to_dict(orient='records')
        
        # Convert datetime objects to strings
        for record in result:
            if 'Date' in record:
                record['Date'] = record['Date'].isoformat()
            elif 'Datetime' in record:
                record['Datetime'] = record['Datetime'].isoformat()
        
        return {"symbol": symbol, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ticker/{symbol}/actions")
async def get_ticker_actions(symbol: str):
    """
    Get dividend and split information for a ticker
    
    Example: /ticker/AAPL/actions
    """
    try:
        ticker = yf.Ticker(symbol)
        actions = ticker.actions
        
        result = actions.reset_index().to_dict(orient='records')
        for record in result:
            if 'Date' in record:
                record['Date'] = record['Date'].isoformat()
        
        return {"symbol": symbol, "actions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ticker/{symbol}/financials")
async def get_ticker_financials(
    symbol: str,
    statement: str = Query("income", description="Type: income, balance, cashflow")
):
    """
    Get financial statements for a ticker
    
    Example: /ticker/AAPL/financials?statement=income
    """
    try:
        ticker = yf.Ticker(symbol)
        
        if statement == "income":
            data = ticker.financials
        elif statement == "balance":
            data = ticker.balance_sheet
        elif statement == "cashflow":
            data = ticker.cashflow
        else:
            raise HTTPException(status_code=400, detail="Invalid statement type. Use: income, balance, or cashflow")
        
        if data is None or data.empty:
            return {"symbol": symbol, "statement": statement, "data": []}
        
        result = data.reset_index().to_dict(orient='records')
        
        return {"symbol": symbol, "statement": statement, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ticker/{symbol}/recommendations")
async def get_ticker_recommendations(symbol: str):
    """
    Get analyst recommendations for a ticker
    
    Example: /ticker/AAPL/recommendations
    """
    try:
        ticker = yf.Ticker(symbol)
        recommendations = ticker.recommendations
        
        if recommendations is None or recommendations.empty:
            return {"symbol": symbol, "recommendations": []}
        
        result = recommendations.reset_index().to_dict(orient='records')
        for record in result:
            if 'Date' in record:
                record['Date'] = record['Date'].isoformat()
        
        return {"symbol": symbol, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download")
async def download_multiple(
    tickers: List[str] = Query(..., description="List of ticker symbols"),
    period: str = Query("1mo", description="Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
):
    """
    Download data for multiple tickers
    
    Example: POST /download?tickers=AAPL&tickers=MSFT&tickers=GOOGL&period=1mo
    """
    try:
        ticker_string = " ".join(tickers)
        data = yf.download(ticker_string, period=period, interval=interval, progress=False)
        
        if data.empty:
            return {"tickers": tickers, "data": []}
        
        result = data.reset_index().to_dict(orient='records')
        for record in result:
            if 'Date' in record:
                record['Date'] = record['Date'].isoformat()
            elif 'Datetime' in record:
                record['Datetime'] = record['Datetime'].isoformat()
        
        return {"tickers": tickers, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_ticker(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, description="Maximum number of results")
):
    """
    Search for tickers by name or symbol
    
    Example: /search?query=apple&max_results=5
    """
    try:
        search = yf.Search(query, max_results=max_results)
        quotes = search.quotes
        
        return {
            "query": query,
            "results": convert_to_json(quotes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/{country_code}")
async def get_market_info(country_code: str):
    """
    Get market information for a specific country
    
    Example: /market/US
    
    Args:
        country_code: Country code (e.g., US, GB, JP)
    """
    try:
        market = yf.Market(country_code)
        
        return {
            "country_code": country_code,
            "info": convert_to_json(market.info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ticker/{symbol}/price")
async def get_current_price(symbol: str):
    """
    Get current price and basic info for a ticker (quick endpoint)
    
    Example: /ticker/AAPL/price
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key price information
        result = {
            "symbol": symbol,
            "currentPrice": info.get('currentPrice'),
            "previousClose": info.get('previousClose'),
            "open": info.get('open'),
            "dayHigh": info.get('dayHigh'),
            "dayLow": info.get('dayLow'),
            "volume": info.get('volume'),
            "marketCap": info.get('marketCap'),
            "currency": info.get('currency'),
            "timestamp": datetime.now().isoformat()
        }
        
        return convert_to_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

