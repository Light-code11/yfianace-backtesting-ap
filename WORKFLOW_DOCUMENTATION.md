# AI Trading Strategy Generator with Backtesting - n8n Workflow

## Overview

This n8n workflow creates an intelligent, self-improving trading strategy system that:
- **Runs weekly** on a schedule
- **Generates trading strategies** using AI (GPT-4)
- **Backtests strategies** against historical data
- **Learns from past performance** to improve over time
- **Provides actionable recommendations** based on results
- **Tracks performance** in Google Sheets for continuous improvement

## Workflow Architecture

```
Weekly Trigger (Monday 9 AM)
    ↓
Set Analysis Parameters (Define tickers, period, interval)
    ↓ ↓
    |  Load Previous Strategy Performance (Google Sheets)
    ↓ /
Fetch Historical Stock Data (YFinance API)
    ↓
AI Strategy Generator (GPT-4 generates 3-5 strategies)
    ↓
Backtest Strategies (Code node simulates trading)
    ↓
Quality Check (IF node - Sharpe ratio > 0.5?)
    ↓ ↓
    |  └── Generate Recommendations (Poor Strategies)
    └── Generate Recommendations (Good Strategies)
         ↓
Save Performance Data (Google Sheets)
    ↓
Format Final Report (Weekly summary)
```

## Key Features

### 1. **AI-Powered Strategy Generation**
- Uses GPT-4 to analyze market data and generate trading strategies
- Considers historical performance data to learn and improve
- Generates 3-5 distinct strategies per week with:
  - Entry/exit conditions
  - Risk management rules
  - Position sizing
  - Stop loss levels
  - Rationale and market analysis

### 2. **Automated Backtesting**
- Simulates strategy execution on historical data
- Calculates key metrics:
  - Total return %
  - Win rate
  - Maximum drawdown
  - Sharpe ratio
  - Number of trades

### 3. **Continuous Learning**
- Stores all strategy performance in Google Sheets
- AI reviews past performance before generating new strategies
- Adapts and improves recommendations based on what worked

### 4. **Quality Assurance**
- Validates strategies meet minimum thresholds
- Routes to different recommendation paths based on quality
- Provides risk warnings for low-quality strategies

### 5. **Weekly Automation**
- Runs every Monday at 9:00 AM
- Completely hands-free after setup
- Generates comprehensive reports automatically

## Node Breakdown

### 1. Weekly Trigger
**Type:** `n8n-nodes-base.scheduleTrigger`
- **Schedule:** Every Monday at 9:00 AM
- **Purpose:** Initiates the weekly strategy generation cycle

### 2. Set Analysis Parameters
**Type:** `n8n-nodes-base.set`
- **Configures:**
  - `tickers`: List of stocks to analyze (AAPL, MSFT, GOOGL, etc.)
  - `period`: Historical data period (6 months)
  - `interval`: Data granularity (daily)

### 3. Fetch Historical Stock Data
**Type:** `n8n-nodes-base.httpRequest`
- **Endpoint:** `POST http://your-yfinance-server:8000/download`
- **Purpose:** Retrieves 6 months of daily price data for all tickers
- **Data includes:** Open, High, Low, Close, Volume

### 4. Load Previous Strategy Performance
**Type:** `n8n-nodes-base.googleSheets`
- **Operation:** Read
- **Purpose:** Loads historical strategy performance for AI learning
- **Enables:** The AI to learn from past successes and failures

### 5. AI Strategy Generator
**Type:** `@n8n/n8n-nodes-langchain.openAi`
- **Model:** GPT-4
- **Input:** Historical data + past performance
- **Output:** 3-5 trading strategies in JSON format
- **Temperature:** 0.7 (balanced creativity)
- **Max Tokens:** 2000

**Strategy Components:**
- Strategy name
- Target tickers
- Entry conditions
- Exit conditions
- Stop loss percentage
- Position sizing
- Holding period
- Rationale

### 6. Backtest Strategies
**Type:** `n8n-nodes-base.code` (JavaScript)
- **Purpose:** Simulates each strategy on historical data
- **Methodology:** Moving average crossover with risk management
- **Calculates:**
  - Total trades executed
  - Winning vs losing trades
  - Total return percentage
  - Maximum drawdown
  - Sharpe ratio
  - Win rate

**Backtesting Logic:**
- Uses 20-period and 5-period simple moving averages
- Enters positions when short MA crosses above long MA
- Exits on stop loss, take profit, or MA cross-down
- Tracks capital, drawdown, and performance metrics
- Applies position sizing rules from strategy

### 7. Quality Check
**Type:** `n8n-nodes-base.if`
- **Condition 1:** Sharpe ratio > 0.5
- **Condition 2:** Total return > 0%
- **Routes:**
  - **TRUE branch:** Good strategies → Detailed recommendations
  - **FALSE branch:** Poor strategies → Risk warnings

### 8. Generate Recommendations (Good Strategies)
**Type:** `@n8n/n8n-nodes-langchain.openAi`
- **Model:** GPT-4
- **Temperature:** 0.3 (conservative, precise)
- **Provides:**
  - Executive summary
  - Specific action items
  - Risk assessment
  - Expected returns
  - Alternative options
  - Market conditions to watch

### 9. Generate Recommendations (Poor Strategies)
**Type:** `@n8n/n8n-nodes-langchain.openAi`
- **Model:** GPT-4
- **Temperature:** 0.3 (conservative)
- **Provides:**
  - Warning about market conditions
  - Reasons for underperformance
  - Recommendation to stay defensive
  - Indicators to monitor
  - Timeline for next review

### 10. Save Performance Data
**Type:** `n8n-nodes-base.googleSheets`
- **Operation:** Append
- **Purpose:** Records all strategy performance for future learning
- **Columns:**
  - Timestamp
  - Strategy name
  - Tickers
  - Total trades
  - Win rate
  - Total return
  - Max drawdown
  - Sharpe ratio
  - Market analysis
  - Recommendation

### 11. Format Final Report
**Type:** `n8n-nodes-base.set`
- **Purpose:** Creates structured weekly report
- **Includes:**
  - Report date
  - Analysis period
  - Tickers analyzed
  - Number of strategies tested
  - Best strategy details
  - AI recommendations
  - Next review date

## Setup Instructions

### Prerequisites

1. **n8n Instance**
   - Self-hosted or cloud version
   - Version 1.0+ recommended

2. **YFinance API Server**
   - Running on `http://your-yfinance-server:8000`
   - See `DEPLOYMENT.md` in this repository for setup

3. **OpenAI API Key**
   - GPT-4 access required
   - Add credentials in n8n: Credentials → New → OpenAI API

4. **Google Sheets**
   - Create a new spreadsheet: "Strategy Performance Tracker"
   - Share with n8n service account
   - Add credentials in n8n: Credentials → New → Google Sheets API

### Installation Steps

1. **Import Workflow**
   ```bash
   # In n8n UI:
   # Workflows → Import from File → Select ai-trading-strategy-workflow.json
   ```

2. **Configure YFinance API Server URL**
   - Open "Fetch Historical Stock Data" node
   - Update URL to your actual server address
   - Example: `http://localhost:8000/download` or `http://your-domain.com:8000/download`

3. **Set Up OpenAI Credentials**
   - Click on any OpenAI node
   - Select "Create New" credential
   - Enter your OpenAI API key
   - Save

4. **Set Up Google Sheets**
   - Create new spreadsheet in Google Sheets
   - Name it: "Strategy Performance Tracker"
   - Add headers: `timestamp, strategy_name, tickers, total_trades, win_rate, total_return, max_drawdown, sharpe_ratio, market_analysis, recommendation`
   - Click on "Load Previous Strategy Performance" node
   - Select your spreadsheet
   - Repeat for "Save Performance Data" node

5. **Customize Stock Tickers** (Optional)
   - Open "Set Analysis Parameters" node
   - Modify the `tickers` array to your preferred stocks
   - Default: `["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "META", "AMZN"]`

6. **Adjust Schedule** (Optional)
   - Open "Weekly Trigger" node
   - Modify day/time as needed
   - Default: Monday at 9:00 AM

7. **Activate Workflow**
   - Toggle "Active" switch in top-right corner
   - Workflow will now run automatically on schedule

## Usage

### Automatic Execution
Once activated, the workflow runs automatically every week. No manual intervention required.

### Manual Execution
To test or run immediately:
1. Click "Execute Workflow" button
2. Wait for completion (2-5 minutes typical)
3. Review results in the execution panel

### Viewing Results

**In n8n:**
- Click on "Format Final Report" node
- View JSON output with complete analysis

**In Google Sheets:**
- Open "Strategy Performance Tracker" spreadsheet
- View historical performance data
- Analyze trends over time

## Customization Options

### 1. Change Analysis Period
```javascript
// In "Set Analysis Parameters" node
"period": "3mo"  // Options: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
```

### 2. Modify Quality Thresholds
```javascript
// In "Quality Check" node
"sharpe_ratio": 0.5  // Increase for stricter quality control
"total_return_pct": 0  // Set minimum acceptable return
```

### 3. Add More Stocks
```javascript
// In "Set Analysis Parameters" node
"tickers": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "META", "AMZN", "NFLX", "DIS"]
```

### 4. Adjust AI Creativity
```javascript
// In "AI Strategy Generator" node
"temperature": 0.7  // 0.0 = conservative, 1.0 = creative
```

### 5. Enhance Backtesting Logic
The backtesting code node can be modified to:
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement different entry/exit strategies
- Include commission and slippage costs
- Add portfolio rebalancing logic
- Implement more sophisticated risk management

## Performance Metrics Explained

### Sharpe Ratio
- Measures risk-adjusted returns
- **> 1.0:** Excellent
- **0.5 - 1.0:** Good
- **< 0.5:** Poor

### Win Rate
- Percentage of profitable trades
- **> 60%:** Strong strategy
- **40-60%:** Average
- **< 40%:** Weak strategy

### Maximum Drawdown
- Largest peak-to-trough decline
- **< 10%:** Conservative
- **10-20%:** Moderate
- **> 20%:** Aggressive

### Total Return
- Overall percentage gain/loss
- Calculated as: `(Final Capital - Initial Capital) / Initial Capital × 100`

## AI Learning Mechanism

### How It Improves Over Time

1. **Week 1:** AI generates initial strategies without historical context
2. **Week 2+:** AI reviews past performance before generating new strategies
3. **Adaptation:** AI identifies successful patterns and avoids failed ones
4. **Refinement:** Strategy parameters improve based on backtesting results

### Learning Factors
- **Strategy performance:** High Sharpe ratio strategies influence future recommendations
- **Market conditions:** AI adapts to changing market environments
- **Risk management:** AI learns optimal stop loss and position sizing
- **Holding periods:** AI optimizes trade duration based on historical results

## Troubleshooting

### Common Issues

**1. OpenAI API Errors**
- **Symptom:** "401 Unauthorized"
- **Solution:** Check API key is valid and has GPT-4 access

**2. YFinance API Connection Failed**
- **Symptom:** "ECONNREFUSED" or timeout
- **Solution:** Verify API server is running and URL is correct

**3. Google Sheets Permission Denied**
- **Symptom:** "403 Forbidden"
- **Solution:** Share spreadsheet with n8n service account email

**4. Backtest Returns Empty Results**
- **Symptom:** All strategies show 0 trades
- **Solution:** Check historical data format and date ranges

**5. Workflow Doesn't Trigger**
- **Symptom:** No automatic execution
- **Solution:** Ensure workflow is Active (toggle in top-right)

## Best Practices

### 1. Start with Paper Trading
- Test strategies before investing real money
- Monitor performance for several weeks
- Validate AI recommendations match your risk tolerance

### 2. Diversify Strategies
- Don't rely on a single strategy
- Use multiple approaches (momentum, mean reversion, etc.)
- Balance aggressive and conservative strategies

### 3. Regular Monitoring
- Review weekly reports
- Check Google Sheets trends monthly
- Adjust thresholds based on market conditions

### 4. Risk Management
- Never invest more than you can afford to lose
- Use appropriate position sizing (typically 5-10% per position)
- Always use stop losses

### 5. Continuous Improvement
- Review AI learning notes
- Analyze failed strategies
- Refine backtesting logic based on real results

## Security Considerations

1. **API Keys:** Store OpenAI API keys securely in n8n credentials
2. **Data Privacy:** Historical performance data contains trading insights
3. **Access Control:** Restrict access to n8n instance and Google Sheets
4. **API Server:** Secure YFinance API server with authentication if exposed

## Cost Estimate

### OpenAI API Costs (per week)
- Strategy generation: ~$0.05-0.10 (2000 tokens)
- Recommendations: ~$0.03-0.06 (1500 tokens)
- **Total per week:** ~$0.08-0.16
- **Annual cost:** ~$4-8

### Infrastructure
- n8n: Free (self-hosted) or ~$20/month (cloud)
- YFinance API: Free
- Google Sheets: Free

## Limitations & Disclaimers

⚠️ **IMPORTANT DISCLAIMERS:**

1. **Not Financial Advice:** This workflow is for educational/informational purposes only
2. **Past Performance:** Does not guarantee future results
3. **Market Risk:** All trading involves risk of loss
4. **Backtesting Limitations:** Historical simulations may not reflect real-world execution
5. **AI Limitations:** AI-generated strategies should be reviewed by humans
6. **No Guarantees:** No warranty of profitability or accuracy

### Known Limitations

- Backtesting uses simplified moving average logic
- Does not account for:
  - Transaction costs/commissions
  - Slippage
  - Overnight gaps
  - Market liquidity
  - Real-world execution challenges
- AI may generate similar strategies week-over-week
- Requires manual review and approval before live trading

## Roadmap & Future Enhancements

### Planned Features
- [ ] Multiple backtesting algorithms (RSI, MACD, Bollinger Bands)
- [ ] Real-time paper trading integration
- [ ] Portfolio optimization
- [ ] Sentiment analysis from news/social media
- [ ] Multi-timeframe analysis
- [ ] Email/Slack notifications
- [ ] Interactive dashboard
- [ ] Advanced risk metrics (VaR, CVaR)
- [ ] Machine learning model training
- [ ] Integration with broker APIs

## Support & Resources

### Documentation
- n8n Documentation: https://docs.n8n.io
- YFinance Documentation: See `README.md` in this repository
- OpenAI API: https://platform.openai.com/docs

### Community
- n8n Community: https://community.n8n.io
- Issues: Create issue in GitHub repository

## License

See `LICENSE.txt` in repository root.

## Version History

### v1.0.0 (Current)
- Initial release
- Weekly scheduling
- AI strategy generation with GPT-4
- Backtesting with performance metrics
- Google Sheets integration
- Quality checks and recommendations
- Continuous learning mechanism

---

**Created:** November 2025  
**Last Updated:** November 2025  
**Workflow Version:** 1.0.0  
**n8n Compatible:** 1.0+

