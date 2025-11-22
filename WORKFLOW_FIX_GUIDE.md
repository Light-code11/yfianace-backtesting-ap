# ✅ Fixed: AI Trading Strategy Workflow

## 🎯 The Problem

The original workflow showed **"?" marks** in n8n because the JSON structure didn't match n8n's expected format. I used the **n8n Workflows MCP** to analyze real working workflows and found the correct structure.

## 🔧 The Solution

Created: **`ai-trading-strategy-workflow-corrected.json`**

### Key Fixes Applied:

1. **Correct Node Structure**
   - Moved `typeVersion` to node level (not inside `parameters`)
   - Flattened `parameters` structure
   - Removed nested configurations

2. **HTTP Request Node Fixed**
   ```json
   // ❌ OLD (caused "?" mark)
   {
     "parameters": {
       "url": "=complex nested expression",
       "typeVersion": 4.3  // WRONG LOCATION
     }
   }
   
   // ✅ NEW (works!)
   {
     "parameters": {
       "method": "POST",
       "url": "={{ $json.api_url }}",  // Simple expression
       "authentication": "none",
       "options": {}
     },
     "type": "n8n-nodes-base.httpRequest",
     "typeVersion": 4.3  // CORRECT LOCATION
   }
   ```

3. **OpenAI Node Structure**
   ```json
   // ✅ Proper structure
   {
     "type": "@n8n/n8n-nodes-langchain.openAi",
     "typeVersion": 2,
     "parameters": {
       "resource": "text",
       "operation": "message",
       "modelId": {
         "__rl": true,
         "mode": "list",
         "value": "gpt-4o"
       },
       "messages": { ... }
     }
   }
   ```

4. **Simplified URL Handling**
   - Created full URL in "Set Analysis Parameters" node
   - HTTP Request just references it: `{{ $json.api_url }}`
   - No more nested expression errors!

5. **Your ngrok URL**
   - ✅ Already configured: `https://darryl-contractile-back.ngrok-free.dev/download`
   - ✅ All 8 tickers included in URL
   - ✅ Period: 6mo, Interval: 1d

## 📥 Import Instructions

### Step 1: Import the Corrected Workflow

1. Open n8n
2. Go to **Workflows** → **Import from File**
3. Select: `ai-trading-strategy-workflow-corrected.json`
4. Click **Import**

### Step 2: Configure Credentials

The workflow needs 2 types of credentials:

#### A. OpenAI Credentials (3 nodes need this)

1. Click on **"AI Strategy Generator"** node
2. Click **"Credential to connect with"** dropdown
3. Select **"Create New Credential"**
4. Choose **"OpenAI API"**
5. Enter your API key from https://platform.openai.com/api-keys
6. Name it: `OpenAI Account`
7. **Save**

8. Click on **"Generate Recommendations (Good)"** node
9. Select the **same credential** you just created
10. Repeat for **"Generate Recommendations (Poor)"** node

#### B. Google Sheets Credentials (2 nodes need this)

1. Create a Google Sheet: **"Strategy Performance Tracker"**
2. Add these column headers in Row 1:
   ```
   timestamp | strategy_name | tickers | total_trades | win_rate | total_return | max_drawdown | sharpe_ratio | market_analysis | recommendation
   ```

3. In n8n, click **"Load Previous Strategy Performance"** node
4. Click **"Credential to connect with"**
5. Create **"Google Sheets OAuth2 API"** credential
6. Follow the OAuth flow
7. Select your spreadsheet
8. Repeat for **"Save Performance Data"** node

### Step 3: Verify Configuration

Check these nodes:

✅ **Set Analysis Parameters**
- Should show your ngrok URL
- Should list 8 stock tickers

✅ **Fetch Historical Stock Data**
- Method: POST
- URL should reference: `{{ $json.api_url }}`

✅ **All nodes should show icons** (no "?" marks)

### Step 4: Test the Workflow

```bash
# First, test your YFinance API directly
curl -X POST 'https://darryl-contractile-back.ngrok-free.dev/download?tickers=AAPL&tickers=MSFT&period=1mo&interval=1d'
```

If that works, in n8n:
1. Click **"Execute Workflow"** button
2. Watch it progress through each node
3. Check **"Format Final Report"** for results

### Step 5: Activate for Weekly Runs

1. Toggle **"Active"** switch (top-right)
2. Workflow will run every **Monday at 9:00 AM** automatically

## 🔍 Why The Original Failed

| Issue | Original | Fixed |
|-------|----------|-------|
| **Node Structure** | `typeVersion` inside `parameters` | `typeVersion` at node level |
| **URL Expression** | Nested `map().join()` in URL | Pre-built URL in Set node |
| **Parameters** | Over-nested structure | Flat structure |
| **Resource Locator** | Missing `__rl` format | Proper `{ "__rl": true, "mode": "list", "value": "..." }` |

## 📊 Workflow Flow

```
Weekly Trigger (Mon 9 AM)
    ↓
Set Analysis Parameters
  → Pre-builds full API URL
  → Sets tickers, period, interval
    ↓ ↓
    |  Load Previous Performance (Google Sheets)
    ↓ /
Fetch Stock Data (YFinance via ngrok)
    ↓
AI Strategy Generator (GPT-4)
  → Analyzes data
  → Generates 3-5 strategies
  → Learns from past performance
    ↓
Backtest Strategies (JavaScript Code)
  → Simulates trading
  → Calculates metrics (Sharpe, return, drawdown)
    ↓
Quality Check (IF node)
  → Sharpe ratio > 0.5?
    ↓ ↓
    |  Generate Recommendations (Poor) ← Risk warning
    ↓
Generate Recommendations (Good) ← Actionable advice
    ↓
Save Performance Data (Google Sheets)
  → Stores for AI learning
    ↓
Format Final Report
  → Weekly summary with recommendations
```

## 🎯 What Each Node Does

1. **Weekly Trigger**
   - Runs every Monday at 9:00 AM
   - Starts the entire workflow

2. **Set Analysis Parameters**
   - Defines stock tickers to analyze
   - Sets analysis period (6 months)
   - Pre-builds the complete API URL

3. **Fetch Historical Stock Data**
   - Calls your ngrok YFinance API
   - Gets 6 months of daily data for 8 stocks
   - Returns OHLCV (Open, High, Low, Close, Volume)

4. **Load Previous Strategy Performance**
   - Reads past results from Google Sheets
   - Provides data for AI to learn from

5. **AI Strategy Generator** (GPT-4)
   - Analyzes historical data
   - Reviews past performance
   - Generates 3-5 trading strategies
   - Includes entry/exit rules, risk management

6. **Backtest Strategies** (Code node)
   - Simulates each strategy on historical data
   - Calculates: Sharpe ratio, win rate, return, drawdown
   - Uses moving average crossover logic

7. **Quality Check** (IF node)
   - Checks if strategies meet thresholds
   - Routes to good or poor recommendations

8. **Generate Recommendations (Good/Poor)** (GPT-4)
   - Good: Detailed action items, risk assessment
   - Poor: Risk warnings, defensive recommendations

9. **Save Performance Data** (Google Sheets)
   - Records all metrics
   - Enables AI learning for next week

10. **Format Final Report**
    - Creates structured weekly summary
    - Includes best strategy and recommendations

## 🛠️ Customization Options

### Change Stock Tickers

Edit **"Set Analysis Parameters"** node:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
  "api_url": "https://darryl-contractile-back.ngrok-free.dev/download?tickers=AAPL&tickers=MSFT&tickers=GOOGL&tickers=NVDA&tickers=META&period=6mo&interval=1d"
}
```

### Change Schedule

Edit **"Weekly Trigger"** node:

```json
{
  "field": "weeks",
  "triggerAtDay": 5,     // Friday
  "triggerAtHour": 16,   // 4 PM
  "triggerAtMinute": 0
}
```

### Change Quality Threshold

Edit **"Quality Check"** node:

```json
{
  "leftValue": "={{ $json.best_strategy.sharpe_ratio }}",
  "rightValue": 1.0  // Stricter: require Sharpe > 1.0
}
```

### Change Analysis Period

Edit **"Set Analysis Parameters"** node:

```json
{
  "period": "3mo",  // Options: 1mo, 3mo, 6mo, 1y, 2y
  "interval": "1wk" // Options: 1d, 1wk, 1mo
}
```

## 🐛 Troubleshooting

### Still Seeing "?" Marks?

1. **Delete** the old workflow completely
2. **Import** the new `ai-trading-strategy-workflow-corrected.json`
3. Don't try to edit the broken one

### Nodes Won't Configure?

- Make sure you're using n8n version 1.0+
- Try restarting n8n
- Clear browser cache

### OpenAI Errors?

- Verify API key has GPT-4 access
- Check you have credits remaining
- Test API key at: https://platform.openai.com/playground

### YFinance API Not Responding?

```bash
# Test directly:
curl -X POST 'https://darryl-contractile-back.ngrok-free.dev/download?tickers=AAPL&period=1mo&interval=1d'

# Check if ngrok is running:
# ngrok should show your tunnel URL is active
```

### Google Sheets Access Denied?

- Share spreadsheet with your n8n service account email
- Give "Editor" permissions
- Email format: `xxx@xxx.iam.gserviceaccount.com`

## 💡 Pro Tips

1. **Start with 1-2 tickers** to test faster
2. **Use "Execute Node"** to test individual nodes
3. **Check execution history** for debugging
4. **Pin data** in nodes to test downstream without re-running
5. **Add sticky notes** to document your customizations

## 📚 Additional Resources

- **Full Documentation**: `WORKFLOW_DOCUMENTATION.md`
- **Quick Start**: `QUICK_START_GUIDE.md`
- **YFinance API**: `QUICKSTART.md`
- **Deployment**: `DEPLOYMENT.md`

## ✅ Success Checklist

- [ ] Imported corrected workflow file
- [ ] All nodes show proper icons (no "?")
- [ ] OpenAI credentials configured (3 nodes)
- [ ] Google Sheets created and connected (2 nodes)
- [ ] YFinance API tested and responding
- [ ] Test execution completed successfully
- [ ] Workflow activated for weekly runs
- [ ] Spreadsheet receiving data

## 🎉 You're All Set!

The workflow is now properly configured and should work without issues. The "?" marks were due to incorrect JSON structure, which is now fixed using the proper n8n format.

**Questions?** Check the troubleshooting section or review the full documentation.

---

**Fixed:** November 2025  
**Based on:** Real n8n workflow templates  
**Validated:** ✅ Using n8n MCP validation tools


