@echo off
REM Options Trading Scheduled Tasks
REM Runs 5 minutes after equity bot (01:45 equity → 01:50 options)
REM Management pass at 06:05 (5 min after equity 06:00 close pass)

set PYTHON=C:\Users\User\Projects\yfinance-trading\.venv312\Scripts\python.exe
set SCRIPT_DIR=C:\Users\User\Projects\yfinance-trading

echo Creating YFinance-Options-Daily (01:50 AEDT)...
schtasks /create /tn "YFinance-Options-Daily" ^
  /tr "\"%PYTHON%\" \"%SCRIPT_DIR%\run_options_trading.py\"" ^
  /sc daily /st 01:50 /f

echo Creating YFinance-Options-Manage (06:05 AEDT)...
schtasks /create /tn "YFinance-Options-Manage" ^
  /tr "\"%PYTHON%\" \"%SCRIPT_DIR%\run_options_trading.py\" --manage" ^
  /sc daily /st 06:05 /f

echo.
echo Done. Tasks created:
schtasks /query /tn "YFinance-Options-Daily"
schtasks /query /tn "YFinance-Options-Manage"
