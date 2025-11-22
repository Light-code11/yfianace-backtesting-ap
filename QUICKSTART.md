# Quick Start Guide (macOS)

## ⚡ Fastest Way to Start

```bash
cd /Users/alanphilip/yfinance
./start_server.sh
```

That's it! The script handles everything automatically.

---

## 📝 Manual Steps

### 1. Navigate to the directory
```bash
cd /Users/alanphilip/yfinance
```

### 2. Install dependencies (first time only)
```bash
# Option 1: Install minimal dependencies (recommended)
pip3 install fastapi 'uvicorn[standard]' python-multipart

# Option 2: Install from file (if you want all extras)
pip3 install -r requirements-api.txt
```

### 3. Run the server
```bash
python3 api_server.py
```

### 4. Open in browser
- API Docs: http://localhost:8000/docs
- Test endpoint: http://localhost:8000/health

---

## 🧪 Test It

```bash
# In another terminal
curl http://localhost:8000/ticker/AAPL/price
```

---

## 🌐 Expose with Ngrok

```bash
# Terminal 1: Run the API
cd /Users/alanphilip/yfinance
python3 api_server.py

# Terminal 2: Expose it
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok.io`) and use it in n8n!

---

## 🔗 Use in n8n

1. Add **HTTP Request** node
2. Set URL: `http://localhost:8000/ticker/AAPL/price` (or your ngrok URL)
3. Method: GET
4. Execute!

---

## 🛑 Stop the Server

Press `Ctrl + C` in the terminal

---

## ❓ Common Issues

**"command not found: python"**
→ Use `python3` instead of `python` on macOS

**"Module not found"**
→ Run: `pip3 install -r requirements-api.txt`

**"Port already in use"**
→ The script will ask if you want to kill it

**"No such file"**
→ Make sure you're in the right directory: `cd /Users/alanphilip/yfinance`

---

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

