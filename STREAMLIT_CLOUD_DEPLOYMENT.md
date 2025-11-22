# Deploying to Streamlit Cloud

This guide will help you deploy your AI Trading Platform to Streamlit Cloud.

## ⚠️ Important Prerequisites

**The Streamlit app requires a backend API server.** You have two options:

### Option 1: Deploy Backend + Frontend (Full Platform)

Deploy both the FastAPI backend and Streamlit frontend.

### Option 2: Frontend Only (Limited Features)

Deploy only the Streamlit app, but it will only show static content without AI strategy generation.

---

## 🚀 Option 1: Deploy Full Platform (Recommended)

### Step 1: Deploy FastAPI Backend

You need to deploy your API server to a publicly accessible URL first.

#### **A. Deploy to Railway.app (Easiest, Free Tier)**

1. **Sign up at** https://railway.app
2. **Create new project** → Deploy from GitHub repo
3. **Select your repository**
4. **Add environment variables:**
   ```
   OPENAI_API_KEY=sk-proj-your-key-here
   DATABASE_URL=sqlite:///./trading_platform.db
   PORT=8000
   ```
5. **Create Procfile** in your repo root:
   ```
   web: uvicorn trading_platform_api:app --host 0.0.0.0 --port $PORT
   ```
6. **Create railway.json** in your repo root:
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "uvicorn trading_platform_api:app --host 0.0.0.0 --port $PORT",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```
7. **Deploy** - Railway will give you a URL like `https://your-app.railway.app`

#### **B. Deploy to Render.com (Free Tier)**

1. **Sign up at** https://render.com
2. **Create New** → Web Service
3. **Connect your GitHub repo**
4. **Configure:**
   - **Build Command:** `pip install -r requirements-trading-platform.txt`
   - **Start Command:** `uvicorn trading_platform_api:app --host 0.0.0.0 --port $PORT`
5. **Add Environment Variables:**
   ```
   OPENAI_API_KEY=sk-proj-your-key-here
   DATABASE_URL=sqlite:///./trading_platform.db
   ```
6. **Deploy** - You'll get a URL like `https://your-app.onrender.com`

---

### Step 2: Deploy Streamlit Frontend

1. **Go to** https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Select:**
   - Repository: Your GitHub repo
   - Branch: main
   - Main file path: `streamlit_app.py`
5. **Click "Advanced settings"**
6. **Add Secrets:** (Click on "Secrets" in advanced settings)
   ```toml
   API_BASE_URL = "https://your-api.railway.app"
   ```
   Replace with your actual API URL from Step 1
7. **Deploy!**

---

## 📝 Step-by-Step Checklist

### Before Deployment:

- [ ] Push your code to GitHub
- [ ] Create `.gitignore` to exclude:
  ```
  .env
  *.db
  __pycache__/
  .streamlit/secrets.toml
  api_server.log
  ```
- [ ] Ensure `requirements.txt` includes all dependencies
- [ ] Test locally first

### Backend Deployment:

- [ ] Deploy FastAPI to Railway/Render
- [ ] Set OPENAI_API_KEY environment variable
- [ ] Test API endpoint: `https://your-api.railway.app/health`
- [ ] Copy the API URL

### Frontend Deployment:

- [ ] Deploy Streamlit app
- [ ] Set API_BASE_URL in Streamlit secrets
- [ ] Test the app
- [ ] Verify it connects to the backend

---

## 🔧 Troubleshooting

### "Module not found" Error

**Solution:** Make sure `requirements.txt` is in the repo root with all dependencies:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
requests>=2.31.0
```

### "Connection refused" or "API Error"

**Solution:**
1. Check that your backend API is running
2. Verify the API_BASE_URL in Streamlit secrets
3. Test the API directly: `curl https://your-api.railway.app/health`

### Backend crashes or times out

**Solution:**
1. Check Railway/Render logs
2. Verify OPENAI_API_KEY is set correctly
3. Ensure database is writable (use PostgreSQL for production)

### OpenAI API errors

**Solution:**
1. Verify API key is valid
2. Check you have credits in your OpenAI account
3. Confirm you're using `gpt-4o-mini` (cheaper than GPT-4)

---

## 💰 Cost Considerations

### Free Tiers Available:

- **Railway:** 500 hours/month free, then ~$5/month
- **Render:** Free tier available (may sleep after 15 min inactivity)
- **Streamlit Cloud:** Free for public apps
- **OpenAI API:** Pay-per-use (~$0.01-0.02 per strategy with gpt-4o-mini)

### Recommended for Production:

- **Backend:** Railway ($5-10/month) or Render ($7/month)
- **Database:** PostgreSQL on Railway/Render (included)
- **Frontend:** Streamlit Cloud (free for public, $20/month for private)
- **OpenAI:** Budget $10-20/month depending on usage

**Total:** ~$15-30/month for production deployment

---

## 🔒 Security Best Practices

1. **Never commit secrets to GitHub**
   - Use `.gitignore` for `.env` files
   - Use platform-specific secrets management

2. **Use environment variables**
   - Railway/Render: Set in dashboard
   - Streamlit: Use secrets.toml (not committed)

3. **Secure your API** (optional but recommended):
   ```python
   # Add API key authentication to trading_platform_api.py
   from fastapi import Header, HTTPException

   def verify_api_key(x_api_key: str = Header(...)):
       if x_api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=401, detail="Invalid API Key")
   ```

4. **Use HTTPS only** in production

---

## 📚 Additional Resources

- **Streamlit Cloud Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Railway Docs:** https://docs.railway.app
- **Render Docs:** https://render.com/docs
- **FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/

---

## ✅ Quick Start Commands

### For Railway Deployment:

```bash
# 1. Create Procfile
echo "web: uvicorn trading_platform_api:app --host 0.0.0.0 --port \$PORT" > Procfile

# 2. Update .gitignore
cat >> .gitignore << EOF
.env
*.db
__pycache__/
.streamlit/secrets.toml
api_server.log
EOF

# 3. Commit and push
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

Then deploy on Railway dashboard.

### For Streamlit Cloud:

1. Make sure code is on GitHub
2. Go to https://share.streamlit.io
3. Click "New app" and select your repo
4. Add API_BASE_URL to secrets
5. Deploy!

---

## 🎉 Success!

Once deployed, your AI Trading Platform will be accessible at:
- **Frontend:** `https://your-app.streamlit.app`
- **API:** `https://your-api.railway.app/docs`

Share your trading platform with others! 📈🚀

---

**Need help?** Check the troubleshooting section or review platform-specific documentation.
