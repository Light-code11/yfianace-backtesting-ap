# 🚀 Quick Deployment Guide

## TL;DR - Deploy in 10 Minutes

### Step 1: Fix Your Current Streamlit Cloud Error

Your error (`import plotly.graph_objects as go`) is because Streamlit Cloud needs the `requirements.txt` file.

✅ **I've already created `requirements.txt`** - just commit and push it to GitHub!

### Step 2: Deploy Backend API (Required!)

Your Streamlit app **needs** the backend API to work. Deploy it first:

#### **Option A: Railway.app (Easiest)**

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Add these environment variables in Railway dashboard:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
6. Railway automatically detects `Procfile` and deploys!
7. Copy your Railway URL (e.g., `https://your-app.railway.app`)

#### **Option B: Render.com**

1. Go to https://render.com
2. Sign up with GitHub
3. New → Web Service → Connect your repo
4. Build Command: `pip install -r requirements-trading-platform.txt`
5. Start Command: `uvicorn trading_platform_api:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `OPENAI_API_KEY=your-key`
7. Deploy and copy your URL

### Step 3: Configure Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. New app → Select your repo → `streamlit_app.py`
4. **Advanced Settings** → **Secrets** → Add:
   ```toml
   API_BASE_URL = "https://your-api.railway.app"
   ```
   (Use your actual Railway/Render URL from Step 2)
5. Click Deploy!

### Step 4: Test

Visit your Streamlit app URL and try generating strategies!

---

## Files Created for Deployment

✅ `requirements.txt` - Dependencies for Streamlit Cloud
✅ `Procfile` - Railway/Render configuration
✅ `railway.json` - Railway-specific config
✅ `.gitignore` - Updated to exclude secrets
✅ `.streamlit/secrets.toml.example` - Example secrets file

---

## Commit and Push

```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

Now deploy on Railway → Streamlit Cloud!

---

## Troubleshooting

**Streamlit Error: "Module not found: plotly"**
- ✅ Solution: `requirements.txt` is now in your repo. Push it to GitHub.

**Streamlit Error: "Connection refused" or "API Error 500"**
- ❌ Backend not deployed yet
- ✅ Solution: Deploy backend to Railway/Render first (Step 2)

**Backend Error: "Invalid API key"**
- ❌ OPENAI_API_KEY not set
- ✅ Solution: Add it in Railway/Render environment variables

**Streamlit can't connect to API**
- ❌ Wrong API_BASE_URL in secrets
- ✅ Solution: Update secrets.toml with correct Railway/Render URL

---

## Cost Summary

- **Railway:** Free 500 hours/month (then ~$5/month)
- **Streamlit Cloud:** Free for public apps
- **OpenAI API:** ~$0.01-0.02 per strategy generation

**Total: $0-5/month** (excluding OpenAI usage)

---

## Need More Details?

See `STREAMLIT_CLOUD_DEPLOYMENT.md` for comprehensive instructions.

---

## 🎉 You're Ready!

Once both are deployed:
- Backend API: `https://your-api.railway.app/docs`
- Frontend: `https://your-app.streamlit.app`

Generate AI trading strategies from anywhere! 📈🚀
