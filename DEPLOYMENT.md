# Deployment Guide — GitHub + Vercel + Koyeb

## Project Structure

```
curator-ai/
├── frontend/          ← React/Vite app (deploy to Vercel)
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vercel.json
│   └── ...
├── backend/           ← Python/FastAPI (deploy to Koyeb)
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## PART 1 — Upload to GitHub

### Step 1: Install Git
1. Go to https://git-scm.com/download/win
2. Download and install with default settings
3. Restart your computer

### Step 2: Create a GitHub account
1. Go to https://github.com
2. Click **Sign up** and follow the steps

### Step 3: Create a repository
1. Log in to GitHub
2. Click **+** (top right) → **New repository**
3. Repository name: `curator-ai`
4. Select **Public**
5. Click **Create repository**

### Step 4: Upload your code
Open PowerShell in your project root (`C:\Users\emran\Downloads\Curator AI`):
```
git init
git add .
git commit -m "Initial commit - CuratorAI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/curator-ai.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username.

---

## PART 2 — Deploy Backend to Koyeb

### Step 1: Create a Koyeb account
1. Go to https://www.koyeb.com
2. Click **Sign up** → sign up with GitHub

### Step 2: Create an app
1. Click **Create App**
2. Select **GitHub/Git** as the build method
3. Name: `curator-backend`

### Step 3: Connect your GitHub repo
1. Under **Source**, select **GitHub**
2. Authorize Koyeb
3. Select your `curator-ai` repository
4. Set these values:
   - **Workdir / Root Directory**: `NEW_UI/backend`
   - **Dockerfile path**: `Dockerfile` (relative to the root directory above)
   - **Port**: `8080`

### Step 4: Set environment variables
Click **Environment Variables** and add:
```
ROBOFLOW_API_KEY = your_roboflow_api_key_here
```
Get your API key from https://app.roboflow.com/settings/api

### Step 5: Deploy
1. Click **Deploy**
2. Wait 2-3 minutes
3. Copy your Koyeb URL (e.g., `https://curator-backend-xxxx.koyeb.app`)

### Step 6: Test
Go to: `https://curator-backend-xxxx.koyeb.app/health`
You should see: `{"status":"ok"}`

---

## PART 3 — Deploy Frontend to Vercel

### Step 1: Create a Vercel account
1. Go to https://vercel.com
2. Click **Sign up** → sign up with GitHub

### Step 2: Import your project
1. Click **Add New → Project**
2. Select **Import Git Repository**
3. Select your `curator-ai` repository

### Step 3: Configure
1. **Framework Preset**: Vite
2. **Root Directory**: `NEW_UI/frontend` — type this manually, match the casing exactly (capital N, capital E, underscore, lowercase front). If Vercel shows a folder picker, navigate into `NEW_UI` then select `frontend`. Do not use `./NEW_UI/frontend` or `new_ui/frontend`.
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`

### Step 4: Set environment variables
Click **Environment Variables** and add:
```
VITE_STORAGE_API_URL = https://curator-backend-xxxx.koyeb.app
```
Replace `xxxx` with your actual Koyeb URL.

### Step 5: Deploy
1. Click **Deploy**
2. Wait 1-2 minutes
3. Copy your Vercel URL (e.g., `https://curator-ai-xxxx.vercel.app`)

---

## Your URLs

| Service | URL |
|---------|-----|
| GitHub | https://github.com/YOUR_USERNAME/curator-ai |
| Backend | https://curator-backend-xxxx.koyeb.app |
| Frontend | https://curator-ai-xxxx.vercel.app |
| Health Check | https://curator-backend-xxxx.koyeb.app/health |

---

## How It Works

- **Blur/blink/duplicate detection** → Runs in your browser (TensorFlow.js, MediaPipe, DINOv2)
- **Object detection** → Frontend sends image to backend → backend calls Roboflow YOLO-World API
- **Caption generation** → Frontend sends image to backend → backend calls Roboflow Florence-2 API
- **ZIP export** → Built entirely in the browser using JSZip (no server needed)

---

## Troubleshooting

**Frontend can't reach backend:**
- Check `VITE_STORAGE_API_URL` in Vercel settings
- Redeploy after changing env vars

**Koyeb app sleeping:**
- Free tier sleeps after inactivity
- First request takes 30-60 seconds to wake up

**Vercel build fails:**
- Make sure Root Directory is `NEW_UI/frontend`
