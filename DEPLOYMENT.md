# Deployment Guide — GitHub + Vercel + Hugging Face Spaces

## Project Structure

```
curator-ai/
├── frontend/          ← React/Vite app (deploy to Vercel)
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vercel.json
│   └── ...
├── backend/           ← Python/FastAPI (deploy to Hugging Face Spaces)
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

## PART 2 — Deploy Backend to Hugging Face Spaces

### Step 1: Create a Hugging Face account
1. Go to https://huggingface.co
2. Click **Sign Up** and follow the steps

### Step 2: Create a new Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `curator-backend`
   - **License**: Choose any (e.g., MIT)
   - **SDK**: Select **Docker**
   - **Visibility**: Public (or Private if you prefer)
3. Click **Create Space**

### Step 3: Add your backend files
After the Space is created, you have two options:

**Option A — Upload via web UI:**
1. In your Space, go to the **Files** tab
2. Click **Upload files**
3. Upload these files from your `NEW_UI/backend/` folder:
   - `main.py`
   - `Dockerfile`
   - `requirements.txt`

**Option B — Push via Git:**
```
cd C:\Users\emran\Downloads\Curator AI\NEW_UI\backend
git init
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/curator-backend
git add .
git commit -m "Deploy backend"
git push space main
```
Replace `YOUR_USERNAME` with your Hugging Face username.

### Step 4: Set environment variables
1. In your Space, go to **Settings** → **Repository secrets**
2. Click **New secret**
3. Add:
   - **Name**: `ROBOFLOW_API_KEY`
   - **Value**: Your Roboflow API key (get it from https://app.roboflow.com/settings/api)

### Step 5: Wait for build
1. Hugging Face will automatically build your Dockerfile
2. Go to the **App** tab to see the build logs
3. Wait 3-5 minutes for the build to complete
4. Once running, your backend URL will be:
   `https://YOUR_USERNAME-curator-backend.hf.space`

### Step 6: Test
Go to: `https://YOUR_USERNAME-curator-backend.hf.space/health`
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
VITE_STORAGE_API_URL = https://YOUR_USERNAME-curator-backend.hf.space
```
Replace `YOUR_USERNAME` with your Hugging Face username.

### Step 5: Deploy
1. Click **Deploy**
2. Wait 1-2 minutes
3. Copy your Vercel URL (e.g., `https://curator-ai-xxxx.vercel.app`)

---

## Your URLs

| Service | URL |
|---------|-----|
| GitHub | https://github.com/YOUR_USERNAME/curator-ai |
| Backend | https://YOUR_USERNAME-curator-backend.hf.space |
| Frontend | https://curator-ai-xxxx.vercel.app |
| Health Check | https://YOUR_USERNAME-curator-backend.hf.space/health |

---

## How It Works

- **Blur/blink/duplicate detection** → Runs in your browser (TensorFlow.js, MediaPipe, DINOv2)
- **Object detection** → Frontend sends image to backend → backend calls Roboflow YOLO-World API
- **Caption generation** → Frontend sends image to backend → backend calls Roboflow Florence-2 API
- **ZIP export** → Built entirely in the browser using JSZip (no server needed)

---

## Important Notes

- **Port 7860**: Hugging Face Spaces internally maps to port 7860. Our Dockerfile already configures uvicorn to listen on this port.
- **Free tier**: HF Spaces free CPU Basic tier will sleep after 48 hours of inactivity. First request after sleep takes 20-30 seconds.
- **No persistent storage**: The backend processes images in-memory. No files are saved to disk.

---

## Troubleshooting

**Frontend can't reach backend:**
- Check `VITE_STORAGE_API_URL` in Vercel settings
- Make sure the URL is `https://YOUR_USERNAME-curator-backend.hf.space` (no trailing slash)
- Redeploy after changing env vars

**HF Spaces build fails:**
- Make sure your `backend/` folder contains `main.py`, `Dockerfile`, and `requirements.txt`
- Check the **Build logs** tab in your Space for errors

**HF Spaces shows 502 Bad Gateway:**
- The app is still building or starting up — wait 30 seconds and refresh
- If it persists, check the **Logs** tab for errors

**Vercel build fails:**
- Make sure Root Directory is `NEW_UI/frontend` (exact casing matters)
