# EmotiTrack — Setup & Run Guide

> Follow every step in order. Do not skip anything.
> Total setup time: ~15 minutes + ~10 minutes for model downloads.

---

## Prerequisites

| Requirement | Minimum |
|---|---|
| OS | Windows 10/11 |
| RAM | 8 GB (16 GB recommended) |
| Disk space | 5 GB free |
| Internet | Required for first-time setup |
| Webcam + Mic | Required for recording (optional if uploading video) |
| GPU | **NOT required** |

---

## Step 1: Install Python 3.10+

1. Download from https://www.python.org/downloads/
2. **IMPORTANT:** Check ✅ **"Add Python to PATH"** during installation
3. Verify:
```
python --version
```
Should show `Python 3.10.x` or higher.

---

## Step 2: Install Node.js 18+

1. Download LTS from https://nodejs.org/
2. Install with default settings
3. Verify:
```
node --version
npm --version
```

---

## Step 3: Install ffmpeg

### Option A — winget (recommended)
```
winget install Gyan.FFmpeg
```
Restart terminal after install.

### Option B — Manual
1. Download from https://www.gyan.dev/ffmpeg/builds/ → "ffmpeg-release-essentials.zip"
2. Extract to `C:\ffmpeg\`
3. Add to PATH:
```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\ffmpeg\bin", "User")
```
4. **Restart terminal**

### Verify
```
ffmpeg -version
```

---

## Step 4: Clone the Repository

```
git clone https://github.com/mansilandge1223/EmotiTrack.git
cd EmotiTrack
```

---

## Step 5: Install Python Dependencies

```
pip install -r requirements.txt
```

If any package fails, try:
```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Verify core packages
```
python -c "import cv2; print('OpenCV OK')"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import librosa; print('librosa OK')"
python -c "import opensmile; print('openSMILE OK')"
python -c "import whisper; print('Whisper OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import yaml; print('PyYAML OK')"
```

If `whisper` or `transformers` is missing:
```
pip install openai-whisper transformers
```

If `fastapi` is missing:
```
pip install fastapi uvicorn python-multipart
```

---

## Step 6: Install Frontend Dependencies

```
cd frontend
npm install
cd ..
```

---

## Step 7: First-Time Model Downloads (one time only)

These models download automatically on first run, but you can pre-download them:

### Whisper base model (~140 MB)
```
python -c "import whisper; whisper.load_model('base'); print('Whisper model downloaded')"
```

### RoBERTa sentiment model (~500 MB)
```
python -c "from transformers import pipeline; p = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); print('Sentiment model downloaded')"
```

### MediaPipe face landmarker (~4 MB)
Downloads automatically when you first upload a video.

---

## Running the Application

### You need 2 terminals open:

### Terminal 1 — Backend (Python server)
```
cd D:\EmotiTrack
set PATH=C:\ffmpeg;%PATH%
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Terminal 2 — Frontend (React app)
```
cd D:\EmotiTrack\frontend
npm start
```

Browser opens automatically at **http://localhost:3000**

### That's it! The app is running.

---

## Using the App

### Option 1: Upload a video
1. Click "Upload Video" or drag & drop a `.mp4` / `.avi` / `.webm` file
2. Wait for the pipeline to process (4 steps, ~1-3 minutes)
3. See results: summary cards + all 62 features

### Option 2: Record from webcam
1. Click "Start Recording"
2. Answer the on-screen questions (5 questions, auto-cycles every 12 sec)
3. Click "Next" to skip to next question, or wait
4. Recording auto-stops at 60 seconds, or click "Stop Recording"
5. Click "Upload & Analyze"
6. Wait for results

---

## Stopping the App

- **Terminal 1:** Press `Ctrl + C` to stop the backend
- **Terminal 2:** Press `Ctrl + C` to stop the frontend

---

## Troubleshooting

### "ffmpeg: not recognized"
ffmpeg is not on PATH. Either:
- Run `set PATH=C:\ffmpeg;%PATH%` before starting the backend
- Or reinstall ffmpeg and restart your terminal

### "react-scripts is not recognized"
Run `npm install` inside the `frontend/` folder:
```
cd frontend
npm install
```

### "ModuleNotFoundError: No module named 'xyz'"
Install the missing package:
```
pip install <package-name>
```

### "Port 8000 already in use"
Another process is using port 8000. Either:
- Close the other process
- Or use a different port: `uvicorn backend.main:app --port 8001`
  (also update `const API = "http://localhost:8001"` in `frontend/src/pages/UploadPage.js`)

### "Port 3000 already in use"
When asked "Would you like to run the app on another port?", type `Y`.

### "Camera access denied"
- Allow camera + microphone in browser permissions
- Chrome: click the lock icon in the address bar → allow camera & mic

### "Upload failed: Network Error"
Backend is not running. Start Terminal 1 first.

### "Processing stuck on face extraction"
Face extraction is the slowest step (~1-2 min for a 15-sec video). Wait for it. If it takes more than 5 minutes, restart the backend.

### torch is too large / taking forever
Use CPU-only version:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Quick Reference

```
# Start backend (Terminal 1)
cd D:\EmotiTrack
set PATH=C:\ffmpeg;%PATH%
uvicorn backend.main:app --reload --port 8000

# Start frontend (Terminal 2)
cd D:\EmotiTrack\frontend
npm start

# Open browser
http://localhost:3000

# Health check
http://localhost:8000/health
```

---

## Project Structure (what each folder does)

```
EmotiTrack/
├── backend/
│   └── main.py              ← FastAPI server (3 endpoints)
├── frontend/
│   └── src/
│       ├── App.js            ← Main React component
│       ├── App.css           ← All styles
│       └── pages/
│           ├── UploadPage.js ← Upload + webcam recording
│           └── ResultsPage.js← Pipeline progress + feature display
├── ml/
│   ├── config.py             ← Reads config.yaml
│   ├── pipeline.py           ← Runs all 3 extractors
│   └── extractors/
│       ├── audio.py          ← 39 voice features
│       ├── face.py           ← 14 face features
│       └── text.py           ← 9 speech features
├── config.yaml               ← All settings
├── requirements.txt          ← Python dependencies
├── PLAN.md                   ← Development plan
└── SETUP.md                  ← This file
```
