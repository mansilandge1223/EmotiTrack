# EmotiTrack — Setup Guide

> Complete installation guide for setting up the development environment.
> Follow every step in order. Do not skip anything.

---

## Prerequisites Checklist

Before you begin, make sure you have:

| Requirement | Minimum | Check command |
|---|---|---|
| Operating System | Windows 10/11, macOS 12+, or Ubuntu 20.04+ | — |
| RAM | 8 GB (16 GB recommended) | — |
| Disk space | 5 GB free (for models and dependencies) | — |
| Internet | Required for first-time model downloads | — |
| Webcam + Microphone | Required for recording sessions | — |
| GPU | **NOT required** — everything runs on CPU | — |

---

## Step 1: Install Python 3.10+

### Windows
1. Download from https://www.python.org/downloads/
2. **IMPORTANT:** Check ✅ "Add Python to PATH" during installation
3. Verify:
```bash
python --version
# Expected: Python 3.10.x or higher
```

### macOS
```bash
brew install python@3.11
python3 --version
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
python3 --version
```

---

## Step 2: Install ffmpeg

ffmpeg is required to split video files into audio and video tracks.

### Windows (Option A — recommended)
```bash
winget install Gyan.FFmpeg
```

### Windows (Option B — manual)
1. Download from https://www.gyan.dev/ffmpeg/builds/
   - Get "ffmpeg-release-essentials.zip"
2. Extract to `C:\ffmpeg\`
3. Add to PATH:
```powershell
# Run in PowerShell (not CMD)
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\ffmpeg\bin", "User")
```
4. **Restart your terminal**

### macOS
```bash
brew install ffmpeg
```

### Ubuntu/Debian
```bash
sudo apt install ffmpeg
```

### Verify
```bash
ffmpeg -version
# Should print: ffmpeg version X.X.X ...

ffprobe -version
# Should print: ffprobe version X.X.X ...
```

---

## Step 3: Install Node.js 18+

Required for the React frontend.

### Windows
Download from https://nodejs.org/ (LTS version)

### macOS
```bash
brew install node
```

### Ubuntu/Debian
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs
```

### Verify
```bash
node --version
# Expected: v18.x.x or higher

npm --version
# Expected: 9.x.x or higher
```

---

## Step 4: Clone the Repository

```bash
git clone git@github.com:mansilandge1223/EmotiTrack.git
cd EmotiTrack
```

If you don't have SSH set up, use HTTPS:
```bash
git clone https://github.com/mansilandge1223/EmotiTrack.git
cd EmotiTrack
```

---

## Step 5: Create Python Virtual Environment

**Always use a virtual environment.** Never install packages globally.

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt. **Every time you open a new terminal, activate the venv again.**

---

## Step 6: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs ~20 packages. First-time install takes 5–10 minutes depending on internet speed.

### Verify core packages
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import opensmile; print('openSMILE:', opensmile.__version__)"
python -c "import whisper; print('Whisper: OK')"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import librosa; print('librosa:', librosa.__version__)"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import yaml; print('PyYAML: OK')"
```

All should print without errors. If any fail, run:
```bash
pip install <package-name>
```

---

## Step 7: Verify Config

```bash
python -c "from ml.config import get_config; c = get_config(); print('Config loaded OK'); print('Sample rate:', c['audio']['sample_rate'])"
```

Expected output:
```
Config loaded OK
Sample rate: 16000
```

---

## Step 8: Generate Test Fixtures

```bash
python -m tests.generate_fixtures
```

Expected output:
```
Generating test_audio.wav (5s sine wave with 1s silence)...
Generating test_silent.wav (5s silence)...
Generating test_video.avi (5s synthetic face)...

Fixtures created in tests/fixtures/
  test_audio.wav (156.3 KB)
  test_silent.wav (156.3 KB)
  test_video.avi (146.9 KB)
```

---

## Step 9: Run Tests

```bash
python -m tests.test_audio
```

Expected output:
```
PASS: test_extract_sine_wave
PASS: test_extract_silent_audio
PASS: test_extract_missing_file
PASS: test_extract_with_transcript

All audio tests passed!
```

---

## Step 10: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Step 11: First-Time Model Downloads

Some models are downloaded automatically on first use. Run these commands once to pre-download them (each takes 1–5 minutes):

### Whisper base model (~140 MB)
```bash
python -c "import whisper; whisper.load_model('base'); print('Whisper base model downloaded')"
```

### RoBERTa sentiment model (~500 MB)
```bash
python -c "from transformers import pipeline; p = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); print('RoBERTa sentiment model downloaded')"
```

### MuRIL model (~900 MB) — only needed for Hindi/Marathi
```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('google/muril-base-cased'); AutoModel.from_pretrained('google/muril-base-cased'); print('MuRIL model downloaded')"
```

> **Note:** These models are cached in `~/.cache/` and only downloaded once. Total: ~1.7 GB.

---

## Running the Application

### Start the backend
```bash
# Make sure venv is activated
uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000/health — should return `{"status": "ok"}`

### Start the frontend (separate terminal)
```bash
cd frontend
npm start
```

Open http://localhost:3000

---

## Project Structure

```
EmotiTrack/
├── backend/                 # FastAPI server
│   ├── main.py             # Entry point
│   ├── routes/             # API endpoint handlers
│   ├── models/             # SQLAlchemy database models
│   └── services/           # Business logic
├── frontend/                # React dashboard
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components (Record, Results, History, Consent)
│   │   └── App.jsx         # Root component
│   └── package.json
├── ml/                      # Machine learning pipeline
│   ├── config.py           # Config loader
│   ├── extractors/
│   │   ├── audio.py        # Voice feature extraction (openSMILE + librosa)
│   │   ├── face.py         # Face feature extraction (Py-Feat + MediaPipe)
│   │   └── text.py         # Text feature extraction (Whisper + sentiment)
│   ├── fusion.py           # Late fusion of modality scores
│   ├── pipeline.py         # End-to-end processing pipeline
│   ├── train.py            # Screening classifier training
│   └── evaluate.py         # Evaluation metrics and plots
├── data/
│   ├── raw/                # Gitignored — raw video uploads
│   ├── features/           # Extracted feature CSVs
│   ├── models/             # Saved classifier weights (.joblib)
│   └── transcripts/        # Cached Whisper transcripts
├── scripts/
│   └── extract_batch.py    # Batch feature extraction CLI
├── tests/
│   ├── fixtures/           # Synthetic test files (gitignored)
│   ├── generate_fixtures.py
│   └── test_audio.py
├── notebooks/               # Jupyter notebooks (exploration only)
├── docs/                    # Documentation, consent forms
├── .gitignore
├── CLAUDE.md               # AI assistant context
├── PLAN.md                 # Development plan and progress tracker
├── SETUP.md                # This file
├── config.yaml             # All paths and model settings
├── requirements.txt        # Python dependencies (pinned versions)
└── README.md               # Project overview
```

---

## Key Configuration (`config.yaml`)

```yaml
paths:
  data_raw: data/raw               # Raw uploads (gitignored)
  data_features: data/features     # Feature CSVs
  data_models: data/models         # Saved models
  transcripts_cache: data/transcripts  # Cached transcripts

audio:
  sample_rate: 16000               # Audio sample rate (Hz)
  energy_threshold: 0.01           # VAD energy threshold
  min_pause_duration: 0.3          # Minimum pause to count (seconds)

face:
  frame_skip: 3                    # Process every Nth frame
  min_detection_rate: 0.5          # Warn if face detected in < 50% frames

text:
  whisper_model: base              # Whisper model size (base or small)
  sentiment_model_en: cardiffnlp/twitter-roberta-base-sentiment-latest
  sentiment_model_indic: google/muril-base-cased
  language_detect: true

training:
  n_folds: 5                       # Cross-validation folds
  test_size: 0.2                   # Test split ratio
  random_state: 42                 # Reproducibility seed

backend:
  database: sqlite:///data/emotitrack.db
  keep_raw: false                  # Delete raw video after processing
  upload_dir: data/uploads
```

---

## Troubleshooting

### "ffmpeg: command not found"
ffmpeg is not on your PATH. See Step 2 above. After adding to PATH, **restart your terminal**.

### "ModuleNotFoundError: No module named 'xyz'"
Your virtual environment is not activated. Run:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

### "pip install fails with build errors"
Try upgrading pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### "torch is too large / taking forever"
PyTorch CPU-only version is smaller. Replace the torch line in requirements.txt:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.1+cpu
```

### "opensmile installation fails on Windows"
Install Visual C++ Build Tools first:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### "MediaPipe not working on M1/M2 Mac"
```bash
pip install mediapipe-silicon
```

### "Whisper model download fails"
Check internet connection. If behind a proxy:
```bash
export HTTPS_PROXY=http://your-proxy:port
```

### "Port 8000 already in use"
```bash
uvicorn backend.main:app --reload --port 8001
```

### "Git push rejected"
Pull first:
```bash
git pull origin main
# Resolve any conflicts, then push again
git push origin main
```

---

## Git Workflow

### Before every coding session
```bash
git pull origin main
```

### After completing a task
```bash
git status                    # Check what changed
git add <specific-files>      # Stage only what you changed
git commit -m "Description"   # Commit with clear message
git push origin main          # Push to remote
```

### Important git rules
- **Never commit** files in `data/raw/`, `*.mp4`, `*.wav`, `*.jpg`
- **Always check** `git status` before pushing
- **Commit after every working phase** — so you can roll back

---

## Team Contact

| Name | Role | GitHub |
|---|---|---|
| | ML pipeline | |
| | Backend | |
| | Frontend | |
| | Data collection | |

---

## Quick Reference Card

```
# Activate environment
venv\Scripts\activate                        # Windows
source venv/bin/activate                     # macOS/Linux

# Run tests
python -m tests.test_audio

# Start backend
uvicorn backend.main:app --reload --port 8000

# Start frontend
cd frontend && npm start

# Generate test data
python -m tests.generate_fixtures

# Extract features from a video
python -c "from ml.pipeline import process_session; print(process_session('video.mp4'))"

# Train classifier
python -m ml.train --features data/features/features.csv --labels data/features/labels.csv
```
