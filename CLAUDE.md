# EmotiTrack — Emotion and Mental State Detection

## What this project does
Multimodal system that analyses a short video of a person speaking and produces:
1. An emotion timeline
2. A screening score for depression and stress

Three signal channels (face, voice, text) are fused using late fusion. All feature extractors are pretrained; only the final classifier is trained by us.

## Tech stack
- **ML:** Python 3.10+, PyTorch, scikit-learn, XGBoost
- **Vision:** OpenCV, MediaPipe Face Mesh, Py-Feat (Action Units)
- **Audio:** openSMILE (eGeMAPS), Whisper, librosa, ffmpeg
- **NLP:** HuggingFace Transformers, MuRIL (Hindi/Marathi), RoBERTa (English)
- **Backend:** FastAPI, SQLite (Postgres-ready schema)
- **Frontend:** React, Recharts

## Project structure
```
EmotiTrack/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── models/
│   └── services/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
├── ml/
│   ├── extractors/
│   │   ├── face.py
│   │   ├── audio.py
│   │   └── text.py
│   ├── fusion.py
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/          # gitignored
│   ├── features/
│   └── models/
├── scripts/
│   └── extract_batch.py
├── notebooks/
├── docs/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

## Key conventions
- All extractor functions expose `extract(path) -> dict | None`
- Feature keys are prefixed by modality: `audio_`, `face_`, `text_`
- Config lives in `config.yaml` — paths, model names, thresholds
- Raw video/audio files are never committed
- Every screening score display must include the disclaimer:
  "This is a screening indicator, not a diagnosis. Please consult a qualified mental health professional."

## Important constraints
- Py-Feat for Action Units (not OpenFace — it doesn't pip-install)
- MediaPipe Face Mesh for head pose and blink rate
- Process every 3rd frame for speed
- Whisper base model on CPU
- Language detection: Hindi/Marathi → MuRIL, English → RoBERTa
- Participant-level train/test splits (never session-level)
- Features-only storage by default; raw video deleted after processing
