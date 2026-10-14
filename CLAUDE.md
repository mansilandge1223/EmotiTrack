# CLAUDE.md

Context for Claude Code working in this repository.

## What this project is

A multimodal system that analyses a short video of a person speaking and produces two outputs:

1. **Emotion timeline** — which of 7 emotions appeared during the session, and when
2. **Mental state screening score** — depression and stress indicators, plus an expressivity measure

Final year engineering project. Target: a working demo plus a measurable result on a public clinical dataset.

## Hard rules — do not violate these

1. **This is a screening tool, never a diagnostic one.** No code, UI text, log message or variable name should state that a person "has depression". Use `depression_indicator`, `screening_score`, `risk_level`. UI must always suggest consulting a professional.
2. **Never commit participant data.** `data/raw/`, `*.mp4`, `*.wav`, `*.jpg` are gitignored. If you see them staged, stop and warn.
3. **Do not train large models.** All feature extractors are pretrained and used as-is. We train exactly one small classifier (Random Forest / XGBoost) on aggregated features. If a task seems to need GPU training, you have misunderstood — ask.
4. **CPU-only must work.** Assume a student laptop, 8 GB RAM, no GPU. Any dependency requiring CUDA is wrong.
5. **Measure, don't interpret.** Facial output is Action Unit intensities (muscle movement), not emotion claims. Emotion labels come from a separate pretrained classifier and are treated as approximate.

## Architecture

```
video.mp4
   ├── video track → MediaPipe / Py-Feat → Action Units, head pose, blink rate
   └── audio track → openSMILE → pitch, pauses, loudness, jitter
                   → Whisper → transcript → MuRIL/RoBERTa → sentiment
                              ↓
                   aggregate to session-level feature vector (~90 features)
                              ↓
              ┌───────────────┴───────────────┐
      pretrained emotion model        our Random Forest
              ↓                               ↓
      emotion timeline              screening score
```

Fusion is **late fusion** — each modality produces a score, then weighted average. Do not implement cross-modal transformers; they do not beat late fusion at our data size and are harder to debug.

## Tech stack

- Python 3.10+, no GPU required
- OpenCV, MediaPipe, Py-Feat (OpenFace alternative that pip-installs)
- openSMILE (eGeMAPS feature set), librosa, ffmpeg
- openai-whisper (base or small model only)
- transformers (MuRIL / IndicBERT for Indian languages)
- scikit-learn for the classifier
- FastAPI backend, React frontend
- PostgreSQL (SQLite acceptable during development)

## Directory layout

```
backend/        FastAPI app
frontend/       React dashboard
ml/
  extractors/   face.py, audio.py, text.py — one wrapper per modality
  fusion.py     late fusion logic
  train.py      trains the screening classifier
  evaluate.py   metrics and plots
data/
  raw/          gitignored
  features/     CSVs of extracted features
  models/       saved classifier weights
notebooks/      exploration only, not production code
tests/
```

## Coding conventions

- Each extractor exposes `extract(path: str) -> dict[str, float]` returning a flat feature dict
- Feature names are prefixed by modality: `face_au12_mean`, `audio_pause_ratio`, `text_sentiment_neg`
- Never hardcode paths; use a `config.yaml`
- Every extractor must handle a failed/empty input and return `None` rather than crashing — videos will have frames where no face is detected
- Log how often face detection fails per session; this is a reported metric, not a hidden error
- Type hints on all public functions
- Tests for feature extraction shapes, not for model accuracy

## Datasets

| Name | Use | Status |
|---|---|---|
| DAIC-WOZ | Primary — depression labels (PHQ-8) | access applied for |
| MELD | Emotion model validation | open |
| RAVDESS | Baseline only — acted, inflates accuracy | open |
| HindiSER | Indian speaker check | on request |

Do not report RAVDESS numbers as headline results. Acted corpora inflate accuracy substantially versus spontaneous speech.

## Evaluation

Report on the **test split only**:
- Emotion: accuracy and macro-F1 on MELD
- Screening: accuracy, F1, AUC on DAIC-WOZ, at participant level not segment level
- Processing time per 5-minute session
- Face detection failure rate

Always report confidence intervals. Our test sets are small; a single accuracy number is misleading.

## Current status

- [x] Repo scaffolding
- [ ] Feature extractors
- [ ] Training pipeline
- [ ] Backend API
- [ ] Frontend dashboard
- [ ] Evaluation and report

Update this checklist as work completes.

## When unsure

Ask before:
- Adding any dependency over 500 MB
- Changing the fusion strategy
- Adding a new output that implies a clinical claim
- Anything that writes to `data/raw/`
