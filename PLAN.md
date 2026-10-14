# EmotiTrack — The Plan

> **One session. 5 minutes. Two outputs.**
> This is the single source of truth. If it's not in this file, we don't build it.

---

## What We Are Building

A person sits in front of a laptop, answers a few prompts for 5 minutes. The system produces:

**Output 1 — Emotion Timeline (moment by moment)**
```
0:00-0:45   Neutral
0:45-1:20   Sad
1:20-2:10   Neutral
2:10-2:40   Happy
Overall: 60% neutral, 25% sad, 15% happy
```

**Output 2 — Mental State Screening Score (one number)**
```
Depression indicator:  0.72  (high)
Stress indicator:      0.55  (moderate)
Expressivity:          Low
Recommendation: suggest speaking to a counsellor
```

That's it. No weekly recording. No longitudinal tracking. One sitting, instant result.

---

## What We Are NOT Building

These are out of scope. If you find yourself working on any of these, stop.

- Multi-session tracking or trends over time
- Real-time live streaming analysis
- Mobile app
- PDF report export
- Body pose / gesture analysis
- Custom deep learning models or fine-tuning
- Any GPU-dependent feature
- Anything that says "you have depression" — we say "screening indicator"

---

## How It Works

```
Person answers prompts for 5 minutes
                ↓
          video.mp4 saved
                ↓
        ┌───────┴───────┐
      face            audio
        │                │
        │          ┌─────┴─────┐
        │       openSMILE    Whisper
        │       (pitch,        │
        │        pauses)     text
        │            │         │
        │            │     sentiment
        └────────────┴─────────┘
                     ↓
          ~90 numbers (feature vector)
                     ↓
          ┌──────────┴──────────┐
    Emotion model          Our classifier
    (pretrained)           (Random Forest)
          │                      │
          ↓                      ↓
    Emotion timeline      Screening score
```

---

## Models — Complete List

### We download and use (DO NOT train these)

| # | Job | Model | Library | Size |
|---|---|---|---|---|
| 1 | Face landmarks + head pose | MediaPipe Face Mesh | `mediapipe` | ~5 MB |
| 2 | Face muscles (Action Units) | Py-Feat detector | `py-feat` | ~150 MB |
| 3 | Emotion from face | HSEmotion / FER | pretrained | ~50 MB |
| 4 | Voice features (88 numbers) | openSMILE eGeMAPS | `opensmile` | ~2 MB |
| 5 | Speech to text | Whisper base | `openai-whisper` | ~140 MB |
| 6 | English sentiment | RoBERTa | `transformers` | ~500 MB |
| 7 | Hindi/Marathi sentiment | MuRIL | `transformers` | ~900 MB |

### We train (the ONLY model we train)

| # | Job | Algorithm | Library | Training time |
|---|---|---|---|---|
| 8 | Depression/stress screening | Random Forest + XGBoost | `scikit-learn`, `xgboost` | < 1 min on CPU |

**Total download: ~1.7 GB. Total training: < 1 minute. No GPU needed.**

---

## Datasets

| Dataset | What it gives us | Access |
|---|---|---|
| **DAIC-WOZ** | 189 clinical interviews + real PHQ-8 depression scores → trains our classifier | Apply at https://dcapswoz.ict.usc.edu/ |
| **MELD** | 13,700 conversation clips + emotion labels → validates emotion timeline | Free download |
| **RAVDESS** | Acted emotion clips → baseline only (inflated accuracy, don't headline) | Free download |
| **HindiSER** | 520 Hindi utterances → checks if pipeline works on Indian speakers | On request |

We do NOT record our own training data. DAIC-WOZ already has everything we need.

---

## Build Phases

### Phase 1: Setup
> Get the dev environment working on every team member's laptop.

- [ ] 1.1 — Python 3.10+ virtual environment created
- [ ] 1.2 — `pip install -r requirements.txt` succeeds
- [ ] 1.3 — `ffmpeg -version` works
- [ ] 1.4 — `config.yaml` loads in Python
- [ ] 1.5 — Create a 5-second test video with audio for testing extractors

---

### Phase 2: Audio Extractor (`ml/extractors/audio.py`)
> Input: `.wav` or `.mp4` → Output: flat dict of ~30 features prefixed `audio_`

**Tools:** openSMILE eGeMAPS + librosa

- [ ] 2.1 — Load audio from .wav or .mp4 (librosa, sr=16000)
- [ ] 2.2 — Energy-based VAD: find speech vs silence segments
- [ ] 2.3 — Pause features: count, mean duration, pause-to-speech ratio
- [ ] 2.4 — Speech rate (words/min, needs transcript as optional arg)
- [ ] 2.5 — Pitch (F0): mean, std, range using librosa.pyin()
- [ ] 2.6 — Loudness: RMS energy mean and std
- [ ] 2.7 — Jitter and shimmer via openSMILE eGeMAPS
- [ ] 2.8 — MFCC 1-13: mean and std of each coefficient
- [ ] 2.9 — Combine all into flat dict, prefix `audio_`
- [ ] 2.10 — Handle edge cases: silent audio, <1s clips, missing track → return None + log
- [ ] 2.11 — Test with synthetic sine wave, verify all keys present

**Output keys (~30):**
```
audio_pause_count, audio_pause_mean_duration, audio_pause_to_speech_ratio,
audio_speech_rate_wpm, audio_f0_mean, audio_f0_std, audio_f0_range,
audio_loudness_mean, audio_loudness_std, audio_jitter, audio_shimmer,
audio_mfcc_01_mean, audio_mfcc_01_std, ... audio_mfcc_13_mean, audio_mfcc_13_std
```

---

### Phase 3: Face Extractor (`ml/extractors/face.py`)
> Input: video file → Output: flat dict of ~20 features prefixed `face_`

**Tools:** Py-Feat (AUs) + MediaPipe Face Mesh (head pose, blinks, gaze)

- [ ] 3.1 — Load video with OpenCV, process every 3rd frame
- [ ] 3.2 — Py-Feat: extract AU intensities per frame
- [ ] 3.3 — AU stats: mean/std for AU04, AU06, AU12, AU15 + total AU activity
- [ ] 3.4 — MediaPipe Face Mesh: initialize, run on sampled frames
- [ ] 3.5 — Head pose: pitch/yaw/roll from landmarks via solvePnP
- [ ] 3.6 — Head movement velocity: mean per-frame change in pose angles
- [ ] 3.7 — Blink detection: eye aspect ratio → blinks/minute
- [ ] 3.8 — Gaze direction variance from iris landmarks
- [ ] 3.9 — Face detection rate (fraction of frames with face). quality_warning if < 0.5
- [ ] 3.10 — Combine into flat dict, prefix `face_`
- [ ] 3.11 — Test with synthetic video, verify dict shape

**Output keys (~20):**
```
face_au04_mean, face_au04_std, face_au06_mean, face_au06_std,
face_au12_mean, face_au12_std, face_au15_mean, face_au15_std,
face_au_total_activity, face_head_movement_velocity,
face_blink_rate_per_min, face_gaze_direction_variance,
face_detection_rate, face_quality_warning
```

---

### Phase 4: Text Extractor (`ml/extractors/text.py`)
> Input: audio file → Output: flat dict of ~10 features prefixed `text_`

**Tools:** Whisper base (transcription) + RoBERTa or MuRIL (sentiment)

- [ ] 4.1 — Transcribe audio with Whisper base model
- [ ] 4.2 — Cache transcript to `data/transcripts/{filename}.json`
- [ ] 4.3 — Detect language from Whisper output → route to correct sentiment model
- [ ] 4.4 — English sentiment: RoBERTa → positive/negative/neutral scores
- [ ] 4.5 — Hindi/Marathi sentiment: MuRIL → same format
- [ ] 4.6 — First-person pronoun ratio (I/me/my/mine/myself / total words)
- [ ] 4.7 — Mean utterance length in words
- [ ] 4.8 — Lexical diversity (type-token ratio)
- [ ] 4.9 — Total word count
- [ ] 4.10 — Silence-adjusted speaking time
- [ ] 4.11 — Combine into flat dict, prefix `text_`
- [ ] 4.12 — Test with known audio, verify caching and keys

**Output keys (~10):**
```
text_word_count, text_first_person_pronoun_ratio,
text_sentiment_positive, text_sentiment_negative, text_sentiment_neutral,
text_mean_utterance_length, text_lexical_diversity,
text_speaking_time_seconds, text_language
```

---

### Phase 5: Pipeline Runner (`ml/pipeline.py`)
> Input: video.mp4 → Output: complete feature vector (~90 numbers) + metadata

- [ ] 5.1 — Split video into video + audio tracks using ffmpeg
- [ ] 5.2 — `process_session(video_path)`: run all 3 extractors, merge dicts
- [ ] 5.3 — Add metadata: duration, timestamp, warnings
- [ ] 5.4 — Handle partial failures: if one extractor fails, return the rest + warning
- [ ] 5.5 — `scripts/extract_batch.py`: process a directory of videos → CSV
- [ ] 5.6 — Resumable: skip videos already in output CSV
- [ ] 5.7 — Progress bar (tqdm) and error log
- [ ] 5.8 — Test end-to-end with test video

---

### Phase 6: Training & Evaluation (`ml/train.py` + `ml/evaluate.py`)
> Train the ONE model we own: screening classifier on DAIC-WOZ features

- [ ] 6.1 — Load features CSV + labels CSV, join on session_id
- [ ] 6.2 — Participant-level split (same person never in train AND test)
- [ ] 6.3 — 5-fold stratified CV using StratifiedGroupKFold
- [ ] 6.4 — Train Random Forest
- [ ] 6.5 — Train XGBoost
- [ ] 6.6 — Report: accuracy, macro-F1, AUC with 95% bootstrap CIs
- [ ] 6.7 — Confusion matrix
- [ ] 6.8 — Feature importances → CSV
- [ ] 6.9 — Save best model to `data/models/` with timestamp
- [ ] 6.10 — `evaluate.py`: ROC curve, confusion matrix plot, top-20 feature importance
- [ ] 6.11 — Late fusion in `fusion.py`: weighted average of 3 modality scores

---

### Phase 7: Backend API (`backend/`)
> FastAPI server: upload video → run pipeline → return results

- [ ] 7.1 — SQLite schema: users, sessions, features (Postgres-swappable)
- [ ] 7.2 — SQLAlchemy models
- [ ] 7.3 — `POST /users` — register with consent flag
- [ ] 7.4 — `DELETE /users/{id}` — delete user + all data (consent withdrawal)
- [ ] 7.5 — `POST /sessions` — upload video, return session_id, start background task
- [ ] 7.6 — Background task: run pipeline, store features, delete raw video
- [ ] 7.7 — `GET /sessions/{id}` — features, emotion timeline, screening score
- [ ] 7.8 — `GET /users/{id}/sessions` — list sessions with scores
- [ ] 7.9 — `GET /sessions/{id}/status` — pending/processing/complete/failed
- [ ] 7.10 — CORS setup + error handling
- [ ] 7.11 — Tests with httpx

---

### Phase 8: Frontend Dashboard (`frontend/`)
> React app: 4 pages, no animations, Recharts for charts

- [ ] 8.1 — React + react-router-dom + axios + recharts setup
- [ ] 8.2 — **Consent page**: must complete before recording is enabled
- [ ] 8.3 — **Record page**: webcam capture, on-screen prompts, countdown, upload on stop
- [ ] 8.4 — **Session Result page**: emotion timeline (band chart), screening score (gauge), feature table
- [ ] 8.5 — **History page**: line chart of past session scores
- [ ] 8.6 — Disclaimer on every score page: "This is a screening indicator, not a diagnosis. Please consult a qualified mental health professional."
- [ ] 8.7 — Navigation bar + routing
- [ ] 8.8 — Clean, readable styling

---

### Phase 9: Integration & Testing
> Make sure everything works together

- [ ] 9.1 — End-to-end: record → upload → pipeline → dashboard shows results
- [ ] 9.2 — Unit tests for each extractor (synthetic clips)
- [ ] 9.3 — Profile pipeline: time each stage for 5-min video
- [ ] 9.4 — Memory check: peak usage < 8 GB
- [ ] 9.5 — Privacy audit: no raw video stored, no PII in logs
- [ ] 9.6 — Edge cases: no face, silent audio, non-English, short clips

---

### Phase 10: Evaluation & Report
> Get real numbers for the report

- [ ] 10.1 — Get DAIC-WOZ access
- [ ] 10.2 — Run pipeline on DAIC-WOZ (extract features for 189 interviews)
- [ ] 10.3 — Train + evaluate screening classifier → accuracy, F1, AUC
- [ ] 10.4 — Run emotion model on MELD → accuracy, macro-F1
- [ ] 10.5 — Test on Indian speakers → compare with English results
- [ ] 10.6 — Fill README results tables
- [ ] 10.7 — Generate plots: ROC, confusion matrix, feature importances

---

## Timeline

```
Week 1      Phase 1 (Setup) + apply for DAIC-WOZ
Week 2-3    Phase 2 (Audio) + Phase 3 (Face) + Phase 4 (Text)  ← parallel
Week 4      Phase 5 (Pipeline runner)
Week 5      Phase 6 (Training logic)
Week 6-7    Phase 7 (Backend) + Phase 8 (Frontend)  ← parallel
Week 8      Phase 9 (Integration testing)
Week 9-10   Phase 10 (Evaluation on real data + report)
```

## Dependencies

```
Phase 2 (Audio) ──┐
Phase 3 (Face)  ──┼── Phase 5 (Pipeline) ── Phase 6 (Training)
Phase 4 (Text)  ──┘         │
                             └── Phase 7 (Backend) ── Phase 8 (Frontend)
                                                            │
                                              Phase 9 (Integration) ── Phase 10 (Report)
```

---

## Target Numbers

| Metric | Target |
|---|---|
| Emotion accuracy (MELD) | 70%+ |
| Depression screening accuracy (DAIC-WOZ) | 80%+ |
| Processing time per 5-min session | < 30 seconds |
| Works on Indian speakers | Yes, with reported results |
| Runs on CPU, 8 GB RAM laptop | Yes |

---

## Risks

| Risk | What happens | Fix |
|---|---|---|
| DAIC-WOZ access delayed | Can't train screening model | Build emotion part first, use synthetic labels to test pipeline |
| Py-Feat slow on CPU | Face extraction too slow | Process every 3rd frame, cap video at 5 min |
| Whisper bad on Indian English | Wrong transcripts | Try whisper small; allow manual transcript override |
| MuRIL has no sentiment head | Hindi sentiment fails | Fine-tune tiny classification head on Hindi sentiment data |
| RAM overflow | Crashes on 8 GB laptop | Load one model at a time, unload after use |

---

## Rules During Development

1. **Check this file before starting any task.** If the task isn't listed here, don't do it.
2. **No scope creep.** If you think of a cool feature, add it to a "maybe later" list, don't build it.
3. **Test each phase before moving on.** A broken extractor will break everything downstream.
4. **Commit after every working phase.** So you can roll back.
5. **Never say "diagnosis."** Always "screening indicator."
6. **Never commit data files.** Check `git status` before every push.
7. **CPU only.** If it needs a GPU, you're doing it wrong.
