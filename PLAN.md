# EmotiTrack — Development Plan

> **Single source of truth for this project.**
> If a task is not in this file, we do not build it.
> Mark each task `[x]` only after it is tested and committed.

---

## Project Summary

| Item | Detail |
|---|---|
| **Project title** | Emotion and Mental State Detection from Audio and Video using Multimodal Deep Learning |
| **One-liner** | A person speaks to the camera for 5 minutes. The system shows which emotions appeared and gives a screening score for depression and stress. |
| **Input** | Single 5-minute webcam recording (video + audio) |
| **Output 1** | Emotion timeline — which of 7 emotions appeared, and when |
| **Output 2** | Mental state screening score — depression indicator, stress indicator, expressivity |
| **Training data** | DAIC-WOZ (189 clinical interviews with PHQ-8 scores) — we do NOT record our own |
| **Models we train** | 1 (Random Forest / XGBoost classifier on ~90 features) |
| **Models we download** | 7 pretrained models, total ~1.7 GB |
| **GPU required** | No — CPU only, 8 GB RAM laptop |
| **This is a** | Screening tool — never a diagnostic tool |

### Example Output

**Emotion Timeline:**
```
0:00–0:45   Neutral
0:45–1:20   Sad
1:20–2:10   Neutral
2:10–2:40   Happy
Overall: 60% neutral, 25% sad, 15% happy
```

**Screening Score:**
```
Depression indicator:  0.72  (high)
Stress indicator:      0.55  (moderate)
Expressivity:          Low
→ "This is a screening indicator, not a diagnosis.
    Please consult a qualified mental health professional."
```

---

## Out of Scope — DO NOT Build

If you find yourself working on any of these, stop immediately.

| Item | Why not |
|---|---|
| Multi-session tracking / trends | Not in project scope |
| Real-time live streaming | Too complex, not needed for demo |
| Mobile app | Web app is sufficient |
| PDF report export | Nice-to-have, not core |
| Body pose / gesture analysis | MediaPipe Pose adds complexity without proven value |
| Custom deep learning / fine-tuning | We use pretrained extractors only |
| Any GPU-dependent feature | Must run on student laptop |
| Diagnostic language | Legal and ethical requirement |

---

## System Architecture

```
Person answers prompts for 5 minutes
                 ↓
           video.mp4 saved
                 ↓
    ┌────────────┴────────────┐
    │                         │
  VIDEO TRACK              AUDIO TRACK
    │                         │
    ▼                    ┌────┴────┐
  Py-Feat              openSMILE  Whisper (base)
  MediaPipe            (eGeMAPS)      │
    │                     │           ▼
    ▼                     ▼       Transcript
  Face features      Audio features   │
  (~20 numbers)      (~30 numbers)    ▼
    │                     │      RoBERTa / MuRIL
    │                     │           │
    │                     │           ▼
    │                     │      Text features
    │                     │      (~10 numbers)
    └─────────┬───────────┴───────────┘
              ▼
    Feature Vector (~90 numbers)
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  Emotion Model     Screening Classifier
  (pretrained)      (our Random Forest)
    │                   │
    ▼                   ▼
  Emotion           Screening
  Timeline          Score
    │                   │
    └─────────┬─────────┘
              ▼
         Dashboard
```

---

## All Models

### Pretrained — download and use as-is (DO NOT train)

| # | Purpose | Model | Library | Download Size | What it outputs |
|---|---|---|---|---|---|
| 1 | Face landmarks, head pose | MediaPipe Face Mesh | `mediapipe` | ~5 MB | 478 landmarks, pitch/yaw/roll |
| 2 | Facial muscle movement | Py-Feat AU detector | `py-feat` | ~150 MB | Action Unit intensities (AU01–AU45) |
| 3 | Emotion classification | HSEmotion / FER+ | pretrained | ~50 MB | 7 emotion probabilities per frame |
| 4 | Acoustic features | openSMILE eGeMAPS v02 | `opensmile` | ~2 MB | 88 acoustic descriptors (pitch, jitter, shimmer, loudness) |
| 5 | Speech → text | Whisper base | `openai-whisper` | ~140 MB | Transcript + detected language |
| 6 | English sentiment | RoBERTa-base-sentiment | `transformers` | ~500 MB | positive / negative / neutral scores |
| 7 | Hindi/Marathi sentiment | MuRIL | `transformers` | ~900 MB | Sentiment scores for Indic languages |

### Trained by us — the ONLY model we train

| # | Purpose | Algorithm | Library | Training time | Input |
|---|---|---|---|---|---|
| 8 | Depression/stress screening | Random Forest + XGBoost | `scikit-learn`, `xgboost` | < 1 minute on CPU | ~90 aggregated features from above |

---

## All Datasets

| Dataset | Size | What we use it for | Access | Status |
|---|---|---|---|---|
| **DAIC-WOZ** | 189 interviews, 142 participants | Train screening classifier (PHQ-8 depression scores) | Apply: https://dcapswoz.ict.usc.edu/ | Applied |
| **MELD** | 13,700 clips | Validate emotion timeline (7 emotion labels) | Free | Available |
| **RAVDESS** | 24 actors, 8 emotions | Baseline only — acted data inflates accuracy | Free | Available |
| **HindiSER** | 520 utterances, 26 speakers | Validate pipeline on Indian speakers | On request | Pending |

**Important:** We do NOT record our own training data. DAIC-WOZ provides real clinical interviews. RAVDESS results must never be headlined — acted data gives inflated accuracy.

---

## All Features We Extract (~90 total)

### Audio Features (~30) — `ml/extractors/audio.py`

| Feature | Key | What it measures | Direction in depression |
|---|---|---|---|
| Pause count | `audio_pause_count` | How many silent gaps | ↑ Increased |
| Mean pause duration | `audio_pause_mean_duration` | Average silence length | ↑ Increased |
| Pause-to-speech ratio | `audio_pause_to_speech_ratio` | Silence vs speech time | ↑ Increased |
| Speech duration | `audio_speech_duration` | Total speaking time | ↓ Decreased |
| Total duration | `audio_total_duration` | Recording length | — |
| Speech rate | `audio_speech_rate_wpm` | Words per minute | ↓ Decreased |
| Pitch mean | `audio_f0_mean` | Average voice pitch | ↓ Decreased |
| Pitch std | `audio_f0_std` | Pitch variability | ↓ Flatter / monotone |
| Pitch range | `audio_f0_range` | Highest minus lowest pitch | ↓ Narrower |
| Loudness mean | `audio_loudness_mean` | Average volume | ↓ Decreased |
| Loudness std | `audio_loudness_std` | Volume variability | ↓ Decreased |
| Jitter | `audio_jitter` | Voice pitch instability | ↑ Increased |
| Shimmer | `audio_shimmer` | Voice amplitude instability | ↑ Increased |
| MFCC 1–13 mean | `audio_mfcc_01_mean` … `audio_mfcc_13_mean` | Spectral shape averages | Varies |
| MFCC 1–13 std | `audio_mfcc_01_std` … `audio_mfcc_13_std` | Spectral shape variability | Varies |

### Face Features (~20) — `ml/extractors/face.py`

| Feature | Key | What it measures | Direction in depression |
|---|---|---|---|
| AU04 mean/std | `face_au04_mean`, `face_au04_std` | Brow lowerer | ↑ More furrowing |
| AU06 mean/std | `face_au06_mean`, `face_au06_std` | Cheek raiser (part of smile) | ↓ Less smiling |
| AU12 mean/std | `face_au12_mean`, `face_au12_std` | Lip corner puller (smile) | ↓ Less smiling |
| AU15 mean/std | `face_au15_mean`, `face_au15_std` | Lip corner depressor | ↑ More frowning |
| Total AU activity | `face_au_total_activity` | Overall expressivity | ↓ Blunted affect |
| Head movement velocity | `face_head_movement_velocity` | How much the head moves | ↓ Less movement |
| Blink rate | `face_blink_rate_per_min` | Blinks per minute | Varies |
| Gaze direction variance | `face_gaze_direction_variance` | Eye movement variability | ↓ Less engagement |
| Face detection rate | `face_detection_rate` | % of frames with detected face | Quality metric |
| Quality warning | `face_quality_warning` | True if detection rate < 50% | Quality metric |

### Text Features (~10) — `ml/extractors/text.py`

| Feature | Key | What it measures | Direction in depression |
|---|---|---|---|
| Word count | `text_word_count` | Total words spoken | ↓ Decreased |
| First-person pronoun ratio | `text_first_person_pronoun_ratio` | I/me/my/mine/myself usage | ↑ Increased self-focus |
| Positive sentiment | `text_sentiment_positive` | Positive tone score | ↓ Decreased |
| Negative sentiment | `text_sentiment_negative` | Negative tone score | ↑ Increased |
| Neutral sentiment | `text_sentiment_neutral` | Neutral tone score | Varies |
| Mean utterance length | `text_mean_utterance_length` | Avg words per sentence | ↓ Shorter sentences |
| Lexical diversity | `text_lexical_diversity` | Unique words / total words | ↓ Decreased |
| Speaking time | `text_speaking_time_seconds` | Total speech time | ↓ Decreased |
| Language | `text_language` | Detected language code | Routing metric |

---

## Build Phases — Progress Tracker

### Legend
- `[x]` = Done, tested, committed
- `[ ]` = Not started
- `[~]` = In progress

---

### Phase 1: Environment Setup ✅ COMPLETE
> **Goal:** Every team member can run the project on their laptop.
> **Completed:** 2026-10-30

- [x] 1.1 — Python 3.10+ installed and verified (we have 3.11.9)
- [x] 1.2 — `requirements.txt` with pinned versions, installable with pip
- [x] 1.3 — ffmpeg installed and on PATH (v9.0.2)
- [x] 1.4 — `config.yaml` loads via `ml/config.py` helper
- [x] 1.5 — Test fixtures created: `test_audio.wav` (sine + silence), `test_silent.wav`, `test_video.avi`

**Files created/modified:**
- `ml/config.py` — Config loader with caching
- `tests/generate_fixtures.py` — Generates synthetic test files
- `tests/fixtures/` — Test audio and video files
- `.gitignore` — Updated with `.avi`
- `requirements.txt` — Added `sqlalchemy`, `httpx`, `joblib`, `soundfile`

---

### Phase 2: Audio Extractor ✅ COMPLETE
> **Goal:** `ml/extractors/audio.py` → takes `.wav` or `.mp4`, returns ~30 features prefixed `audio_`
> **Completed:** 2026-10-30

- [x] 2.1 — Load audio from .wav or .mp4 (extracts audio from video via ffmpeg if needed)
- [x] 2.2 — Energy-based VAD: short-time RMS energy → speech vs silence segments
- [x] 2.3 — Pause features: count, mean duration, pause-to-speech ratio
- [x] 2.4 — Speech rate in words/min (optional transcript argument)
- [x] 2.5 — Pitch (F0): mean, std, range using `librosa.pyin()`
- [x] 2.6 — Loudness: RMS energy mean and std
- [x] 2.7 — Jitter and shimmer via openSMILE eGeMAPS
- [x] 2.8 — MFCC 1–13: mean and std of each coefficient (26 features)
- [x] 2.9 — All keys prefixed `audio_`, returns flat dict
- [x] 2.10 — Edge cases: silent audio → None, short clips → None, missing file → None + log
- [x] 2.11 — 4 tests pass: sine wave, silent audio, missing file, with transcript

**Files created:**
- `ml/extractors/audio.py` — Full implementation (~200 lines)
- `tests/test_audio.py` — 4 test cases

**How it works internally:**
1. `_load_audio()` — loads .wav directly or extracts audio from video via ffmpeg
2. `_compute_vad_segments()` — energy-based voice activity detection
3. `_pause_features()` — counts and measures pauses
4. `_pitch_features()` — F0 via librosa pyin
5. `_loudness_features()` — RMS energy
6. `_opensmile_features()` — jitter/shimmer via eGeMAPS
7. `_mfcc_features()` — 13 MFCCs, mean and std each
8. `extract()` — orchestrates all of the above, returns flat dict or None

---

### Phase 3: Face Extractor ✅ COMPLETE
> **Goal:** `ml/extractors/face.py` → takes video file, returns ~14 features prefixed `face_`
> **Completed:** 2026-10-31

- [x] 3.1 — Load video with OpenCV, sample every 3rd frame (`config.yaml: face.frame_skip`)
- [x] 3.2 — Py-Feat v2: run Detectorv2 on each sampled frame via temp image files
- [x] 3.3 — AU summary stats: mean and std for AU04, AU06, AU12, AU15
- [x] 3.4 — Total AU activity: mean of sum of all AU intensities across frames
- [x] 3.5 — MediaPipe FaceLandmarker (Tasks API v1.2): run on sampled frames
- [x] 3.6 — Head pose estimation: pitch/yaw/roll from landmarks via `cv2.solvePnP`
- [x] 3.7 — Head movement velocity: mean per-frame change in pose angles
- [x] 3.8 — Blink detection: eye aspect ratio from landmarks → blinks per minute
- [x] 3.9 — Gaze direction: iris landmarks variance
- [x] 3.10 — Face detection rate + quality warning if < 0.5
- [x] 3.11 — Edge cases: missing file, unreadable video → return None + log
- [x] 3.12 — 3 tests pass: synthetic video, missing file, feature count

**Files created:**
- `ml/extractors/face.py` — Full implementation (~250 lines)
- `tests/test_face.py` — 3 test cases

**Technical notes:**
- Py-Feat v2 uses `Detectorv2.detect()` with image-based detection (bypasses torchcodec)
- MediaPipe v1.2 uses Tasks API (`FaceLandmarker`), not deprecated `mp.solutions`
- FaceLandmarker model auto-downloads on first run (~3.7 MB)
- Model file gitignored at `data/models/*.task`

---

### Phase 4: Text Extractor ✅ COMPLETE
> **Goal:** `ml/extractors/text.py` → takes audio file, returns ~9 features prefixed `text_`
> **Completed:** 2026-10-31

- [x] 4.1 — Whisper transcription: load base model, transcribe audio file (fp16=False for CPU)
- [x] 4.2 — Cache transcript to `data/transcripts/{filename}.json`
- [x] 4.3 — Language detection from Whisper → route to RoBERTa (English) or MuRIL (Indic)
- [x] 4.4 — Sentiment analysis: positive/negative/neutral scores
- [x] 4.5 — First-person pronoun ratio
- [x] 4.6 — Mean utterance length, lexical diversity, word count
- [x] 4.7 — Speaking time from Whisper segments
- [x] 4.8 — Language encoded as float (0=en, 1=hi, 2=mr, 3=other)
- [x] 4.9 — Edge cases: no speech, empty transcript → return None + log
- [x] 4.10 — 5 tests pass: linguistic features, empty text, missing file, caching, extraction

**Files created:**
- `ml/extractors/text.py` — Full implementation (~250 lines)
- `tests/test_text.py` — 5 test cases

**Technical notes:**
- Whisper base model (~140 MB) auto-downloads on first run, cached in `~/.cache/whisper`
- Sentiment model lazy-loaded and cached in memory — not reloaded per call
- Transcript caching prevents re-running Whisper (saves ~30s per file)
- Handles video files by extracting audio via ffmpeg first

---

### Phase 5: Pipeline Runner ⬜ NOT STARTED
> **Goal:** `ml/pipeline.py` — video in → feature vector out. Batch mode for directories.
> **Dependencies:** Phase 2 + Phase 3 + Phase 4 must be complete

**Tasks:**
- [ ] 5.1 — Split video into video + audio tracks using ffmpeg subprocess
- [ ] 5.2 — `process_session(video_path)` → call all 3 extractors, merge dicts
- [ ] 5.3 — Add metadata: session_id, duration, timestamp, list of warnings
- [ ] 5.4 — Handle partial failures: if one extractor returns None, return others + warning
- [ ] 5.5 — `scripts/extract_batch.py` CLI: `--input-dir`, `--output-csv`
- [ ] 5.6 — Resumable batch: skip session_ids already in output CSV
- [ ] 5.7 — Progress bar with tqdm, error log to `data/extraction_errors.log`
- [ ] 5.8 — Test: process test video end-to-end, verify CSV output has all ~90 columns

**Technical notes:**
- ffmpeg command: `ffmpeg -y -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav`
- Temp files are cleaned up after processing
- Extractors run sequentially (not parallel) to avoid RAM overflow on 8 GB laptops
- CSV uses session filename as session_id

---

### Phase 6: Training & Evaluation ⬜ NOT STARTED
> **Goal:** Train Random Forest + XGBoost on DAIC-WOZ features, evaluate properly.
> **Dependencies:** Phase 5 must be complete. DAIC-WOZ access required for real data.

**Tasks:**
- [ ] 6.1 — Load features CSV + labels CSV, join on `session_id`
- [ ] 6.2 — Participant-level split: group by `participant_id` so same person never in train AND test
- [ ] 6.3 — 5-fold stratified cross-validation using `StratifiedGroupKFold`
- [ ] 6.4 — Train `RandomForestClassifier` with default hyperparameters
- [ ] 6.5 — Train `XGBClassifier` with default hyperparameters
- [ ] 6.6 — Report metrics: accuracy, macro-F1, AUC — each with 95% bootstrap confidence intervals
- [ ] 6.7 — Print confusion matrix to console
- [ ] 6.8 — Save feature importances to `data/models/feature_importances.csv`
- [ ] 6.9 — Save best model to `data/models/screening_model_YYYYMMDD.joblib`
- [ ] 6.10 — `ml/evaluate.py`: load saved model → ROC curve, confusion matrix plot, top-20 feature bar chart
- [ ] 6.11 — `ml/fusion.py`: weighted average of 3 modality scores (configurable weights)

**Technical notes:**
- `StratifiedGroupKFold` ensures participant-level split AND class balance
- Bootstrap CI: resample predictions 1000 times, compute metric each time, take 2.5th and 97.5th percentiles
- Save both RF and XGB, report which one performs better
- Fusion weights default: face=0.4, audio=0.35, text=0.25 (tunable)

---

### Phase 7: Backend API ⬜ NOT STARTED
> **Goal:** FastAPI server that accepts video uploads, runs pipeline, returns results.
> **Dependencies:** Phase 5 must be complete

**Tasks:**
- [ ] 7.1 — Database schema: `users`, `sessions`, `features` tables in SQLite (Postgres-swappable)
- [ ] 7.2 — SQLAlchemy models in `backend/models/database.py`
- [ ] 7.3 — `POST /users` — register participant with consent flag
- [ ] 7.4 — `DELETE /users/{id}` — delete user and ALL their data (consent withdrawal, DPDP Act)
- [ ] 7.5 — `POST /sessions` — upload video file, return session_id immediately
- [ ] 7.6 — Background task: run `process_session()`, store features in DB, delete raw video
- [ ] 7.7 — `GET /sessions/{id}` — return features, emotion timeline, screening score
- [ ] 7.8 — `GET /users/{id}/sessions` — list all sessions for a user with scores
- [ ] 7.9 — `GET /sessions/{id}/status` — return processing status: pending/processing/complete/failed
- [ ] 7.10 — CORS setup: allow frontend origin
- [ ] 7.11 — Error handling: proper HTTP status codes, structured error responses
- [ ] 7.12 — Tests: test each endpoint with `httpx` test client

**Technical notes:**
- SQLite for development, schema designed so Postgres is a one-line swap in `config.yaml`
- Raw video is deleted after feature extraction unless `config.backend.keep_raw = true`
- Background task uses FastAPI's `BackgroundTasks` — no Celery needed
- All timestamps in UTC ISO 8601

---

### Phase 8: Frontend Dashboard ⬜ NOT STARTED
> **Goal:** React app with 4 pages. Clean, readable, no animations.
> **Dependencies:** Phase 7 must be complete (needs API endpoints)

**Tasks:**
- [ ] 8.1 — Project setup: React + react-router-dom + axios + recharts
- [ ] 8.2 — **Consent page**: informed consent form with checkboxes, must complete before recording
- [ ] 8.3 — **Record page**: webcam capture using MediaRecorder API, on-screen prompts, countdown timer
- [ ] 8.4 — Upload flow: POST video to `/sessions`, poll `/sessions/{id}/status`, redirect on completion
- [ ] 8.5 — **Session Result page**: emotion timeline as horizontal band chart, screening score as gauge
- [ ] 8.6 — Feature breakdown table: extracted features grouped by modality (audio/face/text)
- [ ] 8.7 — **History page**: line chart of scores across past sessions for one user
- [ ] 8.8 — Disclaimer text on EVERY page that shows a score (exact text below)
- [ ] 8.9 — Navigation bar + routing between pages
- [ ] 8.10 — Clean, readable styling — no animations per spec

**Required disclaimer (exact text, every score page):**
```
"This is a screening indicator, not a diagnosis.
 Please consult a qualified mental health professional."
```

---

### Phase 9: Integration & Testing ⬜ NOT STARTED
> **Goal:** Everything works end-to-end. No bugs, no data leaks.

- [ ] 9.1 — End-to-end test: record video → upload → pipeline runs → dashboard shows results
- [ ] 9.2 — Unit tests for each extractor with synthetic clips
- [ ] 9.3 — Profile pipeline: time each stage for a 5-minute video, identify bottleneck
- [ ] 9.4 — Memory check: peak usage stays under 8 GB
- [ ] 9.5 — Privacy audit: no raw video stored after processing, no PII in logs, no data in git
- [ ] 9.6 — Edge cases: no face detected, silent audio, non-English speech, very short clips

---

### Phase 10: Evaluation & Report ⬜ NOT STARTED
> **Goal:** Real numbers for the final report.

- [ ] 10.1 — Obtain DAIC-WOZ dataset access
- [ ] 10.2 — Run pipeline on all 189 DAIC-WOZ interviews → extract features
- [ ] 10.3 — Train and evaluate screening classifier → report accuracy, F1, AUC with CIs
- [ ] 10.4 — Run emotion model on MELD → report accuracy, macro-F1
- [ ] 10.5 — Test on Indian speakers → compare with English results, report drop
- [ ] 10.6 — Fill in README results tables (all "—" entries)
- [ ] 10.7 — Generate plots: ROC curve, confusion matrix, feature importance bar chart

**Target numbers:**

| Metric | Target | Dataset |
|---|---|---|
| Emotion accuracy | 70%+ | MELD |
| Depression screening accuracy | 80%+ | DAIC-WOZ |
| Depression screening AUC | 0.75+ | DAIC-WOZ |
| Processing time per 5-min session | < 30 seconds | Local |
| Peak memory | < 8 GB | Local |
| Works on Indian speakers | Yes, with reported numbers | Custom / HindiSER |

---

## Overall Progress

| Phase | Status | Date Completed |
|---|---|---|
| Phase 1: Environment Setup | ✅ Complete | 2026-10-30 |
| Phase 2: Audio Extractor | ✅ Complete | 2026-10-30 |
| Phase 3: Face Extractor | ✅ Complete | 2026-10-31 |
| Phase 4: Text Extractor | ✅ Complete | 2026-10-31 |
| Phase 5: Pipeline Runner | ⬜ Not Started | — |
| Phase 6: Training & Evaluation | ⬜ Not Started | — |
| Phase 7: Backend API | ⬜ Not Started | — |
| Phase 8: Frontend Dashboard | ⬜ Not Started | — |
| Phase 9: Integration Testing | ⬜ Not Started | — |
| Phase 10: Evaluation & Report | ⬜ Not Started | — |

---

## Timeline

```
Week 1       Phase 1 (Setup) + apply for DAIC-WOZ          ✅ Done
Week 2–3     Phase 2 (Audio) + Phase 3 (Face) + Phase 4 (Text)  ← parallel
Week 4       Phase 5 (Pipeline runner)
Week 5       Phase 6 (Training logic)
Week 6–7     Phase 7 (Backend) + Phase 8 (Frontend)  ← parallel
Week 8       Phase 9 (Integration testing)
Week 9–10    Phase 10 (Evaluation on real data + report)
```

## Phase Dependencies

```
Phase 2 (Audio) ──┐
Phase 3 (Face)  ──┼── Phase 5 (Pipeline) ── Phase 6 (Training)
Phase 4 (Text)  ──┘         │
                             └── Phase 7 (Backend) ── Phase 8 (Frontend)
                                                            │
                                              Phase 9 (Integration) ── Phase 10 (Report)
```

- Phases 2, 3, 4 can run **in parallel** (assign to different team members)
- Phases 7 and 8 can run **in parallel** once Phase 5 works

---

## Risks & Mitigations

| # | Risk | Impact | Mitigation | Owner |
|---|---|---|---|---|
| R1 | DAIC-WOZ access delayed | Cannot train screening classifier | Build emotion part first; use synthetic labels to test pipeline | Team |
| R2 | Py-Feat too slow on CPU | Face extraction > 2 min for 5-min video | Process every 3rd frame; fall back to MediaPipe blendshapes | Phase 3 |
| R3 | Whisper inaccurate on Indian English | Wrong transcripts → bad text features | Try whisper small model; allow manual transcript override | Phase 4 |
| R4 | MuRIL has no sentiment classification head | Hindi sentiment scoring fails | Fine-tune a tiny classification head on available Hindi sentiment data | Phase 4 |
| R5 | RAM overflow on 8 GB laptop | OOM crashes | Load one model at a time; unload after use; run extractors sequentially | Phase 5 |
| R6 | Team members blocked on dependencies | Cascading delays | Phases 2/3/4 are independent — assign to different people | Team |

---

## Rules During Development

1. **Check this file before starting any task.** If the task isn't listed here, don't do it.
2. **No scope creep.** Cool feature ideas go in a "maybe later" list, not into code.
3. **Test each phase before moving on.** A broken extractor breaks everything downstream.
4. **Commit after every working phase.** Tag it so you can roll back.
5. **Never say "diagnosis."** Always say "screening indicator."
6. **Never commit data files.** Run `git status` before every push. Check `.gitignore`.
7. **CPU only.** If your code needs a GPU to run, you are doing it wrong.
8. **Mark tasks complete in this file** when they are tested and pushed.
