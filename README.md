# Emotion and Mental State Detection from Audio and Video

A multimodal system that analyses a short video recording of a person speaking and produces two outputs: an emotion timeline, and a screening score for depression and stress.

> **Disclaimer:** This is a research and screening tool. It is **not** a diagnostic system. It cannot diagnose any mental health condition. Results should never replace assessment by a qualified mental health professional.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Models](#models)
- [Extracted Features](#extracted-features)
- [Project Structure](#project-structure)
- [Results](#results)
- [Limitations](#limitations)
- [Ethics and Privacy](#ethics-and-privacy)
- [Roadmap](#roadmap)
- [References](#references)
- [Team](#team)
- [License](#license)

---

## Overview

A person speaks to a webcam for about five minutes, answering a few fixed prompts. The system splits the recording into video and audio, extracts facial, vocal and linguistic features, and returns:

1. **Emotion timeline** — which emotions appeared, and when
2. **Mental state screening score** — indicators for depression and stress, with an expressivity measure

The project combines three independent signal channels (face, voice, text) using late fusion. All feature extractors are pretrained and used as-is; only the final classifier is trained by us.

### Problem

Mental health screening in India relies almost entirely on self-report questionnaires, which require a clinician's time and are easy to answer dishonestly. Observable behavioural signals — how much the face moves, how often a person pauses, how varied their pitch is — carry information that questionnaires do not.

### Research question

> Can facial, vocal and linguistic features extracted from a short video predict clinically validated depression scores, and do features established on English-speaking datasets transfer to Indian speakers?

---

## Features

- Real-time emotion recognition from webcam (7 emotion classes)
- Facial Action Unit extraction (FACS-based muscle movement measurement)
- Vocal feature extraction (pitch, pause patterns, loudness, voice quality)
- Automatic speech transcription and sentiment analysis
- Multilingual text support for Indian languages
- Multimodal late fusion for the final screening score
- Session history and visual dashboard
- Exportable PDF report
- Features-only storage option (no raw video retained)

---

## How It Works

```
                    Video Recording (5 min)
                            |
              +-------------+-------------+
              |                           |
         Video Track                 Audio Track
              |                           |
              v                  +--------+--------+
      MediaPipe / OpenFace       |                 |
              |              openSMILE          Whisper
              v                  |                 |
      Facial Action Units        v                 v
      Head pose, blink rate  Prosodic         Transcript
              |              features              |
              |                  |                 v
              |                  |           MuRIL / RoBERTa
              |                  |                 |
              |                  |                 v
              |                  |          Sentiment scores
              |                  |                 |
              +------------------+-----------------+
                                 |
                                 v
                         Feature Vector
                                 |
                   +-------------+-------------+
                   |                           |
                   v                           v
          Emotion Classifier          Screening Classifier
           (pretrained)                 (trained by us)
                   |                           |
                   v                           v
            Emotion timeline          Depression / stress score
                   |                           |
                   +-------------+-------------+
                                 |
                                 v
                            Dashboard
```

### Pipeline stages

| Stage | Description |
|---|---|
| 1. Capture | Webcam records the session, prompts shown on screen |
| 2. Split | `ffmpeg` separates video and audio tracks |
| 3. Face analysis | Per-frame Action Unit intensities and head pose |
| 4. Voice analysis | eGeMAPS acoustic feature set over the audio |
| 5. Transcription | Speech converted to text, then scored for sentiment |
| 6. Aggregation | Per-frame values summarised into session-level statistics |
| 7. Fusion | Weighted combination of the three modality scores |
| 8. Output | Emotion timeline and screening score rendered to dashboard |

---

## Tech Stack

**Machine Learning**
- Python 3.10+
- PyTorch
- scikit-learn
- NumPy, Pandas

**Computer Vision**
- OpenCV
- MediaPipe (Face Mesh, Face Landmarker, Pose)
- OpenFace 2.0 / Py-Feat

**Audio**
- openSMILE (eGeMAPS feature set)
- Whisper
- librosa
- ffmpeg

**NLP**
- Transformers (Hugging Face)
- MuRIL / IndicBERT for Indian languages

**Backend**
- FastAPI
- PostgreSQL

**Frontend**
- React
- Recharts

---

## Installation

### Prerequisites

- Python 3.10 or higher
- ffmpeg installed and on PATH
- Webcam and microphone
- 8 GB RAM minimum (16 GB recommended)
- GPU optional — the pipeline runs on CPU

### Setup

```bash
# clone the repository
git clone https://github.com/<your-username>/emotion-mental-state-detection.git
cd emotion-mental-state-detection

# create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# verify ffmpeg
ffmpeg -version
```

### requirements.txt

```
opencv-python>=4.9.0
mediapipe>=0.10.9
opensmile>=2.5.0
openai-whisper>=20231117
torch>=2.2.0
transformers>=4.38.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
librosa>=0.10.1
fastapi>=0.110.0
uvicorn>=0.27.0
matplotlib>=3.8.0
```

### OpenFace (optional, for Action Units)

OpenFace is not pip-installable. Build from source or use the prebuilt binaries:
https://github.com/TadasBaltrusaitis/OpenFace/wiki

Py-Feat is an easier Python-native alternative:

```bash
pip install py-feat
```

---

## Usage

### Record and analyse a session

```bash
python record_session.py --user-id U001 --duration 300
python analyse.py --session data/sessions/U001_20261014.mp4
```

### Analyse an existing video file

```bash
python analyse.py --input path/to/video.mp4 --output results/
```

### Run the web app

```bash
# backend
uvicorn backend.main:app --reload

# frontend (separate terminal)
cd frontend && npm install && npm start
```

Open `http://localhost:3000`.

### Extract features only

```bash
python extract_features.py --input-dir data/raw --output features.csv
```

### Train the screening classifier

```bash
python train.py --features features.csv --labels labels.csv --model random_forest
```

---

## Datasets

| Dataset | Content | Used for | Access |
|---|---|---|---|
| DAIC-WOZ | 189 clinical interviews, 142 participants, PHQ-8 scores | Depression screening model | Application required |
| E-DAIC | Extended DAIC with precomputed features | Depression screening model | Application required |
| MELD | 13,700 multimodal conversation clips, 7 emotions | Emotion model validation | Open |
| RAVDESS | 24 actors, 8 emotions, audio + video | Baseline comparison | Open |
| CREMA-D | 91 actors, 6 emotions | Baseline comparison | Open |
| HindiSER | 520 utterances, 26 Hindi speakers, 4 emotions | Indian-speaker validation | On request |
| Custom | Our own recordings from consenting volunteers | Indian-speaker validation | Internal |

**DAIC-WOZ access:** https://dcapswoz.ict.usc.edu/

> **Important:** accuracies reported on acted datasets such as RAVDESS are substantially inflated relative to spontaneous speech. Published work shows the same approach scoring far higher on acted corpora than on in-the-wild data. Report results on MELD or DAIC-WOZ as the primary numbers.

---

## Models

### Pretrained — used as-is, not trained

| Component | Model | Output |
|---|---|---|
| Face landmarks | MediaPipe Face Landmarker | 478 landmarks, 52 blendshapes |
| Action Units | OpenFace 2.0 / Py-Feat | AU presence and intensity |
| Emotion classification | HSEmotion / FER | 7 emotion probabilities |
| Vocal features | openSMILE eGeMAPS | 88 acoustic descriptors |
| Transcription | Whisper (base or small) | Text transcript |
| Text sentiment (EN) | RoBERTa sentiment | Polarity scores |
| Text sentiment (Indic) | MuRIL / IndicBERT | Polarity scores |
| Body pose | MediaPipe Pose | 33 body landmarks |

### Trained by us

| Component | Algorithm | Input |
|---|---|---|
| Screening classifier | Random Forest / XGBoost | ~90 aggregated features |
| Fusion layer | Weighted late fusion | 3 modality scores |

Training the screening classifier takes under a minute on CPU. No GPU required.

---

## Extracted Features

### Facial

| Feature | Description |
|---|---|
| AU04 | Brow lowerer |
| AU06, AU12 | Cheek raiser, lip corner puller (smile) |
| AU15 | Lip corner depressor |
| AU total intensity | Overall expressivity |
| Head movement velocity | Slow movement is associated with blunted affect |
| Blink rate | Fatigue and arousal indicator |
| Gaze direction variance | Attention and engagement |

### Vocal

| Feature | Direction in depression |
|---|---|
| Pause count | Increased |
| Mean pause duration | Increased |
| Pause-to-speech ratio | Increased |
| Speech rate | Decreased |
| Pitch (F0) variability | Decreased — flatter, more monotone |
| Loudness / intensity | Decreased |
| Jitter, shimmer | Increased instability |
| MFCCs 1–13 | Spectral characteristics |

### Linguistic

| Feature | Description |
|---|---|
| First-person pronoun ratio | Self-focus marker |
| Negative sentiment score | Overall polarity |
| Mean utterance length | Verbal fluency |
| Lexical diversity | Type-token ratio |
| Semantic coherence | Topic consistency across utterances |

---

## Project Structure

```
emotion-mental-state-detection/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── routes/
│   ├── models/                 # database models
│   └── services/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
├── ml/
│   ├── extractors/
│   │   ├── face.py             # MediaPipe / OpenFace wrapper
│   │   ├── audio.py            # openSMILE wrapper
│   │   └── text.py             # Whisper + sentiment
│   ├── fusion.py               # late fusion logic
│   ├── train.py                # trains the screening classifier
│   └── evaluate.py             # metrics and plots
├── data/
│   ├── raw/                    # gitignored
│   ├── features/
│   └── models/                 # saved classifier weights
├── notebooks/
│   ├── 01_feature_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results.ipynb
├── docs/
│   ├── consent_form.pdf
│   └── ethics_approval.pdf
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Results

Fill this in as you obtain results. Report the test set, not the training set.

### Emotion recognition

| Dataset | Model | Accuracy | F1 (macro) |
|---|---|---|---|
| MELD | Face only | — | — |
| MELD | Audio only | — | — |
| MELD | Fusion | — | — |

### Depression screening

| Dataset | Modality | Accuracy | F1 | AUC |
|---|---|---|---|---|
| DAIC-WOZ | Audio only | — | — | — |
| DAIC-WOZ | Face only | — | — | — |
| DAIC-WOZ | Audio + Face + Text | — | — | — |

### Indian-speaker transfer

| Test set | Accuracy | Drop vs. English |
|---|---|---|
| Custom (Indian speakers) | — | — |

### Performance

| Metric | Value |
|---|---|
| Processing time per 5-min session | — |
| Peak memory usage | — |
| Runs on CPU only | Yes |

### Reference points from the literature

| Work | Dataset | Reported |
|---|---|---|
| Speech-based depression detection | DAIC-WOZ | 89.4% per segment, 93.9% per subject |
| wav2vec 2.0 transfer learning | DAIC-WOZ | 79% F1 |
| Computerised facial analysis | Ambulatory video | 80–82% agreement with clinician ratings |

---

## Limitations

1. **Not a diagnostic tool.** Output is a screening indicator only.
2. **Emotion inference is contested.** Facial movements do not map reliably onto emotion categories across people, situations and cultures (Barrett et al., 2019). We report measured muscle activity and treat emotion labels as approximate.
3. **Dataset language bias.** DAIC-WOZ is English and US-based. Transfer to Indian speakers is untested and is an explicit research question in this project.
4. **Small evaluation set.** Clinical datasets in this area are small; accuracy figures carry wide confidence intervals.
5. **Recording conditions matter.** Lighting, camera angle, microphone distance and background noise can shift features more than the underlying state does.
6. **Glasses, facial hair and masks** degrade Action Unit extraction.
7. **Acted-data inflation.** Models trained on acted corpora perform substantially worse on spontaneous speech.
8. **No longitudinal validation.** A single session cannot establish a trend.

---

## Ethics and Privacy

- Written informed consent is obtained from every participant before recording.
- Institutional Ethics Committee approval is required before data collection begins.
- Participants may withdraw at any point and request deletion of their data.
- Default storage mode saves **extracted features only**; raw video is deleted after processing.
- All stored data is encrypted at rest.
- The user interface never displays a diagnosis. It displays an indicator and a suggestion to consult a professional.
- This project does not record psychiatric inpatients or any person unable to give informed consent.

Relevant Indian regulation: the Digital Personal Data Protection Act 2023 and the DPDP Rules 2025 govern the processing of personal data, including verifiable guardian consent where a lawful guardian is appointed. Mental health data is additionally subject to the confidentiality provisions of the Mental Healthcare Act 2017.

---

## Roadmap

- [ ] DAIC-WOZ access approval
- [ ] Feature extraction pipeline complete
- [ ] Emotion module working on live webcam
- [ ] Screening classifier trained and evaluated
- [ ] Indian-speaker validation set collected
- [ ] Web dashboard
- [ ] PDF report export
- [ ] On-device inference (no server upload)
- [ ] Hindi and Marathi transcription support
- [ ] Longitudinal mode for multi-session tracking

---

## References

1. Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements. *Psychological Science in the Public Interest*, 20(1), 1–68. https://journals.sagepub.com/doi/10.1177/1529100619832930

2. Gratch, J. et al. (2014). The Distress Analysis Interview Corpus of human and computer interviews. *LREC*. https://dcapswoz.ict.usc.edu/

3. Cohen, A. S. et al. Ambulatory digital phenotyping of blunted affect and alogia using objective facial and vocal analysis: Proof of concept. *Schizophrenia Research*.

4. Computerized analysis of facial expression reveals objective indices of blunted facial affect. *European Archives of Psychiatry and Clinical Neuroscience*. https://link.springer.com/article/10.1007/s00406-023-01696-6

5. Computer Vision-Based Assessment of Motor Functioning in Schizophrenia. https://pmc.ncbi.nlm.nih.gov/articles/PMC7879301/

6. Improving speech depression detection using transfer learning with wav2vec 2.0 in low-resource environments. *Scientific Reports*. https://www.nature.com/articles/s41598-024-60278-1

7. Sagar, R. et al. (2020). The burden of mental disorders across the states of India: GBD Study 1990–2017. *The Lancet Psychiatry*. https://www.thelancet.com/journals/lanpsy/article/PIIS2215-0366(19)30475-4/fulltext

8. National Mental Health Survey of India 2015-16. NIMHANS, Bengaluru.

9. Baltrusaitis, T. et al. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. *IEEE FG*.

10. Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS). *IEEE Transactions on Affective Computing*.

---

## Team

| Name | Role | GitHub |
|---|---|---|
| | ML pipeline | |
| | Backend | |
| | Frontend | |
| | Data collection and evaluation | |

**Guide:**
**Institution:**
**Academic year:** 2026–27

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Datasets used are subject to their own licences and access agreements. DAIC-WOZ, MELD and other third-party corpora are **not** redistributed in this repository.

---

## Acknowledgements

Built on open-source work from MediaPipe, OpenFace, openSMILE, Whisper and Hugging Face Transformers. DAIC-WOZ was collected by the USC Institute for Creative Technologies.