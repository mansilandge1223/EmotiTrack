"""FastAPI entry point for EmotiTrack demo dashboard."""

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="EmotiTrack", version="0.1.0")

# CORS — allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (good enough for demo)
sessions: dict[str, dict] = {}

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _run_pipeline(session_id: str, video_path: str):
    """Run the extraction pipeline in a background thread, updating status at each step."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    try:
        import subprocess
        import tempfile
        from ml.config import get as cfg

        session = sessions[session_id]

        # Step 1: Split audio
        session["current_step"] = 1
        session["step_label"] = "Splitting video into audio + video tracks..."

        audio_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sr = cfg("audio", "sample_rate")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(sr), "-ac", "1", tmp.name],
                capture_output=True, check=True,
            )
            audio_path = tmp.name
        except Exception as e:
            session["warnings"].append(f"Audio split failed: {e}")

        # Step 2: Audio extraction
        session["current_step"] = 2
        session["step_label"] = "Extracting audio features (pitch, pauses, loudness)..."

        audio_features = {}
        if audio_path:
            try:
                from ml.extractors.audio import extract as extract_audio
                result = extract_audio(audio_path)
                if result:
                    audio_features = result
                    session["steps_done"].append({
                        "name": "Audio", "count": len(result), "status": "done"
                    })
                else:
                    session["warnings"].append("Audio extractor returned None")
                    session["steps_done"].append({
                        "name": "Audio", "count": 0, "status": "failed"
                    })
            except Exception as e:
                session["warnings"].append(f"Audio error: {e}")
                session["steps_done"].append({
                    "name": "Audio", "count": 0, "status": "failed"
                })

        # Step 3: Face extraction
        session["current_step"] = 3
        session["step_label"] = "Extracting face features (Action Units, blinks, gaze)..."

        face_features = {}
        try:
            from ml.extractors.face import extract as extract_face
            result = extract_face(video_path)
            if result:
                face_features = result
                session["steps_done"].append({
                    "name": "Face", "count": len(result), "status": "done"
                })
            else:
                session["warnings"].append("Face extractor returned None")
                session["steps_done"].append({
                    "name": "Face", "count": 0, "status": "failed"
                })
        except Exception as e:
            session["warnings"].append(f"Face error: {e}")
            session["steps_done"].append({
                "name": "Face", "count": 0, "status": "failed"
            })

        # Step 4: Text extraction
        session["current_step"] = 4
        session["step_label"] = "Transcribing speech and analyzing sentiment..."

        text_features = {}
        if audio_path:
            try:
                from ml.extractors.text import extract as extract_text
                result = extract_text(audio_path)
                if result:
                    text_features = result
                    session["steps_done"].append({
                        "name": "Text", "count": len(result), "status": "done"
                    })
                else:
                    session["warnings"].append("Text extractor returned None (no speech?)")
                    session["steps_done"].append({
                        "name": "Text", "count": 0, "status": "failed"
                    })
            except Exception as e:
                session["warnings"].append(f"Text error: {e}")
                session["steps_done"].append({
                    "name": "Text", "count": 0, "status": "failed"
                })

        # Clean up temp audio
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)

        # Merge all features
        all_features = {}
        all_features.update(audio_features)
        all_features.update(face_features)
        all_features.update(text_features)

        session["features"] = all_features
        session["status"] = "complete"
        session["current_step"] = 5
        session["step_label"] = "Done!"
        session["total_features"] = len(all_features)

    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["step_label"] = f"Error: {e}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and start processing."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    session_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    # Save uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize session
    sessions[session_id] = {
        "status": "processing",
        "current_step": 0,
        "total_steps": 4,
        "step_label": "Starting pipeline...",
        "steps_done": [],
        "features": {},
        "warnings": [],
        "total_features": 0,
        "filename": file.filename,
    }

    # Run pipeline in background thread
    thread = threading.Thread(target=_run_pipeline, args=(session_id, str(file_path)))
    thread.start()

    return {"session_id": session_id}


@app.get("/status/{session_id}")
def get_status(session_id: str):
    """Get processing status for a session."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    s = sessions[session_id]
    return {
        "session_id": session_id,
        "status": s["status"],
        "current_step": s["current_step"],
        "total_steps": s["total_steps"],
        "step_label": s["step_label"],
        "steps_done": s["steps_done"],
        "total_features": s["total_features"],
        "warnings": s["warnings"],
        "filename": s["filename"],
    }


@app.get("/results/{session_id}")
def get_results(session_id: str):
    """Get extraction results for a completed session."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    s = sessions[session_id]
    if s["status"] != "complete":
        raise HTTPException(400, f"Session not complete. Status: {s['status']}")

    features = s["features"]

    # Group features by category
    audio = {k: v for k, v in features.items() if k.startswith("audio_")}
    face = {k: v for k, v in features.items() if k.startswith("face_")}
    text = {k: v for k, v in features.items() if k.startswith("text_")}

    # Build summary
    summary = {
        "face_detected": f"{features.get('face_detection_rate', 0) * 100:.0f}%",
        "blink_rate": f"{features.get('face_blink_rate_per_min', 0):.0f}/min",
        "smile": f"{features.get('face_au12_mean', 0):.2f}",
        "frown": f"{features.get('face_au15_mean', 0):.2f}",
        "pitch": f"{features.get('audio_f0_mean', 0):.0f} Hz",
        "pauses": f"{features.get('audio_pause_count', 0):.0f}",
        "speech_duration": f"{features.get('audio_speech_duration', 0):.1f}s",
        "total_duration": f"{features.get('audio_total_duration', 0):.1f}s",
        "words": f"{features.get('text_word_count', 0):.0f}",
        "sentiment_negative": f"{features.get('text_sentiment_negative', 0) * 100:.0f}%",
        "pronoun_ratio": f"{features.get('text_first_person_pronoun_ratio', 0) * 100:.0f}%",
    }

    return {
        "session_id": session_id,
        "filename": s["filename"],
        "total_features": len(features),
        "summary": summary,
        "audio_features": audio,
        "face_features": face,
        "text_features": text,
        "warnings": s["warnings"],
    }
