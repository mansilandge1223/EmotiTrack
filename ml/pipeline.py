"""Pipeline runner. Processes a video session through all extractors.

Takes a video file, splits into audio + video tracks, runs all three
extractors, and merges into a single feature vector.
"""

import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ml.config import get as cfg

logger = logging.getLogger(__name__)


def _split_audio(video_path: str) -> str | None:
    """Extract audio track from video file as a temp .wav."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sr = cfg("audio", "sample_rate")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", str(sr), "-ac", "1", tmp.name],
            capture_output=True, check=True,
        )
        return tmp.name
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg audio extraction failed for %s: %s", video_path, e.stderr.decode()[:200])
        return None
    except Exception as e:
        logger.error("Audio split failed for %s: %s", video_path, e)
        return None


def _get_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def process_session(video_path: str) -> dict | None:
    """Run the full extraction pipeline on a video file.

    Args:
        video_path: Path to the video file (.mp4, .avi, etc.)

    Returns:
        Dict with keys:
            'features': merged feature dict from all extractors
            'metadata': session metadata (duration, timestamp, warnings)
        Or None if the video cannot be processed at all.
    """
    from ml.extractors import audio, face, text

    p = Path(video_path)
    if not p.exists():
        logger.error("Video file not found: %s", video_path)
        return None

    session_id = p.stem
    warnings = []

    logger.info("=== Processing session: %s ===", session_id)

    # Get duration
    duration = _get_duration(video_path)

    # Split audio from video
    audio_path = _split_audio(video_path)
    if audio_path is None:
        warnings.append("Failed to extract audio track")

    features = {}

    # --- Audio extractor ---
    logger.info("[1/3] Running audio extractor...")
    if audio_path:
        try:
            audio_features = audio.extract(audio_path)
            if audio_features is not None:
                features.update(audio_features)
                logger.info("  Audio: %d features extracted", len(audio_features))
            else:
                warnings.append("Audio extractor returned None")
                logger.warning("  Audio: returned None")
        except Exception as e:
            warnings.append(f"Audio extractor error: {e}")
            logger.error("  Audio extractor failed: %s", e)
    else:
        warnings.append("No audio track available")

    # --- Face extractor ---
    logger.info("[2/3] Running face extractor...")
    try:
        face_features = face.extract(video_path)
        if face_features is not None:
            features.update(face_features)
            logger.info("  Face: %d features extracted", len(face_features))
        else:
            warnings.append("Face extractor returned None")
            logger.warning("  Face: returned None")
    except Exception as e:
        warnings.append(f"Face extractor error: {e}")
        logger.error("  Face extractor failed: %s", e)

    # --- Text extractor ---
    logger.info("[3/3] Running text extractor...")
    if audio_path:
        try:
            # Pass audio path for transcription; also pass transcript to audio for speech rate
            text_features = text.extract(audio_path)
            if text_features is not None:
                features.update(text_features)
                logger.info("  Text: %d features extracted", len(text_features))
            else:
                warnings.append("Text extractor returned None (no speech detected?)")
                logger.warning("  Text: returned None")
        except Exception as e:
            warnings.append(f"Text extractor error: {e}")
            logger.error("  Text extractor failed: %s", e)
    else:
        warnings.append("No audio for text extraction")

    # Clean up temp audio
    if audio_path:
        Path(audio_path).unlink(missing_ok=True)

    # If no features at all, return None
    if not features:
        logger.error("All extractors failed for %s", video_path)
        return None

    metadata = {
        "session_id": session_id,
        "source_file": str(p.name),
        "duration_seconds": duration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features),
        "warnings": warnings,
    }

    logger.info("=== Session %s complete: %d features, %d warnings ===",
                session_id, len(features), len(warnings))

    return {"features": features, "metadata": metadata}
