"""Audio feature extractor. Extracts session-level vocal features from audio/video files."""

import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import opensmile

from ml.config import get as cfg

logger = logging.getLogger(__name__)

# Minimum duration in seconds to attempt extraction
_MIN_DURATION = 1.0


def _load_audio(path: str) -> tuple[np.ndarray, int] | None:
    """Load audio from .wav or .mp4 file. Returns (signal, sample_rate) or None."""
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        return None

    sr = cfg("audio", "sample_rate")

    # If mp4/video, extract audio to temp wav first
    if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(sr), "-ac", "1", tmp.name],
                capture_output=True, check=True,
            )
            y, sr_out = librosa.load(tmp.name, sr=sr)
            Path(tmp.name).unlink(missing_ok=True)
            return y, sr_out
        except (subprocess.CalledProcessError, Exception) as e:
            logger.error("Failed to extract audio from video %s: %s", path, e)
            Path(tmp.name).unlink(missing_ok=True)
            return None

    # Direct wav/audio load
    try:
        y, sr_out = librosa.load(str(p), sr=sr)
        return y, sr_out
    except Exception as e:
        logger.error("Failed to load audio %s: %s", path, e)
        return None


def _compute_vad_segments(y: np.ndarray, sr: int) -> list[dict]:
    """Energy-based VAD. Returns list of {'type': 'speech'|'pause', 'start': float, 'end': float}."""
    threshold = cfg("audio", "energy_threshold")
    min_pause = cfg("audio", "min_pause_duration")

    # Short-time energy with 25ms frames, 10ms hop
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Binary speech/silence decision
    is_speech = energy > threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

    segments = []
    current_type = "speech" if is_speech[0] else "pause"
    current_start = 0.0

    for i in range(1, len(is_speech)):
        frame_type = "speech" if is_speech[i] else "pause"
        if frame_type != current_type:
            segments.append({"type": current_type, "start": current_start, "end": times[i]})
            current_type = frame_type
            current_start = times[i]

    # Final segment
    segments.append({"type": current_type, "start": current_start, "end": times[-1]})

    # Merge short pauses (< min_pause_duration) into speech
    merged = []
    for seg in segments:
        if seg["type"] == "pause" and (seg["end"] - seg["start"]) < min_pause:
            if merged and merged[-1]["type"] == "speech":
                merged[-1]["end"] = seg["end"]
            else:
                seg["type"] = "speech"
                merged.append(seg)
        else:
            # Merge adjacent same-type segments
            if merged and merged[-1]["type"] == seg["type"]:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg)

    return merged


def _pause_features(segments: list[dict]) -> dict[str, float]:
    """Extract pause-related features from VAD segments."""
    pauses = [s for s in segments if s["type"] == "pause"]
    speech = [s for s in segments if s["type"] == "speech"]

    pause_durations = [p["end"] - p["start"] for p in pauses]
    speech_duration = sum(s["end"] - s["start"] for s in speech)
    total_duration = sum(s["end"] - s["start"] for s in segments)

    pause_total = sum(pause_durations) if pause_durations else 0.0

    return {
        "audio_pause_count": float(len(pauses)),
        "audio_pause_mean_duration": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "audio_pause_to_speech_ratio": pause_total / speech_duration if speech_duration > 0 else 0.0,
        "audio_speech_duration": speech_duration,
        "audio_total_duration": total_duration,
    }


def _speech_rate(transcript: str | None, speech_duration: float) -> dict[str, float]:
    """Compute words per minute from transcript and speech duration."""
    if transcript is None or speech_duration <= 0:
        return {"audio_speech_rate_wpm": 0.0}

    word_count = len(transcript.split())
    wpm = word_count / (speech_duration / 60.0)
    return {"audio_speech_rate_wpm": float(wpm)}


def _pitch_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract F0 (pitch) features using librosa pyin."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

        if len(f0_voiced) == 0:
            return {"audio_f0_mean": 0.0, "audio_f0_std": 0.0, "audio_f0_range": 0.0}

        return {
            "audio_f0_mean": float(np.mean(f0_voiced)),
            "audio_f0_std": float(np.std(f0_voiced)),
            "audio_f0_range": float(np.ptp(f0_voiced)),
        }
    except Exception as e:
        logger.warning("Pitch extraction failed: %s", e)
        return {"audio_f0_mean": 0.0, "audio_f0_std": 0.0, "audio_f0_range": 0.0}


def _loudness_features(y: np.ndarray) -> dict[str, float]:
    """Extract RMS loudness features."""
    rms = librosa.feature.rms(y=y)[0]
    return {
        "audio_loudness_mean": float(np.mean(rms)),
        "audio_loudness_std": float(np.std(rms)),
    }


def _opensmile_features(path: str) -> dict[str, float]:
    """Extract jitter and shimmer using openSMILE eGeMAPS."""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(path)

        features = {}

        # Map openSMILE feature names to our naming convention
        jitter_cols = [c for c in df.columns if "jitter" in c.lower()]
        shimmer_cols = [c for c in df.columns if "shimmer" in c.lower()]

        if jitter_cols:
            features["audio_jitter"] = float(df[jitter_cols[0]].iloc[0])
        else:
            features["audio_jitter"] = 0.0

        if shimmer_cols:
            features["audio_shimmer"] = float(df[shimmer_cols[0]].iloc[0])
        else:
            features["audio_shimmer"] = 0.0

        return features
    except Exception as e:
        logger.warning("openSMILE extraction failed: %s", e)
        return {"audio_jitter": 0.0, "audio_shimmer": 0.0}


def _mfcc_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract MFCC 1-13 mean and std."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = {}
    for i in range(13):
        idx = f"{i + 1:02d}"
        features[f"audio_mfcc_{idx}_mean"] = float(np.mean(mfccs[i]))
        features[f"audio_mfcc_{idx}_std"] = float(np.std(mfccs[i]))
    return features


def extract(path: str, transcript: str | None = None) -> dict[str, float] | None:
    """Extract audio features from a .wav or .mp4 file.

    Args:
        path: Path to audio/video file.
        transcript: Optional transcript text for speech rate calculation.

    Returns:
        Dict of features prefixed with 'audio_', or None on failure.
    """
    result = _load_audio(path)
    if result is None:
        return None

    y, sr = result

    # Check minimum duration
    duration = len(y) / sr
    if duration < _MIN_DURATION:
        logger.warning("Audio too short (%.2fs < %.2fs): %s", duration, _MIN_DURATION, path)
        return None

    # Check if audio is silent
    if np.max(np.abs(y)) < 1e-6:
        logger.warning("Audio is silent: %s", path)
        return None

    features = {}

    # VAD and pause features
    segments = _compute_vad_segments(y, sr)
    pause_feats = _pause_features(segments)
    features.update(pause_feats)

    # Speech rate
    features.update(_speech_rate(transcript, pause_feats["audio_speech_duration"]))

    # Pitch
    features.update(_pitch_features(y, sr))

    # Loudness
    features.update(_loudness_features(y))

    # Jitter and shimmer via openSMILE
    features.update(_opensmile_features(path))

    # MFCCs
    features.update(_mfcc_features(y, sr))

    logger.info("Extracted %d audio features from %s", len(features), path)
    return features
