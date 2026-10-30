"""Text feature extractor. Transcribes audio and extracts linguistic features.

Uses Whisper for transcription + language detection, then RoBERTa (English)
or MuRIL (Hindi/Marathi) for sentiment analysis.
"""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ml.config import get as cfg

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_whisper_model = None
_sentiment_pipeline = None
_sentiment_model_name = None

# First-person pronouns (English)
_FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "myself",
                           "we", "us", "our", "ours", "ourselves"}


def _get_whisper_model():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        model_name = cfg("text", "whisper_model")
        logger.info("Loading Whisper '%s' model...", model_name)
        _whisper_model = whisper.load_model(model_name)
        logger.info("Whisper model loaded.")
    return _whisper_model


def _get_sentiment_pipeline(language: str):
    """Lazy-load sentiment pipeline based on detected language."""
    global _sentiment_pipeline, _sentiment_model_name
    from transformers import pipeline as hf_pipeline

    if language in ("hi", "mr", "bn", "ta", "te", "gu", "kn", "ml", "pa", "ur"):
        model_name = cfg("text", "sentiment_model_indic")
    else:
        model_name = cfg("text", "sentiment_model_en")

    # Reload if different model needed
    if _sentiment_pipeline is None or _sentiment_model_name != model_name:
        logger.info("Loading sentiment model: %s", model_name)
        try:
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                truncation=True,
                max_length=512,
            )
            _sentiment_model_name = model_name
        except Exception as e:
            logger.warning("Failed to load sentiment model %s: %s", model_name, e)
            return None

    return _sentiment_pipeline


def _get_cache_path(audio_path: str) -> Path:
    """Get transcript cache file path."""
    cache_dir = Path(cfg("paths", "transcripts_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    return cache_dir / f"{stem}.json"


def _extract_audio_if_needed(path: str) -> str | None:
    """If path is a video file, extract audio to temp wav. Returns audio path."""
    p = Path(path)
    if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            sr = cfg("audio", "sample_rate")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(sr), "-ac", "1", tmp.name],
                capture_output=True, check=True,
            )
            return tmp.name
        except Exception as e:
            logger.error("Failed to extract audio from %s: %s", path, e)
            return None
    return str(p)


def _transcribe(audio_path: str) -> dict | None:
    """Transcribe audio using Whisper. Returns cached result if available.

    Returns dict with keys: 'text', 'language', 'segments'.
    """
    cache_path = _get_cache_path(audio_path)

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            logger.info("Using cached transcript: %s", cache_path)
            return cached
        except (json.JSONDecodeError, Exception):
            pass

    # Transcribe
    model = _get_whisper_model()
    try:
        result = model.transcribe(audio_path, fp16=False)

        transcript_data = {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "en"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        }

        # Cache to disk
        with open(cache_path, "w") as f:
            json.dump(transcript_data, f, indent=2)
        logger.info("Transcript cached: %s", cache_path)

        return transcript_data
    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return None


def _sentiment_features(text: str, language: str) -> dict[str, float]:
    """Extract sentiment scores from text."""
    pipe = _get_sentiment_pipeline(language)

    if pipe is None or not text.strip():
        return {
            "text_sentiment_positive": 0.0,
            "text_sentiment_negative": 0.0,
            "text_sentiment_neutral": 0.0,
        }

    try:
        # Truncate to avoid token limit issues
        truncated = text[:1000]
        results = pipe(truncated)

        scores = {"text_sentiment_positive": 0.0,
                  "text_sentiment_negative": 0.0,
                  "text_sentiment_neutral": 0.0}

        if results:
            for r in results:
                label = r["label"].lower()
                score = float(r["score"])
                if "pos" in label:
                    scores["text_sentiment_positive"] = score
                elif "neg" in label:
                    scores["text_sentiment_negative"] = score
                elif "neu" in label:
                    scores["text_sentiment_neutral"] = score
                else:
                    # Some models use LABEL_0, LABEL_1, LABEL_2
                    # Map based on common conventions
                    if label in ("label_0",):
                        scores["text_sentiment_negative"] = score
                    elif label in ("label_1",):
                        scores["text_sentiment_neutral"] = score
                    elif label in ("label_2",):
                        scores["text_sentiment_positive"] = score

        return scores
    except Exception as e:
        logger.warning("Sentiment analysis failed: %s", e)
        return {
            "text_sentiment_positive": 0.0,
            "text_sentiment_negative": 0.0,
            "text_sentiment_neutral": 0.0,
        }


def _linguistic_features(text: str) -> dict[str, float]:
    """Extract linguistic features from transcript text."""
    if not text.strip():
        return {
            "text_word_count": 0.0,
            "text_first_person_pronoun_ratio": 0.0,
            "text_mean_utterance_length": 0.0,
            "text_lexical_diversity": 0.0,
        }

    # Tokenize (simple whitespace + lowercase)
    words = text.lower().split()
    word_count = len(words)

    if word_count == 0:
        return {
            "text_word_count": 0.0,
            "text_first_person_pronoun_ratio": 0.0,
            "text_mean_utterance_length": 0.0,
            "text_lexical_diversity": 0.0,
        }

    # First-person pronoun ratio
    pronoun_count = sum(1 for w in words if w.strip(".,!?;:'\"") in _FIRST_PERSON_PRONOUNS)
    pronoun_ratio = pronoun_count / word_count

    # Mean utterance length (split on sentence-ending punctuation)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        utterance_lengths = [len(s.split()) for s in sentences]
        mean_utterance = float(np.mean(utterance_lengths))
    else:
        mean_utterance = float(word_count)  # Treat whole text as one utterance

    # Lexical diversity (type-token ratio)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / word_count

    return {
        "text_word_count": float(word_count),
        "text_first_person_pronoun_ratio": float(pronoun_ratio),
        "text_mean_utterance_length": mean_utterance,
        "text_lexical_diversity": float(lexical_diversity),
    }


def _speaking_time(segments: list[dict]) -> float:
    """Calculate total speaking time from transcript segments."""
    if not segments:
        return 0.0
    return sum(seg["end"] - seg["start"] for seg in segments)


def extract(path: str) -> dict[str, float] | None:
    """Transcribe audio and extract text features.

    Args:
        path: Path to audio or video file.

    Returns:
        Dict of features prefixed with 'text_', or None on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        return None

    # Extract audio from video if needed
    audio_path = _extract_audio_if_needed(path)
    if audio_path is None:
        return None

    is_temp = audio_path != str(p)

    try:
        # Transcribe
        transcript_data = _transcribe(audio_path)
        if transcript_data is None:
            return None

        text = transcript_data["text"]
        language = transcript_data["language"]
        segments = transcript_data["segments"]

        # Empty transcript
        if not text.strip():
            logger.warning("Empty transcript from %s", path)
            return None

        features = {}

        # Linguistic features
        features.update(_linguistic_features(text))

        # Sentiment features
        features.update(_sentiment_features(text, language))

        # Speaking time
        features["text_speaking_time_seconds"] = float(_speaking_time(segments))

        # Language (stored as a string, but we encode as float for consistency)
        # 0 = English, 1 = Hindi, 2 = Marathi, 3 = other
        lang_map = {"en": 0.0, "hi": 1.0, "mr": 2.0}
        features["text_language"] = lang_map.get(language, 3.0)

        logger.info("Extracted %d text features from %s (language: %s, words: %d)",
                     len(features), path, language, int(features["text_word_count"]))

        return features
    finally:
        # Clean up temp audio file
        if is_temp:
            Path(audio_path).unlink(missing_ok=True)
