"""Audio feature extractor. Extracts session-level vocal features from audio/video files."""

import logging

logger = logging.getLogger(__name__)


def extract(path: str, transcript: str | None = None) -> dict | None:
    """Extract audio features from a .wav or .mp4 file.

    Args:
        path: Path to audio/video file.
        transcript: Optional transcript text for speech rate calculation.

    Returns:
        Dict of features prefixed with 'audio_', or None on failure.
    """
    raise NotImplementedError
