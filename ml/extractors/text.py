"""Text feature extractor. Transcribes audio and extracts linguistic features."""

import logging

logger = logging.getLogger(__name__)


def extract(path: str) -> dict | None:
    """Transcribe audio and extract text features.

    Args:
        path: Path to audio/video file.

    Returns:
        Dict of features prefixed with 'text_', or None on failure.
    """
    raise NotImplementedError
