"""Face feature extractor. Extracts session-level facial features from video files."""

import logging

logger = logging.getLogger(__name__)


def extract(path: str) -> dict | None:
    """Extract facial features from a video file.

    Args:
        path: Path to video file.

    Returns:
        Dict of features prefixed with 'face_', or None on failure.
    """
    raise NotImplementedError
