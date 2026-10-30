"""Tests for the face feature extractor."""

from pathlib import Path

from tests.generate_fixtures import generate_test_video

FIXTURES_DIR = Path(__file__).parent / "fixtures"

EXPECTED_KEYS = {
    "face_au04_mean", "face_au04_std",
    "face_au06_mean", "face_au06_std",
    "face_au12_mean", "face_au12_std",
    "face_au15_mean", "face_au15_std",
    "face_au_total_activity",
    "face_head_movement_velocity",
    "face_blink_rate_per_min",
    "face_gaze_direction_variance",
    "face_detection_rate",
    "face_quality_warning",
}


def _ensure_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    video_path = FIXTURES_DIR / "test_video.avi"
    if not video_path.exists():
        generate_test_video(video_path)


def test_extract_synthetic_video():
    """Test extraction on synthetic video. Face may not be detected (drawn circle),
    but all keys should be present and values should be floats."""
    _ensure_fixtures()
    from ml.extractors.face import extract

    result = extract(str(FIXTURES_DIR / "test_video.avi"))

    assert result is not None, "extract() returned None for valid video"
    assert isinstance(result, dict)

    missing = EXPECTED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    for key, val in result.items():
        assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    # Detection rate should be between 0 and 1
    assert 0.0 <= result["face_detection_rate"] <= 1.0

    # Quality warning should be 0.0 or 1.0
    assert result["face_quality_warning"] in (0.0, 1.0)


def test_extract_missing_file():
    """Test that missing file returns None."""
    from ml.extractors.face import extract

    result = extract("nonexistent_video.avi")
    assert result is None


def test_feature_count():
    """Test that we get exactly 14 features."""
    _ensure_fixtures()
    from ml.extractors.face import extract

    result = extract(str(FIXTURES_DIR / "test_video.avi"))
    assert result is not None
    assert len(result) == len(EXPECTED_KEYS), f"Expected {len(EXPECTED_KEYS)} features, got {len(result)}"


if __name__ == "__main__":
    test_extract_synthetic_video()
    print("PASS: test_extract_synthetic_video")

    test_extract_missing_file()
    print("PASS: test_extract_missing_file")

    test_feature_count()
    print("PASS: test_feature_count")

    print("\nAll face tests passed!")
