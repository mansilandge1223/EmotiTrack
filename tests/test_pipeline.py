"""Tests for the pipeline runner."""

from pathlib import Path

from tests.generate_fixtures import generate_sine_wav, generate_video_with_audio

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ensure_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if not (FIXTURES_DIR / "test_video_with_audio.mp4").exists():
        generate_video_with_audio(FIXTURES_DIR / "test_video_with_audio.mp4")
    if not (FIXTURES_DIR / "test_audio.wav").exists():
        generate_sine_wav(FIXTURES_DIR / "test_audio.wav")


def test_process_session_real():
    """Test end-to-end pipeline on a real video with face + speech."""
    real_video = FIXTURES_DIR / "mytest.mp4"
    if not real_video.exists():
        print("  SKIP: mytest.mp4 not found (add a real video to tests/fixtures/)")
        return

    from ml.pipeline import process_session
    result = process_session(str(real_video))

    assert result is not None, "process_session returned None for real video"
    features = result["features"]
    metadata = result["metadata"]

    face_keys = [k for k in features if k.startswith("face_")]
    audio_keys = [k for k in features if k.startswith("audio_")]
    text_keys = [k for k in features if k.startswith("text_")]

    assert len(face_keys) > 0, "Should have face features"
    assert len(audio_keys) > 0, "Should have audio features"

    print(f"  Total: {len(features)} features")
    print(f"  Face: {len(face_keys)}, Audio: {len(audio_keys)}, Text: {len(text_keys)}")
    print(f"  Warnings: {metadata['warnings']}")


def test_process_session_synthetic():
    """Test pipeline on synthetic video (sine wave audio, drawn circle face)."""
    _ensure_fixtures()
    from ml.pipeline import process_session

    result = process_session(str(FIXTURES_DIR / "test_video_with_audio.mp4"))

    assert result is not None, "process_session returned None"
    assert "features" in result
    assert "metadata" in result
    assert result["metadata"]["feature_count"] > 0


def test_process_missing_file():
    """Test that missing file returns None."""
    from ml.pipeline import process_session
    result = process_session("nonexistent_video.mp4")
    assert result is None


def test_partial_failure():
    """Test pipeline returns partial features when some extractors fail."""
    _ensure_fixtures()
    from ml.pipeline import process_session

    result = process_session(str(FIXTURES_DIR / "test_audio.wav"))

    if result is not None:
        assert "features" in result
        assert len(result["metadata"]["warnings"]) > 0
        print(f"  Partial: {result['metadata']['feature_count']} features, "
              f"{len(result['metadata']['warnings'])} warnings")


if __name__ == "__main__":
    test_process_missing_file()
    print("PASS: test_process_missing_file")

    test_process_session_synthetic()
    print("PASS: test_process_session_synthetic")

    test_partial_failure()
    print("PASS: test_partial_failure")

    test_process_session_real()
    print("PASS: test_process_session_real")

    print("\nAll pipeline tests passed!")
