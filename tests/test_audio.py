"""Tests for the audio feature extractor."""

import os
from pathlib import Path

from tests.generate_fixtures import generate_sine_wav, generate_silent_wav

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Expected feature keys from the audio extractor
EXPECTED_KEYS = {
    "audio_pause_count", "audio_pause_mean_duration", "audio_pause_to_speech_ratio",
    "audio_speech_duration", "audio_total_duration", "audio_speech_rate_wpm",
    "audio_f0_mean", "audio_f0_std", "audio_f0_range",
    "audio_loudness_mean", "audio_loudness_std",
    "audio_jitter", "audio_shimmer",
}
# Add MFCC keys
for i in range(1, 14):
    EXPECTED_KEYS.add(f"audio_mfcc_{i:02d}_mean")
    EXPECTED_KEYS.add(f"audio_mfcc_{i:02d}_std")


def _ensure_fixtures():
    """Generate test fixtures if they don't exist."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = FIXTURES_DIR / "test_audio.wav"
    silent_path = FIXTURES_DIR / "test_silent.wav"
    if not wav_path.exists():
        generate_sine_wav(wav_path)
    if not silent_path.exists():
        generate_silent_wav(silent_path)


def test_extract_sine_wave():
    """Test extraction on a 5-second sine wave with 1s silence."""
    _ensure_fixtures()
    from ml.extractors.audio import extract

    result = extract(str(FIXTURES_DIR / "test_audio.wav"))

    assert result is not None, "extract() returned None for valid audio"
    assert isinstance(result, dict)

    # Check all expected keys are present
    missing = EXPECTED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"

    # Check all values are floats
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    # Sanity checks on values
    assert result["audio_total_duration"] > 4.0, "Duration should be ~5s"
    assert result["audio_f0_mean"] > 0, "Should detect pitch in sine wave"
    assert result["audio_loudness_mean"] > 0, "Should detect loudness"
    assert result["audio_pause_count"] >= 1, "Should detect the 1s silence as a pause"


def test_extract_silent_audio():
    """Test that silent audio returns None."""
    _ensure_fixtures()
    from ml.extractors.audio import extract

    result = extract(str(FIXTURES_DIR / "test_silent.wav"))
    assert result is None, "Silent audio should return None"


def test_extract_missing_file():
    """Test that missing file returns None."""
    from ml.extractors.audio import extract

    result = extract("nonexistent_file.wav")
    assert result is None, "Missing file should return None"


def test_extract_with_transcript():
    """Test that speech rate is computed when transcript is provided."""
    _ensure_fixtures()
    from ml.extractors.audio import extract

    result = extract(
        str(FIXTURES_DIR / "test_audio.wav"),
        transcript="hello world this is a test of the speech rate feature"
    )

    assert result is not None
    assert result["audio_speech_rate_wpm"] > 0, "Speech rate should be > 0 with transcript"


if __name__ == "__main__":
    test_extract_sine_wave()
    print("PASS: test_extract_sine_wave")

    test_extract_silent_audio()
    print("PASS: test_extract_silent_audio")

    test_extract_missing_file()
    print("PASS: test_extract_missing_file")

    test_extract_with_transcript()
    print("PASS: test_extract_with_transcript")

    print("\nAll audio tests passed!")
