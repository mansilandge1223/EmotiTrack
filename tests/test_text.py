"""Tests for the text feature extractor."""

from pathlib import Path

from tests.generate_fixtures import generate_sine_wav

FIXTURES_DIR = Path(__file__).parent / "fixtures"

EXPECTED_KEYS = {
    "text_word_count",
    "text_first_person_pronoun_ratio",
    "text_mean_utterance_length",
    "text_lexical_diversity",
    "text_sentiment_positive",
    "text_sentiment_negative",
    "text_sentiment_neutral",
    "text_speaking_time_seconds",
    "text_language",
}


def _ensure_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = FIXTURES_DIR / "test_audio.wav"
    if not wav_path.exists():
        generate_sine_wav(wav_path)


def test_extract_audio():
    """Test extraction on sine wave audio. Whisper may produce empty/gibberish
    transcript for a pure tone, so we test that it handles this gracefully."""
    _ensure_fixtures()
    from ml.extractors.text import extract

    result = extract(str(FIXTURES_DIR / "test_audio.wav"))

    # A pure sine wave might produce no speech — None is acceptable
    if result is not None:
        assert isinstance(result, dict)
        missing = EXPECTED_KEYS - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

        for key, val in result.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

        assert result["text_word_count"] >= 0
        assert 0.0 <= result["text_lexical_diversity"] <= 1.0
    else:
        print("  (returned None — no speech detected in sine wave, OK)")


def test_extract_missing_file():
    """Test that missing file returns None."""
    from ml.extractors.text import extract

    result = extract("nonexistent_audio.wav")
    assert result is None


def test_linguistic_features():
    """Test linguistic feature computation directly."""
    from ml.extractors.text import _linguistic_features

    text = "I am feeling sad today. My life is not going well. I think I need help."
    result = _linguistic_features(text)

    assert result["text_word_count"] == 16.0
    assert result["text_first_person_pronoun_ratio"] > 0, "Should detect first-person pronouns"
    assert result["text_mean_utterance_length"] > 0
    assert 0.0 < result["text_lexical_diversity"] <= 1.0


def test_linguistic_features_empty():
    """Test linguistic features on empty text."""
    from ml.extractors.text import _linguistic_features

    result = _linguistic_features("")
    assert result["text_word_count"] == 0.0
    assert result["text_first_person_pronoun_ratio"] == 0.0


def test_transcript_caching():
    """Test that transcripts are cached to disk."""
    _ensure_fixtures()
    from ml.extractors.text import _transcribe, _get_cache_path

    audio_path = str(FIXTURES_DIR / "test_audio.wav")
    cache_path = _get_cache_path(audio_path)

    # Remove cache if exists
    cache_path.unlink(missing_ok=True)

    # First call — should transcribe and cache
    result1 = _transcribe(audio_path)

    if result1 is not None:
        assert cache_path.exists(), "Transcript should be cached after first call"
        assert "text" in result1
        assert "language" in result1

        # Second call — should use cache
        result2 = _transcribe(audio_path)
        assert result2 is not None
        assert result2["text"] == result1["text"]

    # Cleanup
    cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_linguistic_features()
    print("PASS: test_linguistic_features")

    test_linguistic_features_empty()
    print("PASS: test_linguistic_features_empty")

    test_extract_missing_file()
    print("PASS: test_extract_missing_file")

    test_transcript_caching()
    print("PASS: test_transcript_caching")

    test_extract_audio()
    print("PASS: test_extract_audio")

    print("\nAll text tests passed!")
