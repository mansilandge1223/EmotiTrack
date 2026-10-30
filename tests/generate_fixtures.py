"""Generate synthetic test fixtures for extractor tests.

Run once: python -m tests.generate_fixtures
Requires: numpy, opencv-python (no ffmpeg needed)
"""

import numpy as np
import wave
import struct
import os
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def generate_sine_wav(path: Path, duration: float = 5.0, freq: float = 440.0,
                      sample_rate: int = 16000) -> None:
    """Generate a mono .wav file with a sine wave."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Sine wave with slight amplitude variation to simulate speech-like energy changes
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    # Add 1-second silence in the middle (to test pause detection)
    silence_start = int(2.0 * sample_rate)
    silence_end = int(3.0 * sample_rate)
    signal[silence_start:silence_end] = 0.0

    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for sample in signal:
            wf.writeframes(struct.pack('<h', int(sample * 32767)))


def generate_test_video(path: Path, duration: float = 5.0, fps: int = 30,
                        width: int = 640, height: int = 480) -> None:
    """Generate a simple .avi video with a moving circle (no ffmpeg needed)."""
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed, skipping video generation")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    n_frames = int(duration * fps)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # dark gray background

        # Moving circle to simulate a "face"
        cx = width // 2 + int(30 * np.sin(2 * np.pi * i / n_frames))
        cy = height // 2 + int(20 * np.cos(2 * np.pi * i / n_frames))
        cv2.circle(frame, (cx, cy), 80, (200, 180, 160), -1)  # skin-tone circle
        # Eyes
        cv2.circle(frame, (cx - 25, cy - 15), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx + 25, cy - 15), 8, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 25, cy - 15), 4, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 25, cy - 15), 4, (50, 50, 50), -1)
        # Mouth
        cv2.ellipse(frame, (cx, cy + 25), (20, 10), 0, 0, 180, (100, 100, 100), 2)

        out.write(frame)

    out.release()


def generate_silent_wav(path: Path, duration: float = 5.0,
                        sample_rate: int = 16000) -> None:
    """Generate a completely silent .wav file."""
    n_samples = int(duration * sample_rate)
    with wave.open(str(path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00\x00' * n_samples)


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating test_audio.wav (5s sine wave with 1s silence)...")
    generate_sine_wav(FIXTURES_DIR / "test_audio.wav")

    print("Generating test_silent.wav (5s silence)...")
    generate_silent_wav(FIXTURES_DIR / "test_silent.wav")

    print("Generating test_video.avi (5s synthetic face)...")
    generate_test_video(FIXTURES_DIR / "test_video.avi")

    print(f"\nFixtures created in {FIXTURES_DIR}/")
    for f in sorted(FIXTURES_DIR.iterdir()):
        if f.name != '.gitkeep':
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
