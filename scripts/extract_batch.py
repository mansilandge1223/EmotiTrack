"""Batch feature extraction over a directory of videos.

Usage:
    python -m scripts.extract_batch --input-dir data/raw --output-csv data/features/features.csv

Resumable: skips videos whose session_id already exists in the output CSV.
Logs errors to data/extraction_errors.log instead of stopping.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


def _get_completed_sessions(output_csv: Path) -> set[str]:
    """Read session_ids already in the output CSV."""
    if not output_csv.exists():
        return set()
    completed = set()
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "session_id" in row:
                completed.add(row["session_id"])
    return completed


def main():
    parser = argparse.ArgumentParser(description="Batch feature extraction from videos")
    parser.add_argument("--input-dir", required=True, help="Directory containing video files")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--error-log", default="data/extraction_errors.log",
                        help="Error log file path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    error_log = Path(args.error_log)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Set up error log
    error_log.parent.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(str(error_log))
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(error_handler)

    # Find video files
    videos = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        logger.warning("No video files found in %s", input_dir)
        sys.exit(0)

    logger.info("Found %d video files in %s", len(videos), input_dir)

    # Check which sessions are already done
    completed = _get_completed_sessions(output_csv)
    if completed:
        logger.info("Resuming: %d sessions already processed", len(completed))

    # Filter to pending videos
    pending = [v for v in videos if v.stem not in completed]
    logger.info("Processing %d pending videos (%d already done)", len(pending), len(completed))

    if not pending:
        logger.info("All videos already processed. Nothing to do.")
        return

    # Import pipeline here (lazy — avoids loading models if nothing to do)
    from ml.pipeline import process_session

    # Determine CSV columns from first successful extraction
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_columns = None

    if output_csv.exists():
        with open(output_csv, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                existing_columns = header

    results = []
    failed = 0

    for video_path in tqdm(pending, desc="Extracting features"):
        session_id = video_path.stem
        try:
            result = process_session(str(video_path))

            if result is None:
                logger.error("FAILED: %s — all extractors returned None", session_id)
                failed += 1
                continue

            row = {"session_id": session_id}
            row.update(result["features"])
            row["duration_seconds"] = result["metadata"]["duration_seconds"]
            row["warnings"] = "; ".join(result["metadata"]["warnings"])
            results.append(row)

        except Exception as e:
            logger.error("FAILED: %s — %s", session_id, e)
            failed += 1

    if not results:
        logger.warning("No successful extractions")
        return

    # Determine column order
    if existing_columns:
        columns = existing_columns
        # Add any new columns from results
        for row in results:
            for key in row:
                if key not in columns:
                    columns.append(key)
    else:
        # Use first result to set column order: session_id first, then sorted features
        first_keys = list(results[0].keys())
        columns = ["session_id"] + sorted(k for k in first_keys if k != "session_id")

    # Write/append to CSV
    write_header = not output_csv.exists() or existing_columns is None
    mode = "a" if output_csv.exists() and not write_header else "w"

    # If appending, just append. If new, write everything.
    if output_csv.exists() and existing_columns:
        # Append mode
        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            for row in results:
                writer.writerow(row)
    else:
        # Write fresh
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    logger.info("Done. %d succeeded, %d failed. Output: %s",
                len(results), failed, output_csv)


if __name__ == "__main__":
    main()
