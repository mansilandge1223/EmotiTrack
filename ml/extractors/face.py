"""Face feature extractor. Extracts session-level facial features from video files.

Uses Py-Feat for Action Unit detection and MediaPipe FaceLandmarker (Tasks API)
for head pose, blink rate, and gaze direction.
"""

import logging
import math
import tempfile
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ml.config import get as cfg

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices
_LEFT_EYE = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]

# 3D model points for head pose via solvePnP
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (1)
    (0.0, -330.0, -65.0),     # Chin (152)
    (-225.0, 170.0, -135.0),  # Left eye corner (263)
    (225.0, 170.0, -135.0),   # Right eye corner (33)
    (-150.0, -150.0, -125.0), # Left mouth corner (287)
    (150.0, -150.0, -125.0),  # Right mouth corner (57)
], dtype=np.float64)
_POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

_TARGET_AUS = ["AU04", "AU06", "AU12", "AU15"]

# Model file path for MediaPipe FaceLandmarker
_FACE_LANDMARKER_MODEL = Path("data/models/face_landmarker.task")
_FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Lazy globals
_pyfeat_detector = None


def _ensure_landmarker_model() -> str:
    """Download the FaceLandmarker model if not present."""
    if not _FACE_LANDMARKER_MODEL.exists():
        _FACE_LANDMARKER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe FaceLandmarker model...")
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, str(_FACE_LANDMARKER_MODEL))
        logger.info("FaceLandmarker model downloaded to %s", _FACE_LANDMARKER_MODEL)
    return str(_FACE_LANDMARKER_MODEL)


def _get_pyfeat_detector():
    """Lazy-load Py-Feat detector."""
    global _pyfeat_detector
    if _pyfeat_detector is None:
        from feat import Detectorv2
        _pyfeat_detector = Detectorv2()
        _pyfeat_detector.eval()
    return _pyfeat_detector


def _eye_aspect_ratio(landmarks, eye_indices, w, h) -> float:
    """Compute Eye Aspect Ratio for blink detection."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h_dist = math.dist(pts[0], pts[3])
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


def _estimate_head_pose(landmarks, w, h) -> tuple[float, float, float] | None:
    """Estimate pitch, yaw, roll from face landmarks."""
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in _POSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    yaw = math.degrees(math.atan2(-rmat[2, 0], math.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)))
    roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return pitch, yaw, roll


def _iris_center(landmarks, iris_indices, w, h) -> tuple[float, float]:
    """Get iris center coordinates."""
    xs = [landmarks[i].x * w for i in iris_indices]
    ys = [landmarks[i].y * h for i in iris_indices]
    return float(np.mean(xs)), float(np.mean(ys))


def _extract_au_features(frames: list[np.ndarray]) -> dict[str, float]:
    """Extract Action Unit features from frames using Py-Feat."""
    detector = _get_pyfeat_detector()
    all_aus = {au: [] for au in _TARGET_AUS}
    all_au_sums = []

    for frame in frames:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, frame)
            result = detector.detect(tmp_path, data_type="image", progress_bar=False)

            if result is not None and len(result) > 0:
                au_cols = [c for c in result.columns if c.startswith("AU")]
                row = result.iloc[0]

                for au in _TARGET_AUS:
                    if au in result.columns and not np.isnan(row[au]):
                        all_aus[au].append(float(row[au]))

                au_values = [float(row[c]) for c in au_cols if not np.isnan(row[c])]
                if au_values:
                    all_au_sums.append(sum(au_values))
        except Exception as e:
            logger.debug("Py-Feat failed on frame: %s", e)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    features = {}
    for au in _TARGET_AUS:
        au_lower = au.lower()
        vals = all_aus[au]
        features[f"face_{au_lower}_mean"] = float(np.mean(vals)) if vals else 0.0
        features[f"face_{au_lower}_std"] = float(np.std(vals)) if vals else 0.0

    features["face_au_total_activity"] = float(np.mean(all_au_sums)) if all_au_sums else 0.0
    return features


def _extract_mediapipe_features(frames: list[np.ndarray], fps: float) -> dict[str, float]:
    """Extract head pose, blink rate, and gaze using MediaPipe FaceLandmarker."""
    model_path = _ensure_landmarker_model()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=False,
        num_faces=1,
    )

    poses = []
    ear_values = []
    gaze_positions = []
    face_detected = 0
    total_frames = len(frames)

    with FaceLandmarker.create_from_options(options) as landmarker:
        for frame in frames:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                continue

            face_detected += 1
            landmarks = result.face_landmarks[0]

            # Head pose
            pose = _estimate_head_pose(landmarks, w, h)
            if pose is not None:
                poses.append(pose)

            # Eye aspect ratio
            left_ear = _eye_aspect_ratio(landmarks, _LEFT_EYE, w, h)
            right_ear = _eye_aspect_ratio(landmarks, _RIGHT_EYE, w, h)
            ear_values.append((left_ear + right_ear) / 2.0)

            # Gaze (iris landmarks)
            try:
                left_iris = _iris_center(landmarks, _LEFT_IRIS, w, h)
                right_iris = _iris_center(landmarks, _RIGHT_IRIS, w, h)
                gaze_positions.append((
                    (left_iris[0] + right_iris[0]) / 2.0,
                    (left_iris[1] + right_iris[1]) / 2.0,
                ))
            except (IndexError, Exception):
                pass

    features = {}

    # Detection rate
    detection_rate = face_detected / total_frames if total_frames > 0 else 0.0
    features["face_detection_rate"] = float(detection_rate)
    features["face_quality_warning"] = 1.0 if detection_rate < cfg("face", "min_detection_rate") else 0.0

    if detection_rate == 0:
        logger.warning("No faces detected in any frame")

    # Head movement velocity
    if len(poses) >= 2:
        velocities = [
            math.sqrt(sum((a - b)**2 for a, b in zip(poses[i], poses[i-1])))
            for i in range(1, len(poses))
        ]
        features["face_head_movement_velocity"] = float(np.mean(velocities))
    else:
        features["face_head_movement_velocity"] = 0.0

    # Blink rate
    if len(ear_values) >= 2:
        blink_count = 0
        was_closed = False
        for ear in ear_values:
            if ear < 0.2 and not was_closed:
                blink_count += 1
                was_closed = True
            elif ear >= 0.2:
                was_closed = False

        frame_skip = cfg("face", "frame_skip")
        duration_minutes = (total_frames * frame_skip) / fps / 60.0 if fps > 0 else 0
        features["face_blink_rate_per_min"] = float(blink_count / duration_minutes) if duration_minutes > 0 else 0.0
    else:
        features["face_blink_rate_per_min"] = 0.0

    # Gaze variance
    if len(gaze_positions) >= 2:
        gaze_arr = np.array(gaze_positions)
        features["face_gaze_direction_variance"] = float(np.var(gaze_arr[:, 0]) + np.var(gaze_arr[:, 1]))
    else:
        features["face_gaze_direction_variance"] = 0.0

    return features


def extract(path: str) -> dict[str, float] | None:
    """Extract facial features from a video file.

    Args:
        path: Path to video file.

    Returns:
        Dict of features prefixed with 'face_', or None on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.error("Video file not found: %s", path)
        return None

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = cfg("face", "frame_skip")
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()

    if not frames:
        logger.error("No frames read from video: %s", path)
        return None

    logger.info("Processing %d frames (every %d from %d total) from %s",
                len(frames), frame_skip, frame_idx, path)

    features = {}

    # MediaPipe features
    mp_features = _extract_mediapipe_features(frames, fps)
    features.update(mp_features)

    # Py-Feat AU features
    au_features = _extract_au_features(frames)
    features.update(au_features)

    logger.info("Extracted %d face features (detection rate: %.1f%%)",
                len(features), features.get("face_detection_rate", 0) * 100)

    return features
