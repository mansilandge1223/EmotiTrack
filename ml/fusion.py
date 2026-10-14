"""Late fusion logic for combining modality scores."""


def fuse(face_score: float, audio_score: float, text_score: float,
         weights: tuple[float, float, float] = (0.4, 0.35, 0.25)) -> float:
    """Weighted late fusion of modality scores."""
    return (face_score * weights[0] + audio_score * weights[1] + text_score * weights[2])
