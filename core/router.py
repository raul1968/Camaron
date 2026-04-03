import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def blend_frames(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear alpha blend between two frames."""
    if _CV2_AVAILABLE:
        return cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0)
    # Fallback pure-numpy blend
    return ((1.0 - alpha) * img_a.astype(float) + alpha * img_b.astype(float)).astype(np.uint8)


def ease_in_out(t: float) -> float:
    """Smooth-step easing function."""
    return t * t * (3.0 - 2.0 * t)


def generate_transitions(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    n_frames: int,
) -> list[np.ndarray]:
    """Return list of n_frames interpolated frames (not including endpoints)."""
    # Ensure same shape
    if frame_a.shape != frame_b.shape:
        if _CV2_AVAILABLE:
            h, w = frame_a.shape[:2]
            frame_b = cv2.resize(frame_b, (w, h))
        else:
            frame_b = frame_b[:frame_a.shape[0], :frame_a.shape[1]]

    frames = []
    for i in range(1, n_frames + 1):
        t = ease_in_out(i / (n_frames + 1))
        frames.append(blend_frames(frame_a, frame_b, t))
    return frames
