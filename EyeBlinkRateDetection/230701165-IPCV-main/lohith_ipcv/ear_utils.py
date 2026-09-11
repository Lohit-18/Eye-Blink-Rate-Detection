"""
Eye Aspect Ratio (EAR) helpers for Soukupová–Čech style blink detection
using dlib 68 facial landmarks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance as dist

# Dlib 68-point model: six points per eye in contour order (Rosebrock convention).
LEFT_EYE_IDX = (36, 37, 38, 39, 40, 41)
RIGHT_EYE_IDX = (42, 43, 44, 45, 46, 47)


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    eye: shape (6, 2) — six (x, y) points around one eye.
    """
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    if c < 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


def landmarks_to_eyes(shape) -> tuple[np.ndarray, np.ndarray]:
    """Convert dlib full_object_detection to left/right eye (6, 2) arrays."""
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    left = pts[list(LEFT_EYE_IDX)]
    right = pts[list(RIGHT_EYE_IDX)]
    return left, right


def combined_ear(left: np.ndarray, right: np.ndarray) -> float:
    """Average EAR of both eyes (more stable if one side is partially occluded)."""
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
