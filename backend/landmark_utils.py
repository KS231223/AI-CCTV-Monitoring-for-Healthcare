# landmark_utils.py

import numpy as np


def to_numpy(landmarks, frame_width, frame_height):
    """
    Convert normalized mediapipe coordinates to pixel space
    """

    coords = {}

    for idx, lm in landmarks.items():

        coords[idx] = np.array([
            lm["x"] * frame_width,
            lm["y"] * frame_height,
            lm["z"] * frame_width
        ])

    return coords


def filter_by_visibility(landmarks, threshold=0.7):
    """
    Remove low confidence landmarks
    """

    filtered = {}

    for idx, lm in landmarks.items():

        if lm["visibility"] >= threshold:
            filtered[idx] = lm

    return filtered