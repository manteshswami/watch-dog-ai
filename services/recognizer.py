"""
WatchAI — Face Recognizer
Encodes faces found in a frame and matches them against known criminal encodings.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import face_recognition
import numpy as np
import config


def recognize_faces_in_frame(
    frame_rgb: np.ndarray,
    known_encodings: list,
    known_names: list,
) -> list[tuple[str, float]]:
    """
    Detects and encodes all faces in frame_rgb, then matches each against the
    known criminal encodings using L2 distance.

    NOTE: face_locations are NOT passed explicitly to face_encodings due to a
    dlib 19.24.x crash with certain coordinate orderings. Detection + encoding
    happen in one shot internally.

    Args:
        frame_rgb:       RGB numpy array (H×W×3)
        known_encodings: list of 128-d numpy vectors (one per enrolled person)
        known_names:     parallel list of names

    Returns:
        list of (name, confidence) — one entry per detected face.
        name = "Unknown" when best distance > FACE_MATCH_THRESHOLD.
        confidence = 1.0 - best_distance  (higher is better).
    """
    if not known_encodings:
        return []

    frame_rgb = np.ascontiguousarray(frame_rgb)

    try:
        encodings = face_recognition.face_encodings(
            frame_rgb,
            num_jitters=config.NUM_JITTERS,
        )
    except Exception as e:
        print(f"[RECOGNIZER] Encoding error: {e}")
        return []

    if not encodings:
        return []

    results = []
    for enc in encodings:
        distances  = face_recognition.face_distance(known_encodings, enc)
        best_idx   = int(np.argmin(distances))
        best_dist  = float(distances[best_idx])
        confidence = round(1.0 - best_dist, 2)

        if best_dist <= config.FACE_MATCH_THRESHOLD:
            name = known_names[best_idx]
            print(f"[RECOGNIZER] MATCH: {name} | dist={best_dist:.3f} | conf={confidence:.2f}")
            results.append((name, confidence))
        else:
            results.append(("Unknown", confidence))

    return results
