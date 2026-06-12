"""
WatchAI — Display / Overlay Renderer
Draws bounding boxes, identity labels, risk levels and confidence on frames.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from services.risk_engine import risk_color_bgr

_FONT          = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE    = 0.55
_THICKNESS     = 2
_UNKNOWN_COLOR = (0, 200, 0)   # green for unidentified persons


def draw_result(frame, face_box: tuple, name: str, confidence: float,
                risk_score: int = 0, risk_level: str = "LOW"):
    """
    Draws a labeled bounding box on `frame` (in-place + returned).

    Criminal match  → risk-level color + "! Name | LEVEL (score) conf=X.XX"
    Unknown person  → green box + "Unknown"

    Args:
        frame:      BGR numpy array (modified in-place)
        face_box:   (x1, y1, x2, y2)
        name:       matched criminal name or "Unknown"
        confidence: face recognition confidence
        risk_score: 0–100
        risk_level: "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    """
    x1, y1, x2, y2 = face_box
    is_alert = name != "Unknown"
    color    = risk_color_bgr(risk_level) if is_alert else _UNKNOWN_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=_THICKNESS)

    if is_alert:
        display_name = name.replace("_", " ").title()
        label = f"! {display_name} | {risk_level} ({risk_score}) conf={confidence:.2f}"
    else:
        label = f"Unknown  conf={confidence:.2f}" if confidence > 0 else "Unknown"

    (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _THICKNESS)

    label_y1 = max(y1 - text_h - baseline - 6, 0)
    label_y2 = max(y1, text_h + baseline + 6)

    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y2 - baseline - 2),
                _FONT, _FONT_SCALE, (255, 255, 255), _THICKNESS)

    return frame


def draw_status(frame, text: str, color=(180, 180, 180)):
    """Draws a small status text in the top-left corner."""
    cv2.putText(frame, text, (10, 28), _FONT, 0.65, color, 2)
    return frame
