"""
WatchAI — Motion Detector
MOG2 background subtraction with morphological denoising.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import config

# Stateful background subtractor — lives for the process lifetime
_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=config.MOG2_HISTORY,
    varThreshold=config.MOG2_VAR_THRESHOLD,
    detectShadows=False,
)
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def motion_detected(frame) -> bool:
    """
    Returns True if significant motion is present.
    Uses background subtraction + contour area threshold to filter camera noise.
    """
    fg_mask = _bg_subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, _kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= config.MIN_MOTION_AREA for c in contours)
