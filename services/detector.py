"""
WatchAI — Typed YOLO Detector
Wraps ultralytics YOLOv8 and returns typed Detection dataclasses.
Detects persons, vehicles, and security-relevant objects from a
fixed CCTV camera (bank, street, shop, parking lot, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: List[int]   # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "confidence": round(self.confidence, 3),
            "bbox":       self.bbox,
        }

    def __str__(self) -> str:
        return f"{self.label}({self.confidence:.2f})"


class YOLODetector:
    """Thin wrapper around ultralytics YOLOv8.

    Detects person + security-relevant objects for ground-level CCTV.
    """

    def __init__(self, model_path: str = config.PERSON_YOLO_PATH):
        self._model = None
        self._model_path = model_path
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info(f"YOLO model loaded: {self._model_path}")
        except Exception as exc:
            logger.warning(f"Could not load YOLO model ({exc}). Using simulated detections.")
            self._model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a BGR numpy image. Returns filtered detections."""
        if self._model is None:
            return self._simulated_detect(image)

        try:
            results = self._model(
                image,
                conf=config.YOLO_CONFIDENCE,
                verbose=False,
            )[0]
            detections: List[Detection] = []

            for box in results.boxes:
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                label   = results.names[cls_id]
                xyxy    = box.xyxy[0].tolist()

                if label not in config.YOLO_CLASSES_OF_INTEREST:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                # Clamp to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                ))

            logger.debug(f"YOLO detected {len(detections)} objects")
            return detections

        except Exception as exc:
            logger.error(f"YOLO inference failed: {exc}")
            return self._simulated_detect(image)

    def _simulated_detect(self, image: np.ndarray) -> List[Detection]:
        """Fallback: return a simple dummy detection."""
        h, w = image.shape[:2]
        return [Detection(label="person", confidence=0.85, bbox=[w//4, h//4, w//2, h//2])]

    def person_boxes(self, detections: List[Detection]) -> List[tuple]:
        """Return (x1,y1,x2,y2) tuples for person detections only."""
        return [tuple(d.bbox) for d in detections if d.label == "person"]

    def format_for_prompt(self, detections: List[Detection]) -> str:
        """Format detections as a context string for the VLM prompt."""
        if not detections:
            return "No objects detected."
        parts = ", ".join(str(d) for d in detections)
        return f"YOLO detected: [{parts}]"
