"""
WatchAI — VLM Scene Analyzer
Sends CCTV frames to Gemini 2.5 Flash for ground-level scene understanding.
Works for any fixed-camera location: bank, street, shop, office, parking lot.

Falls back to a rule-based description when GEMINI_API_KEY is not set.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class VLMResult:
    description:      str
    threat_level:     str           # LOW | MEDIUM | HIGH
    key_observations: List[str]
    raw_response:     str
    model_used:       str

    def to_dict(self) -> dict:
        return {
            "description":      self.description,
            "threat_level":     self.threat_level,
            "key_observations": self.key_observations,
            "model_used":       self.model_used,
        }


class VLMAnalyzer:
    """
    Analyzes a BGR frame from a fixed CCTV camera using Gemini 2.5 Flash.
    Prompt is tuned for ground-level locations (not aerial/drone).
    """

    def __init__(self, model: str = config.VLM_MODEL):
        self.model = model
        if config.GEMINI_API_KEY:
            try:
                from google import genai
                self._client = genai.Client(
                    http_options={"timeout": config.VLM_TIMEOUT * 1000}
                )
                self._available = True
                logger.info(f"VLM ready: {self.model}")
            except Exception as exc:
                logger.warning(f"Gemini client init failed: {exc}. Using fallback.")
                self._client = None
                self._available = False
        else:
            self._client = None
            self._available = False
            logger.warning("GEMINI_API_KEY not set. VLM will use fallback descriptions.")

    def analyze(
        self,
        image: np.ndarray,
        detections: list,          # List[Detection] from YOLODetector
        location:   str,
        timestamp:  str,
        camera_id:  str = "CAM-01",
    ) -> VLMResult:
        """
        Analyze a CCTV frame.

        Args:
            image:      BGR numpy array
            detections: list of Detection objects from YOLODetector
            location:   human-readable camera location (e.g. 'Bank Entrance')
            timestamp:  ISO timestamp string
            camera_id:  camera identifier

        Returns:
            VLMResult with description, threat_level, key_observations
        """
        if not self._available:
            return self._fallback(detections, location)

        # Build YOLO context string
        if detections:
            counts: dict = {}
            for d in detections:
                counts[d.label] = counts.get(d.label, 0) + 1
            yolo_ctx = ", ".join(f"{n} {lbl}(s)" for lbl, n in counts.items())
        else:
            yolo_ctx = "none"

        prompt = (
            f"You are an expert CCTV Security Analyst monitoring a live surveillance feed "
            f"from a fixed ground-level camera.\n\n"
            f"--- CAMERA CONTEXT ---\n"
            f"Location: {location} | Camera: {camera_id} | Time: {timestamp}\n"
            f"Object detector findings: {yolo_ctx}\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"1. Analyze the scene from the perspective of a ground-level security camera. "
            f"Describe the environment, people present, and any activities visible.\n"
            f"2. Identify any security concerns: unauthorized access, suspicious behavior, "
            f"loitering, altercations, unattended bags, vandalism, or unusual crowd activity.\n"
            f"3. Assign a THREAT level:\n"
            f"   - LOW: Normal activity, nothing suspicious.\n"
            f"   - MEDIUM: Minor anomaly — someone loitering, unusual gathering, "
            f"vehicle in restricted area, unattended item.\n"
            f"   - HIGH: Active threat — violence, forced entry, dangerous object visible, "
            f"robbery in progress, or serious safety hazard.\n\n"
            f"Respond STRICTLY in this format (no markdown bolding on the labels):\n"
            f"DESCRIPTION: <2-3 sentences about the scene and any security concerns>\n"
            f"THREAT: <LOW|MEDIUM|HIGH>\n"
            f"OBSERVATIONS:\n"
            f"- <specific observation 1>\n"
            f"- <specific observation 2>\n"
            f"- <specific observation 3>"
        )

        # Resize to 640px wide for fast Gemini upload
        h, w = image.shape[:2]
        if w > 640:
            image = cv2.resize(image, (640, int(h * 640 / w)))
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_bytes = buf.tobytes()

        import time
        try:
            from google.genai import types
            start = time.time()
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            elapsed = time.time() - start
            logger.info(f"[VLM] Inference took {elapsed:.2f}s")
            return self._parse(response.text)

        except Exception as exc:
            logger.error(f"VLM inference error: {exc}")
            return self._fallback(detections, location)

    def _parse(self, raw: str) -> VLMResult:
        description   = raw[:300]
        threat_level  = "LOW"
        observations: List[str] = []
        obs_mode = False

        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith("DESCRIPTION:"):
                description = line.removeprefix("DESCRIPTION:").strip()
                obs_mode = False
            elif line.startswith("THREAT:"):
                level = line.removeprefix("THREAT:").strip().upper()
                if level in ("LOW", "MEDIUM", "HIGH"):
                    threat_level = level
                obs_mode = False
            elif line.startswith("OBSERVATIONS:"):
                obs_mode = True
            elif obs_mode and line.startswith("-"):
                observations.append(line[1:].strip())

        return VLMResult(
            description=description,
            threat_level=threat_level,
            key_observations=observations[:3],
            raw_response=raw,
            model_used=self.model,
        )

    def _fallback(self, detections: list, location: str) -> VLMResult:
        """Rule-based fallback when Gemini is not available."""
        labels = [d.label for d in detections]
        if not labels:
            desc   = f"No objects detected at {location}. Scene appears clear."
            threat = "LOW"
            obs    = ["No objects detected", "Scene is clear"]
        else:
            obj_str = ", ".join(set(labels))
            desc    = f"Detected {obj_str} at {location}."
            threat  = "MEDIUM" if "person" in labels else "LOW"
            obs     = [f"{lbl} detected" for lbl in set(labels)][:3]

        return VLMResult(
            description=desc,
            threat_level=threat,
            key_observations=obs,
            raw_response="[FALLBACK — no Gemini API key]",
            model_used="fallback",
        )
