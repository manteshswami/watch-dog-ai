"""
WatchAI — Surveillance Pipeline Core
Processes one video frame through the full AI pipeline.
Called by the Streamlit dashboard (app/streamlit_app.py) on each rerun.

Pipeline per frame:
  1. Motion gate          (MOG2 — skip if nothing moving)
  2. YOLO detection       (persons + security-relevant objects)
  3. Face recognition     (dlib 128-d embeddings vs criminal DB)
  4. Risk scoring         (deterministic rule-based score 0-100)
  5. VLM scene analysis   (Gemini 2.5 Flash — ground-level CCTV prompt)
  6. Event assembly       (merge all outputs into one event dict)
  7. Alert engine         (rule-based alert checks)
  8. FrameIndexer         (SQLite + ChromaDB persistence)

Entry points:
  run_pipeline_step()   — process one frame from a VideoCapture source
  get_camera_location() — resolve camera index / filename → location label
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from services.detector      import YOLODetector, Detection
from services.motion        import motion_detected
from services.recognizer    import recognize_faces_in_frame
from services.risk_engine   import calculate_risk
from services.database      import FrameIndexer
from services.vlm_analyzer  import VLMAnalyzer
from services import alert_engine
from utilis.encoder_preload import load_criminal_encodings


# ── Module-level singletons (loaded once, shared across Streamlit reruns) ─────

_yolo:      Optional[YOLODetector]  = None
_vlm:       Optional[VLMAnalyzer]   = None
_indexer:   Optional[FrameIndexer]  = None
_encodings: Optional[list]          = None
_names:     Optional[list]          = None
_profiles:  Optional[dict]          = None
_last_alerts: dict                  = {}   # name → datetime of last alert (in-memory cooldown)


def load_singletons():
    """Initialize all AI components. Safe to call multiple times (idempotent)."""
    global _yolo, _vlm, _indexer, _encodings, _names, _profiles

    if _yolo is None:
        _yolo = YOLODetector()

    if _vlm is None:
        _vlm = VLMAnalyzer()

    if _indexer is None:
        _indexer = FrameIndexer()

    if _encodings is None:
        _encodings, _names, _profiles = load_criminal_encodings()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_camera_location(source) -> str:
    """
    Resolve a camera source → human-readable location label.
    source can be: int (webcam index), str (video filename stem or path).
    """
    if isinstance(source, int):
        return config.CAMERA_LOCATIONS.get(source, config.DEFAULT_LOCATION)

    stem = Path(str(source)).stem.lower()
    for key, label in config.CAMERA_LOCATIONS.items():
        if isinstance(key, str) and (key in stem or stem in key):
            return label

    return config.DEFAULT_LOCATION


def _on_cooldown(name: str) -> bool:
    """True if this criminal has been alerted in the last ALERT_COOLDOWN_SECONDS."""
    ts = _last_alerts.get(name)
    if ts is None:
        return False
    return (datetime.now() - ts).total_seconds() < config.ALERT_COOLDOWN_SECONDS


def _save_frame(frame: np.ndarray, frame_id: int, location: str) -> str:
    """Save annotated frame to FRAMES_DIR and return the path."""
    safe_loc = location.replace(" ", "_").replace("—", "-").replace("/", "-")[:30]
    fname    = f"frame_{safe_loc}_{frame_id:06d}.jpg"
    out_path = config.FRAMES_DIR / fname
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def _append_log(path: str, data):
    """Append a dict entry to a JSON log file."""
    p = Path(path)
    logs = []
    if p.exists():
        try:
            logs = json.loads(p.read_text())
        except Exception:
            pass
    logs.append(data)
    p.write_text(json.dumps(logs, indent=2, default=str))


# ── Main pipeline step ────────────────────────────────────────────────────────

def run_pipeline_step(
    cap:         cv2.VideoCapture,
    frame_id:    int,
    location:    str,
    camera_id:   str = "CAM-01",
    skip_motion: bool = False,
) -> Optional[dict]:
    """
    Process ONE frame from the VideoCapture source through the full pipeline.

    Args:
        cap          : open cv2.VideoCapture object
        frame_id     : monotonically increasing frame counter
        location     : human-readable camera location string
        camera_id    : camera identifier string
        skip_motion  : if True, bypass the motion gate (always process)

    Returns:
        event dict on success, None if frame read fails or motion gate blocks.
    """
    load_singletons()

    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    if config.WEBCAM_ROTATION:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    timestamp  = datetime.now().isoformat()
    raw_frame  = frame.copy()
    disp_frame = frame.copy()

    # ── 1. Motion gate ───────────────────────────────────────────────────────
    if not skip_motion and not motion_detected(frame):
        return None

    # ── 2. YOLO detection ────────────────────────────────────────────────────
    detections   = _yolo.detect(frame)
    person_boxes = _yolo.person_boxes(detections)

    # ── 3. Face recognition ──────────────────────────────────────────────────
    criminal_name   = ""
    risk_score      = 0
    risk_level      = "LOW"
    face_confidence = 0.0

    if person_boxes and _encodings:
        frame_rgb = np.ascontiguousarray(raw_frame[:, :, ::-1])
        matches   = recognize_faces_in_frame(frame_rgb, _encodings, _names)

        for i, box in enumerate(person_boxes):
            name = "Unknown"
            conf = 0.0
            if i < len(matches):
                name, conf = matches[i]

            if name != "Unknown":
                criminal_name   = name
                face_confidence = conf
                profile         = _profiles.get(name, {})
                risk_score, risk_level = calculate_risk(profile)

                if not _on_cooldown(name):
                    _last_alerts[name] = datetime.now()

    # ── 4. VLM scene analysis ────────────────────────────────────────────────
    vlm_result = _vlm.analyze(
        image=raw_frame,
        detections=detections,
        location=location,
        timestamp=timestamp,
        camera_id=camera_id,
    )

    # ── 5. Annotate display frame ────────────────────────────────────────────
    from app.display import draw_result, draw_status

    for i, box in enumerate(person_boxes):
        name = criminal_name if i == 0 and criminal_name else "Unknown"
        conf = face_confidence if i == 0 and criminal_name else 0.0
        draw_result(disp_frame, box, name, conf, risk_score, risk_level)

    # ── 6. Save annotated frame ──────────────────────────────────────────────
    frame_path = _save_frame(disp_frame, frame_id, location)

    # ── 7. Assemble event dict ───────────────────────────────────────────────
    objects = [d.label for d in detections]

    event = {
        "frame_id":         frame_id,
        "timestamp":        timestamp,
        "location":         location,
        "camera_id":        camera_id,
        "objects":          objects,
        "detections":       [d.to_dict() for d in detections],
        "criminal_name":    criminal_name,
        "face_confidence":  round(face_confidence, 3),
        "risk_score":       risk_score,
        "risk_level":       risk_level,
        "vlm_description":  vlm_result.description,
        "threat_level":     vlm_result.threat_level,
        "key_observations": vlm_result.key_observations,
        "vlm_model":        vlm_result.model_used,
        "frame_path":       frame_path,
        "display_frame":    disp_frame,   # numpy array for st.image()
    }

    # ── 8. Alert engine ──────────────────────────────────────────────────────
    triggered = alert_engine.evaluate(event)
    event["alert_triggered"] = len(triggered) > 0
    event["alerts"]          = triggered

    # ── 9. Persist ───────────────────────────────────────────────────────────
    row_id = _indexer.index_event(event)
    for alert in triggered:
        _indexer.log_alert(row_id, alert)

    _append_log(config.JSON_LOG_PATH, {k: v for k, v in event.items()
                                        if k != "display_frame"})
    if triggered:
        with open(config.ALERT_LOG_PATH, "a") as f:
            f.write(alert_engine.format_alert_log(triggered) + "\n")

    return event


# ── Standalone CLI (headless testing without Streamlit) ───────────────────────

def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="WatchAI — Headless Pipeline Test")
    parser.add_argument("--source",   default=0,               help="Webcam index or video file path")
    parser.add_argument("--location", default="",              help="Camera location label")
    parser.add_argument("--camera",   default="CAM-01",        help="Camera ID")
    parser.add_argument("--frames",   default=100, type=int,   help="Max frames to process")
    args = parser.parse_args()

    src      = int(args.source) if str(args.source).isdigit() else args.source
    location = args.location or get_camera_location(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {src}")
        return

    print(f"[MAIN] Source: {src} | Location: {location}")
    load_singletons()

    frame_id  = 0
    processed = 0

    try:
        while processed < args.frames:
            frame_id += 1
            if frame_id % config.PROCESS_EVERY_N_FRAMES != 0:
                cap.read()
                continue

            event = run_pipeline_step(cap, frame_id, location, args.camera)
            if event is None:
                continue

            processed += 1
            alerted = "🚨 " + ", ".join(a["rule_name"] for a in event.get("alerts", [])) \
                      if event.get("alerts") else "✅"
            print(
                f"[{processed:04d}] {event['timestamp'][:19]} | "
                f"Threat={event['threat_level']} | "
                f"Criminal={event['criminal_name'] or 'none'} | "
                f"{alerted}"
            )
    except KeyboardInterrupt:
        print("\n[MAIN] Stopped.")
    finally:
        cap.release()
        print(f"[MAIN] Done. Processed {processed} frames.")


if __name__ == "__main__":
    _cli_main()
