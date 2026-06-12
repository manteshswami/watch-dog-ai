"""
WatchAI — Alert Engine
Rule-based alert evaluation for ground-level CCTV surveillance.
Covers real-world scenarios: banks, streets, shops, offices, parking lots.

Weapon detection removed — threat assessment is handled by VLM (Gemini).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


def _parse_hour(timestamp: str) -> Optional[int]:
    """Extract hour from an ISO timestamp string."""
    try:
        return datetime.fromisoformat(timestamp).hour
    except Exception:
        return None


def evaluate(event: Dict) -> List[Dict]:
    """
    Evaluate one frame event against all alert rules.

    Parameters
    ----------
    event : dict
        Merged record with keys:
        - objects         : List[str]   — YOLO-detected labels
        - threat_level    : str         — VLM assessment (LOW/MEDIUM/HIGH)
        - vlm_description : str         — VLM scene description
        - timestamp       : str         — ISO 8601
        - location        : str         — e.g. "Bank Entrance"
        - camera_id       : str         — e.g. "CAM-01"
        - frame_id        : int
        - criminal_name   : str | None  — matched criminal, if any
        - risk_score      : int         — 0-100
        - risk_level      : str         — LOW/MEDIUM/HIGH/CRITICAL

    Returns
    -------
    List of alert dicts (may be empty).
    """
    triggered: List[Dict] = []

    objects       = event.get("objects", [])
    threat_level  = event.get("threat_level", "LOW").upper()
    timestamp     = event.get("timestamp", "")
    location      = event.get("location", "Unknown Location")
    camera_id     = event.get("camera_id", "CAM-01")
    frame_id      = event.get("frame_id", 0)
    criminal_name = event.get("criminal_name") or ""
    risk_score    = event.get("risk_score", 0)
    risk_level    = event.get("risk_level", "LOW")
    hour          = _parse_hour(timestamp)

    def _time_str() -> str:
        try:
            return datetime.fromisoformat(timestamp).strftime("%H:%M")
        except Exception:
            return timestamp[:16]

    def _make(rule_name: str, alert_text: str, severity: str) -> Dict:
        return {
            "frame_id":   frame_id,
            "timestamp":  timestamp,
            "location":   location,
            "camera_id":  camera_id,
            "rule_name":  rule_name,
            "alert_text": alert_text,
            "severity":   severity,
            "criminal":   criminal_name,
        }

    # ── Rule 1: After-hours loitering ─────────────────────────────────────────
    if (
        hour is not None
        and (hour >= config.BUSINESS_CLOSE_HOUR or hour < config.BUSINESS_OPEN_HOUR)
        and "person" in objects
    ):
        triggered.append(_make(
            "after_hours_loitering",
            f"Person detected at {location} outside business hours ({_time_str()}). "
            "Security check recommended.",
            "HIGH",
        ))

    # ── Rule 2: Unauthorized vehicle after hours ───────────────────────────────
    vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
    if (
        hour is not None
        and (hour >= config.BUSINESS_CLOSE_HOUR or hour < config.BUSINESS_OPEN_HOUR)
        and any(o in vehicle_labels for o in objects)
    ):
        triggered.append(_make(
            "unauthorized_vehicle_after_hours",
            f"Vehicle detected at {location} after closing hours ({_time_str()}). "
            "Verify if authorized.",
            "MEDIUM",
        ))

    # ── Rule 3: Crowd gathering ────────────────────────────────────────────────
    person_count = objects.count("person")
    if person_count >= 3:
        triggered.append(_make(
            "crowd_gathering",
            f"Crowd of {person_count} people detected at {location} ({_time_str()}). "
            "Monitor for disturbance or unauthorized gathering.",
            "MEDIUM",
        ))

    # ── Rule 4: VLM assessed HIGH threat ──────────────────────────────────────
    if threat_level == "HIGH":
        vlm_desc = event.get("vlm_description", "")[:120]
        triggered.append(_make(
            "vlm_high_threat",
            f"AI scene analysis flagged HIGH threat at {location} ({_time_str()}). "
            f"Scene: {vlm_desc}",
            "HIGH",
        ))

    # ── Rule 5: Known criminal identified ─────────────────────────────────────
    if criminal_name:
        triggered.append(_make(
            "criminal_face_match",
            f"Known criminal '{criminal_name.title()}' identified at {location} "
            f"({_time_str()}). Risk: {risk_level} (score={risk_score}).",
            "HIGH" if risk_level in ("LOW", "MEDIUM") else "CRITICAL",
        ))

    # ── Rule 6: CRITICAL risk score ───────────────────────────────────────────
    if criminal_name and risk_score >= 76:
        triggered.append(_make(
            "critical_risk_criminal",
            f"CRITICAL-risk criminal '{criminal_name.title()}' at {location} "
            f"({_time_str()}). Score={risk_score}. Immediate response required.",
            "CRITICAL",
        ))

    # ── Rule 7: VLM detects dangerous situation + criminal present ────────────
    if criminal_name and threat_level == "HIGH":
        triggered.append(_make(
            "criminal_high_threat_scene",
            f"Known criminal '{criminal_name.title()}' present during HIGH-threat "
            f"incident at {location} ({_time_str()}). "
            "Do NOT approach — contact law enforcement.",
            "CRITICAL",
        ))

    if triggered:
        logger.warning(
            f"[ALERT] {len(triggered)} rule(s) triggered — frame {frame_id} "
            f"at {location} ({camera_id})"
        )
    return triggered


def format_alert_log(alerts: List[Dict]) -> str:
    """Return a human-readable multi-line string for log files."""
    lines = []
    for a in alerts:
        try:
            ts = datetime.fromisoformat(a["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = a.get("timestamp", "?")
        lines.append(
            f"[{a['severity']}] {ts} | {a['location']} ({a.get('camera_id','?')}) "
            f"| {a['rule_name']}\n  → {a['alert_text']}"
        )
    return "\n".join(lines)
