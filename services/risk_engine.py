"""
WatchAI — Risk Engine
Deterministic rule-based scoring system.
Score 0–100, mapped to LOW / MEDIUM / HIGH / CRITICAL.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
import config

# ── Score weights (modular — edit here to retune) ─────────────────────────────
STATUS_SCORES = {
    "wanted":    35,
    "bail":      25,
    "parole":    15,
    "released":   5,
    "imprisoned": 0,   # already incarcerated — lower immediate threat
}

CRIME_TYPE_SCORES = {
    "sexual":      25,
    "violent":     20,
    "drug":        10,
    "fraud":        7,
    "property":     5,
    "non-violent":  2,
}

MAX_CONVICTION_POINTS = 15   # 3 pts each, capped at 5 convictions
MAX_VICTIM_POINTS     = 10   # 1 pt each victim, capped at 10

RECENCY_SCORES = [
    (365,         20),   # < 1 year
    (365 * 3,     12),   # 1–3 years
    (365 * 5,      6),   # 3–5 years
    (float("inf"), 2),   # > 5 years
]


def calculate_risk(profile: dict) -> tuple[int, str]:
    """
    Computes a risk score and level for a criminal profile.

    Factors:
        1. Current legal status     (0–35)
        2. Crime type severity      (0–25)
        3. Weapon used              (0 or +10)
        4. Recency of last crime    (+2 / +6 / +12 / +20)
        5. Prior convictions        (3 pts each, capped at 15)
        6. Victim count             (1 pt each, capped at 10)
        7. Youth + recidivism bonus (0 or +5)

    Returns:
        (score: int, level: str) — score clamped to [0, 100]
    """
    score = 0

    # 1. Legal status
    status = profile.get("current_status", "").lower()
    score += STATUS_SCORES.get(status, 0)

    # 2. Crime type
    crime_type = profile.get("crime_type", "").lower()
    score += CRIME_TYPE_SCORES.get(crime_type, 0)

    # 3. Weapon used
    if profile.get("weapon_used"):
        score += 10

    # 4. Recency of last crime
    last_crime = profile.get("last_crime_date")
    if last_crime:
        days_ago = (date.today() - last_crime).days
        for threshold_days, pts in RECENCY_SCORES:
            if days_ago < threshold_days:
                score += pts
                break

    # 5. Prior convictions (3 pts each, capped)
    convictions = profile.get("num_prior_convictions", 0) or 0
    score += min(convictions * 3, MAX_CONVICTION_POINTS)

    # 6. Victim count (1 pt each, capped)
    victims = profile.get("victim_count", 0) or 0
    score += min(victims, MAX_VICTIM_POINTS)

    # 7. Youth + multiple convictions → higher recidivism risk
    age = profile.get("age")
    if age and age < 25 and convictions >= 2:
        score += 5

    score = max(0, min(score, 100))

    level = "LOW"
    for threshold, label in config.RISK_LEVELS:
        if score >= threshold:
            level = label
            break

    return score, level


def risk_color_bgr(level: str) -> tuple[int, int, int]:
    """
    Returns an OpenCV BGR color for each risk level.
    LOW→green, MEDIUM→yellow, HIGH→orange-red, CRITICAL→dark red.
    """
    return {
        "LOW":      (0,   200,   0),
        "MEDIUM":   (0,   200, 255),
        "HIGH":     (0,    80, 220),
        "CRITICAL": (30,   0,  180),
    }.get(level, (0, 255, 0))
