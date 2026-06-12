"""
WatchAI — Criminal Encoding Preloader
Runs once at startup. Builds known_encodings + known_names from:
  - criminal_images/{name}/  (multi-image folder → averaged encoding)
  - criminal_images/{photo}  (single image from Excel photo column)
  - criminal_images/{name}.jpg  (fallback)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import face_recognition
import numpy as np
import config
from utilis.excel_loader import load_criminal_profiles

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SUPPORTED   = (".jpg", ".jpeg", ".png", ".bmp")


def _encodings_from_dir(person_dir: str) -> list:
    """
    Load all face images from a directory and return a list of valid 128-d encodings.
    Images with no detected face are skipped with a warning.
    """
    encodings = []
    image_files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith(_SUPPORTED))

    if not image_files:
        return encodings

    for fname in image_files:
        fpath = os.path.join(person_dir, fname)
        try:
            image = face_recognition.load_image_file(fpath)
            enc   = face_recognition.face_encodings(image, num_jitters=config.NUM_JITTERS)
            if enc:
                encodings.append(enc[0])
                print(f"[ENCODER]     + {fname} ✓")
            else:
                print(f"[ENCODER]     - {fname} — no face detected, skipping")
        except Exception as e:
            print(f"[ENCODER]     ! {fname} — error: {e}")

    return encodings


def _encoding_from_file(filepath: str, label: str):
    """Load a single image and return its first face encoding, or None."""
    try:
        image = face_recognition.load_image_file(filepath)
        enc   = face_recognition.face_encodings(image, num_jitters=config.NUM_JITTERS)
        if enc:
            return enc[0]
        print(f"[ENCODER] WARNING: No face detected in {label}")
    except Exception as e:
        print(f"[ENCODER] ERROR loading {label}: {e}")
    return None


def load_criminal_encodings() -> tuple[list, list, dict]:
    """
    Returns:
        known_encodings  — list of 128-d numpy float32 vectors
        known_names      — parallel list of criminal name strings
        profiles         — dict[name → profile dict] from Excel
    """
    known_encodings: list = []
    known_names:     list = []

    profiles = load_criminal_profiles()

    if not os.path.exists(config.CRIMINALS_DIR):
        print(f"[ENCODER] criminal_images/ not found: {config.CRIMINALS_DIR}")
        return known_encodings, known_names, profiles

    for name, profile in profiles.items():
        person_dir = os.path.join(config.CRIMINALS_DIR, name)

        # ── Multi-image path: criminal_images/{name}/ ─────────────────────────
        if os.path.isdir(person_dir):
            print(f"[ENCODER] '{name}' — multi-image folder found")
            encs = _encodings_from_dir(person_dir)
            if not encs:
                print(f"[ENCODER] WARNING: No valid faces in {person_dir}/ — skipping '{name}'")
                continue
            avg_enc = np.mean(encs, axis=0).astype(np.float64)
            known_encodings.append(avg_enc)
            known_names.append(name)
            print(f"[ENCODER] '{name}' → averaged {len(encs)} encoding(s) ✓")
            continue

        # ── Single-image path ─────────────────────────────────────────────────
        photo = profile.get("photo", "").strip()
        if not photo:
            photo = f"{name}.jpg"

        # Support both "john.jpg" and "watchai/criminal_images/john.jpg"
        if os.sep in photo or "/" in photo:
            filepath = os.path.normpath(os.path.join(PROJECT_ROOT, photo))
        else:
            filepath = os.path.join(config.CRIMINALS_DIR, photo)

        if not os.path.exists(filepath):
            print(f"[ENCODER] WARNING: Photo not found → {filepath} — skipping '{name}'")
            continue

        enc = _encoding_from_file(filepath, f"{name}/{photo}")
        if enc is not None:
            known_encodings.append(enc)
            known_names.append(name)
            print(f"[ENCODER] '{name}' ← {photo} ✓")

    print(f"[ENCODER] Total criminals loaded: {len(known_names)}")
    return known_encodings, known_names, profiles
