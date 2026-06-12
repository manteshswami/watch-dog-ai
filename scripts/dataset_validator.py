#!/usr/bin/env python3
"""
WatchAI — Dataset Validator
Audits the criminal dataset end-to-end:
  1. Excel profile integrity (required fields, valid values)
  2. Image file presence and readability
  3. HOG face detectability per image
  4. Risk score distribution

Usage:
    uv run python scripts/dataset_validator.py
    uv run python scripts/dataset_validator.py --skip-face   # skip slow face check
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utilis.excel_loader import load_criminal_profiles
from services.risk_engine import calculate_risk

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; RST = "\033[0m"
def ok(m):     print(f"  {G}[OK]{RST}   {m}")
def fail(m):   print(f"  {R}[FAIL]{RST} {m}")
def warn(m):   print(f"  {Y}[WARN]{RST} {m}")
def info(m):   print(f"  {C}[INFO]{RST} {m}")
def header(m): print(f"\n{B}{'='*62}\n  {m}\n{'='*62}{RST}")

_SUPPORTED      = (".jpg", ".jpeg", ".png", ".bmp")
_VALID_STATUSES = {"wanted", "bail", "parole", "released", "imprisoned"}
_VALID_CRIMES   = {"violent", "drug", "fraud", "property", "sexual", "non-violent"}


# ── Step 1: Excel ─────────────────────────────────────────────────────────────

def validate_excel() -> tuple[dict, bool]:
    header("Step 1 — Excel Profile Validation")
    profiles = load_criminal_profiles()

    if not profiles:
        fail("No profiles loaded. Run:  uv run python scripts/build_dataset.py")
        return {}, False

    ok(f"Loaded {len(profiles)} profile(s)")
    issues = 0

    for name, p in profiles.items():
        if not p.get("age") or not (10 <= p["age"] <= 100):
            warn(f"'{name}' — suspicious age: {p.get('age')}")
            issues += 1

        if p.get("current_status") and p["current_status"] not in _VALID_STATUSES:
            warn(f"'{name}' — unknown status: '{p['current_status']}'")
            issues += 1

        if p.get("crime_type") and p["crime_type"] not in _VALID_CRIMES:
            warn(f"'{name}' — unknown crime_type: '{p['crime_type']}'")

    if issues == 0:
        ok("All profiles pass basic validation")

    return profiles, True


# ── Step 2: Image files ───────────────────────────────────────────────────────

def validate_images(profiles: dict) -> dict:
    header("Step 2 — Image File Validation")

    if not os.path.exists(config.CRIMINALS_DIR):
        fail(f"criminal_images/ directory not found: {config.CRIMINALS_DIR}")
        return {}

    image_map = {}   # name → list of absolute image paths

    for name, profile in profiles.items():
        person_dir = os.path.join(config.CRIMINALS_DIR, name)

        if os.path.isdir(person_dir):
            imgs = sorted(
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith(_SUPPORTED)
            )
            if imgs:
                ok(f"'{name}' → folder  {len(imgs)} image(s)")
                image_map[name] = imgs
            else:
                fail(f"'{name}' → empty folder: {person_dir}/")
                image_map[name] = []
            continue

        # Single-image fallback
        photo = profile.get("photo", "").strip() or f"{name}.jpg"
        fpath = os.path.join(config.CRIMINALS_DIR, photo)
        if os.path.exists(fpath):
            ok(f"'{name}' → single file  {photo}")
            image_map[name] = [fpath]
        else:
            fail(f"'{name}' → image not found: {fpath}")
            image_map[name] = []

    total  = len(profiles)
    with_i = sum(1 for v in image_map.values() if v)
    print(f"\n  Images found: {with_i}/{total}")

    return image_map


# ── Step 3: Face detectability ────────────────────────────────────────────────

def validate_face_detection(image_map: dict):
    header("Step 3 — Face Detectability (HOG detector)")
    print("  Checks if dlib HOG can find a face in each enrolled image.\n")

    try:
        import face_recognition
    except ImportError:
        warn("face_recognition not installed — skipping")
        return

    total, detected = 0, 0

    for name, paths in image_map.items():
        for path in paths[:2]:   # only first 2 per person (speed)
            total += 1
            fname = os.path.basename(path)
            try:
                img  = face_recognition.load_image_file(path)
                locs = face_recognition.face_locations(img, model="hog")
                if locs:
                    ok(f"'{name}/{fname}' → {len(locs)} face(s)")
                    detected += 1
                else:
                    fail(f"'{name}/{fname}' → 0 faces — will be SKIPPED at runtime")
            except Exception as e:
                fail(f"'{name}/{fname}' — {e}")

    if total == 0:
        warn("No images to check.")
        return

    rate    = detected / total * 100
    color   = G if rate >= 80 else (Y if rate >= 50 else R)
    print(f"\n  Detection rate: {color}{detected}/{total} ({rate:.1f}%){RST}")

    if detected < total:
        print(f"\n  {Y}TIP:{RST} HOG needs clear, front-facing photos.")
        print(f"  Replace failed images with higher-quality frontal shots.")
        print(f"  Or set FACE_DETECT_MODEL = 'cnn' in config.py (GPU required).")


# ── Step 4: Risk distribution ─────────────────────────────────────────────────

def validate_risk_scores(profiles: dict):
    header("Step 4 — Risk Score Distribution")

    from collections import Counter
    level_counts: Counter = Counter()

    print(f"  {'Name':<30} {'Score':>6}  Level")
    print(f"  {'-'*30} {'-'*6}  -----")

    for name, p in profiles.items():
        score, level = calculate_risk(p)
        level_counts[level] += 1
        level_colors = {"LOW": G, "MEDIUM": Y, "HIGH": "\033[33m", "CRITICAL": R}
        color = level_colors.get(level, RST)
        print(f"  {name:<30} {score:>6}  {color}{level}{RST}")

    print(f"\n  Distribution:")
    for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = level_counts[lvl]
        if cnt:
            bar = "█" * cnt
            print(f"    {lvl:<10}  {cnt:>3}  {bar}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(profiles: dict, image_map: dict):
    header("SUMMARY")
    total      = len(profiles)
    with_imgs  = sum(1 for v in image_map.values() if v)
    no_imgs    = total - with_imgs

    if no_imgs == 0:
        ok(f"All {total} profiles have images")
    else:
        warn(f"{no_imgs}/{total} profile(s) missing images")

    ready = with_imgs
    if ready > 0:
        ok(f"{ready} criminal(s) ready for enrollment")
    else:
        fail("No criminals ready — run build_dataset.py")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WatchAI Dataset Validator")
    parser.add_argument("--skip-face", action="store_true",
                        help="Skip the face detectability check (faster)")
    args = parser.parse_args()

    print(f"\n{B}WatchAI — Dataset Validator{RST}\n")

    profiles, excel_ok = validate_excel()
    if not excel_ok:
        sys.exit(1)

    image_map = validate_images(profiles)

    if not args.skip_face:
        validate_face_detection(image_map)
    else:
        print(f"\n{Y}[SKIP] Face detection check skipped (--skip-face){RST}")

    validate_risk_scores(profiles)
    print_summary(profiles, image_map)
