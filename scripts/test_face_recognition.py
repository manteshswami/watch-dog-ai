#!/usr/bin/env python3
"""
WatchAI

"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import face_recognition
import config
from utilis.encoder_preload import load_criminal_encodings
from services.recognizer import recognize_faces_in_frame

# ── ANSI colors ───────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; RST = "\033[0m"
def ok(m):     print(f"  {G}[OK]{RST}   {m}")
def fail(m):   print(f"  {R}[FAIL]{RST} {m}")
def warn(m):   print(f"  {Y}[WARN]{RST} {m}")
def info(m):   print(f"  {C}[INFO]{RST} {m}")
def header(m): print(f"\n{B}{'='*62}\n  {m}\n{'='*62}{RST}")

_SUPPORTED = (".jpg", ".jpeg", ".png", ".bmp")


def _collect_images(name: str) -> list[str]:
    paths = []
    person_dir = os.path.join(config.CRIMINALS_DIR, name)
    if os.path.isdir(person_dir):
        paths = [os.path.join(person_dir, f)
                 for f in sorted(os.listdir(person_dir))
                 if f.lower().endswith(_SUPPORTED)]
    return paths


# ── Test 1: Self-recognition ──────────────────────────────────────────────────

def test_self_recognition(known_encodings, known_names) -> tuple[int, int, list]:
    header("Test 1 — Self-Recognition (enrolled images → should match self)")

    total, passed = 0, 0
    failures = []

    for name in known_names:
        images = _collect_images(name)
        if not images:
            warn(f"'{name}' — no images found in {config.CRIMINALS_DIR}/{name}/")
            continue

        for img_path in images:
            total += 1
            fname = os.path.basename(img_path)
            try:
                img_rgb = face_recognition.load_image_file(img_path)
                results = recognize_faces_in_frame(img_rgb, known_encodings, known_names)

                if not results:
                    fail(f"{name}/{fname} — no face detected in image")
                    failures.append((name, fname, "no_face"))
                    continue

                matched, conf = results[0]
                if matched == name:
                    ok(f"{name}/{fname} → self-match ✓  conf={conf:.2f}")
                    passed += 1
                else:
                    fail(f"{name}/{fname} → matched '{matched}'  conf={conf:.2f}")
                    failures.append((name, fname, matched))

            except Exception as e:
                fail(f"{name}/{fname} — exception: {e}")
                failures.append((name, fname, str(e)))

    if total == 0:
        warn("No images to test. Run build_dataset.py first.")
        return 0, 0, []

    acc = passed / total * 100
    color = G if acc >= 80 else (Y if acc >= 50 else R)
    print(f"\n{B}  Result: {passed}/{total} passed  →  {color}{acc:.1f}%{RST}")
    return passed, total, failures


# ── Test 2: Threshold sensitivity ─────────────────────────────────────────────

def test_threshold_sensitivity(known_encodings, known_names, max_persons: int = 6):
    header("Test 2 — Threshold Sensitivity Analysis")
    print(f"  (Shows per-image distances; helps tune FACE_MATCH_THRESHOLD)\n")
    print(f"  {'Name':<25} {'File':<14} {'Self dist':>10} {'Near other':>11} {'Margin':>8}")
    print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*11} {'-'*8}")

    rows = []
    for name in known_names[:max_persons]:
        images = _collect_images(name)[:2]   # first 2 images only
        if name not in known_names:
            continue
        enc_idx = known_names.index(name)

        for img_path in images:
            fname = os.path.basename(img_path)
            try:
                img_rgb = face_recognition.load_image_file(img_path)
                encs    = face_recognition.face_encodings(img_rgb)
                if not encs:
                    warn(f"  {name}/{fname} — no face, skipping")
                    continue
                enc = encs[0]

                distances  = face_recognition.face_distance(known_encodings, enc)
                self_dist  = float(distances[enc_idx])
                other_dist = [float(d) for i, d in enumerate(distances) if i != enc_idx]
                near_other = min(other_dist) if other_dist else 1.0
                margin     = near_other - self_dist

                m_color = G if margin > 0.15 else (Y if margin > 0 else R)
                print(f"  {name:<25} {fname:<14} {self_dist:>10.3f} {near_other:>11.3f} "
                      f"{m_color}{margin:>8.3f}{RST}")
                rows.append({"self": self_dist, "other": near_other, "margin": margin})

            except Exception as e:
                warn(f"  {name}/{fname} — {e}")

    if rows:
        avg_self  = np.mean([r["self"]   for r in rows])
        avg_other = np.mean([r["other"]  for r in rows])
        avg_mar   = np.mean([r["margin"] for r in rows])
        print(f"\n  Averages — self={avg_self:.3f}  nearest-other={avg_other:.3f}  margin={avg_mar:.3f}")
        thr = config.FACE_MATCH_THRESHOLD
        verdict = "good" if avg_self < thr < avg_other else "may need tuning"
        print(f"  Current FACE_MATCH_THRESHOLD = {thr}  →  {verdict}")


# ── Test 3: External image ────────────────────────────────────────────────────

def test_external_image(image_path: str, known_encodings, known_names):
    header(f"Test 3 — External Image: {os.path.basename(image_path)}")

    if not os.path.exists(image_path):
        fail(f"File not found: {image_path}")
        return

    img_rgb = face_recognition.load_image_file(image_path)
    results = recognize_faces_in_frame(img_rgb, known_encodings, known_names)

    if not results:
        info("No faces detected in the image.")
        return

    for name, conf in results:
        color = R if name != "Unknown" else G
        print(f"  Face → {color}{name}{RST}  conf={conf:.2f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WatchAI Face Recognition Test Suite")
    parser.add_argument("--image",   type=str, help="Path to an external image to test")
    parser.add_argument("--verbose", action="store_true", help="Show recognizer debug logs")
    args = parser.parse_args()

    if not args.verbose:
        # Suppress per-match debug prints from recognizer
        import services.recognizer as _rec
        _rec_orig = _rec.print
        # (print suppression not strictly needed — logs go to stdout anyway)

    print(f"\n{B}WatchAI — Face Recognition Test Suite{RST}")
    print(f"  FACE_MATCH_THRESHOLD : {config.FACE_MATCH_THRESHOLD}")
    print(f"  NUM_JITTERS          : {config.NUM_JITTERS}")
    print(f"  FACE_DETECT_MODEL    : {config.FACE_DETECT_MODEL}\n")

    print("[LOAD] Loading criminal encodings...")
    known_encodings, known_names, _ = load_criminal_encodings()

    if not known_names:
        print(f"\n{R}No criminals loaded. Run:  uv run python scripts/build_dataset.py{RST}\n")
        sys.exit(1)

    info(f"Enrolled: {len(known_names)} person(s)")

    passed, total, failures = test_self_recognition(known_encodings, known_names)
    test_threshold_sensitivity(known_encodings, known_names)

    if args.image:
        test_external_image(args.image, known_encodings, known_names)

    header("SUMMARY")
    if total == 0:
        warn("No test images found.")
    elif passed == total:
        ok(f"All {total} enrolled images recognized correctly.")
    else:
        missed = total - passed
        warn(f"{missed}/{total} image(s) failed self-recognition.")
        print(f"\n  {Y}Common causes:{RST}")
        print(f"    — Non-frontal or low-resolution images (HOG misses face)")
        print(f"    — Only 1 enrolled image → averaged encoding may drift")
        print(f"    — Lower FACE_MATCH_THRESHOLD in config.py to be more permissive")
        if failures:
            print(f"\n  Failed images:")
            for n, f, reason in failures[:8]:
                print(f"    {R}- {n}/{f} : {reason}{RST}")
    print()
