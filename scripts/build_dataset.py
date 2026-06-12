#!/usr/bin/env python3
"""
WatchAI — Dataset Builder
Generates a synthetic criminal dataset with profile data and face images.

Downloads portrait photos from randomuser.me (free, no API key needed).
Falls back to colored placeholder images if download fails.
Creates criminal_images/{name}/ folders (multi-image, 3 per person).

Usage:
    uv run python scripts/build_dataset.py
    uv run python scripts/build_dataset.py --count 20
    uv run python scripts/build_dataset.py --count 10 --no-download
"""
import os
import sys
import random
import argparse
import urllib.request
import urllib.error
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openpyxl
import cv2
import numpy as np
import config

# ── Synthetic name pool ────────────────────────────────────────────────────────
FIRST_NAMES = [
    "Ahmed", "Bilal", "Carlos", "David", "Elena",
    "Fatima", "George", "Hassan", "Ivan", "Jasmine",
    "Kevin", "Layla", "Marcus", "Nadia", "Omar",
    "Pavel", "Quinn", "Rania", "Stefan", "Tanya",
    "Usman", "Vera", "Walter", "Xena", "Yusuf",
]
LAST_NAMES = [
    "Khan", "Smith", "Rodriguez", "Johnson", "Malik",
    "Brown", "Ali", "Petrov", "Wilson", "Hassan",
    "Torres", "Lee", "Nguyen", "Patel", "Silva",
]

CRIME_TYPES  = ["violent", "drug", "fraud", "property", "sexual", "non-violent"]
STATUS_TYPES = ["wanted", "bail", "parole", "released", "imprisoned"]
GENDERS      = ["M", "F"]

IMAGES_PER_PERSON = 3    # number of face images to download per criminal

# randomuser.me portrait IDs (1–99)
_MALE_POOL   = list(range(1, 99))
_FEMALE_POOL = list(range(1, 99))


# ── Profile generation ─────────────────────────────────────────────────────────

def _random_date(years_back_max: int = 8) -> date:
    days = random.randint(30, years_back_max * 365)
    return date.today() - timedelta(days=days)


def _generate_profile(idx: int, first: str, last: str, gender: str) -> dict:
    name = f"{first.lower()}_{last.lower()}"
    return {
        "id":                    idx + 1,
        "name":                  name,
        "photo":                 "",           # blank → multi-image folder used
        "age":                   random.randint(18, 55),
        "gender":                gender,
        "crime_type":            random.choice(CRIME_TYPES),
        "num_prior_convictions": random.randint(0, 7),
        "weapon_used":           random.choice([True, False]),
        "current_status":        random.choice(STATUS_TYPES),
        "last_crime_date":       _random_date(),
        "victim_count":          random.randint(0, 8),
    }


# ── Image acquisition ──────────────────────────────────────────────────────────

def _download(url: str, save_path: str, timeout: int = 12) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:   # suspiciously small — treat as failure
            return False
        with open(save_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    [DL] Failed {url} — {e}")
        return False


def _make_placeholder(save_path: str, name: str, idx: int, img_idx: int):
    """
    Creates a colored 200×200 placeholder with name text.
    NOTE: These will NOT pass dlib's HOG face detector — replace with real
    face photos for actual face recognition to work.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    color = ((idx * 37 + img_idx * 13) % 256,
             (idx * 83 + img_idx * 29) % 256,
             (idx * 131 + img_idx * 47) % 256)
    img[:] = color
    label = name[:12]
    cv2.putText(img, label, (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"#{idx+1}-{img_idx+1}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.imwrite(save_path, img)
    print(f"    [PLACEHOLDER] {os.path.basename(save_path)}")


def _portrait_urls(gender: str, count: int, used: set) -> list[tuple]:
    pool = _MALE_POOL if gender == "M" else _FEMALE_POOL
    available = [i for i in pool if (gender, i) not in used]
    random.shuffle(available)
    chosen = available[:count]
    g_key = "men" if gender == "M" else "women"
    return [(f"https://randomuser.me/api/portraits/{g_key}/{uid}.jpg", (gender, uid))
            for uid in chosen]


# ── Excel writer ───────────────────────────────────────────────────────────────

def _write_excel(profiles: list):
    os.makedirs(os.path.dirname(config.CRIMINALS_XLSX), exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Criminals"

    headers = ["id", "name", "photo", "age", "gender", "crime_type",
               "num_prior_convictions", "weapon_used", "current_status",
               "last_crime_date", "victim_count"]
    ws.append(headers)

    for p in profiles:
        ws.append([
            p["id"], p["name"], p["photo"], p["age"], p["gender"],
            p["crime_type"], p["num_prior_convictions"], p["weapon_used"],
            p["current_status"], str(p["last_crime_date"]), p["victim_count"],
        ])

    wb.save(config.CRIMINALS_XLSX)
    print(f"[BUILD] Excel saved → {config.CRIMINALS_XLSX}  ({len(profiles)} rows)")


# ── Main builder ───────────────────────────────────────────────────────────────

def build_dataset(count: int = 15, download_images: bool = True):
    os.makedirs(config.CRIMINALS_DIR, exist_ok=True)

    random.seed(42)   # reproducible dataset
    names = list(zip(
        random.sample(FIRST_NAMES, min(count, len(FIRST_NAMES))),
        [random.choice(LAST_NAMES) for _ in range(count)],
    ))
    # Pad if count > FIRST_NAMES pool
    while len(names) < count:
        names.append((random.choice(FIRST_NAMES), random.choice(LAST_NAMES)))

    profiles       = []
    used_portraits: set = set()

    print(f"\n[BUILD] Generating {count} criminal profiles + images...\n")

    for i, (first, last) in enumerate(names[:count]):
        gender  = random.choice(GENDERS)
        profile = _generate_profile(i, first, last, gender)
        name    = profile["name"]

        print(f"[BUILD] {i+1:02d}. {name:<30} "
              f"status={profile['current_status']:<12} crime={profile['crime_type']}")

        person_dir = os.path.join(config.CRIMINALS_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        if download_images:
            urls = _portrait_urls(gender, IMAGES_PER_PERSON, used_portraits)
            for img_idx, (url, uid_key) in enumerate(urls):
                save_path = os.path.join(person_dir, f"{img_idx+1:03d}.jpg")
                if _download(url, save_path):
                    used_portraits.add(uid_key)
                    print(f"    [DL] {img_idx+1}/{len(urls)} ✓")
                else:
                    _make_placeholder(save_path, name, i, img_idx)
        else:
            for img_idx in range(IMAGES_PER_PERSON):
                save_path = os.path.join(person_dir, f"{img_idx+1:03d}.jpg")
                _make_placeholder(save_path, name, i, img_idx)

        profiles.append(profile)

    _write_excel(profiles)

    print(f"\n[BUILD] ✓ Dataset ready:")
    print(f"  {len(profiles)} criminals  |  {count * IMAGES_PER_PERSON} images")
    print(f"  {config.CRIMINALS_XLSX}")
    print(f"  {config.CRIMINALS_DIR}/")
    print(f"\n  Next steps:")
    print(f"    uv run python scripts/dataset_validator.py")
    print(f"    uv run python main.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WatchAI Dataset Builder")
    parser.add_argument("--count",       type=int, default=15,
                        help="Number of criminals to generate (default: 15)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip image downloads (creates placeholders only)")
    args = parser.parse_args()
    build_dataset(args.count, download_images=not args.no_download)
