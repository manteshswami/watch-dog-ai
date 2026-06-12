#!/usr/bin/env python3
"""
WatchAI — Evaluation & Metrics System
Runs the surveillance pipeline and collects quantitative metrics.

Metrics collected:
  - Person detection rate (frames with person / motion frames)
  - Face recognition: match rate, per-person counts, avg confidence
  - Alert statistics with cooldown tracking
  - Inference timing (avg, P95 ms/frame)

Usage:
    uv run python scripts/evaluate.py                          # 30s webcam
    uv run python scripts/evaluate.py --source video.mp4
    uv run python scripts/evaluate.py --duration 60
    uv run python scripts/evaluate.py --source 0 --duration 45 --output data/logs/run1.json
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import config
from services.detector import YOLODetector
from services.motion import motion_detected
from services.recognizer import recognize_faces_in_frame
from services.risk_engine import calculate_risk
from utilis.encoder_preload import load_criminal_encodings

B = "\033[1m"; G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; RST = "\033[0m"


# ── Metrics container ─────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self._t0              = time.time()
        self.total_frames     = 0
        self.processed_frames = 0
        self.motion_frames    = 0
        self.person_frames    = 0       # frames where ≥1 person detected
        self.total_persons    = 0       # cumulative person count
        self.face_attempts    = 0
        self.face_matches     = defaultdict(lambda: {"count": 0, "conf_sum": 0.0})
        self.unknown_count    = 0
        self.risk_dist        = defaultdict(int)    # level → count
        self.alert_counts     = defaultdict(int)    # name → alert count
        self.inference_ms     = []

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def record_match(self, name: str, conf: float, level: str):
        self.face_matches[name]["count"]    += 1
        self.face_matches[name]["conf_sum"] += conf
        self.risk_dist[level]               += 1

    def report(self) -> str:
        e   = self.elapsed
        fps = self.total_frames / e if e > 0 else 0

        lines = [
            f"\n{B}{'='*65}",
            f"  WatchAI Evaluation Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*65}{RST}",

            f"\n{B}  OVERVIEW{RST}",
            f"  Duration         : {e:.1f}s",
            f"  Total frames     : {self.total_frames}  ({fps:.1f} fps)",
            f"  Processed (1/{config.PROCESS_EVERY_N_FRAMES}) : {self.processed_frames}",
            f"  Motion frames    : {self.motion_frames}",

            f"\n{B}  PERSON DETECTION{RST}",
        ]

        p_rate = (self.person_frames / self.motion_frames * 100) if self.motion_frames else 0
        color  = G if p_rate >= 70 else (Y if p_rate >= 40 else R)
        lines += [
            f"  Frames w/ person : {self.person_frames} / {self.motion_frames}  "
            f"({color}{p_rate:.1f}%{RST})",
            f"  Total persons    : {self.total_persons}",
            f"  Avg / active frame: {(self.total_persons/self.person_frames):.2f}"
            if self.person_frames else "  Avg / active frame: —",
        ]

        lines.append(f"\n{B}  FACE RECOGNITION{RST}")
        total_matches = sum(v["count"] for v in self.face_matches.values())
        mr = (total_matches / self.face_attempts * 100) if self.face_attempts else 0
        mr_color = G if mr >= 60 else (Y if mr >= 30 else R)
        lines += [
            f"  Face attempts    : {self.face_attempts}",
            f"  Criminal matches : {total_matches}",
            f"  Unknown faces    : {self.unknown_count}",
            f"  Match rate       : {mr_color}{mr:.1f}%{RST}",
        ]
        if self.face_matches:
            lines.append(f"  Per person:")
            for name, d in sorted(self.face_matches.items(), key=lambda x: -x[1]["count"]):
                avg_c = d["conf_sum"] / d["count"] if d["count"] else 0
                lines.append(f"    {name:<28} n={d['count']:>4}  avg_conf={avg_c:.2f}")

        lines.append(f"\n{B}  RISK DISTRIBUTION{RST}")
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            cnt = self.risk_dist.get(lvl, 0)
            if cnt:
                lines.append(f"  {lvl:<10}: {cnt}")

        lines.append(f"\n{B}  ALERTS (cooldown={config.ALERT_COOLDOWN_SECONDS}s){RST}")
        total_alerts = sum(self.alert_counts.values())
        lines.append(f"  Total alerts     : {total_alerts}")
        for name, cnt in self.alert_counts.items():
            lines.append(f"    {name}: {cnt}")

        lines.append(f"\n{B}  TIMING{RST}")
        if self.inference_ms:
            lines += [
                f"  Avg inference    : {np.mean(self.inference_ms):.1f} ms/frame",
                f"  P95 inference    : {np.percentile(self.inference_ms, 95):.1f} ms/frame",
                f"  Min / Max        : {min(self.inference_ms):.0f} / {max(self.inference_ms):.0f} ms",
            ]

        lines.append(f"\n{B}{'='*65}{RST}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "timestamp":         datetime.now().isoformat(),
            "elapsed_seconds":   round(self.elapsed, 2),
            "total_frames":      self.total_frames,
            "processed_frames":  self.processed_frames,
            "motion_frames":     self.motion_frames,
            "person_frames":     self.person_frames,
            "total_persons":     self.total_persons,
            "face_attempts":     self.face_attempts,
            "face_matches":      {k: dict(v) for k, v in self.face_matches.items()},
            "unknown_count":     self.unknown_count,
            "risk_distribution": dict(self.risk_dist),
            "alert_counts":      dict(self.alert_counts),
            "avg_inference_ms":  round(float(np.mean(self.inference_ms)), 2) if self.inference_ms else 0,
            "p95_inference_ms":  round(float(np.percentile(self.inference_ms, 95)), 2) if self.inference_ms else 0,
        }


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run(source, duration: int, known_encodings: list, known_names: list,
        profiles: dict) -> Metrics:

    m    = Metrics()
    yolo = YOLODetector()
    cap  = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"{R}[EVAL] Cannot open: {source}{RST}")
        return m

    is_webcam    = isinstance(source, int)
    frame_count  = 0
    last_alerts: dict = {}
    next_progress = 5.0

    print(f"[EVAL] Running for {duration}s...  (press Ctrl+C to stop early)\n")

    try:
        while True:
            if is_webcam and m.elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    break
                continue

            if is_webcam and config.WEBCAM_ROTATION:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            m.total_frames += 1
            frame_count    += 1

            if frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                continue

            m.processed_frames += 1
            t0 = time.time()

            if not motion_detected(frame):
                continue

            m.motion_frames += 1

            # Person detection
            detections   = yolo.detect(frame)
            person_boxes = yolo.person_boxes(detections)

            if not person_boxes:
                m.inference_ms.append((time.time() - t0) * 1000)
                continue

            m.person_frames += 1
            m.total_persons += len(person_boxes)

            # Face recognition
            if known_encodings:
                frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
                matches   = recognize_faces_in_frame(frame_rgb, known_encodings, known_names)
                m.face_attempts += len(matches)

                for name, conf in matches:
                    if name == "Unknown":
                        m.unknown_count += 1
                    else:
                        profile      = profiles.get(name, {})
                        score, level = calculate_risk(profile)
                        m.record_match(name, conf, level)

                        now  = time.time()
                        last = last_alerts.get(name, 0)
                        if now - last >= config.ALERT_COOLDOWN_SECONDS:
                            m.alert_counts[name] += 1
                            last_alerts[name]    = now
                            print(f"  {R}[ALERT]{RST} {name.upper()} | {level} ({score}) "
                                  f"conf={conf:.2f}")

            m.inference_ms.append((time.time() - t0) * 1000)

            # Progress report every 5s
            if m.elapsed >= next_progress:
                pct = min(100, m.elapsed / duration * 100)
                avg = np.mean(m.inference_ms[-20:]) if m.inference_ms else 0
                print(f"  [{pct:5.1f}%] frames={m.total_frames:>5}  "
                      f"persons={m.person_frames:>4}  "
                      f"matches={sum(v['count'] for v in m.face_matches.values()):>3}  "
                      f"inf={avg:.0f}ms")
                next_progress += 5.0

    except KeyboardInterrupt:
        print("\n[EVAL] Stopped by user.")
    finally:
        cap.release()

    return m


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WatchAI Evaluation")
    parser.add_argument("--source",   default="0",
                        help="Webcam index or video path (default: 0)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Evaluation duration in seconds (default: 30)")
    parser.add_argument("--output",   default="data/logs/eval_report.json",
                        help="JSON output path")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    print(f"\n{B}WatchAI — Evaluation System{RST}")
    print(f"  Source   : {'webcam ' + args.source if isinstance(src, int) else src}")
    print(f"  Duration : {args.duration}s")
    print(f"  Output   : {args.output}\n")

    print("[EVAL] Loading criminal encodings...")
    known_encodings, known_names, profiles = load_criminal_encodings()
    if not known_names:
        print(f"{Y}[EVAL] No criminals loaded — running in detection-only mode{RST}")

    metrics = run(src, args.duration, known_encodings, known_names, profiles)

    print(metrics.report())

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"[EVAL] JSON report saved → {out_path}\n")
