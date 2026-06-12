"""
WatchAI — Centralized Configuration
All tunable parameters live here. Modify this file to adjust system behavior
without touching service code.

Ground-level CCTV surveillance for real locations:
banks, streets, shops, offices, parking lots, etc.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
CRIMINALS_DIR  = str(BASE_DIR / "criminal_images")
CRIMINALS_XLSX = str(BASE_DIR / "criminals" / "criminals.xlsx")
MODELS_DIR     = str(BASE_DIR / "models")

# Data directories (flybase pattern)
DATA_DIR   = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
LOGS_DIR   = DATA_DIR / "logs"
DB_DIR     = DATA_DIR / "db"

for _d in [FRAMES_DIR, LOGS_DIR, DB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging paths ──────────────────────────────────────────────────────────────
JSON_LOG_PATH   = str(LOGS_DIR / "events.json")
OBJECT_LOG_PATH = str(LOGS_DIR / "object_detected.log")
ALERT_LOG_PATH  = str(LOGS_DIR / "alerts.log")

# ── Frame Processing ──────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 3      # process 1 in N frames (webcam mode)
WEBCAM_ROTATION        = False  # apply 90° CW if webcam is mounted sideways

# ── Motion Detection (MOG2) ───────────────────────────────────────────────────
MOG2_HISTORY       = 500
MOG2_VAR_THRESHOLD = 50
MIN_MOTION_AREA    = 3000       # minimum contour area (px²) to count as real motion

# ── Face Recognition ──────────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD = 0.55     # L2 distance; lower = stricter (dlib default = 0.6)
NUM_JITTERS          = 1        # >1 improves accuracy but slows encoding linearly
FACE_DETECT_MODEL    = "hog"    # "hog" (CPU fast) or "cnn" (GPU accurate)

# ── YOLO Person Detection ─────────────────────────────────────────────────────
PERSON_YOLO_PATH      = str(BASE_DIR / "yolov8n.pt")
PERSON_CONF_THRESHOLD = 0.50
YOLO_MODEL            = "yolov8n.pt"   # for YOLODetector class (auto-downloads)
YOLO_CONFIDENCE       = 0.35

YOLO_CLASSES_OF_INTEREST = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "backpack", "handbag", "suitcase", "cell phone",
]

# ── Alert / Logging ───────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = 8          # suppress duplicate alerts per person (in-memory)

# ── Risk Engine ───────────────────────────────────────────────────────────────
RISK_LEVELS = [
    (76, "CRITICAL"),
    (56, "HIGH"),
    (31, "MEDIUM"),
    (0,  "LOW"),
]

# ── VLM / Gemini ──────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
VLM_MODEL       = os.getenv("VLM_MODEL", "gemini-2.5-flash")
VLM_TIMEOUT     = 60  # seconds

# ── Ollama (local LLM for agent + embeddings) ─────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_LLM_MODEL  = os.getenv("AGENT_LLM_MODEL", "gemma4:e2b")
EMBEDDING_MODEL  = "nomic-embed-text"

# ── Databases (SQLite + ChromaDB) ────────────────────────────────────────────
SQLITE_DB_PATH        = str(DB_DIR / "watchai_security.db")
CHROMA_PERSIST_DIR    = str(DB_DIR / "chromadb")
CHROMA_COLLECTION_NAME = "watchai_events"

# ── Alert Rules ───────────────────────────────────────────────────────────────
ALERT_RULES = {
    "midnight_loitering": {
        "description": "Person detected between 22:00 and 05:00",
        "hours":       list(range(22, 24)) + list(range(0, 6)),
        "objects":     ["person"],
        "severity":    "HIGH",
    },
    "unauthorized_vehicle_night": {
        "description": "Vehicle detected after hours (20:00–06:00)",
        "hours":       list(range(20, 24)) + list(range(0, 7)),
        "objects":     ["car", "truck", "bus", "motorcycle"],
        "severity":    "MEDIUM",
    },
    "crowd_detected": {
        "description": "3 or more people detected simultaneously",
        "min_count":   3,
        "objects":     ["person"],
        "severity":    "MEDIUM",
    },
    "high_threat_vlm": {
        "description": "VLM assessed threat level as HIGH",
        "threat_level": "HIGH",
        "severity":    "HIGH",
    },
    "criminal_face_match": {
        "description": "Known criminal identified by face recognition",
        "severity":    "HIGH",
    },
    "critical_risk_score": {
        "description": "Criminal risk score reached CRITICAL threshold (≥76)",
        "min_score":   76,
        "severity":    "CRITICAL",
    },
    "criminal_high_threat_scene": {
        "description": "Known criminal present during a HIGH-threat VLM scene",
        "severity":    "CRITICAL",
    },
}

# ── Camera / Video Source ───────────────────────────────────────────────────
DATA_SAMPLE_DIR = BASE_DIR / "data_sample"   # local video files (.mp4/.avi)

# Named location labels for each camera source.
# Key: webcam device index (int) or video filename stem (str)
# Value: human-readable location label shown in dashboard and alerts
CAMERA_LOCATIONS: dict = {
    0:             "Main Entrance",
    1:             "Parking Lot",
    2:             "Street Corner",
    "bank":        "Bank Branch — Counter Area",
    "street":      "Street — Junction",
    "shop":        "Retail Shop — Floor",
    "office":      "Office — Reception",
    "parking":     "Parking Lot — Zone A",
    "atm":         "ATM Vestibule",
    "warehouse":   "Warehouse — Loading Bay",
}
DEFAULT_LOCATION = "Surveillance Zone"   # fallback when no label is configured

# Business closing hours — used by night-vehicle and loitering rules
BUSINESS_CLOSE_HOUR = 21   # 9 PM
BUSINESS_OPEN_HOUR  = 6    # 6 AM

# Process every Nth frame from a video file (speed vs. detail)
VIDEO_FRAME_STEP    = 5
SIMULATION_FPS      = 1    # seconds between Streamlit reruns in live mode

# ── Display / Streamlit ─────────────────────────────────────────────────
APP_TITLE      = "WatchAI — CCTV Surveillance"
APP_ICON       = "🎯"
DISPLAY_WINDOW = "WatchAI — Surveillance Feed"  # kept for OpenCV fallback
SHOW_DISPLAY   = True
