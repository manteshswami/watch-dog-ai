# 🎯 WatchAI — CCTV Surveillance System

AI-powered ground-level CCTV surveillance for real-world locations:
**banks, streets, shops, offices, parking lots, ATMs, and more.**

Combines criminal face recognition with live scene understanding (Gemini VLM),
risk scoring, and a conversational security agent.

---

## 🏗️ Architecture

```
Webcam / CCTV Video File
        │
        ▼
  Motion Gate (MOG2)
        │
        ▼
  YOLO Detection
  (persons, vehicles, bags)
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
  Face Recognition              VLM Analyzer
  (dlib vs criminal DB)         Gemini 2.5 Flash
        │                       ground-level CCTV
        ▼                                 │
  Risk Scoring (0–100) ◄──────────────────┘
        │
        ▼
  Alert Engine (7 rules)
  (loitering, crowd, VLM HIGH, criminal match…)
        │
        ▼
  FrameIndexer
  ┌─────────────┐  ┌────────────┐
  │   SQLite    │  │  ChromaDB  │
  │ frames+alerts│  │  vectors   │
  └─────────────┘  └────────────┘
        │
        ▼
  LangGraph Agent (Ollama gemma4:e2b)
        │
        ▼
  Streamlit Dashboard
  Live Monitor | Alerts | Criminal Log | Events | Agent Chat
```

---

## 🚀 Setup

### Prerequisites
- Python 3.12+
- `uv` package manager
- Ollama running locally with `gemma4:e2b` + `nomic-embed-text`
- Google Gemini API key (optional — VLM falls back gracefully)

### Install
```bash
uv sync
```

### Configure
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Add criminals
```bash
# Put criminal photos in criminal_images/
# Update criminals/criminals.xlsx with profiles
uv run python scripts/build_dataset.py
```

### Run
```bash
uv run streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`

---

## 📁 Project Structure

```
watchai/
├── app/
│   ├── streamlit_app.py    # Streamlit 5-tab dashboard
│   └── display.py          # OpenCV frame annotation
├── services/
│   ├── detector.py         # YOLODetector (typed Detection dataclass)
│   ├── motion.py           # MOG2 motion gate
│   ├── recognizer.py       # Face recognition (dlib)
│   ├── risk_engine.py      # Risk scoring (0–100)
│   ├── database.py         # FrameIndexer (SQLite + ChromaDB)
│   ├── vlm_analyzer.py     # Gemini 2.5 Flash scene analysis
│   ├── alert_engine.py     # 7-rule alert evaluator
│   ├── security_agent.py   # LangGraph ReAct agent
│   └── embeddings.py       # Ollama ChromaDB embeddings
├── utilis/
│   ├── encoder_preload.py  # Criminal face encoding loader
│   └── excel_loader.py     # criminals.xlsx parser
├── scripts/
│   ├── build_dataset.py    # Enroll criminals into DB
│   ├── dataset_validator.py
│   ├── test_face_recognition.py
│   └── evaluate.py
├── criminal_images/        # Criminal mugshots
├── criminals/
│   └── criminals.xlsx      # Criminal profiles
├── models/                 # (empty — no custom models)
├── data_sample/            # Optional: local CCTV video files (.mp4)
├── config.py               # Central configuration
├── main.py                 # Pipeline core (run_pipeline_step)
├── pyproject.toml
└── .env.example
```

---

## 🧩 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🎥 **Live Monitor** | Real-time feed with YOLO + face match overlays, VLM scene description, detection details |
| 🚨 **Alert Feed** | All triggered alerts with severity badges, filterable by location + severity |
| 👤 **Criminal Log** | Every face match: name, confidence, risk score, annotated frame |
| 📋 **Event Log** | Full frame-by-frame table with CSV download + object frequency chart |
| 💬 **Agent Chat** | LangGraph security agent — ask in plain English about events, criminals, alerts |

### Sidebar location presets
🏦 Bank | 🛣️ Street | 🛒 Shop | 🅿️ Parking | 🏧 ATM

---

## 🔔 Alert Rules

| Rule | Trigger | Severity |
|------|---------|---------|
| After-hours loitering | Person detected outside business hours | HIGH |
| Unauthorized vehicle | Vehicle after closing time | MEDIUM |
| Crowd gathering | 3+ people simultaneously | MEDIUM |
| VLM HIGH threat | Gemini assessed scene as HIGH | HIGH |
| Criminal face match | Known criminal identified | HIGH |
| Critical risk score | Criminal risk score ≥ 76 | CRITICAL |
| Criminal + HIGH threat | Criminal present during HIGH-threat scene | CRITICAL |

---

## 🤖 AI Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Person/object detection | YOLOv8n (COCO) | Persons, vehicles, bags |
| Face recognition | dlib 128-d HOG | Criminal matching |
| Scene analysis | Gemini 2.5 Flash | Full scene threat assessment |
| Semantic search | nomic-embed-text (Ollama) | ChromaDB embeddings |
| Security agent | gemma4:e2b (Ollama) | LangGraph ReAct |
