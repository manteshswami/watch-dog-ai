"""
WatchAI — Streamlit CCTV Surveillance Dashboard
5 tabs: Live Monitor | Alert Feed | Criminal Log | Event Log | Agent Chat

Launch:
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import os
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Suppress noisy warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import config
from main import run_pipeline_step, get_camera_location, load_singletons

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar keep-open via st.iframe (replaces deprecated components.v1.html) ──
_sidebar_js_b64 = (
    "data:text/html;base64,"
    + __import__("base64").b64encode(b"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;">
<script>
(function keepSidebarOpen() {
    try {
        var doc = window.parent.document;
        var sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'none';
            sidebar.style.left = '0';
        }
        var btn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (btn) btn.style.display = 'none';
        var ctrl = doc.querySelector('[data-testid="collapsedControl"]');
        if (ctrl) ctrl.style.display = 'none';
    } catch(e) {}
    setTimeout(keepSidebarOpen, 400);
})();
</script>
</body></html>""").decode()
)
st.iframe(src=_sidebar_js_b64, height=1)

# ── Custom CSS (Dark Theme + Dossier UI) ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #CBD5E1 !important;
}

.main, .stApp {
    background: #0F172A !important; /* Deep Slate Background */
}

/* ── Always Visible Sidebar ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[kind="header"] { display: none !important; }

section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
    transform: none !important;
    left: 0 !important;
    visibility: visible !important;
    display: block !important;
    background: #1E293B !important;
    border-right: 1px solid #334155;
}

.block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 1rem !important;
}
[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; visibility: hidden; }

/* Global Text Overrides */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown b, .stMarkdown strong, .stMarkdown em,
.stText, .stCaption, label, .stRadio label,
[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #CBD5E1 !important;
}

/* ── Page Title ── */
.watchai-title {
    text-align: center;
    padding: 0.5rem 0 0.75rem;
    margin-bottom: 0.5rem;
}
.watchai-title h1 {
    font-family: 'Syne', 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.15;
    background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.watchai-title .watchai-icon {
    font-size: 2.2rem;
    vertical-align: middle;
    margin-right: 6px;
    -webkit-text-fill-color: initial;
}
.watchai-title p {
    font-size: 0.92rem;
    color: #94A3B8;
    margin-top: 7px;
    font-weight: 400;
}
.live-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #10B981;
    margin-left: 10px;
    animation: pulse-dot 2s infinite;
    vertical-align: middle;
}
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ── Sidebar Brand ── */
[data-testid="stSidebar"] .sidebar-brand {
    font-family: 'Syne', 'Inter', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #06B6D4 !important;
    padding: 0.25rem 0 0.5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1E293B;
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 4px;
    border: 1px solid #334155;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8 !important;
    border-radius: 8px !important;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 8px 16px !important;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #334155 !important;
    color: #F1F5F9 !important;
}
.stTabs [aria-selected="true"] {
    background: #334155 !important;
    color: #06B6D4 !important;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(6, 182, 212, 0.15);
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    border: none !important;
    padding: 16px 0 0 0 !important;
}

/* ── Severity badges ── */
.badge-CRITICAL { background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.badge-HIGH     { background: #7C2D12; color: #FDBA74; border: 1px solid #EA580C; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.badge-MEDIUM   { background: #713F12; color: #FDE047; border: 1px solid #CA8A04; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.badge-LOW      { background: #14532D; color: #86EFAC; border: 1px solid #16A34A; padding: 3px 12px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* ── Alert cards ── */
.alert-card-CRITICAL { border-left: 4px solid #EF4444; background: #450A0A; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; color: #F1F5F9; border: 1px solid #334155; border-left: 4px solid #EF4444; }
.alert-card-HIGH     { border-left: 4px solid #F97316; background: #431407; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; color: #F1F5F9; border: 1px solid #334155; border-left: 4px solid #F97316; }
.alert-card-MEDIUM   { border-left: 4px solid #EAB308; background: #422006; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; color: #F1F5F9; border: 1px solid #334155; border-left: 4px solid #EAB308; }
.alert-card-LOW      { border-left: 4px solid #22C55E; background: #052E16; border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px; color: #F1F5F9; border: 1px solid #334155; border-left: 4px solid #22C55E; }

/* ── Stat boxes & Event Cards ── */
.stat-box, .event-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    color: #F1F5F9;
}
.event-card { text-align: left; margin-bottom: 12px; }
.stat-box:hover, .event-card:hover {
    transform: translateY(-2px);
    border-color: #06B6D4;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.15);
}
.stat-number        { font-size: 2.2rem; font-weight: 700; color: #06B6D4; line-height: 1.2; }
.stat-number-orange { font-size: 2.2rem; font-weight: 700; color: #FB923C; line-height: 1.2; }
.stat-number-red    { font-size: 2.2rem; font-weight: 700; color: #F87171; line-height: 1.2; }
.stat-number-amber  { font-size: 2.2rem; font-weight: 700; color: #FBBF24; line-height: 1.2; }
.stat-label         { font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }

/* ── Advanced Dossier Profiles (Criminal Log) ── */
.dossier-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-left: 4px solid #EF4444; /* Red accent for criminals */
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    transition: transform 0.2s;
}
.dossier-card:hover {
    transform: translateY(-2px);
    border-left-color: #F87171;
    box-shadow: 0 6px 16px rgba(239, 68, 68, 0.15);
}
.dossier-header {
    font-size: 1.3rem;
    font-weight: 800;
    color: #F1F5F9;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
    margin-bottom: 12px;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.02em;
}
.dossier-label {
    color: #64748B;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.dossier-value {
    color: #F1F5F9;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── VLM Scene Intelligence Tags ── */
.vlm-tag {
    background: #0F172A;
    border: 1px solid #3B82F6;
    color: #60A5FA;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 6px;
    margin-bottom: 6px;
    display: inline-block;
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.15);
}

/* ── Camera metadata pill ── */
.cam-pill {
    display: inline-block;
    background: #134E4A;
    border: 1px solid #0F766E;
    border-radius: 8px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #5EEAD4;
    margin: 3px;
}

/* ── Chat bubbles ── */
.chat-user  { background: #134E4A; border: 1px solid #0F766E; border-radius: 14px 14px 4px 14px; padding: 12px 16px; margin: 6px 0; color: #F1F5F9; }
.chat-agent { background: #1E293B; border: 1px solid #334155; border-radius: 14px 14px 14px 4px; padding: 12px 16px; margin: 6px 0; color: #F1F5F9; }

/* ── Section title ── */
.section-title {
    font-size: 1.0rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 12px;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
}

/* ── Buttons & Inputs ── */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid #475569;
    color: #F1F5F9 !important;
    background: #1E293B;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #22D3EE !important;
    color: #22D3EE !important;
    box-shadow: 0 2px 8px rgba(6,182,212,0.15);
}
.stButton > button[kind="primary"] {
    background: #0D9488 !important;
    color: #FFFFFF !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0F766E !important;
    box-shadow: 0 4px 14px rgba(13,148,136,0.3) !important;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    background-color: #1E293B !important;
    border-color: #334155 !important;
    color: #F1F5F9 !important;
}
.stTextInput input {
    background: #1E293B !important;
    border-color: #334155 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
}
.streamlit-expanderHeader {
    color: #F1F5F9 !important;
    font-weight: 500;
    background: #1E293B !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}
.streamlit-expanderContent {
    background: #0F172A !important;
    border-left: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    border-radius: 0 0 10px 10px;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #334155;
}
.stAlert { background: #1E293B; border: 1px solid #334155; border-radius: 10px; }
code {
    background: #1E293B !important;
    color: #22D3EE !important;
    border-radius: 4px;
    padding: 2px 6px;
}
.stCodeBlock {
    background: #1E293B !important;
    border: 1px solid #334155;
    border-radius: 8px;
}
.stCodeBlock code { color: #CBD5E1 !important; }
hr { border-color: #334155 !important; }
.stRadio div { color: #CBD5E1 !important; }

/* Hide sidebar JS iframe */
iframe[height="1"] { display: none !important; position: absolute !important; }
</style>
""", unsafe_allow_html=True)


# ── Centered Page Title ────────────────────────────────────────────────________
st.markdown(
    f"""
    <div class="watchai-title">
        <h1>
            <span class="watchai-icon">{config.APP_ICON}</span>{config.APP_TITLE}<span class="live-dot"></span>
        </h1>
        <p>Intelligent CCTV monitoring &mdash; powered by AI vision</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Cached singletons ──────────────────────────────────────────────────────────

@st.cache_resource
def _load_pipeline():
    load_singletons()
    return True

@st.cache_resource
def _load_agent():
    from services.security_agent import build_agent
    return build_agent()

@st.cache_resource
def _get_indexer():
    from services.database import FrameIndexer
    return FrameIndexer()


# ── Session state init ─────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "frame_id":        0,
        "running":         False,
        "processed":       [],
        "alerts":          [],
        "chat_history":    [],
        "cap":             None,
        "location":        config.DEFAULT_LOCATION,
        "camera_id":       "CAM-LIVE",
        "source":          0,
        "agent":           None,
        "behavior_buffer": [],
        "loiter_timers":   {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _badge(severity: str) -> str:
    return f'<span class="badge-{severity}">{severity}</span>'

def _threat_color(level: str) -> str:
    return {
        "HIGH":     "#FB923C", # Orange 400
        "MEDIUM":   "#FBBF24", # Amber 400
        "LOW":      "#4ADE80", # Green 400
        "CRITICAL": "#F87171", # Red 400
    }.get(level, "#94A3B8")

def _risk_color(level: str) -> str:
    return _threat_color(level)

def _open_cap(source) -> cv2.VideoCapture:
    if st.session_state.get("cap") is not None:
        try:
            st.session_state["cap"].release()
        except:
            pass
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    st.session_state["cap"] = cap
    return cap


def _render_stats():
    processed       = st.session_state["processed"]
    alerts          = st.session_state["alerts"]
    high            = sum(1 for e in processed if e.get("threat_level") == "HIGH")
    criminal_events = [e for e in processed if e.get("criminal_name")]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(processed)}</div>'
            f'<div class="stat-label">Frames Processed</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number-orange">{len(alerts)}</div>'
            f'<div class="stat-label">Alerts Triggered</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number-red">{high}</div>'
            f'<div class="stat-label">High Threat Events</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="stat-box"><div class="stat-number-amber">{len(criminal_events)}</div>'
            f'<div class="stat-label">Criminal Sightings</div></div>',
            unsafe_allow_html=True,
        )


def _render_camera_meta(event: dict):
    ts = event.get("timestamp", "")[:19].replace("T", " ")
    st.markdown(
        f'<span class="cam-pill">📷 {event.get("camera_id","?")}</span>'
        f'<span class="cam-pill">📍 {event.get("location","?")}</span>'
        f'<span class="cam-pill">🕐 {ts}</span>'
        f'<span class="cam-pill">🎯 Threat: {event.get("threat_level","LOW")}</span>'
        + (
            f'<span class="cam-pill" style="color:#FCA5A5;background:#7F1D1D;border-color:#DC2626">'
            f'⚠️ {event.get("criminal_name","").title()}</span>'
            if event.get("criminal_name") else ""
        ),
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">{config.APP_ICON} {config.APP_TITLE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### 📷 Camera Selection")
    
    # Let user toggle between Hardware/Preconfigured nodes or an explicit file upload interface
    feed_mode = st.radio("Select Input Logic", ["Configured Feeds", "Upload Video File"])
    
    if feed_mode == "Configured Feeds":
        # Auto discover local test items
        sample_dir = config.DATA_SAMPLE_DIR
        videos = sorted(sample_dir.glob("*.mp4")) + sorted(sample_dir.glob("*.avi")) \
                 if sample_dir.exists() else []
        
        # Map professional names directly to configurations
        camera_map = {
            "📷 Primary Live Webcam": {
                "source": 0,
                "location": "Control Center — Hardware Cam",
                "camera_id": "CAM-LIVE"
            }
        }
        
        if videos:
            for idx, vid in enumerate(videos):
                clean_name = vid.stem.replace("_", " ").title()
                camera_map[f"📹 {clean_name} Feed"] = {
                    "source": str(vid),
                    "location": get_camera_location(str(vid)),
                    "camera_id": f"CAM-VID-{idx+1:02d}"
                }
        else:
            # High quality fallback presets if directory is empty
            camera_map.update({
                "🏦 Bank Vault Branch": {"source": 0, "location": "Bank Branch — Counter Area", "camera_id": "CAM-BANK"},
                "🛒 Retail Floor Shop": {"source": 0, "location": "Retail Shop — Floor", "camera_id": "CAM-SHOP"},
                "🅿️ Parking Lot Zone A": {"source": 0, "location": "Parking Lot — Zone A", "camera_id": "CAM-PARK"},
                "🏧 ATM Vestibule Node": {"source": 0, "location": "ATM Vestibule Area", "camera_id": "CAM-ATM"}
            })

        selected_node = st.selectbox("Select Active Feed Node", list(camera_map.keys()))
        node_data = camera_map[selected_node]

        st.session_state["source"]    = node_data["source"]
        st.session_state["location"]  = node_data["location"]
        st.session_state["camera_id"] = node_data["camera_id"]
        
    else:
        # File Upload Mode
        uploaded_video = st.file_uploader("Upload Video File Target", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            # Persistent temporary write out to local system disk path for continuous OpenCV ingestion parsing
            temp_path = Path("app_uploaded_temp.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            
            st.session_state["source"]    = str(temp_path)
            st.session_state["location"]  = "Uploaded Workspace Instance File"
            st.session_state["camera_id"] = "CAM-UPLOAD"
        else:
            st.session_state["source"]    = None
            st.info("Upload video stream framework processing asset file above to initialize.")

    st.markdown("---")
    st.markdown("#### ⚙️ Surveillance Controls")

    col_s, col_r = st.columns(2)
    with col_s:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            if st.session_state["source"] is not None:
                _open_cap(st.session_state["source"])
                st.session_state["running"] = True
            else:
                st.error("No valid active source file logic loaded to start computation pipeline.")
    with col_r:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state["running"] = False

    if st.button("🔄 Reset System Dashboard", use_container_width=True):
        if st.session_state.get("cap"):
            st.session_state["cap"].release()
        st.session_state.update({
            "frame_id": 0, "processed": [], "alerts": [],
            "running": False, "cap": None, "behavior_buffer": [], "loiter_timers": {},
        })
        st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Monitor",
    "🚨 Alert Feed",
    "👤 Criminal Log",
    "📋 Event Log",
    "💬 Agent Chat",
])

_load_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    _render_stats()
    st.markdown("---")

    left_col, right_col = st.columns([3, 2])

    # Instantiate explicitly structured UI container layout blocks for persistent state injection
    frame_ph = left_col.empty()
    meta_ph  = left_col.empty()
    alert_ph = left_col.empty()

    with right_col:
        st.markdown('<div class="section-title">🧠 Advanced Scene Intelligence</div>', unsafe_allow_html=True)
        vlm_ph = st.empty()
        st.markdown('<div class="section-title">📡 Detection Details</div>', unsafe_allow_html=True)
        det_ph = st.empty()

    # Dynamic Frame Ingestion Execution Loop
    if st.session_state["running"]:
        cap = st.session_state.get("cap")
        if cap is None or not cap.isOpened():
            st.error("❌ Source offline or uninitialized. Verify webcam access or file availability.")
            st.session_state["running"] = False
        else:
            # 1. Dynamically calculate the hardware video file or camera preset baseline frames-per-second
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            target_delay = 1.0 / video_fps if video_fps > 0 else 0.033 # Fallback to ~30 FPS sequence layout

            while st.session_state["running"]:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    # Automatically handle video loop termination elegantly
                    alert_ph.info("🏁 Video feed parsing terminal endpoint reached.")
                    st.session_state["running"] = False
                    break
                
                st.session_state["frame_id"] += 1
                fid = st.session_state["frame_id"]

                # Fast frame filtering/skipping logic handles quick background parsing
                if fid % config.PROCESS_EVERY_N_FRAMES == 0:
                    event = run_pipeline_step(
                        cap=cap,
                        frame_id=fid,
                        location=st.session_state["location"],
                        camera_id=st.session_state["camera_id"],
                    )

                    if event is not None:
                        st.session_state["processed"].append(event)
                        for a in event.get("alerts", []):
                            st.session_state["alerts"].append(a)

                        # ─── REAL-TIME BEHAVIORAL PATTERN EXTRACTOR (Stateful Temporal tracking) ───
                        current_time = datetime.now()
                        st.session_state["behavior_buffer"].append({
                            "timestamp": current_time,
                            "detections": event.get("detections", []),
                            "threat": event.get("threat_level", "LOW"),
                            "vlm_obs": event.get("key_observations", [])
                        })
                        
                        # Restrict tracking registry sliding lookback frame window matrix context scale
                        if len(st.session_state["behavior_buffer"]) > 30:
                            st.session_state["behavior_buffer"].pop(0)
                        
                        recent_events = st.session_state["behavior_buffer"]
                        high_threat_count = sum(1 for e in recent_events if e["threat"] in ["MEDIUM", "HIGH", "CRITICAL"])
                        avg_person_count = np.mean([sum(1 for d in e["detections"] if d['label'] == 'person') for e in recent_events]) if recent_events else 0
                        
                        # Behavioral Classification State Matrix Definition
                        behavior_status = "STABLE / NORMAL"
                        behavior_color = "#10B981" # Green
                        
                        if high_threat_count > 12:
                            behavior_status = "ANOMALOUS / AGGRESSIVE SYSTEM STATE"
                            behavior_color = "#EF4444" # Red
                        elif avg_person_count >= 3:
                            behavior_status = "CROWD ACCUMULATION IN PROGRESS"
                            behavior_color = "#F59E0B" # Orange
                        elif len(recent_events) >= 10 and any(len(e["detections"]) > 0 for e in recent_events):
                            behavior_status = "PERSISTENT LOITERING ACTIVITY DETECTED"
                            behavior_color = "#3B82F6" # Blue

                        disp_rgb = cv2.cvtColor(event["display_frame"], cv2.COLOR_BGR2RGB)
                        
                        # Direct image container rendering injection drops blinking entirely
                        frame_ph.image(disp_rgb, use_container_width=True,
                                       caption=f"Frame {fid} — {event['location']}")

                        with meta_ph.container():
                            _render_camera_meta(event)

                        if event.get("alert_triggered"):
                            # Flush container then display items
                            alert_ph.empty()
                            for a in event["alerts"]:
                                alert_ph.error(f"🚨 **[{a['severity']}]** — {a['alert_text']}")
                        else:
                            alert_ph.success("✅ Surveillance clear — No threats logged")

                        # Advanced VLM Display
                        threat = event.get("threat_level", "LOW")
                        color  = _threat_color(threat)
                        
                        obs_list = event.get("key_observations", [])
                        if not obs_list:
                            obs_list = ["System scanning environment for anomalies..."]
                            
                        tags_html = "".join(f"<span class='vlm-tag'>[{str(o).split()[0].upper()}]</span>" for o in obs_list[:4])

                        vlm_ph.markdown(
                            f'<div class="event-card" style="border-top: 3px solid #3B82F6;">'
                            f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #334155; padding-bottom:8px; margin-bottom:10px;">'
                            f'<span style="font-weight:600; color:#E2E8F0; font-family:\'JetBrains Mono\', monospace;">&gt; VLM_SCAN_ACTIVE <span class="live-dot" style="width:6px;height:6px;background:#3B82F6;"></span></span>'
                            f'<span style="font-size:0.8rem; color:{color}; font-weight:800; letter-spacing:0.05em;">{threat} THREAT</span>'
                            f'</div>'
                            f'<div style="color:#94A3B8; font-size:0.95rem; line-height:1.6; margin-bottom:14px;">'
                            f'{event.get("vlm_description","Scanning surrounding viewport for profile matches...")}'
                            f'</div>'
                            f'<div><span style="font-size:0.75rem; color:#64748B; text-transform:uppercase; margin-bottom:4px; display:block;">Detected Context:</span>{tags_html}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        dets = event.get("detections", [])
                        if dets:
                            det_str = "  ".join(
                                f"`{d['label']}` ({d['confidence']:.2f})" for d in dets
                            )
                            det_ph.markdown(
                                f'<div class="event-card" style="font-size:0.85rem; border-left: 4px solid {behavior_color};">'
                                f'<div style="display:flex; justify-content:space-between; margin-bottom:8px;">'
                                f'<b style="color:#D1D5DB">🎭 LIVE BEHAVIOR SIGNATURE:</b>'
                                f'<span style="color:{behavior_color}; font-weight:700;">{behavior_status}</span>'
                                f'</div>'
                                f'<hr style="border-color:#334155; margin: 4px 0 10px 0;">'
                                f'<b style="color:#D1D5DB">YOLO Object Matrix:</b> {len(dets)} objects tracking<br>'
                                f'<span style="color:#94A3B8">{det_str}</span><br><br>'
                                f'<b style="color:#D1D5DB">Biometric Match:</b> '
                                f'<span style="color:#94A3B8">'
                                f'{event.get("criminal_name","None") or "None"}</span>'
                                f' &nbsp;|&nbsp; '
                                f'<b style="color:#D1D5DB">Risk Profile:</b> '
                                f'<span style="color:{_risk_color(event.get("risk_level","LOW"))};'
                                f'font-weight:600">'
                                f'{event.get("risk_level","LOW")} (Score: {event.get("risk_score",0)})</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            det_ph.markdown(
                                f'<div class="event-card" style="font-size:0.85rem; border-left: 4px solid {behavior_color};">'
                                f'<div style="display:flex; justify-content:space-between; margin-bottom:8px;">'
                                f'<b style="color:#D1D5DB">🎭 LIVE BEHAVIOR SIGNATURE:</b>'
                                f'<span style="color:{behavior_color}; font-weight:700;">{behavior_status}</span>'
                                f'</div>'
                                f'<hr style="border-color:#334155; margin: 4px 0 10px 0;">'
                                f'<span style="color:#94A3B8">No optical metadata bounding layers tracked this frame.</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                
                # 2. Synchronized Processing Pacing & Frame Dropping Engine Logic
                process_time = time.time() - loop_start
                
                if process_time < target_delay:
                    # Ingestion sequence operating faster than native video framework timing -> Apply sleep constraint
                    time.sleep(target_delay - process_time)
                else:
                    # CPU pipeline processing is bound -> Drop trailing video buffer stack to maintain true real-time execution
                    frames_to_skip = int(process_time / target_delay)
                    for _ in range(frames_to_skip):
                        cap.grab() # Low level rapid grab drops pixel decoding overhead entirely

    else:
        processed = st.session_state["processed"]
        if not processed:
            frame_ph.info("▶ Press **Start** in the sidebar node interface to begin surveillance tracking.")
        else:
            ev       = processed[-1]
            disp_rgb = cv2.cvtColor(ev["display_frame"], cv2.COLOR_BGR2RGB)
            frame_ph.image(disp_rgb, use_container_width=True,
                           caption=f"Last logged frame — {ev['location']}")
            with meta_ph.container():
                _render_camera_meta(ev)

    if st.session_state["processed"]:
        st.markdown("---")
        st.markdown("### 📼 Historical Archival Feed")
        for ev in reversed(st.session_state["processed"][-10:]):
            with st.expander(
                f"Frame {ev['frame_id']} — {ev['location']} "
                f"| {ev['timestamp'][:19].replace('T',' ')} "
                f"| Threat Matrix: {ev['threat_level']}"
                + (" 🚨" if ev.get("alert_triggered") else " ✅"),
                expanded=False,
            ):
                fc1, fc2 = st.columns([2, 1])
                img_path = ev.get("frame_path")
                if img_path and Path(img_path).exists():
                    img = cv2.imread(img_path)
                    if img is not None:
                        fc1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                  use_container_width=True)
                dets = ev.get("detections", [])
                fc2.markdown(
                    f'<div style="font-size:0.83rem;color:#CBD5E1;line-height:1.9">'
                    f'<b>Objects Tracked:</b> {", ".join(d["label"] for d in dets) or "none"}<br>'
                    f'<b>Target ID Match:</b> {ev.get("criminal_name") or "none"}<br>'
                    f'<b>Risk Computation:</b> {ev.get("risk_level","LOW")} ({ev.get("risk_score",0)})<br>'
                    f'<b>VLM Intelligence Summary:</b> <span style="color:#94A3B8">'
                    f'{ev.get("vlm_description","")[:120]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALERT FEED
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    alerts = st.session_state["alerts"]

    if not alerts:
        st.info("No network incidents logged yet. Stream feeds to generate operational events.")
    else:
        sev_counts  = Counter(a["severity"] for a in alerts)
        SEV_ORDER   = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        sorted_sevs = sorted(
            sev_counts.items(),
            key=lambda x: SEV_ORDER.index(x[0]) if x[0] in SEV_ORDER else 99,
        )
        cols = st.columns(max(len(sorted_sevs), 1))
        for col, (sev, cnt) in zip(cols, sorted_sevs):
            col.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-number" style="color:{_threat_color(sev)}">{cnt}</div>'
                f'<div class="stat-label">{sev}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_sev, col_loc = st.columns(2)
        sev_filter = col_sev.selectbox(
            "Filter by severity", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            key="alert_sev_filter",
        )
        locations  = sorted(set(a.get("location", "") for a in alerts))
        loc_filter = col_loc.selectbox(
            "Filter by location node", ["All"] + locations,
            key="alert_loc_filter",
        )

        filtered = alerts
        if sev_filter != "All":
            filtered = [a for a in filtered if a["severity"] == sev_filter]
        if loc_filter != "All":
            filtered = [a for a in filtered if a.get("location") == loc_filter]

        st.markdown(
            f'<p style="color:#94A3B8;font-size:0.85rem">'
            f'<b style="color:#F1F5F9">{len(filtered)}</b> intelligence alert(s) found</p>',
            unsafe_allow_html=True,
        )

        for a in reversed(filtered):
            try:
                ts = datetime.fromisoformat(a["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = a.get("timestamp", "?")[:19]
            criminal_tag = f" | 👤 {a['criminal'].title()}" if a.get("criminal") else ""
            st.markdown(
                f'<div class="alert-card-{a["severity"]}">'
                f'{_badge(a["severity"])} &nbsp; '
                f'<span style="color:#94A3B8;font-size:0.8rem">{ts} &nbsp;|&nbsp; '
                f'📍 {a.get("location","?")} &nbsp;|&nbsp; 📷 {a.get("camera_id","?")}'
                f'{criminal_tag}</span><br>'
                f'<span style="font-size:0.9rem;color:#F1F5F9">{a["alert_text"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CRIMINAL LOG (Dossier Mode)
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    criminal_events = [e for e in st.session_state["processed"] if e.get("criminal_name")]

    if not criminal_events:
        st.info("No biometric face matches recorded on active layers yet.")
    else:
        st.markdown(
            f'<h3 style="color:#F1F5F9;font-size:1.1rem;font-weight:600;margin-bottom:12px">'
            f'👤 {len(criminal_events)} Active BOLO Subject Dossier Profiles Captured</h3>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        for ev in reversed(criminal_events):
            name  = ev.get("criminal_name", "Unknown Subject").title()
            conf  = ev.get("face_confidence", 0.85) * 100 
            score = ev.get("risk_score", 0)
            level = ev.get("risk_level", "HIGH")
            ts    = ev.get("timestamp", "")[:19].replace("T", " ")
            loc   = ev.get("location", "Unknown Location")
            cam   = ev.get("camera_id", "CAM-UNKNOWN")
            color = _risk_color(level)
            
            mock_id = f"ID-{abs(hash(name)) % 100000:05d}"
            mock_warrant = "Active Warrant - Armed & Dangerous" if level in ["CRITICAL", "HIGH"] else "Person of Interest (Monitor)"

            st.markdown('<div class="dossier-card">', unsafe_allow_html=True)
            col_img, col_data = st.columns([1, 3])
            
            with col_img:
                img_path = ev.get("frame_path")
                if img_path and Path(img_path).exists():
                    img = cv2.imread(img_path)
                    if img is not None:
                        st.image(
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                            caption=f"Capture Target: {ts[-8:]}"
                        )
                else:
                    st.markdown(
                        '<div style="height:150px; background:#0F172A; display:flex; align-items:center; justify-content:center; color:#475569; border-radius:8px; border:1px dashed #334155;">'
                        'No Image Captured</div>', 
                        unsafe_allow_html=True
                    )

            with col_data:
                st.markdown(
                    f'<div class="dossier-header">{name}'
                    f'<span style="float:right; font-size:1.0rem; color:{color}; font-family:\'JetBrains Mono\';">Match Confidence: {conf:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f'<div style="display:flex; flex-wrap:wrap; gap:24px;">'
                    f'<div><div class="dossier-label">Subject Identification ID</div><div class="dossier-value">{mock_id}</div></div>'
                    f'<div><div class="dossier-label">Legal Warrant Status</div><div class="dossier-value" style="color:{"#EF4444" if "Warrant" in mock_warrant else "#FBBF24"};">{mock_warrant}</div></div>'
                    f'<div><div class="dossier-label">Threat Risk Profile</div><div class="dossier-value" style="color:{color};">{level} (Score: {score}/100)</div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown('<hr style="margin: 12px 0; border-color: #334155;">', unsafe_allow_html=True)
                
                st.markdown(
                    f'<div style="display:flex; flex-wrap:wrap; gap:24px;">'
                    f'<div><div class="dossier-label">Chronological Sighting</div><div class="dossier-value" style="color:#94A3B8;">{ts}</div></div>'
                    f'<div><div class="dossier-label">Source Camera Location Node</div><div class="dossier-value" style="color:#94A3B8;">📍 {loc}</div></div>'
                    f'<div><div class="dossier-label">System Hardware Identifier</div><div class="dossier-value" style="color:#94A3B8;">📷 {cam}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    processed = st.session_state["processed"]

    if not processed:
        st.info("Log database clear. Initiate feeds to index metadata.")
    else:
        import pandas as pd

        rows = []
        for e in processed:
            rows.append({
                "Frame":    e["frame_id"],
                "Time":     e["timestamp"][:19].replace("T", " "),
                "Location": e["location"],
                "Camera":   e["camera_id"],
                "Objects":  ", ".join(e.get("objects", [])) or "—",
                "Threat":   e.get("threat_level", "LOW"),
                "Criminal": e.get("criminal_name", "").title() or "—",
                "Risk":     f"{e.get('risk_level','LOW')} ({e.get('risk_score',0)})",
                "Alert":    "🚨" if e.get("alert_triggered") else "✅",
                "VLM":      e.get("vlm_description", "")[:80],
            })

        df = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        loc_filter = c1.selectbox(
            "Filter by target node",
            ["All"] + sorted(df["Location"].unique().tolist()),
            key="event_loc_filter",
        )
        thr_filter = c2.selectbox(
            "Filter by risk vector",
            ["All", "HIGH", "MEDIUM", "LOW"],
            key="event_thr_filter",
        )
        crm_filter = c3.selectbox(
            "Filter by target match",
            ["All"] + sorted(set(r for r in df["Criminal"].tolist() if r != "—")),
            key="event_crm_filter",
        )

        if loc_filter != "All":
            df = df[df["Location"] == loc_filter]
        if thr_filter != "All":
            df = df[df["Threat"] == thr_filter]
        if crm_filter != "All":
            df = df[df["Criminal"] == crm_filter]

        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button("⬇ Export Ledger Logs to CSV", csv, "watchai_events.csv", "text/csv")

        st.markdown("---")
        st.markdown(
            '<div class="section-title">📊 Visual Classification Matrix Frequency</div>',
            unsafe_allow_html=True,
        )
        all_objects = []
        for e in processed:
            all_objects.extend(e.get("objects", []))
        if all_objects:
            obj_df = pd.Series(all_objects).value_counts().reset_index()
            obj_df.columns = ["Object Layer Class", "Frequency Count"]
            st.bar_chart(obj_df.set_index("Object Layer Class"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENT Chat
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown(
        '<div class="section-title">💬 Query WatchAI Intelligence Analyst Node</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Interrogate indexed database telemetry files through descriptive natural phrasing. "
        "Examples: *Identify incidents flagged at bank counter* | *Isolate high severity risk vectors*"
    )

    if st.session_state.get("agent") is None:
        with st.spinner("Compiling security agent operational layers..."):
            st.session_state["agent"] = _load_agent()

    for role, text in st.session_state["chat_history"]:
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(text)
        else:
            with st.chat_message("assistant", avatar="🎯"):
                st.markdown(text)

    st.markdown(
        '<p style="font-size:0.82rem;font-weight:600;color:#CBD5E1;margin-bottom:8px">'
        'Quick Console Directives:</p>',
        unsafe_allow_html=True,
    )
    qcols = st.columns(4)
    quick_prompts = [
        "Summarise today",
        "Show CRITICAL alerts",
        "Find all criminals spotted",
        "Show HIGH threat events",
    ]
    for col, prompt in zip(qcols, quick_prompts):
        if col.button(prompt, use_container_width=True):
            st.session_state["_pending_prompt"] = prompt

    user_input = st.chat_input("Input intelligence query...")

    pending  = st.session_state.pop("_pending_prompt", None)
    question = user_input or pending

    if question:
        st.session_state["chat_history"].append(("user", question))
        agent = st.session_state["agent"]

        with st.spinner("🎯 Analyzing data blocks..."):
            try:
                if agent:
                    result = agent.invoke(
                        {"messages": [("user", question)]},
                        config={"configurable": {"thread_id": "watchai-session-1"}},
                    )
                    raw    = result["messages"][-1].content
                    answer = (
                        "\n".join(
                            b.get("text", "") for b in raw
                            if isinstance(b, dict) and "text" in b
                        )
                        if isinstance(raw, list) else str(raw)
                    )
                    if not answer.strip():
                        answer = "No response generated. Rephrase search request criteria."
                else:
                    from services.security_agent import fallback_agent_response
                    answer = fallback_agent_response(question)
            except Exception:
                from services.security_agent import fallback_agent_response
                answer = f"*[Agent error — fallback mode]*\n\n{fallback_agent_response(question)}"

        st.session_state["chat_history"].append(("agent", answer))
        st.rerun()

    with st.expander("📊 High-Level Daily Operations Brief"):
        if st.button("Compile Operational Overview"):
            indexer = _get_indexer()
            summary = indexer.get_daily_summary()
            obj_str = (
                ", ".join(f"{v} {k}" for k, v in summary["object_counts"].items())
                or "none"
            )
            st.markdown(
                f'<div style="font-size:0.88rem;color:#CBD5E1;line-height:2">'
                f'<b>Operations Date:</b> {summary["date"]}<br>'
                f'<b>Metrics Processed:</b> {summary["total_frames"]} frames<br>'
                f'<b>Incidents Logged:</b> {summary["total_alerts"]} alerts<br>'
                f'<b>High Risk Threshold Breaches:</b> {summary["high_threat_frames"]}<br>'
                f'<b>Target BOLO Sightings Identified:</b> {summary.get("criminal_matches", 0)}<br>'
                f'<b>Classified Matrix Instances:</b> {obj_str}'
                f'</div>',
                unsafe_allow_html=True,
            )