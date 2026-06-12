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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main  { background-color: #0d1117; }
.stApp { background-color: #0d1117; }

/* Severity badges */
.badge-CRITICAL { background:#7c1c1c; color:#ff6b6b; border:1px solid #c94040; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-HIGH     { background:#4a1e1e; color:#ff8c42; border:1px solid #c0622a; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-MEDIUM   { background:#3a3010; color:#f0c040; border:1px solid #a08020; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.badge-LOW      { background:#0f2c1e; color:#4caf80; border:1px solid #2a7a50; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-weight:600; font-family:'JetBrains Mono',monospace; }

/* Alert cards */
.alert-card-CRITICAL { border-left:4px solid #ff2020; background:#200808; border-radius:8px; padding:12px 16px; margin-bottom:8px; }
.alert-card-HIGH     { border-left:4px solid #ff6b35; background:#1a1208; border-radius:8px; padding:12px 16px; margin-bottom:8px; }
.alert-card-MEDIUM   { border-left:4px solid #f0c040; background:#1a1a08; border-radius:8px; padding:12px 16px; margin-bottom:8px; }
.alert-card-LOW      { border-left:4px solid #4caf80; background:#081a10; border-radius:8px; padding:12px 16px; margin-bottom:8px; }

/* Stat boxes */
.stat-box    { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:16px; text-align:center; }
.stat-number { font-size:2rem; font-weight:700; color:#58a6ff; }
.stat-label  { font-size:0.78rem; color:#8b949e; font-weight:500; text-transform:uppercase; letter-spacing:0.05em; }

/* Event card */
.event-card { background:#161b22; border:1px solid #21262d; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.event-card:hover { border-color:#388bfd; transition:border-color 0.2s; }

/* Camera metadata pill */
.cam-pill { display:inline-block; background:#1c2128; border:1px solid #30363d; border-radius:6px; padding:4px 10px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#79c0ff; margin:2px; }

/* Chat bubbles */
.chat-user  { background:#1c2a3a; border-radius:12px 12px 4px 12px; padding:10px 14px; margin:6px 0; color:#cdd9e5; }
.chat-agent { background:#161b22; border:1px solid #21262d; border-radius:12px 12px 12px 4px; padding:10px 14px; margin:6px 0; color:#e6edf3; }

.section-title { font-size:1.1rem; font-weight:600; color:#e6edf3; margin-bottom:12px; border-bottom:1px solid #21262d; padding-bottom:8px; }
</style>
""", unsafe_allow_html=True)


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
        "frame_id":       0,
        "running":        False,
        "processed":      [],
        "alerts":         [],
        "chat_history":   [],
        "cap":            None,
        "location":       config.DEFAULT_LOCATION,
        "camera_id":      "CAM-01",
        "source":         0,
        "agent":          None,
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
        "HIGH":     "#ff6b35",
        "MEDIUM":   "#f0c040",
        "LOW":      "#4caf80",
        "CRITICAL": "#ff2020",
    }.get(level, "#8b949e")

def _risk_color(level: str) -> str:
    return _threat_color(level)

def _open_cap(source) -> cv2.VideoCapture:
    """Open a VideoCapture; auto-release any existing one."""
    if st.session_state.get("cap") is not None:
        st.session_state["cap"].release()
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    st.session_state["cap"] = cap
    return cap


def _render_stats():
    processed = st.session_state["processed"]
    alerts    = st.session_state["alerts"]
    high      = sum(1 for e in processed if e.get("threat_level") == "HIGH")

    criminal_events = [e for e in processed if e.get("criminal_name")]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(processed)}</div><div class="stat-label">Frames Processed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff6b35">{len(alerts)}</div><div class="stat-label">Alerts Triggered</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#ff2020">{high}</div><div class="stat-label">High Threat Events</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-box"><div class="stat-number" style="color:#f0c040">{len(criminal_events)}</div><div class="stat-label">Criminal Sightings</div></div>', unsafe_allow_html=True)


def _render_camera_meta(event: dict):
    ts = event.get("timestamp", "")[:19].replace("T", " ")
    st.markdown(
        f'<span class="cam-pill">📷 {event.get("camera_id","?")}</span>'
        f'<span class="cam-pill">📍 {event.get("location","?")}</span>'
        f'<span class="cam-pill">🕐 {ts}</span>'
        f'<span class="cam-pill">🎯 Threat: {event.get("threat_level","LOW")}</span>'
        + (f'<span class="cam-pill" style="color:#ff6b6b">⚠️ {event.get("criminal_name","").title()}</span>'
           if event.get("criminal_name") else ""),
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## {config.APP_ICON} {config.APP_TITLE}")
    st.markdown("---")

    # ── Source selection ───────────────────────────────────────────────────────
    st.markdown("#### 📷 Camera Source")
    source_type = st.radio("Source type", ["Webcam", "Video File"], horizontal=True)

    if source_type == "Webcam":
        cam_idx  = st.selectbox("Camera index", [0, 1, 2, 3], index=0)
        source   = cam_idx
        location = st.text_input(
            "Location label",
            value=config.CAMERA_LOCATIONS.get(cam_idx, config.DEFAULT_LOCATION),
        )
        camera_id = st.text_input("Camera ID", value=f"CAM-0{cam_idx+1}")
    else:
        # List video files from data_sample/
        sample_dir = config.DATA_SAMPLE_DIR
        videos = sorted(sample_dir.glob("*.mp4")) + sorted(sample_dir.glob("*.avi")) \
                 if sample_dir.exists() else []
        if videos:
            vid_choice = st.selectbox("Video file", [v.name for v in videos])
            source     = str(sample_dir / vid_choice)
            location   = st.text_input(
                "Location label",
                value=get_camera_location(source),
            )
        else:
            st.warning("No videos found in data_sample/. Switch to Webcam.")
            source   = 0
            location = config.DEFAULT_LOCATION
        camera_id = st.text_input("Camera ID", value="CAM-VID")

    st.session_state["source"]    = source
    st.session_state["location"]  = location
    st.session_state["camera_id"] = camera_id

    st.markdown("---")
    st.markdown("#### ⚙️ Controls")

    col_s, col_r = st.columns(2)
    with col_s:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            _open_cap(source)
            st.session_state["running"] = True
    with col_r:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state["running"] = False

    if st.button("🔄 Reset", use_container_width=True):
        if st.session_state.get("cap"):
            st.session_state["cap"].release()
        st.session_state.update({
            "frame_id": 0, "processed": [], "alerts": [],
            "running": False, "cap": None,
        })
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📡 Model Config")
    st.code(
        f"VLM:   {config.VLM_MODEL}\n"
        f"Agent: {config.AGENT_LLM_MODEL}\n"
        f"Embed: {config.EMBEDDING_MODEL}",
        language="text",
    )

    st.markdown("---")
    st.markdown("#### 🏢 Location Presets")
    presets = {
        "🏦 Bank":    ("Bank Branch — Counter Area", "CAM-BANK"),
        "🛣️ Street":  ("Street — Junction",          "CAM-STR"),
        "🛒 Shop":    ("Retail Shop — Floor",         "CAM-SHOP"),
        "🅿️ Parking": ("Parking Lot — Zone A",        "CAM-PARK"),
        "🏧 ATM":     ("ATM Vestibule",               "CAM-ATM"),
    }
    for label, (loc, cid) in presets.items():
        if st.button(label, use_container_width=True):
            st.session_state["location"]  = loc
            st.session_state["camera_id"] = cid
            st.rerun()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎥 Live Monitor",
    "🚨 Alert Feed",
    "👤 Criminal Log",
    "📋 Event Log",
    "💬 Agent Chat",
])

_load_pipeline()   # init AI models once


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    _render_stats()
    st.markdown("---")

    left_col, right_col = st.columns([3, 2])

    frame_ph = left_col.empty()
    meta_ph  = left_col.empty()
    alert_ph = left_col.empty()

    with right_col:
        st.markdown('<div class="section-title">🧠 VLM Scene Analysis</div>', unsafe_allow_html=True)
        vlm_ph   = st.empty()
        st.markdown('<div class="section-title">📡 Detection Details</div>', unsafe_allow_html=True)
        det_ph   = st.empty()

    # ── Run one pipeline step ─────────────────────────────────────────────────
    if st.session_state["running"]:
        cap = st.session_state.get("cap")
        if cap is None or not cap.isOpened():
            st.warning("Camera not opened. Press ▶ Start.")
            st.session_state["running"] = False
        else:
            st.session_state["frame_id"] += 1
            fid = st.session_state["frame_id"]

            # Skip non-process frames (frame_step)
            if fid % config.PROCESS_EVERY_N_FRAMES == 0:
                event = run_pipeline_step(
                    cap=cap,
                    frame_id=fid,
                    location=st.session_state["location"],
                    camera_id=st.session_state["camera_id"],
                )

                if event is None:
                    # Motion gate fired or end of video
                    frame_ph.info("⏳ No motion detected — waiting...")
                else:
                    st.session_state["processed"].append(event)
                    for a in event.get("alerts", []):
                        st.session_state["alerts"].append(a)

                    # Display annotated frame
                    disp = event["display_frame"]
                    disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                    frame_ph.image(disp_rgb, use_container_width=True,
                                   caption=f"Frame {fid} — {event['location']}")

                    # Camera metadata strip
                    with meta_ph.container():
                        _render_camera_meta(event)

                    # Alerts banner
                    if event.get("alert_triggered"):
                        for a in event["alerts"]:
                            alert_ph.error(f"🚨 **[{a['severity']}]** — {a['alert_text']}")
                    else:
                        alert_ph.success("✅ No alerts this frame")

                    # VLM panel
                    threat = event.get("threat_level", "LOW")
                    color  = _threat_color(threat)
                    vlm_ph.markdown(
                        f'<div class="event-card">'
                        f'<b>Threat:</b> <span style="color:{color};font-weight:700">{threat}</span><br><br>'
                        f'<b>Description:</b><br>{event.get("vlm_description","—")}<br><br>'
                        + (
                            '<b>Observations:</b><ul>'
                            + "".join(f"<li>{o}</li>" for o in event.get("key_observations", []))
                            + "</ul>"
                            if event.get("key_observations") else ""
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Detection details
                    dets = event.get("detections", [])
                    if dets:
                        det_str = "  ".join(f"`{d['label']}` ({d['confidence']:.2f})" for d in dets)
                        det_ph.markdown(
                            f"**YOLO:** {len(dets)} objects<br>{det_str}<br><br>"
                            f"**Criminal:** {event.get('criminal_name','None') or 'None'} "
                            f"| **Risk:** {event.get('risk_level','LOW')} ({event.get('risk_score',0)})",
                            unsafe_allow_html=True,
                        )
                    else:
                        det_ph.markdown("**YOLO:** No objects detected")

        time.sleep(config.SIMULATION_FPS)
        st.rerun()

    else:
        processed = st.session_state["processed"]
        if not processed:
            frame_ph.info("▶ Press **Start** in the sidebar to begin surveillance.")
        else:
            ev = processed[-1]
            disp_rgb = cv2.cvtColor(ev["display_frame"], cv2.COLOR_BGR2RGB)
            frame_ph.image(disp_rgb, use_container_width=True,
                           caption=f"Last processed frame — {ev['location']}")
            with meta_ph.container():
                _render_camera_meta(ev)

    # ── Processed frame history ───────────────────────────────────────────────
    if st.session_state["processed"]:
        st.markdown("---")
        st.markdown("### 📼 Processed Frame History")
        for ev in reversed(st.session_state["processed"][-10:]):   # last 10
            with st.expander(
                f"Frame {ev['frame_id']} — {ev['location']} "
                f"| {ev['timestamp'][:19].replace('T',' ')} "
                f"| Threat: {ev['threat_level']}"
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
                    f"**Objects:** {', '.join(d['label'] for d in dets) or 'none'}<br>"
                    f"**Criminal:** {ev.get('criminal_name') or 'none'}<br>"
                    f"**Risk:** {ev.get('risk_level','LOW')} ({ev.get('risk_score',0)})<br>"
                    f"**VLM:** {ev.get('vlm_description','')[:120]}",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALERT FEED
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    alerts = st.session_state["alerts"]

    if not alerts:
        st.info("No alerts yet. Start surveillance to generate events.")
    else:
        sev_counts = Counter(a["severity"] for a in alerts)
        cols = st.columns(max(len(sev_counts), 1))
        for col, (sev, cnt) in zip(cols, sorted(sev_counts.items(), key=lambda x: ["LOW","MEDIUM","HIGH","CRITICAL"].index(x[0]) if x[0] in ["LOW","MEDIUM","HIGH","CRITICAL"] else 0, reverse=True)):
            col.markdown(
                f'<div class="stat-box">'
                f'<div class="stat-number" style="color:{_threat_color(sev)}">{cnt}</div>'
                f'<div class="stat-label">{sev}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_sev, col_loc = st.columns(2)
        sev_filter = col_sev.selectbox("Filter by severity", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        locations  = sorted(set(a.get("location","") for a in alerts))
        loc_filter = col_loc.selectbox("Filter by location", ["All"] + locations)

        filtered = alerts
        if sev_filter != "All":
            filtered = [a for a in filtered if a["severity"] == sev_filter]
        if loc_filter != "All":
            filtered = [a for a in filtered if a.get("location") == loc_filter]

        st.markdown(f"**{len(filtered)} alert(s)**")

        for a in reversed(filtered):
            try:
                ts = datetime.fromisoformat(a["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts = a.get("timestamp", "?")[:19]
            criminal_tag = f" | 👤 {a['criminal'].title()}" if a.get("criminal") else ""
            st.markdown(
                f'<div class="alert-card-{a["severity"]}">'
                f'{_badge(a["severity"])} &nbsp; {ts} &nbsp;|&nbsp; '
                f'📍 {a.get("location","?")} &nbsp;|&nbsp; 📷 {a.get("camera_id","?")}'
                f'{criminal_tag}<br>'
                f'<span style="font-size:0.92rem">{a["alert_text"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CRIMINAL LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    criminal_events = [
        e for e in st.session_state["processed"]
        if e.get("criminal_name")
    ]

    if not criminal_events:
        st.info("No criminal face matches yet. Start surveillance to detect matches.")
    else:
        st.markdown(f"### 👤 {len(criminal_events)} Criminal Sighting(s) Detected")
        st.markdown("---")

        for ev in reversed(criminal_events):
            name  = ev.get("criminal_name", "").title()
            conf  = ev.get("face_confidence", 0)
            score = ev.get("risk_score", 0)
            level = ev.get("risk_level", "LOW")
            ts    = ev.get("timestamp", "")[:19].replace("T", " ")
            loc   = ev.get("location", "?")
            cam   = ev.get("camera_id", "?")

            color = _risk_color(level)

            with st.container():
                st.markdown(
                    f'<div class="event-card">'
                    f'<span style="font-size:1.1rem;font-weight:700;color:#e6edf3">! {name}</span>'
                    f'<span style="float:right;color:{color};font-weight:600">{level} ({score})</span>'
                    f'<br>'
                    f'<span style="color:#8b949e;font-size:0.85rem">'
                    f'📷 {cam} &nbsp;|&nbsp; 📍 {loc} &nbsp;|&nbsp; 🕐 {ts} &nbsp;|&nbsp; '
                    f'Face conf: {conf:.2f}'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

                img_path = ev.get("frame_path")
                if img_path and Path(img_path).exists():
                    img = cv2.imread(img_path)
                    if img is not None:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                 use_container_width=True,
                                 caption=f"{name} — {ts}")
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    processed = st.session_state["processed"]

    if not processed:
        st.info("No events yet. Start surveillance.")
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
        loc_filter = c1.selectbox("Filter by location", ["All"] + sorted(df["Location"].unique().tolist()))
        thr_filter = c2.selectbox("Filter by threat",   ["All", "HIGH", "MEDIUM", "LOW"])
        crm_filter = c3.selectbox("Filter by criminal", ["All"] + sorted(
            set(r for r in df["Criminal"].tolist() if r != "—")
        ))

        if loc_filter != "All":
            df = df[df["Location"] == loc_filter]
        if thr_filter != "All":
            df = df[df["Threat"] == thr_filter]
        if crm_filter != "All":
            df = df[df["Criminal"] == crm_filter]

        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "watchai_events.csv", "text/csv")

        st.markdown("---")
        st.markdown("#### 📊 Object Detection Frequency")
        all_objects = []
        for e in processed:
            all_objects.extend(e.get("objects", []))
        if all_objects:
            obj_df = pd.Series(all_objects).value_counts().reset_index()
            obj_df.columns = ["Object", "Count"]
            st.bar_chart(obj_df.set_index("Object"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AGENT CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("#### 💬 Ask the WatchAI Security Analyst")
    st.caption(
        "Query the surveillance database in natural language. "
        "Examples: *Who was spotted at the bank today?* | *Show all CRITICAL alerts* | "
        "*Find John Doe sightings* | *What happened at the parking lot after 9pm?*"
    )

    # Init agent once
    if st.session_state.get("agent") is None:
        with st.spinner("Loading security agent..."):
            st.session_state["agent"] = _load_agent()

    # Render chat history
    for role, text in st.session_state["chat_history"]:
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(text)
        else:
            with st.chat_message("assistant", avatar="🎯"):
                st.markdown(text)

    # Quick prompts
    st.markdown("**Quick queries:**")
    qcols = st.columns(4)
    quick_prompts = [
        "Summarize today",
        "Show CRITICAL alerts",
        "Find all criminals spotted",
        "Show HIGH threat events",
    ]
    for col, prompt in zip(qcols, quick_prompts):
        if col.button(prompt, use_container_width=True):
            st.session_state["_pending_prompt"] = prompt

    user_input = st.chat_input("Ask a security question...")

    pending  = st.session_state.pop("_pending_prompt", None)
    question = user_input or pending

    if question:
        st.session_state["chat_history"].append(("user", question))
        agent = st.session_state["agent"]

        with st.spinner("🎯 Analysing..."):
            try:
                if agent:
                    result = agent.invoke(
                        {"messages": [("user", question)]},
                        config={"configurable": {"thread_id": "watchai-session-1"}},
                    )
                    raw = result["messages"][-1].content
                    answer = (
                        "\n".join(b.get("text", "") for b in raw if isinstance(b, dict) and "text" in b)
                        if isinstance(raw, list) else str(raw)
                    )
                    if not answer.strip():
                        answer = "No response generated. Try rephrasing your query."
                else:
                    from services.security_agent import fallback_agent_response
                    answer = fallback_agent_response(question)
            except Exception as exc:
                from services.security_agent import fallback_agent_response
                answer = f"*[Agent error — fallback mode]*\n\n{fallback_agent_response(question)}"

        st.session_state["chat_history"].append(("agent", answer))
        st.rerun()

    # Daily summary expander
    with st.expander("📊 Daily Summary"):
        if st.button("Generate Summary"):
            indexer = _get_indexer()
            summary = indexer.get_daily_summary()
            obj_str = ", ".join(f"{v} {k}" for k, v in summary["object_counts"].items()) or "none"
            st.markdown(f"""
**Date:** {summary['date']}
- **Frames processed:**   {summary['total_frames']}
- **Alerts triggered:**   {summary['total_alerts']}
- **High-threat frames:** {summary['high_threat_frames']}
- **Criminal sightings:** {summary.get('criminal_matches', 0)}
- **Objects detected:**   {obj_str}
            """)
