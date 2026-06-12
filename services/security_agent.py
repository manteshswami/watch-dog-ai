"""
WatchAI — Security Analyst Agent
LangGraph ReAct agent backed by local Ollama (gemma4:e2b).
Tuned for ground-level CCTV surveillance queries:
banks, streets, shops, offices, parking lots.

Tools:
  - summarize_today      : daily security summary
  - semantic_search      : ChromaDB natural language search
  - query_criminal_alerts: filter alerts by criminal name
  - get_risk_breakdown   : risk score profile for a named criminal
  - get_alerts           : recent alerts filtered by severity
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

import config

logger = logging.getLogger(__name__)


# ── Lazy indexer ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_indexer():
    from services.database import FrameIndexer
    return FrameIndexer()


# ── Tool implementations ──────────────────────────────────────────────────────

def _summarize_today(_: str = "") -> str:
    """Generate today's CCTV security summary."""
    try:
        idx     = _get_indexer()
        summary = idx.get_daily_summary()
        obj_str = (
            ", ".join(f"{v} {k}(s)" for k, v in summary["object_counts"].items())
            or "none"
        )
        return (
            f"Security Summary — {summary['date']}:\n"
            f"- Frames processed   : {summary['total_frames']}\n"
            f"- Alerts triggered   : {summary['total_alerts']}\n"
            f"- High-threat events : {summary['high_threat_frames']}\n"
            f"- Criminal sightings : {summary.get('criminal_matches', 0)}\n"
            f"- Objects detected   : {obj_str}"
        )
    except Exception as exc:
        return f"Error generating summary: {exc}"


def _semantic_search(query: str) -> str:
    """
    Semantic search over all indexed CCTV events using natural language.
    E.g. 'suspicious person near bank entrance after 9pm' or 'crowd at ATM'.
    """
    try:
        idx     = _get_indexer()
        results = idx.semantic_search(query, n_results=5)
        if not results:
            return "No matching events found in the surveillance database."
        lines = []
        for r in results:
            meta = r.get("metadata", {})
            criminal_part = f" | Criminal: {meta['criminal']}" if meta.get("criminal") else ""
            lines.append(
                f"[{meta.get('timestamp', '?')[:16]}] "
                f"{meta.get('location', '?')} "
                f"| Threat: {meta.get('threat_level', '?')} "
                f"| Objects: {meta.get('objects', 'none')}"
                f"{criminal_part}"
            )
        return f"Semantic search results for '{query}':\n" + "\n".join(lines)
    except Exception as exc:
        return f"Semantic search error: {exc}"


def _query_criminal_alerts(name: str) -> str:
    """
    Find all alerts and sightings for a specific criminal by name.
    Input: the criminal's name (partial match supported).
    """
    try:
        idx    = _get_indexer()
        events = idx.query_by_criminal(name)
        alerts = idx.get_all_alerts(criminal=name)
        if not events and not alerts:
            return f"No records found for criminal '{name}'."
        lines = [f"Records for '{name}':"]
        if events:
            lines.append(f"\nSightings ({len(events)} frames):")
            for e in events[:8]:
                lines.append(
                    f"  [{e['timestamp'][:16]}] {e['location']} "
                    f"| Risk: {e.get('risk_level','?')} ({e.get('risk_score',0)})"
                )
        if alerts:
            lines.append(f"\nAlerts triggered ({len(alerts)}):")
            for a in alerts[:5]:
                lines.append(f"  [{a['severity']}] {a['timestamp'][:16]} — {a['alert_text'][:80]}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error querying criminal '{name}': {exc}"


def _get_risk_breakdown(name: str) -> str:
    """
    Return the risk profile breakdown for a named criminal.
    Input: criminal name.
    """
    try:
        from utilis.excel_loader import load_criminal_profiles
        profiles = load_criminal_profiles()
        name_key = name.strip().lower()
        profile  = profiles.get(name_key)
        if profile is None:
            matches = [k for k in profiles if name_key in k]
            if matches:
                profile  = profiles[matches[0]]
                name_key = matches[0]
            else:
                return f"No criminal profile found for '{name}'."

        from services.risk_engine import calculate_risk
        score, level = calculate_risk(profile)

        lines = [
            f"Risk Breakdown — {name_key.title()}:",
            f"  Status       : {profile.get('current_status', '?')}",
            f"  Crime type   : {profile.get('crime_type', '?')}",
            f"  Convictions  : {profile.get('num_prior_convictions', 0)}",
            f"  Victim count : {profile.get('victim_count', 0)}",
            f"  Age          : {profile.get('age', '?')}",
            f"  Last crime   : {profile.get('last_crime_date', '?')}",
            f"  ─────────────────────",
            f"  Risk score   : {score}/100",
            f"  Risk level   : {level}",
        ]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving risk breakdown for '{name}': {exc}"


def _get_all_alerts(severity: str = "") -> str:
    """Return recent alerts, optionally filtered by severity (HIGH/MEDIUM/LOW/CRITICAL)."""
    try:
        idx = _get_indexer()
        sev = severity.strip().upper() or None
        if sev not in (None, "HIGH", "MEDIUM", "LOW", "CRITICAL"):
            sev = None
        alerts = idx.get_all_alerts(severity=sev)
        if not alerts:
            return f"No alerts found{f' with severity {sev}' if sev else ''}."
        lines = []
        for a in alerts[:10]:
            ts = a["timestamp"][:16]
            lines.append(
                f"[{a['severity']}] {ts} | {a['location']} → {a['alert_text'][:100]}"
            )
        return f"Found {len(alerts)} alert(s):\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving alerts: {exc}"


# ── LangChain tool wrappers ───────────────────────────────────────────────────

from langchain_core.tools import tool

@tool
def summarize_today(dummy_arg: str = "") -> str:
    """Generate a summary of today's CCTV security activity: total frames, alerts, high-threat events, criminal sightings, and detected objects. Input: empty string."""
    return _summarize_today(dummy_arg)

@tool
def semantic_search(query: str) -> str:
    """Search CCTV surveillance events using natural language. Works for queries like: 'suspicious person near bank entrance', 'crowd near ATM', 'criminal near parking lot after 9pm'. Input: descriptive query."""
    return _semantic_search(query)

@tool
def query_criminal_alerts(name: str) -> str:
    """Find all CCTV sightings and triggered alerts for a specific criminal by name. Shows timestamp, location, and risk level. Input: criminal name (partial match supported)."""
    return _query_criminal_alerts(name)

@tool
def get_risk_breakdown(name: str) -> str:
    """Get the complete risk score breakdown for a named criminal from the database. Shows crime type, legal status, convictions, victim count, and final risk score/level. Input: criminal name."""
    return _get_risk_breakdown(name)

@tool
def get_alerts(severity: str = "") -> str:
    """Retrieve recent security alerts, optionally filtered by severity level. Input: severity (HIGH, MEDIUM, LOW, CRITICAL) or empty string for all alerts."""
    return _get_all_alerts(severity)


TOOLS = [summarize_today, semantic_search, query_criminal_alerts, get_risk_breakdown, get_alerts]

SYSTEM_PROMPT = """You are WatchAI ,an expert CCTV Security Analyst AI for a ground-level surveillance system.
You monitor real-world locations including banks, street junctions, retail shops, office lobbies, parking lots, and ATMs.
You have access to a live database of CCTV events captured by fixed security cameras.

Your capabilities:
- Summarize daily security activity across all camera locations
- Search for events involving specific objects (people, vehicles)
- Find all sightings and alerts for specific known criminals
- Retrieve detailed risk profiles for criminals in the database
- Answer questions about alert history and security incidents

Always:
- Cite specific timestamps and locations when referencing events
- Flag HIGH and CRITICAL alerts prominently
- Suggest concrete actions (e.g. "contact law enforcement", "review camera footage from 21:30-22:00")
- Be direct and precise — this is a security-critical system

CRITICAL: Never return an empty response. Always call a tool first to gather data, then provide a clear answer."""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(llm_model: str = config.AGENT_LLM_MODEL):
    """Build and return the LangGraph ReAct agent (Ollama-backed)."""
    try:
        from langchain_ollama import ChatOllama
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        llm = ChatOllama(
            model=llm_model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )
        memory = MemorySaver()
        agent  = create_react_agent(
            model=llm,
            tools=TOOLS,
            checkpointer=memory,
            prompt=SYSTEM_PROMPT,
        )
        logger.info(f"Security agent built with model: {llm_model}")
        return agent

    except Exception as exc:
        logger.error(f"Failed to build agent: {exc}")
        return None


def fallback_agent_response(question: str) -> str:
    """Rule-based fallback when LLM is unavailable."""
    q = question.lower()
    if any(w in q for w in ["vehicle", "truck", "car", "bus", "motorcycle"]):
        return _semantic_search("vehicle detection")
    if any(w in q for w in ["person", "people", "loiter", "crowd"]):
        return _query_criminal_alerts(q) if "criminal" in q else _semantic_search("person detected")
    if any(w in q for w in ["criminal", "wanted", "match", "suspect"]):
        name = q.split("criminal")[-1].strip().split()[0] if "criminal" in q else ""
        return _query_criminal_alerts(name) if name else _get_all_alerts("HIGH")
    if any(w in q for w in ["alert", "warning", "threat"]):
        sev = "HIGH" if "high" in q else "CRITICAL" if "critical" in q else ""
        return _get_all_alerts(sev)
    if any(w in q for w in ["summar", "today", "daily"]):
        return _summarize_today("")
    return _semantic_search(question)
