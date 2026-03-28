"""
pages/7_🔧_Admin.py — Study Helper v2
Admin panel for system monitoring, debugging, and maintenance.
Gated behind require_auth + ADMIN_USER_ID env var.
"""
import streamlit as st
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from styles.theme import inject_css, sidebar_header
from model_manager import MODELS, get_llm, _is_ollama_running
from supabase_client import get_supabase
from auth import require_auth
from failed_queries_db import get_failed_queries, delete_failed_query

st.set_page_config(
    page_title="Admin Panel — Study Helper",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()
user_id = require_auth()

ADMIN_IDS = [s.strip() for s in os.getenv("ADMIN_USER_IDS", "").split(",") if s.strip()]
if ADMIN_IDS and user_id not in ADMIN_IDS:
    st.error("⛔ Access denied — admin privileges required.")
    st.stop()

sidebar_header(active_page="Admin")
st.markdown("# 🔧 Admin Panel")

# ── Helper ────────────────────────────────────────────────────────────────────
def _safe_count(table: str) -> int:
    """Return row count for a Supabase table, 0 on error."""
    try:
        resp = get_supabase().table(table).select("*", count="exact").limit(0).execute()
        return resp.count or 0
    except Exception:
        return 0

# ── 1. Model Health ──────────────────────────────────────────────────────────
st.markdown("## 🤖 Model Health")

rows = []
for m in MODELS:
    label, provider, key_env = m["label"], m["provider"], m["key_env"]
    if key_env is None:
        if provider == "ollama":
            ok = _is_ollama_running()
            rows.append({"Model": label, "Provider": provider,
                         "Status": "✅ Running" if ok else "❌ Offline",
                         "Details": "" if ok else "Run: ollama serve"})
        else:
            rows.append({"Model": label, "Provider": provider,
                         "Status": "⚠️ Unknown", "Details": ""})
    else:
        has_key = bool(os.getenv(key_env))
        rows.append({"Model": label, "Provider": provider,
                     "Status": "✅ Key set" if has_key else "❌ No key",
                     "Details": "" if has_key else f"Set {key_env}"})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if st.button("🧪 Live-test all models"):
    from langchain_core.messages import HumanMessage
    with st.spinner("Testing…"):
        for m in MODELS:
            try:
                llm = get_llm(m["label"])
                r = llm.invoke([HumanMessage(content="Say OK")])
                st.write(f"✅ **{m['label']}** — {r.content.strip()[:40]}")
            except Exception as e:
                st.write(f"❌ **{m['label']}** — {str(e)[:60]}")

# ── 2. Database Stats ────────────────────────────────────────────────────────
st.markdown("## 📊 Database Stats")

tables = {
    "Documents":     "sh_documents",
    "Flashcards":    "sh_flashcards",
    "Quiz History":  "sh_quiz_history",
    "Conversations": "sh_conversations",
    "Failed Queries":"sh_failed_queries",
    "Canvas Cache":  "sh_canvas_cache",
}

cols = st.columns(len(tables))
for col, (nice, tbl) in zip(cols, tables.items()):
    col.metric(nice, _safe_count(tbl))

# ── 3. Failed Queries ────────────────────────────────────────────────────────
st.markdown("## ❌ Failed Queries")

failed = get_failed_queries(user_id, limit=25)
if failed:
    for fq in failed:
        with st.expander(f"{fq.get('query', '?')[:80]}  —  {fq.get('created_at', '')[:16]}"):
            st.write(f"**Score:** {fq.get('top_score', 'N/A')}")
            st.write(f"**User:** {fq.get('user_id', '?')}")
            if st.button("🗑️ Delete", key=f"del_fq_{fq['id']}"):
                delete_failed_query(fq["id"])
                st.rerun()
else:
    st.caption("No failed queries logged yet.")

# ── 4. Canvas Debug ──────────────────────────────────────────────────────────
st.markdown("## 📅 Canvas Cache")

try:
    sb = get_supabase()
    resp = sb.table("sh_canvas_cache").select("*").order("cached_at", desc=True).limit(5).execute()
    cache_rows = resp.data or []
    if cache_rows:
        for row in cache_rows:
            with st.expander(f"Key: {row.get('cache_key', '?')}  —  cached {row.get('cached_at', '')[:16]}"):
                st.json(row.get("data", row))
    else:
        st.caption("Cache is empty.")
except Exception as e:
    st.warning(f"Canvas cache read error: {e}")

# ── 5. ElevenLabs ────────────────────────────────────────────────────────────
st.markdown("## 🎙️ TTS Status")

el_key = os.getenv("ELEVENLABS_API_KEY")
el_voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
if el_key:
    st.success(f"✅ ElevenLabs key set  ·  Voice ID: `{el_voice}`")
else:
    st.info("ElevenLabs not configured — gTTS fallback active.")

# ── 6. System Info ───────────────────────────────────────────────────────────
st.markdown("## ℹ️ System Info")

c1, c2 = st.columns(2)
c1.metric("Ollama", "✅ Running" if _is_ollama_running() else "❌ Offline")
c2.metric("Environment", os.getenv("ENVIRONMENT", "development"))

st.json({
    "python": sys.version,
    "streamlit": st.__version__,
    "timestamp": datetime.now(timezone.utc).isoformat(),
})
