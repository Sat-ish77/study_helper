"""
pages/6_📊_Dashboard.py — Study Helper v2
Simple, honest dashboard: what's due, what to review, how you're doing.
"""
import streamlit as st
from datetime import datetime, timezone

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from supabase_client import get_supabase
from flashcard_db import get_flashcard_stats
from quiz_db import get_quiz_stats, get_recent_quiz_scores
from canvas_api import get_upcoming_events, load_ical_url, fetch_canvas_events
from ingest import get_user_documents

st.set_page_config(
    page_title="Dashboard — Study Helper",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
user_id = require_auth()
if not user_id:
    st.stop()

sidebar_header(active_page="Dashboard")

st.markdown("# 📊 Study Dashboard")

# ── Fetch data once ──────────────────────────────────────────────────────────
fc = get_flashcard_stats(user_id)
qz = get_quiz_stats(user_id)
docs = get_user_documents(user_id)

# ── Top metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documents", len(docs) if docs else 0)
c2.metric("🃏 Cards Due", fc["due"])
c3.metric("📝 Quiz Avg", f"{qz.get('average_score', 0):.0f}%")
c4.metric("🃏 Total Cards", fc["total"], f"{fc['learned']} mastered")

st.markdown("---")

# ── 1. What's Due Soon (Canvas) ─────────────────────────────────────────────
st.markdown("### 📅 What's Due Soon")

ical_url = load_ical_url(user_id)
if ical_url:
    try:
        raw_events = fetch_canvas_events(ical_url, user_id)
        events = get_upcoming_events(raw_events, days_ahead=14)
    except Exception:
        events = []

    if events:
        for ev in events[:8]:
            start = ev.get("start")
            if start:
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                days = (start.date() - datetime.now(timezone.utc).date()).days
                if days <= 0:
                    badge = "🔴 Today"
                elif days == 1:
                    badge = "🟠 Tomorrow"
                elif days <= 3:
                    badge = f"🟡 In {days} days"
                else:
                    badge = f"🟢 In {days} days"
                date_str = start.strftime("%b %d")
            else:
                badge = "⚪ No date"
                date_str = ""

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.05); padding:12px 16px;'
                f' border-radius:8px; margin:6px 0; display:flex;'
                f' justify-content:space-between; align-items:center;">'
                f'<div><strong>{ev.get("title", "Untitled")}</strong>'
                f' <span style="color:#6b7280; font-size:0.85rem;">— {date_str}</span></div>'
                f'<div style="font-size:0.85rem;">{badge}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No upcoming events in the next 2 weeks.")
else:
    st.info("Add your Canvas iCal URL in Study Helper sidebar → Calendar to see upcoming deadlines here.")

st.markdown("---")

# ── 2. Flashcards to Review ─────────────────────────────────────────────────
st.markdown("### 🃏 Flashcards to Review Today")

if fc["due"] > 0:
    st.markdown(
        f'<div style="background:rgba(245,166,35,0.1); border:1px solid rgba(245,166,35,0.3);'
        f' border-radius:10px; padding:20px; text-align:center; margin:8px 0;">'
        f'<div style="font-size:2.5rem; font-weight:bold; color:#f5a623;">{fc["due"]}</div>'
        f'<div style="color:#9ca3af; margin-top:4px;">cards due for review</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/5_🃏_Flashcards.py", label="Start Review →", icon="🃏")
elif fc["total"] > 0:
    st.success("All caught up! No flashcards due right now.")
else:
    st.caption("No flashcards yet. Save one from a Q&A answer or generate them on the Flashcards page.")

st.markdown("---")

# ── 3. Recent Quiz Scores ───────────────────────────────────────────────────
st.markdown("### 📝 Recent Quiz Scores")

recent = get_recent_quiz_scores(user_id, limit=6)
if recent:
    for q in recent:
        pct = round(q["score"] / q["total"] * 100) if q["total"] else 0
        clr = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.05); padding:12px 16px;'
            f' border-radius:8px; margin:6px 0; display:flex;'
            f' justify-content:space-between; align-items:center;">'
            f'<div><strong>{q.get("course_name") or q.get("topic", "General")}</strong>'
            f' <span style="color:#6b7280; font-size:0.85rem;">'
            f'— {q["score"]}/{q["total"]} correct · {q["created_at"][:10]}</span></div>'
            f'<div style="color:{clr}; font-weight:bold; font-size:1.1rem;">{pct}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("No quizzes taken yet. Head to Quiz Lab to test yourself!")

st.markdown("---")

# ── 4. Your Documents ────────────────────────────────────────────────────────
st.markdown("### 📄 Your Documents")

if docs:
    for doc in docs:
        st.markdown(
            f'<div style="background:rgba(232,164,74,0.08); padding:6px 10px;'
            f' border-radius:6px; margin:3px 0; border-left:2px solid #e8a44a;'
            f' font-size:0.85rem; color:#e2e4e9;">📄 {doc}</div>',
            unsafe_allow_html=True,
        )
else:
    st.caption("No documents uploaded yet. Visit Upload Docs to add study materials.")
