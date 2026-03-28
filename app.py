"""
app.py — Study Helper v2
Landing page + auth gate.
Keeps: welcome audio, feature cards, all original personality.
Adds: Supabase auth + app password, production UI.
"""

import streamlit as st
import io
import base64

st.set_page_config(
    page_title="Study Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth, get_current_user
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector

inject_css()
user_id = require_auth()

# ── Welcome audio (kept exactly from original) ────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

WELCOME_TEXT = """
Hey! I'm Study Helper — your AI-powered study companion. Upload your PDFs, Word docs, PowerPoint slides,
or even images, and ask me anything. I'll find answers from your notes with exact citations, or search the web if needed.
Switch between AI models like GPT, Claude, Groq, and Gemini. Get explanations simplified, technical, or in your language.
 Generate quizzes,create images, create explainer videos,  and deep dive into any topic. Let's start studying!
"""

@st.cache_data
def generate_welcome_audio():
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=WELCOME_TEXT, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_header(active_page="Home")
with st.sidebar:
    st.markdown("---")
    render_model_selector()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="sh-display" style="text-align:center; margin-top:1.5rem;">'
    'Your Personal<br><em>Study Assistant</em></h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align:center; color:#6b7280; font-size:0.85rem;'
    ' text-transform:uppercase; letter-spacing:0.1em; margin:0.75rem 0 0.5rem;">'
    'Upload · Ask · Understand · Ace</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sh-divider" style="margin:0 auto 2rem;"></div>',
    unsafe_allow_html=True
)

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, "6", "File formats"),
    (c2, "6", "LLM providers"),
    (c3, "6", "Languages"),
    (c4, "5", "Question types"),
]:
    with col:
        st.markdown(
            f'<div class="sh-stat"><span class="sh-stat-num">{num}</span>'
            f'<span class="sh-stat-label">{label}</span></div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Welcome box + audio (kept from original) ──────────────────────────────────
st.markdown(
    '<div style="background:linear-gradient(135deg,rgba(17,19,24,0.95) 0%,'
    'rgba(13,15,20,0.98) 100%); border-radius:16px; padding:24px;'
    ' border:1px solid rgba(232,164,74,0.25); max-width:700px; margin:0 auto;'
    ' text-align:center;">'
    '<div style="font-family:\'DM Serif Display\',serif; font-size:1.3rem;'
    ' color:#f0ede8; margin-bottom:8px;">👋 Welcome to Study Helper</div>'
    '<p style="color:#6b7280; font-size:0.875rem; margin-bottom:0;">'
    'Click below to hear what I can do for you</p>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("▶️ Play Introduction", width='stretch', type="primary"):
        with st.spinner("Generating audio..."):
            audio_data = generate_welcome_audio()
            if audio_data:
                st.session_state.show_welcome_audio = True
                st.session_state.welcome_audio_data = audio_data
            else:
                st.warning("Install gTTS: pip install gTTS")

if st.session_state.get("show_welcome_audio") and st.session_state.get("welcome_audio_data"):
    st.markdown(
        f'<div style="text-align:center; margin:1rem 0;">'
        f'<audio controls autoplay style="width:80%; max-width:500px;">'
        f'<source src="data:audio/mp3;base64,{st.session_state.welcome_audio_data}"'
        f' type="audio/mp3"></audio></div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔇 Close Audio", width='stretch', key="close_audio"):
            st.session_state.show_welcome_audio = False
            st.rerun()

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; margin:2rem 0;">'
    '<a href="/Study_Helper" target="_self"'
    ' style="font-family:\'DM Sans\',sans-serif; font-size:0.9rem; font-weight:500;'
    ' letter-spacing:0.06em; text-transform:uppercase; display:inline-flex;'
    ' align-items:center; gap:0.5rem; padding:0.875rem 2.5rem;'
    ' background:linear-gradient(135deg,#e8a44a 0%,#d4943a 100%);'
    ' color:#0d0f14 !important; border-radius:4px; text-decoration:none;">'
    '🚀 Start Studying →</a></div>',
    unsafe_allow_html=True
)

# ── Features ──────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 class="sh-section"><span class="sh-section-accent">✦</span>Features</h2>',
    unsafe_allow_html=True
)

features = [
    ("📄", "Multi-Format Support",
     "Upload PDF, DOCX, PPTX, TXT, or images. Each user gets a private document space."),
    ("🎯", "Smart Citations",
     "Every answer includes exact file name and page or slide number."),
    ("🌐", "Web Fallback",
     "When your notes don't have the answer, the AI searches the web automatically."),
    ("🤖", "Multi-LLM Router",
     "Switch between GPT-4o, Claude, Groq (free), Gemini (free), Databricks (free)."),
    ("🎬", "Video Explainer",
     "Turn any concept into a narrated MP4 video in your chosen language."),
    ("🧪", "Quiz Lab",
     "5 question types, difficulty levels, language support, and saved history."),
]

for i in range(0, len(features), 3):
    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, features[i:i+3]):
        with col:
            st.markdown(
                f'<div class="sh-card">'
                f'<div style="font-size:1.5rem; margin-bottom:0.75rem;">{icon}</div>'
                f'<div class="sh-card-title">{title}</div>'
                f'<div class="sh-card-body">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)

# ── How it works (v2 flow) ────────────────────────────────────────────────────
st.markdown(
    '<h2 class="sh-section"><span class="sh-section-accent">✦</span>How It Works</h2>',
    unsafe_allow_html=True
)

steps = [
    ("1", "Create your account",     "Sign up with email. Your data stays private to you."),
    ("2", "Upload your study files",  "Drag and drop files on the Upload Docs page — no terminal needed."),
    ("3", "Ask anything",             "Get cited answers from your notes, or from the web if needed."),
    ("4", "Use Quick Actions",        "Simplify, go technical, translate, listen via TTS, or deep dive."),
    ("5", "Test yourself",            "Generate quizzes from your notes and track progress in Quiz Lab."),
]

for num, title, detail in steps:
    st.markdown(
        f'<div style="display:flex; align-items:flex-start; gap:1.25rem;'
        f' padding:1rem 0; border-bottom:1px solid #1a1c22;">'
        f'<div style="flex-shrink:0; width:32px; height:32px; background:#1a1c22;'
        f' border:1px solid #2a2d36; border-radius:50%; display:flex;'
        f' align-items:center; justify-content:center; font-size:0.8rem;'
        f' font-weight:600; color:#e8a44a; margin-top:2px;">{num}</div>'
        f'<div><div style="font-size:0.95rem; font-weight:500; color:#e2e4e9;'
        f' margin-bottom:0.2rem;">{title}</div>'
        f'<div style="font-size:0.83rem; color:#4b5563;">{detail}</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ── 24hr Deadline Banner (Idea C) ───────────────────────────────────────────────────
from canvas_api import get_upcoming_events, load_ical_url, fetch_canvas_events
from datetime import datetime, timezone, timedelta

if user_id:
    # First fetch the raw events
    ical_url = load_ical_url(user_id)
    if ical_url:
        raw_events = fetch_canvas_events(ical_url, user_id)
        events = get_upcoming_events(raw_events, days_ahead=2)
    else:
        events = []
else:
    events = []
urgent_events = []

for event in events:
    start = event.get('start')
    if start:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_until = (start - now).total_seconds() / 3600
        
        # Show events within 24 hours
        if 0 <= hours_until <= 24:
            urgent_events.append((event, hours_until))

if urgent_events:
    # Sort by urgency (closest first)
    urgent_events.sort(key=lambda x: x[1])
    
    st.markdown("---")
    st.markdown("### ⚠️ Upcoming Deadlines")
    
    for event, hours_until in urgent_events[:3]:  # Show max 3 urgent events
        if hours_until <= 1:
            time_str = "Less than 1 hour"
            urgency_color = "#ef4444"
        elif hours_until <= 6:
            time_str = f"{int(hours_until)} hours"
            urgency_color = "#f59e0b"
        else:
            time_str = f"{int(hours_until)} hours"
            urgency_color = "#e8a44a"
        
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.1); border:1px solid {urgency_color}; '
            f'border-radius:8px; padding:12px; margin:8px 0;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div><strong>{event.get("title", "Untitled")}</strong></div>'
            f'<div style="color:{urgency_color}; font-weight:bold;">{time_str}</div>'
            f'</div>'
            f'<div style="font-size:0.9rem; color:#9ca3af; margin-top:4px;">'
            f'Due: {start.strftime("%b %d, %I:%M %p")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
user = get_current_user()
email_str = f" · {user['email']}" if user else ""
st.markdown(
    f'<div class="sh-footer">📚 STUDY HELPER &nbsp;·&nbsp; LangChain'
    f' &nbsp;·&nbsp; Supabase &nbsp;·&nbsp; Streamlit{email_str}<br>'
    f'<span style="font-size:0.68rem;">Built by Satish Wagle</span></div>',
    unsafe_allow_html=True
)