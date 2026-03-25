"""
pages/1_📚_Study_Helper.py — Study Helper v2
Keeps ALL original features:
  - 4 color themes (Night, Ocean, Forest, Purple)
  - TTS / Listen button
  - Simpler / Technical / language translation quick actions
  - Deep Dive chat panel
  - process_question() full RAG logic
Adds:
  - Supabase pgvector (per user_id) instead of ChromaDB
  - Multi-LLM model selector from sidebar
  - 6 language support (not just Nepali)
  - Auth gate
"""

from __future__ import annotations

import os
import io
import base64
import re
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Study Helper · Ask",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth, get_current_user
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector, get_llm
from agents.study_agent import LANGUAGES
from ingest import get_user_documents
from main import (
    SYSTEM_RULES,
    build_tagged_context,
    build_tagged_web_context,
    context_is_sufficient,
    detect_length_mode,
    make_user_prompt,
    retrieve_docs,
    tavily_search,
    is_comparison_question,
    coverage_check_for_comparison,
)
from langchain_core.messages import HumanMessage, SystemMessage

inject_css()
user_id = require_auth()

# ── 4 Themes (kept exactly from original) ────────────────────────────────────
THEMES = {
    "🌙 Night Study": {
        "bg_primary":   "#1a1a1a",
        "bg_secondary": "#242424",
        "bg_card":      "rgba(42,42,42,0.8)",
        "accent":       "#f5a623",
        "accent_hover": "#ffb84d",
        "text_primary": "#e8e8e8",
        "text_secondary":"#888888",
        "border":       "rgba(245,166,35,0.3)",
    },
    "🌊 Ocean Blue": {
        "bg_primary":   "#0f172a",
        "bg_secondary": "#1e293b",
        "bg_card":      "rgba(30,41,59,0.8)",
        "accent":       "#06b6d4",
        "accent_hover": "#22d3ee",
        "text_primary": "#e2e8f0",
        "text_secondary":"#94a3b8",
        "border":       "rgba(6,182,212,0.3)",
    },
    "🌲 Forest Green": {
        "bg_primary":   "#14201b",
        "bg_secondary": "#1a2e25",
        "bg_card":      "rgba(26,46,37,0.8)",
        "accent":       "#22c55e",
        "accent_hover": "#4ade80",
        "text_primary": "#e8f5e9",
        "text_secondary":"#81c784",
        "border":       "rgba(34,197,94,0.3)",
    },
    "🔮 Purple Haze": {
        "bg_primary":   "#1a1625",
        "bg_secondary": "#2d2640",
        "bg_card":      "rgba(45,38,64,0.8)",
        "accent":       "#a855f7",
        "accent_hover": "#c084fc",
        "text_primary": "#ede9fe",
        "text_secondary":"#a78bfa",
        "border":       "rgba(168,85,247,0.3)",
    },
}


def inject_theme_css(theme_name: str):
    t = THEMES.get(theme_name, THEMES["🌙 Night Study"])
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {t['bg_primary']}; color: {t['text_primary']}; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg,{t['bg_primary']} 0%,{t['bg_secondary']} 100%) !important;
        border-right: 1px solid {t['border']} !important;
    }}
    .answer-container {{
        background: linear-gradient(135deg,{t['bg_card']} 0%,{t['bg_secondary']}ee 100%);
        backdrop-filter: blur(12px);
        border-radius: 16px; padding: 24px;
        border: 1px solid {t['border']}; margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    .source-badge {{
        display: inline-block; background: {t['accent']}22; color: {t['accent']};
        padding: 6px 14px; border-radius: 20px; margin: 4px; font-size: 0.9em;
        border: 1px solid {t['border']}; font-weight: 500;
    }}
    .web-source-badge {{
        display: inline-block; background: rgba(52,152,219,0.15); color: #3498db;
        padding: 6px 14px; border-radius: 20px; margin: 4px; font-size: 0.9em;
        border: 1px solid rgba(52,152,219,0.3); font-weight: 500;
    }}
    .deep-dive-header {{
        color: {t['accent']}; font-size: 1.2rem; font-weight: 600;
        margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid {t['border']};
    }}
    .stButton > button {{
        background: {t['accent']}22 !important; color: {t['accent']} !important;
        border: 1px solid {t['border']} !important; border-radius: 20px !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: {t['accent']}44 !important; border-color: {t['accent']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── TTS (kept exactly from original) ─────────────────────────────────────────
def text_to_speech(text: str) -> str | None:
    try:
        from gtts import gTTS
        clean = re.sub(r'\[S\d+\]|\[W\d+\]', '', text)
        clean = re.sub(r'\*\*|\*|#', '', clean)[:3000]
        tts   = gTTS(text=clean, lang='en', slow=False)
        buf   = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return None


# ── Re-explain functions (kept from original, extended to all 6 languages) ────
def reexplain_simpler(llm, answer: str, question: str) -> str:
    prompt = f"""You previously answered "{question}":

{answer}

Re-explain in MUCH SIMPLER terms:
- Use everyday language, no jargon
- Use analogies and examples
- Explain like teaching a 10-year-old
- Keep it short and clear
- Don't add new information, just simplify"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def reexplain_technical(llm, answer: str, question: str) -> str:
    prompt = f"""You previously answered "{question}":

{answer}

Re-explain with MORE TECHNICAL DEPTH:
- Use proper scientific/technical terminology
- Include specific mechanisms and details
- Be precise and academic in tone
- Keep the same core information, just more detailed"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


def reexplain_in_language(llm, answer: str, question: str, language: str) -> str:
    prompt = f"""You previously answered "{question}":

{answer}

Now explain this same concept in {language}:
- Use simple, natural {language} language
- Use familiar examples
- Make it easy to understand
- Write your response entirely in {language}"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ── Deep Dive (kept exactly from original) ───────────────────────────────────
def get_deep_dive_response(llm, context: str, question: str, history: List[Dict]) -> str:
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    ])
    prompt = f"""You are having a focused conversation about a specific topic.

ORIGINAL CONTEXT:
{context}

CONVERSATION SO FAR:
{history_text}

USER'S NEW QUESTION:
{question}

Respond helpfully, staying focused on the topic. If asked to generate a quiz, create 3-5 questions.
If asked to summarize, provide a concise summary of the discussion so far."""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ── process_question (kept from original, swapped vectordb → user_id) ────────
def process_question(
    question: str,
    user_id: str,
    llm,
    web_fallback: bool,
    default_mode: str,
) -> Dict:
    length_mode = detect_length_mode(question)
    if length_mode == "medium":
        length_mode = default_mode

    rr         = retrieve_docs(user_id, question)
    file_cites = []
    web_cites  = []
    answer_text = ""

    cant_answer_phrases = [
        "does not contain", "do not contain", "cannot provide", "can't provide",
        "no information", "not found", "not mentioned", "doesn't mention",
        "unable to find", "couldn't find", "could not find"
    ]

    if not rr.docs:
        if web_fallback:
            web_results = tavily_search(question)
            if web_results:
                web_context, web_cites = build_tagged_web_context(web_results)
                user_prompt = make_user_prompt(
                    question, length_mode,
                    file_context="(No file context found)",
                    web_context=web_context
                )
                resp = llm.invoke([
                    SystemMessage(content=SYSTEM_RULES),
                    HumanMessage(content=user_prompt)
                ])
                answer_text = resp.content.strip()
            else:
                answer_text = "I can't find anything in your uploaded files and web search returned nothing."
        else:
            answer_text = "I can't find this in your uploaded files. Try enabling web fallback in the sidebar."

    else:
        # Comparison coverage check
        if is_comparison_question(question):
            ok, reason = coverage_check_for_comparison(question, rr.docs)
            if not ok and web_fallback:
                web_results = tavily_search(question)
                if web_results:
                    web_context, web_cites = build_tagged_web_context(web_results)

        sufficient = context_is_sufficient(rr, length_mode)

        if sufficient:
            file_context, file_cites = build_tagged_context(rr.docs)
            user_prompt = make_user_prompt(
                question, length_mode,
                file_context=file_context,
                web_context=None
            )
            resp = llm.invoke([
                SystemMessage(content=SYSTEM_RULES),
                HumanMessage(content=user_prompt)
            ])
            answer_text = resp.content.strip()

            llm_says_no = any(p in answer_text.lower() for p in cant_answer_phrases)
            if llm_says_no and web_fallback:
                web_results = tavily_search(question)
                if web_results:
                    web_context, web_cites = build_tagged_web_context(web_results)
                    user_prompt = make_user_prompt(
                        question, length_mode,
                        file_context="(No relevant file context)",
                        web_context=web_context
                    )
                    resp = llm.invoke([
                        SystemMessage(content=SYSTEM_RULES),
                        HumanMessage(content=user_prompt)
                    ])
                    answer_text = (
                        "📄 I couldn't find this in your documents.\n"
                        "🌐 Here's what I found from the web:\n\n"
                        + resp.content.strip()
                    )
                    file_cites = []
        else:
            if web_fallback:
                web_results = tavily_search(question)
                if web_results:
                    web_context, web_cites = build_tagged_web_context(web_results)
                    user_prompt = make_user_prompt(
                        question, length_mode,
                        file_context="(No relevant file context)",
                        web_context=web_context
                    )
                    resp = llm.invoke([
                        SystemMessage(content=SYSTEM_RULES),
                        HumanMessage(content=user_prompt)
                    ])
                    answer_text = (
                        "📄 Couldn't find enough in your documents.\n"
                        "🌐 Here's what I found from the web:\n\n"
                        + resp.content.strip()
                    )
                else:
                    answer_text = "Found some related info but not enough to answer confidently. Try a more specific question."
            else:
                answer_text = "Couldn't find relevant information in your documents. Try enabling web fallback."

    return {
        "answer":      answer_text,
        "file_cites":  file_cites,
        "web_cites":   web_cites,
        "question":    question,
    }


# ── Quick actions renderer ────────────────────────────────────────────────────
def render_quick_actions(llm, answer: str, question: str, msg_index: int, language: str):
    col1, col2, col3, col4, col5 = st.columns(5)

    if col1.button("😊 Simpler",    key=f"btn_simpler_{msg_index}",   use_container_width=True):
        with st.spinner("Simplifying..."):
            result = reexplain_simpler(llm, answer, question)
        st.session_state[f"reexplain_{msg_index}"] = ("😊 Simplified", result)
        st.rerun()

    if col2.button("🔬 Technical",  key=f"btn_technical_{msg_index}", use_container_width=True):
        with st.spinner("Adding detail..."):
            result = reexplain_technical(llm, answer, question)
        st.session_state[f"reexplain_{msg_index}"] = ("🔬 Technical", result)
        st.rerun()

    lang_label = f"🌐 {language}" if language != "English" else "🇳🇵 नेपाली"
    if col3.button(lang_label,      key=f"btn_lang_{msg_index}",      use_container_width=True):
        target = language if language != "English" else "Nepali"
        with st.spinner(f"Translating to {target}..."):
            result = reexplain_in_language(llm, answer, question, target)
        st.session_state[f"reexplain_{msg_index}"] = (f"🌐 {target}", result)
        st.rerun()

    if col4.button("🔊 Listen",     key=f"btn_tts_{msg_index}",       use_container_width=True):
        with st.spinner("Generating audio..."):
            audio_data = text_to_speech(answer)
        if audio_data:
            st.session_state[f"audio_{msg_index}"] = audio_data
            st.rerun()
        else:
            st.warning("Install gTTS: pip install gTTS")

    if col5.button("💬 Deep Dive",  key=f"btn_dd_{msg_index}",        use_container_width=True):
        st.session_state.show_deep_dive      = True
        st.session_state.deep_dive_context   = answer
        st.session_state.deep_dive_topic     = question
        st.session_state.deep_dive_messages  = []
        st.rerun()

    # Re-explained answer
    if f"reexplain_{msg_index}" in st.session_state:
        mode, content = st.session_state[f"reexplain_{msg_index}"]
        st.markdown(f"### {mode}")
        st.markdown(
            f'<div class="answer-container">{content}</div>',
            unsafe_allow_html=True
        )
        if st.button("❌ Close", key=f"close_reexplain_{msg_index}"):
            del st.session_state[f"reexplain_{msg_index}"]
            st.rerun()

    # Audio player
    if f"audio_{msg_index}" in st.session_state:
        st.markdown(
            f'<audio controls autoplay style="width:100%; margin-top:10px;">'
            f'<source src="data:audio/mp3;base64,{st.session_state[f"audio_{msg_index}"]}"'
            f' type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )
        if st.button("🔇 Close Audio", key=f"close_audio_{msg_index}"):
            del st.session_state[f"audio_{msg_index}"]
            st.rerun()


# ── Deep Dive panel (kept exactly from original) ─────────────────────────────
def render_deep_dive_panel(llm):
    st.markdown(
        '<div class="deep-dive-header">💬 Deep Dive Chat</div>',
        unsafe_allow_html=True
    )
    topic = st.session_state.get("deep_dive_topic", "this topic")
    st.markdown(f"**Topic:** {topic[:80]}{'...' if len(topic) > 80 else ''}")

    for msg in st.session_state.get("deep_dive_messages", []):
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"{icon} **{msg['role'].title()}:** {msg['content']}")

    st.markdown("---")

    with st.form(key="deep_dive_form", clear_on_submit=True):
        deep_input = st.text_input(
            "Ask a follow-up...",
            placeholder="Type your follow-up question...",
            key="deep_dive_input"
        )
        col_send, col_close = st.columns([3, 1])
        submitted    = col_send.form_submit_button("Send", use_container_width=True)
        close_clicked= col_close.form_submit_button("✕ Close", use_container_width=True)

    if submitted and deep_input:
        if "deep_dive_messages" not in st.session_state:
            st.session_state.deep_dive_messages = []
        st.session_state.deep_dive_messages.append({"role": "user", "content": deep_input})
        with st.spinner("Thinking..."):
            resp = get_deep_dive_response(
                llm,
                st.session_state.get("deep_dive_context", ""),
                deep_input,
                st.session_state.deep_dive_messages,
            )
        st.session_state.deep_dive_messages.append({"role": "assistant", "content": resp})
        st.rerun()

    if close_clicked:
        st.session_state.show_deep_dive     = False
        st.session_state.deep_dive_messages = []
        st.rerun()


# ── Session init ──────────────────────────────────────────────────────────────
if "messages"         not in st.session_state: st.session_state.messages         = []
if "show_deep_dive"   not in st.session_state: st.session_state.show_deep_dive   = False
if "current_theme"    not in st.session_state: st.session_state.current_theme    = "🌙 Night Study"

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_header(active_page="Study Helper")

with st.sidebar:
    st.markdown("---")
    model_label = render_model_selector() 

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">Theme</div>',
        unsafe_allow_html=True
    )
    selected_theme = st.selectbox(
        "Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.current_theme),
        label_visibility="collapsed",
        key="theme_selector"
    )
    st.session_state.current_theme = selected_theme

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">Settings</div>',
        unsafe_allow_html=True
    )
    web_fallback = st.toggle(
        "🌐 Web fallback",
        value=st.session_state.get("web_fallback", True),
        key="web_fallback_toggle"
    )
    st.session_state.web_fallback = web_fallback

    answer_mode = st.radio(
        "Answer mode",
        ["Short", "Medium", "Long"],
        index=1,
        key="answer_mode_radio"
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">Response language</div>',
        unsafe_allow_html=True
    )
    language = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        label_visibility="collapsed",
        key="study_language"
    )

    st.markdown("---")
    docs = get_user_documents(user_id)
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.5rem;">Your documents</div>',
        unsafe_allow_html=True
    )
    if docs:
        for doc in docs:
            st.markdown(
                f'<div style="background:rgba(232,164,74,0.08); padding:6px 10px;'
                f' border-radius:6px; margin:3px 0; border-left:2px solid #e8a44a;'
                f' font-size:0.82rem; color:#e2e4e9;">📄 {doc}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="sh-info">No documents yet. Visit Upload Docs to add files.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True, key="btn_clear"):
        st.session_state.messages = []
        st.rerun()

# ── Apply theme CSS ───────────────────────────────────────────────────────────
inject_theme_css(selected_theme)
theme = THEMES.get(selected_theme, THEMES["🌙 Night Study"])

# ── Main layout ───────────────────────────────────────────────────────────────
if st.session_state.get("show_deep_dive"):
    main_col, deep_col = st.columns([2, 1])
else:
    main_col = st.container()
    deep_col = None

with main_col:
    st.markdown(
        f'<h1 style="color:{theme["accent"]}; margin-bottom:8px;">📚 Study Helper</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color:{theme["text_secondary"]}; margin-bottom:1.5rem;">'
        f'Model: <b style="color:{theme["accent"]}">{model_label}</b>'
        f' &nbsp;·&nbsp; Language: {language}</p>',
        unsafe_allow_html=True
    )

    # Chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if msg.get("file_sources"):
                    with st.expander("📄 File Sources", expanded=False):
                        badges = "".join(
                            f'<span class="source-badge">{s}</span>'
                            for s in msg["file_sources"]
                        )
                        st.markdown(
                            f'<div style="margin-top:8px;">{badges}</div>',
                            unsafe_allow_html=True
                        )

                if msg.get("web_sources"):
                    with st.expander("🌐 Web Sources", expanded=False):
                        for cite in msg["web_sources"]:
                            parts = cite.split(" — ")
                            if len(parts) == 2:
                                st.markdown(f"🔗 [{parts[0]}]({parts[1]})")
                            else:
                                st.markdown(f"🔗 {cite}")

                # Quick actions on latest assistant message only
                if i == len(st.session_state.messages) - 1:
                    st.markdown("---")
                    st.markdown(
                        f'<p style="font-size:0.78rem; color:{theme["text_secondary"]}; margin:0 0 8px;">✨ Quick Actions</p>',
                        unsafe_allow_html=True
                    )
                    render_quick_actions(
                        get_llm(model_label),
                        msg["content"],
                        msg.get("original_question", ""),
                        i,
                        language,
                    )

    # Chat input
    if question := st.chat_input("Ask anything about your notes…"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching your notes..."):
            llm    = get_llm(model_label)
            result = process_question(
                question=question,
                user_id=user_id,
                llm=llm,
                web_fallback=web_fallback,
                default_mode=answer_mode.lower(),
            )

        st.session_state.messages.append({
            "role":              "assistant",
            "content":           result["answer"],
            "file_sources":      result["file_cites"],
            "web_sources":       result["web_cites"],
            "original_question": question,
        })
        st.rerun()

if deep_col:
    with deep_col:
        render_deep_dive_panel(get_llm(model_label))