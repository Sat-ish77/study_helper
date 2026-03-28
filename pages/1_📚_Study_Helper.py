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
from learn_more import render_learn_more_section
from canvas_api import render_daily_digest, load_ical_url, save_ical_url, clear_ical_url
from flashcard_db import save_flashcard_from_qa
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


# ── TTS (gTTS primary, ElevenLabs fallback) ─────────────────────────────────────
def text_to_speech(text: str, lang: str = "en") -> str | None:
    """Generate speech using gTTS (free), fallback to ElevenLabs if gTTS fails."""
    # Clean text for TTS — remove citations, markdown, emoji source lines
    clean = re.sub(r'\[S\d+\]', '', text)
    clean = re.sub(r'\[W\d+\]', '', clean)
    clean = re.sub(r'\*\*|\*|#+', '', clean)
    clean = re.sub(r'📄.*?(?=\n|$)', '', clean)
    clean = re.sub(r'🌐.*?(?=\n|$)', '', clean)
    clean = clean.strip()[:3000]

    # PRIMARY: gTTS (free)
    try:
        from gtts import gTTS
        tts = gTTS(text=clean, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.session_state.tts_engine = "🔊 gTTS"
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"[TTS] gTTS failed: {e}")

    # FALLBACK: ElevenLabs
    try:
        from elevenlabs.client import ElevenLabs
        el_key = os.getenv("ELEVENLABS_API_KEY")
        if not el_key:
            raise ValueError("No ElevenLabs key")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        client = ElevenLabs(api_key=el_key)
        audio = client.text_to_speech.convert(
            text=clean,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio)
        audio_io = io.BytesIO(audio_bytes)
        audio_io.seek(0)
        st.session_state.tts_engine = "🎙️ ElevenLabs"
        return base64.b64encode(audio_io.read()).decode()
    except Exception as e:
        print(f"[TTS] ElevenLabs failed: {e}")

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
    prompt = f"""Translate the following text to {language}.

Strict rules:
1. Use simple, everyday {language} words that a high school student would understand
2. Keep ALL technical/scientific terms in English (e.g. "photosynthesis", "algorithm", "RAM", "DNA", "HTTP")
3. Keep ALL code snippets, formulas, and equations in English
4. Keep ALL proper nouns (names, brands, places) in English  
5. Write naturally — how a {language} teacher would explain this to a student, not a word-for-word translation
6. Do NOT translate any text inside backticks or code blocks

Text to translate:
{answer}"""
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ── process_question (kept from original, swapped vectordb → user_id) ────────
def process_question(
    question: str,
    user_id: str,
    llm,
    web_fallback: bool,
    default_mode: str,
    history: List[Dict] | None = None,
) -> Dict:
    length_mode = detect_length_mode(question)
    if length_mode == "medium":
        length_mode = default_mode

    # Build dynamic system prompt with strong length instruction
    LENGTH_INSTRUCTIONS = {
        "short":  "ANSWER LENGTH: Answer in 2-3 sentences maximum. Be concise. No long paragraphs.",
        "medium": "ANSWER LENGTH: Answer in 1-2 clear paragraphs. Moderate detail.",
        "long":   "ANSWER LENGTH: Answer in detail with examples, explanation, and full context. Use structured paragraphs and bullet points.",
    }
    system_prompt = SYSTEM_RULES + "\n\n" + LENGTH_INSTRUCTIONS.get(length_mode, LENGTH_INSTRUCTIONS["medium"])

    # Build history context ONLY when there are ≥2 real exchanges
    # (i.e. at least 2 assistant messages exist before the current query).
    # Never inject "CONVERSATION HISTORY:" with empty/thin content —
    # that causes the LLM to hallucinate fake prior discussions.
    history_block = ""
    if history:
        prior = [m for m in history if m["role"] == "assistant"]
        if len(prior) >= 2:
            recent = history[-6:]
            lines = []
            for m in recent:
                role = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"{role}: {m['content'][:300]}")
            history_block = "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"

    rr         = retrieve_docs(user_id, question)
    file_cites = []
    web_cites  = []
    raw_sources = []
    answer_text = ""

    # Extract raw source metadata for detailed display
    for d in rr.docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        page = meta.get("page")
        try:
            page = int(page) + 1 if page is not None else None
        except Exception:
            pass
        raw_sources.append({
            "source_file": meta.get("filename", "Unknown"),
            "page_num": page,
            "similarity": meta.get("similarity", 0.0),
        })

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
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=history_block + user_prompt)
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
                SystemMessage(content=system_prompt),
                HumanMessage(content=history_block + user_prompt)
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
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=history_block + user_prompt)
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
                    resp= llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=history_block + user_prompt)
                    ])
                    answer_text= (
                        "📄 Couldn't find enough in your documents.\n"
                        "🌐 Here's what I found from the web:\n\n"
                        + resp.content.strip()
                    )
                    file_cites=[]

    return {
        "answer":      answer_text,
        "file_cites":  file_cites,
        "web_cites":   web_cites,
        "raw_sources": raw_sources,
        "question":    question,
    }


# ── Visualize helper ─────────────────────────────────────────────────────────
DIAGRAM_KEYWORDS = [
    "flowchart", "diagram", "chart", "flow", "process",
    "steps", "workflow", "cycle", "structure", "how does", "how do",
]


def _is_diagram_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in DIAGRAM_KEYWORDS)


def _generate_viz(llm, answer: str, question: str) -> dict:
    """Return {image_bytes, explanation} for the Visualize button."""
    from image_generator import generate_flowchart, generate_dalle_image

    if _is_diagram_query(question):
        # Ask LLM to extract steps from the answer
        step_prompt = (
            "Extract 4-8 short steps (one sentence each) from this text. "
            "Return ONLY a Python list of strings, no explanation.\n\n"
            f"{answer[:2000]}"
        )
        raw = llm.invoke([HumanMessage(content=step_prompt)]).content.strip()
        # Parse list
        import ast
        try:
            steps = ast.literal_eval(raw)
            if not isinstance(steps, list):
                steps = [s.strip("- ") for s in raw.splitlines() if s.strip()]
        except Exception:
            steps = [s.strip("- •*") for s in raw.splitlines() if s.strip()]
        steps = [s for s in steps if len(s) > 2][:8]
        if not steps:
            steps = ["Step 1", "Step 2", "Step 3"]

        title = question[:60] + ("…" if len(question) > 60 else "")
        img_bytes = generate_flowchart(title, steps)
    else:
        # DALL-E with forced English labels
        prompt = (
            "Create this image with ALL text and labels strictly in English only: "
            + answer[:500]
        )
        img_bytes = generate_dalle_image(prompt)

    # Generate simple explanation
    explain_prompt = (
        "You just created a visual for this answer:\n"
        f"{answer[:1000]}\n\n"
        "Write a 4-6 sentence explanation of the image for a 10-year-old. "
        "Describe what the shapes/arrows represent, what the flow or sequence is, "
        "and why it matters. No jargon. "
        "Write the explanation in the same language the user is using."
    )
    explanation = llm.invoke([HumanMessage(content=explain_prompt)]).content.strip()

    return {"image_bytes": img_bytes, "explanation": explanation}


# ── Quick actions renderer ────────────────────────────────────────────────────
def render_quick_actions(llm, answer: str, question: str, msg_index: int, language: str, user_id: str):
    LANGUAGES = {
        "English": "English",
        "Nepali": "नेपाली",
        "Spanish": "Español",
        "French": "Français",
        "German": "Deutsch",
        "Chinese": "中文",
        "Japanese": "日本語",
        "Hindi": "हिन्दी"
    }

    lang_options = list(LANGUAGES.keys())

    # Row 1: [Simpler] [Technical] [Translate] [lang▾] [Listen] [lang▾]
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns([1, 1, 1, 0.8, 1, 0.8])

    with r1c4:
        translate_lang = st.selectbox(
            "Translate lang",
            options=lang_options,
            key=f"translate_lang_{msg_index}",
            index=0,
            label_visibility="collapsed",
        )
    with r1c6:
        listen_lang = st.selectbox(
            "Listen lang",
            options=lang_options,
            key=f"listen_lang_{msg_index}",
            index=0,
            label_visibility="collapsed",
        )

    if r1c1.button("😊 Simpler", key=f"btn_simpler_{msg_index}", use_container_width=True):
        with st.spinner("Simplifying..."):
            result = reexplain_simpler(llm, answer, question)
        st.session_state[f"reexplain_{msg_index}"] = ("😊 Simplified", result)
        st.rerun()

    if r1c2.button("🔬 Technical", key=f"btn_technical_{msg_index}", use_container_width=True):
        with st.spinner("Adding detail..."):
            result = reexplain_technical(llm, answer, question)
        st.session_state[f"reexplain_{msg_index}"] = ("🔬 Technical", result)
        st.rerun()

    if r1c3.button("🌐 Translate", key=f"btn_lang_{msg_index}", use_container_width=True):
        target_lang = LANGUAGES[translate_lang]
        if target_lang == "English":
            result = answer
            mode = "🌐 English"
        else:
            with st.spinner(f"Translating to {target_lang}..."):
                result = reexplain_in_language(llm, answer, question, target_lang)
            mode = f"🌐 {target_lang}"
        st.session_state[f"reexplain_{msg_index}"] = (mode, result)
        st.rerun()

    if r1c5.button("🔊 Listen", key=f"btn_tts_{msg_index}", use_container_width=True):
        target_lang = LANGUAGES.get(listen_lang, "English")
        if target_lang != "English":
            with st.spinner(f"Translating to {target_lang} for audio..."):
                text_for_audio = reexplain_in_language(llm, answer, question, target_lang)
        else:
            text_for_audio = answer
        with st.spinner("Generating audio..."):
            lang_code = target_lang.lower()[:2] if target_lang != "English" else "en"
            audio_data = text_to_speech(text_for_audio, lang=lang_code)
        if audio_data:
            st.session_state[f"audio_{msg_index}"] = audio_data
            st.rerun()
        else:
            st.warning("Install gTTS: pip install gTTS")

    # Row 2: [Visualize] [Flashcard]
    r2c1, r2c2 = st.columns([1, 1])

    if r2c1.button("📊 Visualize", key=f"btn_visualize_{msg_index}", use_container_width=True):
        with st.spinner("Generating visualization..."):
            try:
                viz = _generate_viz(llm, answer, question)
                st.session_state[f"viz_{msg_index}"] = viz
            except Exception as e:
                st.error(f"Visualization failed: {e}")
        st.rerun()

    if r2c2.button("🃏 Flashcard", key=f"btn_flashcard_{msg_index}", use_container_width=True):
        card_id = save_flashcard_from_qa(user_id, question, answer)
        if card_id:
            st.success("✅ Saved as flashcard!")
        else:
            st.error("Failed to save flashcard")

    # ── Expanded panels below buttons ────────────────────────────────────────

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
        engine_indicator = st.session_state.get("tts_engine", "🔊 gTTS")
        st.markdown(
            f'<div style="font-size:0.75rem; color:#6b7280; margin-bottom:5px;">'
            f'{engine_indicator} Audio</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<audio controls autoplay style="width:100%; margin-top:10px;">'
            f'<source src="data:audio/mp3;base64,{st.session_state[f"audio_{msg_index}"]}"'
            f' type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )
        if st.button("🔇 Close Audio", key=f"close_audio_{msg_index}"):
            del st.session_state[f"audio_{msg_index}"]
            st.rerun()

    # Visualization display
    if f"viz_{msg_index}" in st.session_state:
        viz = st.session_state[f"viz_{msg_index}"]
        st.markdown("### 📊 Visualization")
        img = viz.get("image_bytes")
        if img:
            st.image(img, use_container_width=True)
        explanation = viz.get("explanation", "")
        if explanation:
            st.markdown(explanation)
        if st.button("❌ Close Visualization", key=f"close_viz_{msg_index}"):
            del st.session_state[f"viz_{msg_index}"]
            st.rerun()


if "messages"         not in st.session_state: st.session_state.messages         = []
if "current_theme"    not in st.session_state: st.session_state.current_theme    = "🌙 Night Study"
if "current_conv_id"  not in st.session_state: st.session_state.current_conv_id  = None

# Auto-save flag — set after appending an assistant message, cleared after save
if "needs_save" not in st.session_state:
    st.session_state.needs_save = False

if st.session_state.needs_save and st.session_state.messages:
    from conversation_db import create_conversation, update_conversation
    conv_id = st.session_state.current_conv_id
    messages = st.session_state.messages
    if any(m["role"] == "assistant" for m in messages):
        if conv_id:
            update_conversation(conv_id, messages)
        else:
            first_q = next((m["content"] for m in messages if m["role"] == "user"), "")
            if first_q:
                title = first_q[:50] + ("..." if len(first_q) > 50 else "")
                conv_id = create_conversation(user_id, title, messages)
                if conv_id:
                    st.session_state.current_conv_id = conv_id
    st.session_state.needs_save = False

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
    
    # Canvas Calendar integration
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">Canvas Calendar</div>',
        unsafe_allow_html=True
    )
    existing_url = load_ical_url(user_id) or ""
    ical_input = st.text_input(
        "iCal URL",
        value=existing_url,
        key="ical_url_input",
        placeholder="https://canvas.instructure.com/feeds/calendars/...",
        label_visibility="collapsed"
    )
    col_save, col_clear = st.columns([3, 1])
    with col_save:
        if st.button("💾 Save Calendar URL", use_container_width=True, key="btn_save_ical"):
            if ical_input and ical_input.strip():
                save_ical_url(user_id, ical_input.strip())
                st.success("✅ Saved!")
                st.rerun()
            else:
                st.warning("Please enter a URL first")
    with col_clear:
        if st.button("🗑️", key="btn_clear_ical", use_container_width=True):
            clear_ical_url(user_id)
            st.rerun()

    if existing_url:
        render_daily_digest(user_id)

    # Conversations section
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">Conversations</div>',
        unsafe_allow_html=True
    )

    if st.button("➕ New Chat", use_container_width=True, key="btn_new_chat"):
        st.session_state.messages = []
        st.session_state.current_conv_id = None
        st.rerun()

    from conversation_db import list_conversations, load_conversation, delete_conversation
    conversations = list_conversations(user_id, limit=10)

    if conversations:
        for conv in conversations:
            col_title, col_delete = st.columns([4, 1])
            with col_title:
                label = conv["title"][:30] + ("..." if len(conv["title"]) > 30 else "")
                if st.button(
                    label,
                    key=f"conv_{conv['id']}",
                    help=f"Created: {conv['created_at'][:10]}",
                    use_container_width=True,
                ):
                    full_conv = load_conversation(conv["id"])
                    if full_conv and full_conv.get("messages"):
                        st.session_state.messages = full_conv["messages"]
                        st.session_state.current_conv_id = conv["id"]
                        st.rerun()

            with col_delete:
                if st.button("✕", key=f"del_{conv['id']}", help="Delete conversation"):
                    if delete_conversation(conv["id"]):
                        st.rerun()
    else:
        st.caption("No saved conversations yet")

# ── Apply theme CSS ───────────────────────────────────────────────────────────
inject_theme_css(selected_theme)
theme = THEMES.get(selected_theme, THEMES["🌙 Night Study"])

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(
    f'<h1 style="color:{theme["accent"]}; margin-bottom:8px;">📚 Study Helper</h1>',
    unsafe_allow_html=True
)
st.markdown(
    f'<p style="color:{theme["text_secondary"]}; margin-bottom:1.5rem;">'
    f'Model: <b style="color:{theme["accent"]}">{model_label}</b></p>',
    unsafe_allow_html=True
)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            has_file = msg.get("file_sources") or msg.get("raw_sources")
            has_web  = msg.get("web_sources")
            if has_file or has_web:
                with st.expander("📄 Sources", expanded=False):
                    # Detailed file sources (raw_sources with page + similarity)
                    raw = msg.get("raw_sources", [])
                    if raw:
                        for src in raw:
                            page_str = f"Page {src['page_num']}" if src.get("page_num") else "—"
                            sim = src.get("similarity", 0)
                            st.caption(
                                f"📄 {src.get('source_file', 'Unknown')} "
                                f"| {page_str} "
                                f"| Score: {sim:.2f}"
                            )
                    elif msg.get("file_sources"):
                        for s in msg["file_sources"]:
                            st.caption(f"📄 {s}")

                    # Web sources
                    if has_web:
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
                    "English",
                    user_id
                )

                # Learn More section — lazy load on button click
                has_sources = (msg.get("file_sources") or
                               msg.get("web_sources") or
                               msg.get("raw_sources"))
                if has_sources:
                    if st.button("📚 Learn More about this topic", key=f"learn_more_btn_{i}"):
                        render_learn_more_section(
                            msg.get("original_question", ""),
                            user_id,
                            msg["content"]
                        )

# Chat input
if question := st.chat_input("Ask anything about your notes…"):
    st.session_state.messages.append({"role": "user", "content": question})

    # Check if this is a conversational message
    from main import is_conversational, answer_conversationally, is_follow_up, answer_follow_up

    if is_conversational(question):
        with st.spinner("Thinking..."):
            llm = get_llm(model_label)
            conversational_response = answer_conversationally(question, llm)

        st.session_state.messages.append({
            "role": "assistant",
            "content": conversational_response,
            "file_sources": [],
            "web_sources": [],
            "original_question": question,
        })
    elif is_follow_up(question, st.session_state.messages):
        with st.spinner("Thinking..."):
            llm = get_llm(model_label)
            follow_up_response = answer_follow_up(question, st.session_state.messages, llm)

        st.session_state.messages.append({
            "role": "assistant",
            "content": follow_up_response,
            "file_sources": [],
            "web_sources": [],
            "original_question": question,
        })
    else:
        with st.spinner("Searching your notes..."):
            llm    = get_llm(model_label)
            result = process_question(
                question=question,
                user_id=user_id,
                llm=llm,
                web_fallback=web_fallback,
                default_mode=answer_mode.lower(),
                history=st.session_state.messages,
            )

        st.session_state.messages.append({
            "role":              "assistant",
            "content":           result["answer"],
            "file_sources":      result["file_cites"],
            "web_sources":       result["web_cites"],
            "raw_sources":       result.get("raw_sources", []),
            "original_question": question,
        })
    st.session_state.needs_save = True
    st.rerun()