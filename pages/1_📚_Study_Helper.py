"""
Study Helper - Main Q&A Page
Enhanced with Quick Actions, Deep Dive Chat, Text-to-Speech, Themes
"""

from __future__ import annotations

import os
import io
import base64
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from gtts import gTTS

# Import backend functions from main.py
from main import (
    DB_DIR,
    DEFAULT_MODEL,
    SYSTEM_RULES,
    TEMPERATURE,
    build_tagged_context,
    build_tagged_web_context,
    context_is_sufficient,
    detect_length_mode,
    make_user_prompt,
    retrieve_docs,
    tavily_search,
)

load_dotenv()

# Page config
st.set_page_config(
    page_title="Study Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------
# Theme Definitions
# -----------------------
THEMES = {
    "🌙 Night Study": {
        "bg_primary": "#1a1a1a",
        "bg_secondary": "#242424",
        "bg_card": "rgba(42, 42, 42, 0.8)",
        "accent": "#f5a623",
        "accent_hover": "#ffb84d",
        "text_primary": "#e8e8e8",
        "text_secondary": "#888888",
        "border": "rgba(245, 166, 35, 0.3)",
    },
    "🌊 Ocean Blue": {
        "bg_primary": "#0f172a",
        "bg_secondary": "#1e293b",
        "bg_card": "rgba(30, 41, 59, 0.8)",
        "accent": "#06b6d4",
        "accent_hover": "#22d3ee",
        "text_primary": "#e2e8f0",
        "text_secondary": "#94a3b8",
        "border": "rgba(6, 182, 212, 0.3)",
    },
    "🌲 Forest Green": {
        "bg_primary": "#14201b",
        "bg_secondary": "#1a2e25",
        "bg_card": "rgba(26, 46, 37, 0.8)",
        "accent": "#22c55e",
        "accent_hover": "#4ade80",
        "text_primary": "#e8f5e9",
        "text_secondary": "#81c784",
        "border": "rgba(34, 197, 94, 0.3)",
    },
    "🔮 Purple Haze": {
        "bg_primary": "#1a1625",
        "bg_secondary": "#2d2640",
        "bg_card": "rgba(45, 38, 64, 0.8)",
        "accent": "#a855f7",
        "accent_hover": "#c084fc",
        "text_primary": "#ede9fe",
        "text_secondary": "#a78bfa",
        "border": "rgba(168, 85, 247, 0.3)",
    },
}


# -----------------------
# Custom CSS with Theme Support
# -----------------------
def inject_custom_css(theme_name: str):
    theme = THEMES.get(theme_name, THEMES["🌙 Night Study"])
    
    st.markdown(
        f"""
        <style>
        /* Global theme */
        .stApp {{
            background-color: {theme["bg_primary"]};
            color: {theme["text_primary"]};
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {theme["bg_primary"]} 0%, {theme["bg_secondary"]} 100%);
            border-right: 1px solid {theme["border"]};
        }}
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {theme["accent"]} !important;
            font-weight: 600;
        }}
        
        /* Answer container */
        .answer-container {{
            background: linear-gradient(135deg, {theme["bg_card"]} 0%, {theme["bg_secondary"]}ee 100%);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid {theme["border"]};
            margin: 15px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        /* Source badges */
        .source-badge {{
            display: inline-block;
            background: {theme["accent"]}22;
            color: {theme["accent"]};
            padding: 6px 14px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9em;
            border: 1px solid {theme["border"]};
            font-weight: 500;
        }}
        
        .web-source-badge {{
            display: inline-block;
            background: rgba(52, 152, 219, 0.15);
            color: #3498db;
            padding: 6px 14px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9em;
            border: 1px solid rgba(52, 152, 219, 0.3);
            font-weight: 500;
        }}
        
        /* Chat message styling */
        .stChatMessage {{
            background: {theme["bg_card"]};
            border-radius: 12px;
        }}
        
        /* File list styling */
        .file-item {{
            background: {theme["accent"]}11;
            padding: 8px 12px;
            border-radius: 6px;
            margin: 4px 0;
            border-left: 3px solid {theme["accent"]};
            font-size: 0.9em;
            color: {theme["text_primary"]};
        }}
        
        /* Deep dive panel */
        .deep-dive-header {{
            color: {theme["accent"]};
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid {theme["border"]};
        }}
        
        /* Quick action buttons styling */
        .stButton > button {{
            background: {theme["accent"]}22;
            color: {theme["accent"]};
            border: 1px solid {theme["border"]};
            border-radius: 20px;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background: {theme["accent"]}44;
            border-color: {theme["accent"]};
            transform: translateY(-2px);
        }}
        
        /* Links */
        a {{
            color: {theme["accent"]};
        }}
        
        a:hover {{
            color: {theme["accent_hover"]};
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {theme["bg_primary"]};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {theme["accent"]};
            border-radius: 4px;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background: {theme["bg_card"]} !important;
            border-radius: 8px !important;
            border: 1px solid {theme["border"]} !important;
            color: {theme["accent"]} !important;
        }}
        
        /* Text color fixes */
        p, span, div {{
            color: {theme["text_primary"]};
        }}
        
        /* Input styling */
        .stTextInput input, .stTextArea textarea {{
            background-color: {theme["bg_secondary"]} !important;
            color: {theme["text_primary"]} !important;
            border-color: {theme["border"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# Text-to-Speech Function
# -----------------------
def text_to_speech(text: str) -> str:
    """Convert text to speech and return base64 audio data"""
    try:
        # Clean text for TTS (remove citations like [S1], [W1], etc.)
        import re
        clean_text = re.sub(r'\[S\d+\]|\[W\d+\]', '', text)
        clean_text = re.sub(r'\*\*|\*|#', '', clean_text)  # Remove markdown
        clean_text = clean_text[:3000]  # Limit length
        
        # Generate speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # Save to bytes
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        return audio_base64
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None


# -----------------------
# Cache resources
# -----------------------
@st.cache_resource
def init_vectordb():
    """Initialize Chroma vectordb with OpenAI embeddings"""
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def init_llm():
    """Initialize ChatOpenAI LLM"""
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=TEMPERATURE)


@st.cache_data
def get_indexed_files(_vectordb) -> List[str]:
    """Extract unique filenames from vectordb metadata"""
    try:
        collection = _vectordb._collection
        all_metadata = collection.get()["metadatas"]
        filenames = set()
        for meta in all_metadata:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
            elif meta and "source" in meta:
                source = meta["source"]
                if "/" in source or "\\" in source:
                    source = os.path.basename(source)
                filenames.add(source)
        return sorted(list(filenames))
    except Exception:
        return []


# -----------------------
# Re-explain Functions
# -----------------------
def reexplain_simpler(llm, original_answer: str, question: str) -> str:
    """Re-explain the answer in simpler terms"""
    prompt = f"""You previously gave this answer to the question "{question}":

{original_answer}

Now, re-explain this in MUCH SIMPLER terms:
- Use everyday language, no jargon
- Use analogies and examples
- Explain like teaching a 10-year-old
- Keep it short and clear
- Don't add new information, just simplify what's already there
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def reexplain_technical(llm, original_answer: str, question: str) -> str:
    """Re-explain the answer with more technical detail"""
    prompt = f"""You previously gave this answer to the question "{question}":

{original_answer}

Now, re-explain this with MORE TECHNICAL DEPTH:
- Use proper scientific terminology
- Include specific mechanisms and pathways
- Add molecular/cellular details where relevant
- Be precise and academic in tone
- Keep the same core information, just more detailed
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


def reexplain_nepali(llm, original_answer: str, question: str) -> str:
    """Re-explain the answer in Nepali"""
    prompt = f"""You previously gave this answer to the question "{question}":

{original_answer}

Now, explain this same concept in NEPALI (नेपाली):
- Use simple, natural Nepali language
- Avoid English scientific terms where possible - use Nepali equivalents or explain them
- Use familiar examples that would make sense to a Nepali student
- Make it easy to understand, like explaining to a friend
- Focus on helping the student truly understand the concept

Write your response entirely in Nepali (Devanagari script).
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


# -----------------------
# Deep Dive Chat Functions
# -----------------------
def get_deep_dive_response(llm, context: str, question: str, chat_history: List[Dict]) -> str:
    """Get response for deep dive follow-up questions"""
    history_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in chat_history[-6:]
    ])
    
    prompt = f"""You are having a focused conversation about a specific topic.

ORIGINAL CONTEXT:
{context}

CONVERSATION SO FAR:
{history_text}

USER'S NEW QUESTION:
{question}

Respond helpfully, staying focused on the topic. If asked to generate a quiz, create 3-5 questions.
If asked to summarize, provide a concise summary of the discussion so far.
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content.strip()


# -----------------------
# UI Components
# -----------------------
def render_sidebar(vectordb):
    """Render sidebar with controls and file list"""
    with st.sidebar:
        st.markdown("# 📚 Study Helper")
        st.markdown("*Ask anything about your notes*")
        st.markdown("---")
        
        # Theme selector
        st.markdown("### 🎨 Theme")
        selected_theme = st.selectbox(
            "Choose theme:",
            options=list(THEMES.keys()),
            index=0,
            key="theme_selector",
            label_visibility="collapsed"
        )
        st.session_state.current_theme = selected_theme
        
        st.markdown("---")
        
        # Web fallback toggle
        st.markdown("### 🌐 Settings")
        web_fallback = st.toggle(
            "Web Fallback",
            value=st.session_state.get("web_fallback", False),
            help="Search the web when your notes don't have the answer",
            key="web_fallback_toggle"
        )
        st.session_state.web_fallback = web_fallback
        
        # Answer mode radio
        answer_mode = st.radio(
            "Answer Mode",
            options=["Short", "Medium", "Long"],
            index=1,
            help="Control answer length and detail",
            key="answer_mode_radio"
        )
        
        st.markdown("---")
        
        # Indexed files section
        st.markdown("### 📄 Indexed Files")
        indexed_files = get_indexed_files(vectordb)
        
        if indexed_files:
            for filename in indexed_files:
                st.markdown(f'<div class="file-item">📄 {filename}</div>', unsafe_allow_html=True)
        else:
            st.info("No files indexed yet. Run `python ingest.py` to add files.")
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("last_answer_index", None)
            st.rerun()
        
        st.markdown("---")
        
        # Footer
        st.markdown(
            '<div style="text-align: center; color: #888; font-size: 0.85em;">'
            'Powered by LangChain + OpenAI'
            '</div>',
            unsafe_allow_html=True
        )
    
    return web_fallback, answer_mode.lower(), selected_theme


def render_source_badges(sources: List[str], source_type: str = "file"):
    """Render source citations as styled badges"""
    if not sources:
        return
    
    badge_class = "source-badge" if source_type == "file" else "web-source-badge"
    html = '<div style="margin-top: 12px;">'
    for source in sources:
        html += f'<span class="{badge_class}">{source}</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_web_sources(web_cites: List[str]):
    """Render web sources as clickable links"""
    for cite in web_cites:
        parts = cite.split(" — ")
        if len(parts) == 2:
            tag_title = parts[0]
            url = parts[1]
            st.markdown(f"🔗 [{tag_title}]({url})")
        else:
            st.markdown(f"🔗 {cite}")


def render_quick_actions(llm, answer: str, question: str, msg_index: int):
    """Render quick action buttons below an answer"""
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("😊 Simpler", key=f"btn_simpler_{msg_index}", use_container_width=True):
            with st.spinner("Simplifying..."):
                simpler = reexplain_simpler(llm, answer, question)
                st.session_state[f"reexplain_{msg_index}"] = ("😊 Simplified", simpler)
                st.rerun()
    
    with col2:
        if st.button("🔬 Technical", key=f"btn_technical_{msg_index}", use_container_width=True):
            with st.spinner("Adding detail..."):
                technical = reexplain_technical(llm, answer, question)
                st.session_state[f"reexplain_{msg_index}"] = ("🔬 Technical", technical)
                st.rerun()
    
    with col3:
        if st.button("🇳🇵 नेपाली", key=f"btn_nepali_{msg_index}", use_container_width=True):
            with st.spinner("Translating..."):
                nepali = reexplain_nepali(llm, answer, question)
                st.session_state[f"reexplain_{msg_index}"] = ("🇳🇵 नेपाली", nepali)
                st.rerun()
    
    with col4:
        if st.button("🔊 Listen", key=f"btn_tts_{msg_index}", use_container_width=True):
            with st.spinner("Generating audio..."):
                audio_data = text_to_speech(answer)
                if audio_data:
                    st.session_state[f"audio_{msg_index}"] = audio_data
                    st.rerun()
    
    with col5:
        if st.button("💬 Deep Dive", key=f"btn_dd_{msg_index}", use_container_width=True):
            st.session_state.show_deep_dive = True
            st.session_state.deep_dive_context = answer
            st.session_state.deep_dive_topic = question
            st.session_state.deep_dive_messages = []
            st.rerun()
    
    # Show re-explained answer if available
    reexplain_key = f"reexplain_{msg_index}"
    if reexplain_key in st.session_state:
        mode, content = st.session_state[reexplain_key]
        st.markdown(f"### {mode}")
        st.markdown(f'<div class="answer-container">{content}</div>', unsafe_allow_html=True)
        if st.button("❌ Close", key=f"close_reexplain_{msg_index}"):
            del st.session_state[reexplain_key]
            st.rerun()
    
    # Show audio player if available
    audio_key = f"audio_{msg_index}"
    if audio_key in st.session_state:
        audio_html = f'''
        <audio controls autoplay style="width: 100%; margin-top: 10px;">
            <source src="data:audio/mp3;base64,{st.session_state[audio_key]}" type="audio/mp3">
        </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
        if st.button("🔇 Close Audio", key=f"close_audio_{msg_index}"):
            del st.session_state[audio_key]
            st.rerun()


def render_deep_dive_panel(llm):
    """Render the deep dive chat panel"""
    st.markdown('<div class="deep-dive-header">💬 Deep Dive Chat</div>', unsafe_allow_html=True)
    
    topic = st.session_state.get("deep_dive_topic", "this topic")
    st.markdown(f"**Topic:** {topic[:80]}..." if len(topic) > 80 else f"**Topic:** {topic}")
    
    # Display deep dive chat history
    for msg in st.session_state.get("deep_dive_messages", []):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")
    
    st.markdown("---")
    
    # Deep dive input - use a form to properly clear input after submit
    with st.form(key="deep_dive_form", clear_on_submit=True):
        deep_input = st.text_input(
            "Ask a follow-up question...",
            placeholder="Type your follow-up question here...",
            key="deep_dive_text_input"
        )
        
        col_send, col_close = st.columns([3, 1])
        
        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)
        
        with col_close:
            close_clicked = st.form_submit_button("✕ Close", use_container_width=True)
    
    if submitted and deep_input:
        if "deep_dive_messages" not in st.session_state:
            st.session_state.deep_dive_messages = []
        
        st.session_state.deep_dive_messages.append({
            "role": "user",
            "content": deep_input
        })
        
        with st.spinner("Thinking..."):
            context = st.session_state.get("deep_dive_context", "")
            response = get_deep_dive_response(
                llm, 
                context, 
                deep_input, 
                st.session_state.deep_dive_messages
            )
            
            st.session_state.deep_dive_messages.append({
                "role": "assistant",
                "content": response
            })
        
        st.rerun()
    
    if close_clicked:
        st.session_state.show_deep_dive = False
        st.session_state.deep_dive_messages = []
        st.rerun()


def process_question(question: str, vectordb, llm, web_fallback: bool, default_mode: str) -> Dict:
    """Process a question and return answer with metadata"""
    length_mode = detect_length_mode(question)
    if length_mode == "medium":
        length_mode = default_mode
    
    rr = retrieve_docs(vectordb, question)
    
    file_cites = []
    web_cites = []
    answer_text = ""
    
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
                answer_text = "I can't find anything in your uploaded files, and web search is unavailable."
        else:
            answer_text = "I can't find this in your uploaded files. Try enabling web fallback in the sidebar."
    else:
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
            
            cant_answer_phrases = [
                "does not contain", "do not contain", "cannot provide",
                "can't provide", "no information", "not found",
                "not mentioned", "doesn't mention", "unable to find",
                "couldn't find", "could not find"
            ]
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
                    answer_text = "📄 I couldn't find this in your documents.\n🌐 Here's what I found from the web:\n\n" + resp.content.strip()
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
                    answer_text = "📄 I couldn't find relevant information in your documents.\n🌐 Here's what I found from the web:\n\n" + resp.content.strip()
                else:
                    answer_text = "I found some related info in your files, but not enough to answer confidently. Try a more specific question."
            else:
                answer_text = "I couldn't find relevant information in your documents for this question. Try enabling web fallback."
    
    return {
        "answer": answer_text,
        "file_cites": file_cites,
        "web_cites": web_cites,
        "question": question
    }


# -----------------------
# Main App
# -----------------------
def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Missing `OPENAI_API_KEY` in your .env file. Please add it to continue.")
        st.stop()
    
    # Initialize resources
    try:
        vectordb = init_vectordb()
        llm = init_llm()
    except Exception as e:
        st.error(f"⚠️ Error initializing resources: {str(e)}")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_deep_dive" not in st.session_state:
        st.session_state.show_deep_dive = False
    if "current_theme" not in st.session_state:
        st.session_state.current_theme = "🌙 Night Study"
    
    # Render sidebar
    web_fallback, default_mode, selected_theme = render_sidebar(vectordb)
    
    # Inject CSS with selected theme
    inject_custom_css(selected_theme)
    
    # Main layout
    if st.session_state.get("show_deep_dive"):
        main_col, deep_dive_col = st.columns([2, 1])
    else:
        main_col = st.container()
        deep_dive_col = None
    
    with main_col:
        # Header
        theme = THEMES.get(selected_theme, THEMES["🌙 Night Study"])
        st.markdown(
            f'<h1 style="color: {theme["accent"]}; margin-bottom: 20px;">📚 Study Helper</h1>',
            unsafe_allow_html=True
        )
        
        # Display chat history with quick actions
        for i, msg in enumerate(st.session_state.messages):
            role = msg["role"]
            content = msg["content"]
            
            with st.chat_message(role):
                st.markdown(content)
                
                if role == "assistant":
                    # Show sources
                    if msg.get("file_sources"):
                        with st.expander("📄 File Sources", expanded=False):
                            render_source_badges(msg["file_sources"], source_type="file")
                    
                    if msg.get("web_sources"):
                        with st.expander("🌐 Web Sources", expanded=False):
                            render_web_sources(msg["web_sources"])
                    
                    # Quick actions (only for the last assistant message)
                    if i == len(st.session_state.messages) - 1:
                        st.markdown("---")
                        st.markdown("**✨ Quick Actions:**")
                        original_question = msg.get("original_question", "")
                        render_quick_actions(llm, content, original_question, i)
        
        # Chat input
        if question := st.chat_input("Ask anything about your notes…"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Process question
            with st.spinner("🔍 Searching your notes..."):
                result = process_question(question, vectordb, llm, web_fallback, default_mode)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "file_sources": result["file_cites"],
                "web_sources": result["web_cites"],
                "original_question": question
            })
            
            st.rerun()
    
    # Deep dive panel (if open)
    if deep_dive_col:
        with deep_dive_col:
            render_deep_dive_panel(llm)


if __name__ == "__main__":
    main()
