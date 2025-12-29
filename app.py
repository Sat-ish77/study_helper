"""
Study Helper - Streamlit UI
Modern dark-themed chat interface for RAG-based study assistance
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


# -----------------------
# Custom CSS for dark theme with glassmorphism
# -----------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global dark theme */
        .stApp {
            background-color: #1a1a1a;
            color: #e8e8e8;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a1a 0%, #242424 100%);
            border-right: 1px solid #333;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #f5a623 !important;
            font-weight: 600;
        }

        /* Glassy card effect */
        .glass-card {
            background: rgba(42, 42, 42, 0.6);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(245, 166, 35, 0.2);
            margin: 10px 0;
        }

        /* Answer container */
        .answer-container {
            background: linear-gradient(135deg, rgba(42, 42, 42, 0.8) 0%, rgba(32, 32, 32, 0.9) 100%);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(245, 166, 35, 0.3);
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        /* Source badges */
        .source-badge {
            display: inline-block;
            background: rgba(245, 166, 35, 0.15);
            color: #f5a623;
            padding: 6px 14px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9em;
            border: 1px solid rgba(245, 166, 35, 0.3);
            font-weight: 500;
        }

        .web-source-badge {
            display: inline-block;
            background: rgba(52, 152, 219, 0.15);
            color: #3498db;
            padding: 6px 14px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9em;
            border: 1px solid rgba(52, 152, 219, 0.3);
            font-weight: 500;
        }

        /* Links styling */
        a {
            color: #f5a623;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        a:hover {
            color: #ffb84d;
            text-decoration: underline;
        }

        /* Chat message styling */
        .stChatMessage {
            background: rgba(42, 42, 42, 0.4);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }

        /* Input styling */
        .stChatInput input {
            background-color: rgba(42, 42, 42, 0.8) !important;
            border: 1px solid rgba(245, 166, 35, 0.3) !important;
            border-radius: 24px !important;
            color: #e8e8e8 !important;
            padding: 12px 20px !important;
        }

        .stChatInput input:focus {
            border-color: #f5a623 !important;
            box-shadow: 0 0 0 2px rgba(245, 166, 35, 0.2) !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(42, 42, 42, 0.6) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(245, 166, 35, 0.2) !important;
            color: #f5a623 !important;
            font-weight: 600 !important;
        }

        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #f5a623 0%, #d88e1a 100%);
            color: #1a1a1a;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(245, 166, 35, 0.4);
        }

        /* Toggle and radio styling */
        [data-testid="stToggle"] {
            color: #f5a623;
        }

        /* Markdown in answers */
        .answer-container p {
            line-height: 1.7;
            margin-bottom: 12px;
        }

        .answer-container ul, .answer-container ol {
            margin-left: 20px;
            line-height: 1.8;
        }

        .answer-container code {
            background: rgba(245, 166, 35, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            color: #f5a623;
        }

        /* File list styling */
        .file-item {
            background: rgba(245, 166, 35, 0.05);
            padding: 8px 12px;
            border-radius: 6px;
            margin: 4px 0;
            border-left: 3px solid #f5a623;
            font-size: 0.9em;
        }

        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #f5a623 !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1a1a1a;
        }

        ::-webkit-scrollbar-thumb {
            background: #f5a623;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #ffb84d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        # Get all documents from the collection
        collection = _vectordb._collection
        all_metadata = collection.get()["metadatas"]

        # Extract unique filenames
        filenames = set()
        for meta in all_metadata:
            if meta and "filename" in meta:
                filenames.add(meta["filename"])
            elif meta and "source" in meta:
                # Fallback to source field
                source = meta["source"]
                # Extract just the filename
                if "/" in source or "\\" in source:
                    source = os.path.basename(source)
                filenames.add(source)

        return sorted(list(filenames))
    except Exception:
        return []


# -----------------------
# UI Components
# -----------------------
def render_sidebar(vectordb):
    """Render sidebar with controls and file list"""
    with st.sidebar:
        # Title with emoji
        st.markdown("# 📚 Study Helper")
        st.markdown("*Late-night study session mode*")
        st.markdown("---")

        # Web fallback toggle
        st.markdown("### 🌐 Settings")
        web_fallback = st.toggle(
            "Web Fallback",
            value=False,
            help="Enable Tavily web search when file context is insufficient"
        )

        # Answer mode radio
        answer_mode = st.radio(
            "Answer Mode",
            options=["Short", "Medium", "Long"],
            index=1,  # Default to Medium
            help="Control the length and detail of answers"
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

        # Footer
        st.markdown(
            '<div style="text-align: center; color: #888; font-size: 0.85em; margin-top: 40px;">'
            'Powered by LangChain + OpenAI'
            '</div>',
            unsafe_allow_html=True
        )

    return web_fallback, answer_mode.lower()


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
    if not web_cites:
        return

    for cite in web_cites:
        # Parse citation format: [W1] Title — URL
        parts = cite.split(" — ")
        if len(parts) == 2:
            tag_title = parts[0]
            url = parts[1]
            st.markdown(f"🔗 [{tag_title}]({url})")
        else:
            st.markdown(f"🔗 {cite}")


# -----------------------
# Main app
# -----------------------
def main():
    # Page config
    st.set_page_config(
        page_title="Study Helper",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    inject_custom_css()

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

    # Render sidebar and get settings
    web_fallback, default_mode = render_sidebar(vectordb)

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Main header
    st.markdown(
        '<h1 style="color: #f5a623; text-align: center; margin-bottom: 30px;">'
        '📚 Study Helper</h1>',
        unsafe_allow_html=True
    )

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            st.markdown(content)

            # Show sources if available
            if role == "assistant":
                if "file_sources" in msg and msg["file_sources"]:
                    with st.expander("📄 File Sources", expanded=False):
                        render_source_badges(msg["file_sources"], source_type="file")

                if "web_sources" in msg and msg["web_sources"]:
                    with st.expander("🌐 Web Sources", expanded=False):
                        render_web_sources(msg["web_sources"])

    # Chat input
    if question := st.chat_input("Ask anything about your notes…"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        # Process question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Detect length mode from question or use default
                    length_mode = detect_length_mode(question)
                    if length_mode == "medium":
                        length_mode = default_mode

                    # Retrieve documents from vectordb
                    rr = retrieve_docs(vectordb, question)

                    file_cites = []
                    web_cites = []
                    answer_text = ""

                    # Handle case: no documents retrieved
                    if not rr.docs:
                        if web_fallback:
                            web_results = tavily_search(question)
                            if web_results:
                                web_context, web_cites = build_tagged_web_context(web_results)
                                user_prompt = make_user_prompt(
                                    question,
                                    length_mode,
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
                            answer_text = "I can't find this in your uploaded files. Try enabling web fallback."

                    # Handle case: documents retrieved
                    else:
                        # Check if context is sufficient
                        sufficient = context_is_sufficient(rr, length_mode)

                        if sufficient:
                            # Answer from files only
                            file_context, file_cites = build_tagged_context(rr.docs)
                            user_prompt = make_user_prompt(
                                question,
                                length_mode,
                                file_context=file_context,
                                web_context=None
                            )
                            resp = llm.invoke([
                                SystemMessage(content=SYSTEM_RULES),
                                HumanMessage(content=user_prompt)
                            ])
                            answer_text = resp.content.strip()

                            # Post-answer check: did LLM say it couldn't answer?
                            cant_answer_phrases = [
                                "does not contain", "do not contain", "cannot provide",
                                "can't provide", "no information", "not found",
                                "not mentioned", "doesn't mention", "unable to find",
                                "couldn't find", "could not find"
                            ]
                            llm_says_no = any(p in answer_text.lower() for p in cant_answer_phrases)

                            if llm_says_no and web_fallback:
                                # Try web fallback
                                web_results = tavily_search(question)
                                if web_results:
                                    web_context, web_cites = build_tagged_web_context(web_results)
                                    user_prompt = make_user_prompt(
                                        question,
                                        length_mode,
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
                            # Context insufficient
                            if web_fallback:
                                web_results = tavily_search(question)
                                if web_results:
                                    web_context, web_cites = build_tagged_web_context(web_results)
                                    file_context, file_cites = build_tagged_context(rr.docs)
                                    user_prompt = make_user_prompt(
                                        question,
                                        length_mode,
                                        file_context=file_context,
                                        web_context=web_context
                                    )
                                    resp = llm.invoke([
                                        SystemMessage(content=SYSTEM_RULES),
                                        HumanMessage(content=user_prompt)
                                    ])
                                    answer_text = resp.content.strip()
                                else:
                                    answer_text = "I found some related info in your files, but not enough to answer confidently. Try a more specific question or upload more relevant notes."
                            else:
                                answer_text = "I couldn't find relevant information in your documents for this question. Try enabling web fallback or upload notes covering this topic."

                    # Display answer
                    st.markdown(
                        f'<div class="answer-container">{answer_text}</div>',
                        unsafe_allow_html=True
                    )

                    # Display sources
                    if file_cites:
                        with st.expander("📄 File Sources", expanded=True):
                            render_source_badges(file_cites, source_type="file")

                    if web_cites:
                        with st.expander("🌐 Web Sources", expanded=True):
                            render_web_sources(web_cites)

                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "file_sources": file_cites,
                        "web_sources": web_cites
                    })

                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "file_sources": [],
                        "web_sources": []
                    })


if __name__ == "__main__":
    main()
