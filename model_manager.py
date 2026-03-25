"""
model_manager.py — Study Helper v2
Flat model list — every model stands alone, no provider grouping.
"""

from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Flat model registry ───────────────────────────────────────────────────────
# Each model is independent — no provider nesting.
# label: what user sees in dropdown
# provider: which API to call
# model_id: actual API model name
# key_env: env var needed (None = local)
# ctx: context window in tokens
# free: show green Free badge

MODELS = [
    {"label": "Llama 3.3 70B",        "provider": "groq",        "model_id": "llama-3.3-70b-versatile",                      "key_env": "GROQ_API_KEY",       "ctx": 32768,   "free": True},
    {"label": "Llama 3.1 8B",         "provider": "groq",        "model_id": "llama-3.1-8b-instant",                         "key_env": "GROQ_API_KEY",       "ctx": 131072,  "free": True},
    {"label": "Mixtral 8x7B",         "provider": "groq",        "model_id": "mixtral-8x7b-32768",                           "key_env": "GROQ_API_KEY",       "ctx": 32768,   "free": True},
    {"label": "Gemini 2.0 Flash",     "provider": "gemini",      "model_id": "gemini-2.0-flash",                             "key_env": "GEMINI_API_KEY",     "ctx": 1048576, "free": True},
    {"label": "Gemini 1.5 Flash",     "provider": "gemini",      "model_id": "gemini-1.5-flash-latest",                      "key_env": "GEMINI_API_KEY",     "ctx": 1048576, "free": True},
    {"label": "Databricks Llama 70B", "provider": "databricks",  "model_id": "databricks-meta-llama-3-1-70b-instruct",       "key_env": "DATABRICKS_TOKEN",   "ctx": 128000,  "free": True},
    {"label": "Databricks Mixtral",   "provider": "databricks",  "model_id": "databricks-mixtral-8x7b-instruct",             "key_env": "DATABRICKS_TOKEN",   "ctx": 32768,   "free": True},
    {"label": "Databricks DBRX",      "provider": "databricks",  "model_id": "databricks-dbrx-instruct",                    "key_env": "DATABRICKS_TOKEN",   "ctx": 32768,   "free": True},
    {"label": "GPT-4o",               "provider": "openai",      "model_id": "gpt-4o",                                       "key_env": "OPENAI_API_KEY",     "ctx": 128000,  "free": False},
    {"label": "GPT-4o mini",          "provider": "openai",      "model_id": "gpt-4o-mini",                                  "key_env": "OPENAI_API_KEY",     "ctx": 128000,  "free": False},
    {"label": "Claude Sonnet",        "provider": "anthropic",   "model_id": "claude-sonnet-4-20250514",                     "key_env": "ANTHROPIC_API_KEY",  "ctx": 200000,  "free": False},
    {"label": "Llama 3.2 (Local)",    "provider": "ollama",      "model_id": "llama3.2",                                     "key_env": None,                 "ctx": 128000,  "free": True},
    {"label": "Mistral (Local)",      "provider": "ollama",      "model_id": "mistral",                                      "key_env": None,                 "ctx": 32768,   "free": True},
]

CHARS_PER_TOKEN = 4


# ── Availability ──────────────────────────────────────────────────────────────

def _is_ollama_running() -> bool:
    try:
        return requests.get("http://localhost:11434", timeout=1).status_code == 200
    except Exception:
        return False


def _is_available(m: dict) -> bool:
    if m["provider"] == "ollama":
        return _is_ollama_running()
    if m["key_env"] is None:
        return True
    return bool(os.getenv(m["key_env"]))


def get_available_models() -> list[dict]:
    """Returns list of available model dicts."""
    available = [m for m in MODELS if _is_available(m)]
    # Always include at least GPT-4o if OPENAI_API_KEY set, else first available
    return available if available else [MODELS[8]]  # GPT-4o as fallback


def get_model_by_label(label: str) -> dict | None:
    return next((m for m in MODELS if m["label"] == label), None)


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(model_label: str, temperature: float = 0):
    """Get LangChain LLM by model label string."""
    m = get_model_by_label(model_label)
    if not m:
        # fallback to GPT-4o
        m = MODELS[8]

    provider = m["provider"]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m["model_id"], temperature=temperature,
                          openai_api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=m["model_id"], temperature=temperature,
                             anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=m["model_id"], temperature=temperature,
                        groq_api_key=os.getenv("GROQ_API_KEY"))

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=m["model_id"], temperature=temperature,
                                      google_api_key=os.getenv("GEMINI_API_KEY"))

    elif provider == "databricks":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=m["model_id"], temperature=temperature,
            api_key=os.getenv("DATABRICKS_TOKEN"),
            base_url=f"https://{os.getenv('DATABRICKS_HOST', '')}/serving-endpoints",
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=m["model_id"], temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}")


def get_embeddings():
    """Databricks BGE if available, else OpenAI."""
    token = os.getenv("DATABRICKS_TOKEN")
    host  = os.getenv("DATABRICKS_HOST")
    if token and host:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="databricks-bge-large-en",
            api_key=token,
            base_url=f"https://{host}/serving-endpoints",
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ── History truncation ────────────────────────────────────────────────────────

def truncate_history(messages: list, model_label: str,
                     system_prompt: str = "", reserve_tokens: int = 2000) -> list:
    m = get_model_by_label(model_label)
    ctx_tokens   = m["ctx"] if m else 4096
    usable_chars = (ctx_tokens - reserve_tokens) * CHARS_PER_TOKEN - len(system_prompt)

    kept, used = [], 0
    for msg in reversed(messages):
        content   = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        msg_chars = len(str(content))
        if used + msg_chars > usable_chars:
            break
        kept.append(msg)
        used += msg_chars

    kept.reverse()
    return kept


# ── Sidebar widget ────────────────────────────────────────────────────────────

def render_model_selector() -> str:
    """
    Renders a single flat model dropdown in the sidebar.
    Returns selected model label (str).
    Handles history carry-over on model switch.
    """
    import streamlit as st

    available = get_available_models()
    labels    = [m["label"] for m in available]

    if not labels:
        st.sidebar.warning("No models available. Check your API keys in .env")
        return ""

    # Default to first free model
    default_label = next((m["label"] for m in available if m["free"]), labels[0])

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = default_label

    # Keep selection valid if model becomes unavailable
    if st.session_state.llm_model not in labels:
        st.session_state.llm_model = default_label

    st.sidebar.markdown(
        '<div style="font-size:0.72rem; color:#4b5563; text-transform:uppercase;'
        ' letter-spacing:0.08em; margin-bottom:0.4rem;">AI Model</div>',
        unsafe_allow_html=True
    )

    prev_model = st.session_state.llm_model

    selected = st.sidebar.selectbox(
        "Model",
        labels,
        index=labels.index(st.session_state.llm_model),
        label_visibility="collapsed",
        key="sb_model_flat",
    )

    st.session_state.llm_model = selected

    # Show free badge
    model_info = get_model_by_label(selected)
    if model_info and model_info["free"]:
        st.sidebar.markdown(
            '<span style="font-size:0.7rem; background:#0d2618; color:#4ade80;'
            ' border:1px solid #166534; padding:1px 6px; border-radius:99px;">🟢 Free</span>',
            unsafe_allow_html=True
        )

    # Carry history over on switch, truncate if needed
    if selected != prev_model:
        history = st.session_state.get("chat_history", [])
        if history:
            truncated = truncate_history(history, selected)
            st.session_state.chat_history = truncated
            if len(truncated) < len(history):
                dropped = len(history) - len(truncated)
                st.sidebar.caption(f"ℹ️ {dropped} older message(s) trimmed for {selected}.")

    # Ollama warning
    if model_info and model_info["provider"] == "ollama" and not _is_ollama_running():
        st.sidebar.warning("Ollama not running. Install from ollama.com and run `ollama serve`.")

    return selected