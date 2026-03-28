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
MODELS = [
    {"label": "Llama 3.3 70B",        "provider": "groq",        "model_id": "llama-3.3-70b-versatile",                "key_env": "GROQ_API_KEY",       "ctx": 32768,   "free": True},
    {"label": "Llama 3.1 8B",         "provider": "groq",        "model_id": "llama-3.1-8b-instant",                  "key_env": "GROQ_API_KEY",       "ctx": 131072,  "free": True},
    {"label": "Gemini 2.0 Flash",     "provider": "gemini",      "model_id": "gemini-2.0-flash-exp",                  "key_env": "GEMINI_API_KEY",     "ctx": 1048576, "free": True},
    {"label": "Gemini 1.5 Flash",     "provider": "gemini",      "model_id": "gemini-1.5-flash",                      "key_env": "GEMINI_API_KEY",     "ctx": 1048576, "free": True},
    {"label": "Databricks Llama 70B", "provider": "databricks",  "model_id": "databricks-meta-llama-3-1-70b-instruct","key_env": "DATABRICKS_TOKEN",   "ctx": 128000,  "free": True},
    {"label": "Databricks Mixtral",   "provider": "databricks",  "model_id": "databricks-mixtral-8x7b-instruct",      "key_env": "DATABRICKS_TOKEN",   "ctx": 32768,   "free": True},
    {"label": "Databricks DBRX",      "provider": "databricks",  "model_id": "databricks-dbrx-instruct",              "key_env": "DATABRICKS_TOKEN",   "ctx": 32768,   "free": True},
    {"label": "GPT-4o",               "provider": "openai",      "model_id": "gpt-4o",                                "key_env": "OPENAI_API_KEY",     "ctx": 128000,  "free": False},
    {"label": "GPT-4o mini",          "provider": "openai",      "model_id": "gpt-4o-mini",                           "key_env": "OPENAI_API_KEY",     "ctx": 128000,  "free": False},
    {"label": "Claude Sonnet",        "provider": "anthropic",   "model_id": "claude-sonnet-4-20250514",              "key_env": "ANTHROPIC_API_KEY",  "ctx": 200000,  "free": False},
    {"label": "Llama 3.2 (Local)",    "provider": "ollama",      "model_id": "llama3.2",                              "key_env": None,                 "ctx": 128000,  "free": True},
    {"label": "Mistral (Local)",      "provider": "ollama",      "model_id": "mistral",                               "key_env": None,                 "ctx": 32768,   "free": True},
]

CHARS_PER_TOKEN = 4


# ── Availability ──────────────────────────────────────────────────────────────

def _is_ollama_running() -> bool:
    try:
        return requests.get("http://localhost:11434", timeout=1).status_code == 200
    except Exception:
        return False


def get_available_models() -> list[dict]:
    """
    Returns models that have API keys set.
    Does NOT ping any APIs — just checks env vars.
    Live testing is only done in Admin panel health check.
    """
    available = []
    for m in MODELS:
        if m["provider"] == "ollama":
            if _is_ollama_running():
                available.append(m)
            continue
        if m["key_env"] is None or os.getenv(m["key_env"]):
            available.append(m)
    return available if available else [MODELS[0]]


def test_all_models() -> list[dict]:
    """
    Full health check — pings every model with a test message.
    Only call this from Admin panel, never on page load.
    """
    import time
    results = []
    TEST = "Reply with one word: working"
    for m in MODELS:
        result = {
            "Model": m["label"],
            "Provider": m["provider"],
            "Status": "",
            "Latency": "—",
            "Preview": ""
        }
        if m["provider"] == "ollama" and not _is_ollama_running():
            result["Status"] = "🔌 offline"
            results.append(result)
            continue
        if m["key_env"] and not os.getenv(m["key_env"]):
            result["Status"] = "❌ no key"
            results.append(result)
            continue
        try:
            from langchain_core.messages import HumanMessage
            llm = get_llm(m["label"])
            start = time.time()
            response = llm.invoke([HumanMessage(content=TEST)])
            elapsed = round(time.time() - start, 1)
            result["Status"] = "✅ working"
            result["Latency"] = f"{elapsed}s"
            result["Preview"] = response.content[:40]
        except Exception as e:
            err = str(e).lower()
            if "403" in str(e) or "401" in str(e) or "permission" in err:
                result["Status"] = "❌ auth error"
            elif "timeout" in err:
                result["Status"] = "⏱️ timeout"
            elif "connect" in err or "refused" in err:
                result["Status"] = "🔌 offline"
            elif "not_found" in err or "404" in str(e):
                result["Status"] = "❌ model not found"
            elif "resource_exhausted" in err or "quota" in err:
                result["Status"] = "⚠️ quota exceeded"
            else:
                result["Status"] = "❌ error"
            result["Preview"] = str(e)[:60]
        results.append(result)
    return results


def get_model_by_label(label: str) -> dict | None:
    return next((m for m in MODELS if m["label"] == label), None)


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(model_label: str, temperature: float = 0):
    """Get LangChain LLM by model label string."""
    m = get_model_by_label(model_label)
    if not m:
        m = next((x for x in MODELS if x["label"] == "GPT-4o mini"), MODELS[0])

    provider = m["provider"]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=m["model_id"],
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=m["model_id"],
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=m["model_id"],
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=m["model_id"],
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    elif provider == "databricks":
        from databricks_langchain import ChatDatabricks
        # ChatDatabricks reads DATABRICKS_HOST and DATABRICKS_TOKEN
        # from environment automatically — ensure correct format
        host = os.getenv("DATABRICKS_HOST", "")
        if host and not host.startswith("http"):
            os.environ["DATABRICKS_HOST"] = f"https://{host}"
        return ChatDatabricks(
            endpoint=m["model_id"],
            temperature=temperature,
            max_tokens=1000
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=m["model_id"],
            temperature=temperature
        )

    raise ValueError(f"Unknown provider: {provider}")


# ── History truncation ────────────────────────────────────────────────────────

def truncate_history(messages: list, model_label: str,
                     system_prompt: str = "", reserve_tokens: int = 2000) -> list:
    m = get_model_by_label(model_label)
    ctx_tokens = m["ctx"] if m else 4096
    usable_chars = (ctx_tokens - reserve_tokens) * CHARS_PER_TOKEN - len(system_prompt)

    kept, used = [], 0
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
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
    Renders a flat model dropdown in the sidebar.
    Returns selected model label.
    Fast — no API calls, just checks env vars.
    """
    import streamlit as st

    available = get_available_models()
    labels = [m["label"] for m in available]

    if not labels:
        st.sidebar.warning("No models available. Check your API keys in .env")
        return ""

    default_label = next((m["label"] for m in available if m["free"]), labels[0])

    if "llm_model" not in st.session_state:
        st.session_state.llm_model = default_label

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

    model_info = get_model_by_label(selected)
    if model_info and model_info["free"]:
        st.sidebar.markdown(
            '<span style="font-size:0.7rem; background:#0d2618; color:#4ade80;'
            ' border:1px solid #166534; padding:1px 6px; border-radius:99px;">🟢 Free</span>',
            unsafe_allow_html=True
        )

    if selected != prev_model:
        history = st.session_state.get("chat_history", [])
        if history:
            truncated = truncate_history(history, selected)
            st.session_state.chat_history = truncated
            if len(truncated) < len(history):
                dropped = len(history) - len(truncated)
                st.sidebar.caption(f"ℹ️ {dropped} older message(s) trimmed for {selected}.")

    if model_info and model_info["provider"] == "ollama" and not _is_ollama_running():
        st.sidebar.warning("Ollama not running. Install from ollama.com and run `ollama serve`.")

    return selected