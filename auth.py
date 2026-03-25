"""
auth.py — Study Helper v2
Two-layer auth:
  Layer 1: Supabase email/password
  Layer 2: Hardcoded app password (env var APP_PASSWORD)
Session persists across ALL pages until logout or browser close.
"""

from __future__ import annotations
import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL      = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
APP_PASSWORD      = os.getenv("APP_PASSWORD")


@st.cache_resource
def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment.")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _render_login_tab():
    st.markdown("#### Sign in")
    email    = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Sign in", use_container_width=True, type="primary", key="btn_login"):
        if not email or not password:
            st.warning("Please enter email and password.")
            return
        try:
            sb  = get_supabase()
            res = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state.user_id    = str(res.user.id)
            st.session_state.user_email = res.user.email
            st.session_state.logged_in  = True
            st.rerun()
        except Exception as e:
            err = str(e).lower()
            if "invalid" in err or "credentials" in err:
                st.error("Incorrect email or password.")
            elif "confirmed" in err:
                st.error("Please confirm your email address first.")
            else:
                st.error(f"Login failed: {e}")


def _render_register_tab():
    st.markdown("#### Create account")
    email    = st.text_input("Email", key="reg_email")
    password = st.text_input("Password (min 6 chars)", type="password", key="reg_password")
    confirm  = st.text_input("Confirm password", type="password", key="reg_confirm")

    if st.button("Create account", use_container_width=True, type="primary", key="btn_register"):
        if not email or not password:
            st.warning("Please fill in all fields.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        try:
            sb = get_supabase()
            sb.auth.sign_up({"email": email, "password": password})
            st.success("Account created! Check your email to confirm, then sign in.")
        except Exception as e:
            st.error(f"Registration failed: {e}")


def _render_supabase_gate():
    st.markdown('<div style="max-width:420px; margin:3rem auto 0;">', unsafe_allow_html=True)
    tab_login, tab_register = st.tabs(["Sign in", "Create account"])
    with tab_login:
        _render_login_tab()
    with tab_register:
        _render_register_tab()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def _render_app_password_gate():
    email = st.session_state.get("user_email", "")
    st.markdown(
        f'<p style="text-align:center; color:#6b7280; font-size:0.85rem; margin-bottom:1rem;">'
        f'Signed in as {email}</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="max-width:380px; margin:1rem auto 0; text-align:center;">'
        '<p style="color:#6b7280; font-size:0.875rem; margin-bottom:1.25rem;">'
        'Enter the access code to continue.</p></div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        entered = st.text_input(
            "Access code", type="password",
            key="app_password_input", placeholder="••••••••"
        )
        if st.button("Unlock", use_container_width=True, type="primary", key="btn_unlock"):
            if not APP_PASSWORD:
                st.session_state.app_unlocked = True
                st.rerun()
            elif entered == APP_PASSWORD:
                st.session_state.app_unlocked = True
                st.rerun()
            else:
                st.error("Incorrect access code.")
    st.stop()


def require_auth() -> str:
    """
    Call at the TOP of every page before any other code.
    Session persists across all pages — password asked only ONCE per session.
    Returns user_id (str) when both layers pass.
    """
    # Always set defaults so keys exist — prevents None edge cases
    for key, default in [
        ("logged_in",    False),
        ("app_unlocked", False),
        ("user_id",      ""),
        ("user_email",   ""),
    ]:
        st.session_state.setdefault(key, default)

    # Layer 1
    if not st.session_state.logged_in:
        _render_supabase_gate()
        return ""

    # Layer 2
    if not st.session_state.app_unlocked:
        _render_app_password_gate()
        return ""

    return st.session_state.user_id


def get_current_user() -> dict | None:
    if not st.session_state.get("logged_in"):
        return None
    return {
        "id":    st.session_state.get("user_id", ""),
        "email": st.session_state.get("user_email", ""),
    }


def logout():
    try:
        get_supabase().auth.sign_out()
    except Exception:
        pass
    for key in ["logged_in", "user_id", "user_email", "app_unlocked"]:
        st.session_state.pop(key, None)
    st.rerun()


def render_logout_button():
    if st.button("🔒 Sign out", use_container_width=True, key="btn_signout"):
        logout()