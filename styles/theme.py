"""
styles/theme.py — Study Helper v2
Central CSS design system. All pages call inject_css() once.
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0d0f14 !important;
    color: #e2e4e9 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
.eyeqlp51 { display: none !important; }
.st-emotion-cache-zq5wmm { display: none !important; }
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1080px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2028 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #6b7280 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Typography ── */
.sh-display {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #f0ede8;
    margin: 0;
}
.sh-display em {
    font-style: italic;
    color: #e8a44a;
}
.sh-serif {
    font-family: 'DM Serif Display', serif !important;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0ede8 !important;
}

/* ── Section headers ── */
.sh-section {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f0ede8;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1e2028;
    margin: 2rem 0 1.25rem;
}
.sh-section-accent { color: #e8a44a; margin-right: 0.4rem; }

/* ── Cards ── */
.sh-card {
    background: #111318;
    border: 0.5px solid #1e2028;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    transition: border-color 0.2s ease;
}
.sh-card:hover { border-color: rgba(232,164,74,0.3); }
.sh-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #f0ede8;
    margin-bottom: 0.4rem;
}
.sh-card-body {
    font-size: 0.875rem;
    color: #6b7280;
    line-height: 1.65;
}

/* ── Source badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 99px;
    letter-spacing: 0.04em;
    margin-right: 4px;
}
.badge-rag  { background:#0d2618; color:#4ade80; border:1px solid #166534; }
.badge-web  { background:#0d1a2e; color:#60a5fa; border:1px solid #1d4ed8; }
.badge-llm  { background:#1e1a0d; color:#fbbf24; border:1px solid #92400e; }
.badge-ocr  { background:#1a0d2e; color:#c084fc; border:1px solid #6b21a8; }

/* ── Model pill ── */
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1c22;
    border: 1px solid #2a2d36;
    border-radius: 99px;
    padding: 4px 10px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #e8a44a;
}

/* ── Chat messages ── */
.chat-user {
    background: #1a1c22;
    border: 0.5px solid #2a2d36;
    border-radius: 12px 12px 4px 12px;
    padding: 0.875rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #e2e4e9;
}
.chat-assistant {
    background: #111318;
    border: 0.5px solid #1e2028;
    border-radius: 12px 12px 12px 4px;
    padding: 0.875rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #e2e4e9;
    line-height: 1.7;
}
.chat-sources {
    font-size: 0.78rem;
    color: #4b5563;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 0.5px solid #1e2028;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #1a1c22 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 8px !important;
    color: #e2e4e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #e8a44a !important;
    box-shadow: 0 0 0 2px rgba(232,164,74,0.15) !important;
}
.stSelectbox > div > div {
    background: #1a1c22 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 8px !important;
    color: #e2e4e9 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e8a44a 0%, #d4943a 100%) !important;
    color: #0d0f14 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(232,164,74,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: #1a1c22 !important;
    color: #e8a44a !important;
    border: 1px solid #2a2d36 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #e8a44a !important;
    background: #1e2028 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2028 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #e8a44a !important;
    border-bottom-color: #e8a44a !important;
}

/* ── Progress / spinner ── */
.stProgress > div > div > div {
    background: #e8a44a !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #111318 !important;
    border: 0.5px solid #1e2028 !important;
    border-radius: 8px !important;
    color: #e2e4e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111318 !important;
    border: 1px dashed #2a2d36 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #e8a44a !important;
}

/* ── Divider ── */
.sh-divider {
    width: 48px; height: 2px;
    background: linear-gradient(90deg, #e8a44a, transparent);
    margin: 0.75rem 0 1.5rem;
}

/* ── Stat block ── */
.sh-stat {
    background: #111318;
    border: 0.5px solid #1e2028;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.sh-stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    color: #e8a44a;
    display: block;
}
.sh-stat-label {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Alert / info boxes ── */
.sh-info {
    background: #0d1a2e;
    border: 0.5px solid #1d4ed8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #93c5fd;
    margin: 0.75rem 0;
}
.sh-warn {
    background: #1e1a0d;
    border: 0.5px solid #92400e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.875rem;
    color: #fbbf24;
    margin: 0.75rem 0;
}

/* ── Footer ── */
.sh-footer {
    text-align: center;
    font-size: 0.72rem;
    color: #2d3748;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid #1a1c22;
    margin-top: 3rem;
    letter-spacing: 0.05em;
}

/* ── Sidebar logo ── */
.sh-sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #f0ede8;
    padding: 0.75rem 0 0.15rem;
}
.sh-sidebar-sub {
    font-size: 0.72rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.25rem;
}

/* ── Language selector label ── */
.lang-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}

/* ── Quiz ── */
.quiz-question {
    background: #111318;
    border: 0.5px solid #1e2028;
    border-left: 3px solid #e8a44a;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    color: #e2e4e9;
}
.quiz-correct {
    background: #0d2618;
    border: 0.5px solid #166534;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #4ade80;
    font-size: 0.875rem;
}
.quiz-wrong {
    background: #1e0d0d;
    border: 0.5px solid #7f1d1d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #f87171;
    font-size: 0.875rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2d36; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #e8a44a; }
</style>
"""


def inject_css():
    """Call once at the top of every page after set_page_config."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def sidebar_header(active_page: str = ""):
    """Render consistent sidebar logo + user info across all pages."""
    import streamlit as st
    from auth import get_current_user, render_logout_button

    with st.sidebar:
        st.markdown(
            '<div class="sh-sidebar-logo">📚 Study Helper</div>'
            '<div class="sh-sidebar-sub">AI Study Platform</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        # User info
        user = get_current_user()
        if user:
            st.markdown(
                f'<div style="font-size:0.78rem; color:#4b5563; margin-bottom:0.5rem;">'
                f'Signed in as<br>'
                f'<span style="color:#9ca3af;">{user["email"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown("---")

        render_logout_button()