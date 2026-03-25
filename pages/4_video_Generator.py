"""
pages/4_🎥_Video_Generator.py — Study Helper v2
Topic → narrated MP4 video.
"""

import streamlit as st

st.set_page_config(
    page_title="Study Helper · Video",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector, get_llm
from video_generator import render_video_generator_page

inject_css()
user_id = require_auth()
sidebar_header(active_page="Video Generator")

with st.sidebar:
    st.markdown("---")
    model_label = render_model_selector() 
    st.markdown("---")
    st.markdown(
        '<div class="sh-info">'
        '🎬 Videos are generated from your notes when available, '
        'with web or AI fallback.</div>',
        unsafe_allow_html=True
    )

llm = get_llm(model_label)
render_video_generator_page(llm, user_id)