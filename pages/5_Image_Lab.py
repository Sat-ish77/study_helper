"""
pages/5_🖼️_Image_Lab.py — Study Helper v2
Concept → DALL-E 3 image or matplotlib diagram.
"""

import streamlit as st

st.set_page_config(
    page_title="Study Helper · Image Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector, get_llm
from image_generator import render_image_lab_page

inject_css()
user_id = require_auth()
sidebar_header(active_page="Image Lab")

with st.sidebar:
    st.markdown("---")
    model_label = render_model_selector() 
    st.markdown("---")
    st.markdown(
        '<div class="sh-info">'
        '🎨 <b>Real images</b> use DALL-E 3 (~$0.04 each).<br>'
        '📊 <b>Charts &amp; flowcharts</b> use matplotlib — free.</div>',
        unsafe_allow_html=True
    )

llm = get_llm(model_label)
render_image_lab_page(llm)