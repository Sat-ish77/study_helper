"""
pages/4_🖼️_Image_Lab.py — Study Helper v2
Image Lab for custom diagrams and visualizations.
Video generation removed in favor of Learn More section.
"""
import streamlit as st
from styles.theme import inject_css, sidebar_header
from auth import require_auth
from model_manager import render_model_selector, get_llm
from image_generator import render_image_lab_page

# Page config
st.set_page_config(
    page_title="Image Lab — Study Helper",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Theme + auth
inject_css()
user_id = require_auth()
if not user_id:
    st.stop()

# Sidebar
sidebar_header()
model_label = render_model_selector()
llm = get_llm(model_label)

# Render Image Lab
render_image_lab_page(llm)
