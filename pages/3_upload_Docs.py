"""
pages/3_📤_Upload_Docs.py — Study Helper v2
Per-user document upload → Supabase pgvector.
Supports PDF, DOCX, PPTX, TXT, PNG, JPG, WEBP.
"""

import streamlit as st

st.set_page_config(
    page_title="Study Helper · Upload",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

from auth import require_auth
from styles.theme import inject_css, sidebar_header
from model_manager import render_model_selector
from ingest import (
    ingest_uploaded_file,
    get_user_documents,
    delete_user_document,
)

inject_css()
user_id = require_auth()
sidebar_header(active_page="Upload Docs")

with st.sidebar:
    st.markdown("---")
    render_model_selector()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h2 style="font-family:\'DM Serif Display\',serif; color:#f0ede8;">📤 Upload Documents</h2>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#6b7280; margin-bottom:1.5rem;">'
    'Your documents are private — only you can see and search them.</p>',
    unsafe_allow_html=True
)

# ── Supported formats info ────────────────────────────────────────────────────
st.markdown(
    '<div class="sh-info" style="margin-bottom:1.5rem;">'
    '📄 <b>PDF</b> &nbsp;·&nbsp; 📝 <b>DOCX</b> &nbsp;·&nbsp;'
    ' 📊 <b>PPTX</b> &nbsp;·&nbsp; 📃 <b>TXT</b>'
    ' &nbsp;·&nbsp; 🖼️ <b>PNG / JPG / WEBP</b> (OCR extracted)'
    '</div>',
    unsafe_allow_html=True
)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Drop your study files here",
    type=["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=f"uploader_{user_id}",
    help="All file types supported. Images are processed with OCR."
)

if uploaded_files:
    st.markdown(
        f'<div style="font-size:0.85rem; color:#9ca3af; margin:0.5rem 0 1rem;">'
        f'Processing {len(uploaded_files)} file(s)...</div>',
        unsafe_allow_html=True
    )

    progress_bar = st.progress(0)
    results      = []

    for i, uf in enumerate(uploaded_files):
        with st.spinner(f"Processing {uf.name}..."):
            result = ingest_uploaded_file(uf, user_id)
            results.append((uf.name, result))
        progress_bar.progress((i + 1) / len(uploaded_files))

    progress_bar.empty()

    # Show results
    success_count = sum(1 for _, r in results if r["success"])
    fail_count    = len(results) - success_count

    if success_count:
        st.markdown(
            f'<div style="background:#0d2618; border:0.5px solid #166534; border-radius:8px;'
            f' padding:12px 16px; margin-bottom:0.5rem;">'
            f'<span style="color:#4ade80; font-weight:500;">✓ {success_count} file(s) indexed successfully</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    for name, result in results:
        if result["success"]:
            st.markdown(
                f'<div style="font-size:0.85rem; color:#6b7280; padding:4px 0;">'
                f'📄 <b style="color:#e2e4e9;">{result["filename"]}</b>'
                f' — {result["pages"]} pages, {result["chunks"]} chunks indexed</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(f"✗ {name}: {result['error']}")

    if fail_count:
        st.markdown(
            '<div class="sh-warn">Some files failed. '
            'For images, make sure pytesseract is installed.</div>',
            unsafe_allow_html=True
        )

# ── Existing documents ────────────────────────────────────────────────────────
st.markdown(
    '<h2 class="sh-section" style="margin-top:2.5rem;">'
    '<span class="sh-section-accent">✦</span>Your Documents</h2>',
    unsafe_allow_html=True
)

docs = get_user_documents(user_id)

if docs:
    st.markdown(
        f'<div style="font-size:0.78rem; color:#4b5563; margin-bottom:1rem;">'
        f'{len(docs)} document(s) in your knowledge base</div>',
        unsafe_allow_html=True
    )

    for doc in docs:
        col1, col2 = st.columns([6, 1])
        with col1:
            # File type icon
            ext  = doc.rsplit(".", 1)[-1].lower() if "." in doc else ""
            icon = {
                "pdf": "📄", "docx": "📝", "pptx": "📊",
                "txt": "📃", "png": "🖼️", "jpg": "🖼️",
                "jpeg": "🖼️", "webp": "🖼️",
            }.get(ext, "📄")
            st.markdown(
                f'<div style="background:#111318; border:0.5px solid #1e2028;'
                f' border-radius:8px; padding:10px 14px; font-size:0.875rem;'
                f' color:#e2e4e9;">{icon} {doc}</div>',
                unsafe_allow_html=True
            )
        with col2:
            if st.button("Remove", key=f"del_{doc}", use_container_width=True):
                if delete_user_document(user_id, doc):
                    st.success(f"Removed {doc}")
                    st.rerun()
                else:
                    st.error("Failed to remove.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sh-info">'
        '💡 After uploading, head to <b>Study Helper</b> to ask questions '
        'or <b>Quiz Lab</b> to generate a quiz from your notes.'
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align:center; padding:3rem 1rem; color:#4b5563;">'
        '<div style="font-size:2rem; margin-bottom:0.75rem;">📭</div>'
        '<div style="font-size:0.95rem;">No documents uploaded yet.</div>'
        '<div style="font-size:0.83rem; margin-top:0.4rem;">'
        'Upload your study files above to get started.</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── OCR note ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About image OCR"):
    st.markdown("""
    When you upload image files (PNG, JPG, WEBP), the app uses **pytesseract OCR** to
    extract text automatically. For best results:
    - Use clear, well-lit images of printed text
    - Avoid handwritten notes (OCR accuracy is lower)
    - After upload, check the chunk count — low counts may mean OCR found little text
    
    To enable image OCR on your local machine:
    ```bash
    pip install pytesseract pillow
    # macOS: brew install tesseract
    # Ubuntu: apt install tesseract-ocr
    # Windows: download installer from github.com/UB-Mannheim/tesseract
    ```
    """)