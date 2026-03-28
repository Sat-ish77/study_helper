from supabase import create_client
from dotenv import load_dotenv
from pathlib import Path
import os
import streamlit as st

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

@st.cache_resource
def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError(f".env not loaded. Looked in: {env_path}")
    return create_client(url, key)