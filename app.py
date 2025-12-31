"""
Study Helper - Home Page
Landing page with app introduction and features overview
"""

import streamlit as st
import io
import base64

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Study Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import gTTS for welcome message
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


# -----------------------
# Welcome Message TTS
# -----------------------
WELCOME_TEXT = """
Hi! I am Study Helper, your personal AI study assistant.

Here's what I can do for you:

First, I can answer questions from your study notes. Just upload your PDFs, Word documents, or PowerPoint files, and I'll find the answers with exact page and slide citations.

Second, if your notes don't have the answer, I can search the web and clearly show you where the information came from.

Third, I can explain things in different ways. Want it simpler? Click the Simpler button. Need more technical detail? Click Technical. Or I can explain in Nepali for easier understanding.

Fourth, you can have a deeper conversation about any topic using the Deep Dive feature.

And finally, visit the Quiz Lab to test yourself with auto-generated quizzes from your notes.

Let's start studying! Click on Study Helper in the sidebar to begin.
"""

@st.cache_data
def generate_welcome_audio():
    """Generate welcome message audio"""
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=WELCOME_TEXT, lang='en', slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        return audio_base64
    except Exception:
        return None


# -----------------------
# Custom CSS
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
        
        /* Hero section */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f5a623 0%, #ffb84d 50%, #f5a623 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .hero-subtitle {
            font-size: 1.3rem;
            color: #888;
            text-align: center;
            margin-bottom: 20px;
        }
        
        /* Welcome box */
        .welcome-box {
            background: linear-gradient(135deg, rgba(42, 42, 42, 0.9) 0%, rgba(32, 32, 32, 0.95) 100%);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 30px;
            border: 2px solid rgba(245, 166, 35, 0.4);
            margin: 20px auto;
            max-width: 800px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(245, 166, 35, 0.2);
        }
        
        .welcome-text {
            font-size: 1.5rem;
            color: #f5a623;
            margin-bottom: 15px;
        }
        
        .play-button {
            background: linear-gradient(135deg, #f5a623 0%, #d88e1a 100%);
            color: #1a1a1a;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .play-button:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 25px rgba(245, 166, 35, 0.5);
        }
        
        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, rgba(42, 42, 42, 0.8) 0%, rgba(32, 32, 32, 0.9) 100%);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(245, 166, 35, 0.2);
            height: 100%;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            border-color: rgba(245, 166, 35, 0.5);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(245, 166, 35, 0.2);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        
        .feature-title {
            color: #f5a623;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .feature-desc {
            color: #aaa;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* CTA Button */
        .cta-container {
            text-align: center;
            margin: 40px 0;
        }
        
        .cta-button {
            background: linear-gradient(135deg, #f5a623 0%, #d88e1a 100%);
            color: #1a1a1a !important;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.2rem;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            border: none;
        }
        
        .cta-button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(245, 166, 35, 0.4);
        }
        
        /* Section headers */
        .section-header {
            color: #f5a623;
            font-size: 1.8rem;
            font-weight: 600;
            text-align: center;
            margin: 50px 0 30px 0;
        }
        
        /* How it works steps */
        .step-number {
            background: linear-gradient(135deg, #f5a623 0%, #d88e1a 100%);
            color: #1a1a1a;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            margin-right: 15px;
        }
        
        .step-text {
            color: #e8e8e8;
            font-size: 1.1rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #666;
            margin-top: 60px;
            padding: 20px;
            border-top: 1px solid #333;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("# 📚 Study Helper")
        st.markdown("*Your AI Study Assistant*")
        st.markdown("---")
        st.markdown("### 📑 Navigation")
        st.markdown("Use the pages above to navigate:")
        st.markdown("- **🏠 Home** - You are here")
        st.markdown("- **📚 Study Helper** - Ask questions")
        st.markdown("- **🧪 Quiz Lab** - Take quizzes")
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; color: #888; font-size: 0.85em;">'
            'Built with ❤️ using LangChain + OpenAI'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Hero Section
    st.markdown('<h1 class="hero-title">📚 Study Helper</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Your Personal AI Study Assistant — Upload Notes, Ask Questions, Ace Exams</p>',
        unsafe_allow_html=True
    )
    
    # Welcome Box with Play Button
    st.markdown(
        '''
        <div class="welcome-box">
            <div class="welcome-text">👋 Welcome to Study Helper!</div>
            <p style="color: #aaa; margin-bottom: 20px;">Click the play button to hear what I can do for you</p>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Play button and audio
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("▶️ Play Introduction", use_container_width=True, type="primary"):
            with st.spinner("Generating audio..."):
                audio_data = generate_welcome_audio()
                if audio_data:
                    st.session_state.show_welcome_audio = True
                    st.session_state.welcome_audio_data = audio_data
                else:
                    st.warning("Audio generation not available. Please read the features below!")
    
    # Show audio player if available
    if st.session_state.get("show_welcome_audio") and st.session_state.get("welcome_audio_data"):
        audio_html = f'''
        <div style="text-align: center; margin: 20px 0;">
            <audio controls autoplay style="width: 80%; max-width: 500px;">
                <source src="data:audio/mp3;base64,{st.session_state.welcome_audio_data}" type="audio/mp3">
            </audio>
        </div>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)
        if st.button("🔇 Close Audio", key="close_welcome_audio"):
            st.session_state.show_welcome_audio = False
            st.rerun()
    
    # CTA Button
    st.markdown(
        '''
        <div class="cta-container">
            <a href="/Study_Helper" target="_self" class="cta-button">🚀 Start Studying</a>
        </div>
        ''',
        unsafe_allow_html=True
    )
    
    # Features Section
    st.markdown('<h2 class="section-header">✨ Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Multi-Format Support</div>
                <div class="feature-desc">
                    Upload your study notes in PDF, DOCX, or PPTX format. 
                    We'll extract and index everything for instant search.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Smart Citations</div>
                <div class="feature-desc">
                    Every answer includes citations with exact file names and page/slide numbers. 
                    Never wonder "where did this come from?"
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Web Fallback</div>
                <div class="feature-desc">
                    If your notes don't have the answer, we'll search the web 
                    and clearly mark which info came from where.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🇳🇵</div>
                <div class="feature-title">Nepali Explanations</div>
                <div class="feature-desc">
                    Don't understand in English? Click to get the same concept 
                    explained naturally in Nepali.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Deep Dive Chat</div>
                <div class="feature-desc">
                    Want to explore a topic further? Open the Deep Dive panel 
                    for back-and-forth conversation on any answer.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            '''
            <div class="feature-card">
                <div class="feature-icon">🧪</div>
                <div class="feature-title">Quiz Generation</div>
                <div class="feature-desc">
                    Test yourself with auto-generated quizzes from your notes. 
                    Multiple choice, true/false, and more.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    # How It Works Section
    st.markdown('<h2 class="section-header">🔧 How It Works</h2>', unsafe_allow_html=True)
    
    steps = [
        ("1", "📁 Drop your study files into the `data/raw/` folder"),
        ("2", "⚡ Run `python ingest.py` to build your knowledge base"),
        ("3", "🚀 Open the Study Helper page and start asking questions"),
        ("4", "📝 Use Quick Actions to simplify, translate, or dive deeper"),
    ]
    
    for num, text in steps:
        st.markdown(
            f'''
            <div style="display: flex; align-items: center; margin: 15px 0; padding: 15px; 
                        background: rgba(42, 42, 42, 0.4); border-radius: 10px;">
                <span class="step-number">{num}</span>
                <span class="step-text">{text}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    # Footer
    st.markdown(
        '''
        <div class="footer">
            <p>📚 Study Helper — Personalized study app built by Satish</p>
            <p style="font-size: 0.85em;">Powered by LangChain • OpenAI • ChromaDB • Streamlit</p>
        </div>
        ''',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
