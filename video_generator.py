"""
video_generator.py — Study Helper v2
Pillow slide rendering + gTTS narration + MoviePy assembly → MP4
Supports MoviePy 1.x and 2.x automatically.
No ImageMagick dependency — Streamlit Cloud safe.
"""
from __future__ import annotations

import os
import textwrap
import tempfile
from pathlib import Path

SUPPORTED_LANGUAGES = {
    "English":  "en",
    "Nepali":   "ne",
    "Spanish":  "es",
    "Hindi":    "hi",
    "French":   "fr",
    "Japanese": "ja",
}

READING_LEVELS = ["Simple (5th grade)", "Standard", "Technical / College"]

WIDTH  = 1280
HEIGHT = 720

BG_COLOR      = (13,  15,  20)
TITLE_COLOR   = (240, 237, 232)
BULLET_COLOR  = (180, 182, 188)
ACCENT_COLOR  = (232, 164, 74)
COUNTER_COLOR = (75,  85,  99)


# ── Detect MoviePy version ────────────────────────────────────────────────────

def _get_moviepy_version() -> int:
    """Returns major version number: 1 or 2."""
    try:
        import moviepy
        version = getattr(moviepy, "__version__", "1.0.0")
        return int(version.split(".")[0])
    except Exception:
        return 1


# ── Slide rendering ───────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = False):
    from PIL import ImageFont
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Default font — always works
    return ImageFont.load_default()


def render_slide(title: str, bullets: list[str],
                 slide_num: int, total_slides: int):
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (WIDTH, HEIGHT), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Accent bar top
    draw.rectangle([(0, 0), (WIDTH, 6)], fill=ACCENT_COLOR)

    # Slide counter
    counter_font = _get_font(18)
    draw.text((WIDTH - 90, 20), f"{slide_num} / {total_slides}",
              font=counter_font, fill=COUNTER_COLOR)

    # Title
    title_font = _get_font(52, bold=True)
    title_y    = 80
    for line in textwrap.wrap(title, width=36)[:2]:
        draw.text((80, title_y), line, font=title_font, fill=TITLE_COLOR)
        title_y += 65

    # Divider
    div_y = title_y + 20
    draw.rectangle([(80, div_y), (220, div_y + 3)], fill=ACCENT_COLOR)

    # Bullets
    bullet_font = _get_font(30)
    bullet_y    = div_y + 45

    for bullet in bullets[:4]:
        draw.ellipse([(80, bullet_y + 10), (93, bullet_y + 23)], fill=ACCENT_COLOR)
        for i, line in enumerate(textwrap.wrap(bullet, width=60)[:2]):
            draw.text((110, bullet_y + (i * 34)), line,
                      font=bullet_font, fill=BULLET_COLOR)
        bullet_y += 34 * min(len(textwrap.wrap(bullet, 60)), 2) + 24

    # Progress bar bottom
    bar_y = HEIGHT - 8
    bar_w = int(WIDTH * (slide_num / total_slides))
    draw.rectangle([(0, bar_y), (WIDTH, HEIGHT)], fill=(25, 27, 34))
    draw.rectangle([(0, bar_y), (bar_w, HEIGHT)], fill=ACCENT_COLOR)

    # Watermark
    wm_font = _get_font(16)
    draw.text((WIDTH - 160, HEIGHT - 30), "Study Helper",
              font=wm_font, fill=(40, 42, 50))

    return img


# ── Audio generation ──────────────────────────────────────────────────────────

def generate_audio(text: str, lang_code: str, output_path: str) -> bool:
    # Try ElevenLabs first if key available
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
    if elevenlabs_key:
        try:
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=elevenlabs_key)
            audio  = client.text_to_speech.convert(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
                text=text,
                model_id="eleven_multilingual_v2",
            )
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"[ElevenLabs] failed, falling back to gTTS: {e}")

    # gTTS fallback
    try:
        from gtts import gTTS
        gTTS(text=text, lang=lang_code, slow=False).save(output_path)
        return True
    except Exception as e:
        print(f"[gTTS] error: {e}")
        return False


# ── Video assembly ────────────────────────────────────────────────────────────

def build_video(slides: list[dict], lang_code: str, output_path: str) -> bool:
    """
    Supports both MoviePy 1.x and 2.x.
    Returns True on success, False on failure.
    """
    moviepy_version = _get_moviepy_version()
    print(f"[video_generator] MoviePy version: {moviepy_version}")

    try:
        if moviepy_version >= 2:
            # MoviePy 2.x API
            from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
            def make_clip(img_path, audio_path, fallback_duration=4):
                audio = AudioFileClip(audio_path)
                clip  = ImageClip(img_path).with_duration(audio.duration).with_audio(audio)
                return clip
        else:
            # MoviePy 1.x API
            from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
            def make_clip(img_path, audio_path, fallback_duration=4):
                audio = AudioFileClip(audio_path)
                clip  = ImageClip(img_path).set_duration(audio.duration).set_audio(audio)
                return clip

    except ImportError as e:
        print(f"[video_generator] MoviePy import failed: {e}")
        return False

    clips = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, slide in enumerate(slides):
            # Render slide
            img      = render_slide(
                title=slide.get("title", ""),
                bullets=slide.get("bullets", []),
                slide_num=i + 1,
                total_slides=len(slides),
            )
            img_path   = os.path.join(tmpdir, f"slide_{i}.png")
            audio_path = os.path.join(tmpdir, f"audio_{i}.mp3")
            img.save(img_path)

            narrator = slide.get("narrator_text", slide.get("title", ""))
            ok = generate_audio(narrator, lang_code, audio_path)

            if ok and os.path.exists(audio_path):
                try:
                    clip = make_clip(img_path, audio_path)
                    clips.append(clip)
                except Exception as e:
                    print(f"[video_generator] clip creation failed slide {i}: {e}")
                    # Still add a silent clip so video doesn't lose slides
                    if moviepy_version >= 2:
                        clips.append(ImageClip(img_path).with_duration(4))
                    else:
                        clips.append(ImageClip(img_path).set_duration(4))
            else:
                if moviepy_version >= 2:
                    clips.append(ImageClip(img_path).with_duration(4))
                else:
                    clips.append(ImageClip(img_path).set_duration(4))

        if not clips:
            print("[video_generator] No clips generated")
            return False

        try:
            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(
                output_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                logger=None,
                temp_audiofile=os.path.join(tmpdir, "temp_audio.m4a"),
            )
            return True
        except Exception as e:
            print(f"[video_generator] write_videofile failed: {e}")
            return False


# ── Streamlit page ────────────────────────────────────────────────────────────

def render_video_generator_page(llm, user_id: str):
    import streamlit as st
    from agents.video_agent import VideoAgent
    from main import retrieve_docs, build_tagged_context, tavily_search, build_tagged_web_context

    st.markdown(
        '<h2 style="font-family:\'DM Serif Display\',serif; color:#f0ede8;">'
        '🎬 Video Generator</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="color:#6b7280; margin-bottom:1.5rem;">'
        'Type a topic and get a narrated explainer video generated from your notes.</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_area(
            "Topic or concept",
            placeholder="e.g. Python loops, TCP/IP, Cell division",
            height=100,
            key="video_topic"
        )
    with col2:
        language = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), key="video_lang")
        level    = st.selectbox("Reading level", READING_LEVELS, index=1, key="video_level")

    if not st.button("Generate Video", type="primary", use_container_width=True, key="btn_gen_video"):
        return
    if not topic.strip():
        st.warning("Please enter a topic.")
        return

    lang_code = SUPPORTED_LANGUAGES[language]

    with st.status("Generating your video...", expanded=True) as status:
        st.write("🔍 Searching your notes...")
        rr           = retrieve_docs(user_id, topic)
        file_context = ""
        source_label = "llm"

        if rr.docs:
            file_context, _ = build_tagged_context(rr.docs)
            source_label     = "rag"
        else:
            st.write("📄 Not in your notes — trying web...")
            web_results = tavily_search(topic)
            if web_results:
                file_context, _ = build_tagged_web_context(web_results)
                source_label     = "web"

        st.write("✍️ Writing script...")
        agent       = VideoAgent(llm)
        slides, src = agent.generate_slides(
            topic=topic,
            context=file_context,
            language=language,
            level=level,
        )

        if not slides:
            st.error("Could not generate slides. Try a different topic.")
            status.update(label="Failed", state="error")
            return

        source_text = {
            "rag": "📚 Generated from your notes",
            "web": "🌐 Generated from web sources",
            "llm": "🤖 Generated from AI general knowledge",
        }.get(source_label, "")
        st.markdown(
            f'<span style="font-size:0.8rem; color:#4ade80;">{source_text}</span>',
            unsafe_allow_html=True
        )

        st.write(f"🎬 Rendering {len(slides)} slides...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            output_path = tmp.name

        ok = build_video(slides, lang_code, output_path)

        if ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            os.unlink(output_path)
            status.update(label="Video ready!", state="complete")
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download MP4",
                data=video_bytes,
                file_name=f"{topic[:30].replace(' ', '_')}_explainer.mp4",
                mime="video/mp4",
            )
        else:
            st.warning("Video render failed — generating audio narration instead.")
            full_script = " ".join(s.get("narrator_text", "") for s in slides)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as atmp:
                audio_path = atmp.name
            ok_audio = generate_audio(full_script, lang_code, audio_path)
            if ok_audio:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(audio_path)
                status.update(label="Audio ready", state="complete")
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button("⬇️ Download MP3", data=audio_bytes,
                                   file_name="narration.mp3", mime="audio/mp3")
            else:
                status.update(label="Failed", state="error")
                st.error("Generation failed. Check terminal for details.")