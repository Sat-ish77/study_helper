"""
backend/services/tts_service.py
TTS helpers extracted from routers/speech.py.
"""
from __future__ import annotations

import io
import os
import re
import base64


LANGUAGE_CODES = {
    "English":    "en",
    "Nepali":     "ne",
    "Hindi":      "hi",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Chinese":    "zh",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Arabic":     "ar",
    "Portuguese": "pt",
}


def clean_for_tts(text: str) -> str:
    """Remove citations, markdown, emojis before speaking."""
    # Remove citation tags
    text = re.sub(r'\[S\d+\]|\[W\d+\]', '', text)
    # Remove markdown
    text = re.sub(r'\*\*|\*|#+', '', text)
    # Remove emoji lines
    text = re.sub(r'[📄🌐]\s.*?(?=\n|$)', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    return text[:3000]


def gtts_synthesize(text: str, lang: str = "en") -> str | None:
    """Generate speech with gTTS. Returns base64 MP3 or None."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"[tts_service] gTTS failed: {e}")
        return None


def elevenlabs_synthesize(text: str) -> str | None:
    """Generate speech with ElevenLabs. Returns base64 MP3 or None."""
    try:
        from elevenlabs.client import ElevenLabs
        key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

        if not key:
            return None

        client = ElevenLabs(api_key=key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        audio_bytes = b"".join(audio)
        return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        print(f"[tts_service] ElevenLabs failed: {e}")
        return None