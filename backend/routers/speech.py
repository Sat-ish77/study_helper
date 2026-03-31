"""
backend/routers/speech.py
Voice endpoints — NEW feature not in Streamlit.

POST /api/v1/speech/transcribe  → Whisper (audio → text)
POST /api/v1/speech/synthesize  → gTTS/ElevenLabs (text → audio)

How voice input works in React:
1. User clicks mic button
2. React records audio using MediaRecorder API
3. Sends audio blob to POST /speech/transcribe
4. Gets back text
5. Sends text to POST /rag/ask or /rag/ask/stream
6. Gets answer
7. Optionally sends answer to POST /speech/synthesize
8. Plays audio back

This replaces the Web Speech API approach and gives much 
better accuracy, especially for technical/academic content.
"""

import os
import io
import re
import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from dependencies import get_current_user

router = APIRouter()

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


# ── Voice Input (Whisper) ─────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Convert speech to text using OpenAI Whisper.
    
    Accepts: audio/webm, audio/mp4, audio/wav, audio/m4a
    Returns: {text: "transcribed text"}
    
    React sends: FormData with audio blob from MediaRecorder
    """
    from openai import OpenAI

    allowed_types = [
        "audio/webm", "audio/mp4", "audio/wav",
        "audio/mpeg", "audio/m4a", "audio/ogg"
    ]

    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio.content_type}"
        )

    audio_bytes = await audio.read()

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Whisper needs a file-like object with a name
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = audio.filename or "recording.webm"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        return {"text": transcript.strip()}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


# ── Voice Output (TTS) ────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    language: str = "English"
    engine: str = "gtts"  # gtts or elevenlabs


@router.post("/synthesize")
async def synthesize(
    body: SynthesizeRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Convert text to speech.
    Returns audio as base64-encoded MP3.
    
    Primary: gTTS (free, unlimited)
    Fallback: ElevenLabs (paid, better quality)
    
    Text is cleaned before TTS:
    - Removes citation tags [S1], [W2]
    - Removes markdown formatting
    - Removes emoji source lines
    """
    text = _clean_for_tts(body.text)
    lang_code = LANGUAGE_CODES.get(body.language, "en")

    # Try gTTS first
    audio_b64 = _gtts(text, lang_code)
    engine_used = "gtts"

    # Fallback to ElevenLabs if gTTS fails
    if not audio_b64:
        audio_b64 = _elevenlabs(text)
        engine_used = "elevenlabs"

    if not audio_b64:
        raise HTTPException(
            status_code=500,
            detail="Both gTTS and ElevenLabs failed"
        )

    return {
        "audio": audio_b64,  # base64 encoded MP3
        "engine": engine_used,
        "language": body.language,
    }


def _clean_for_tts(text: str) -> str:
    """Remove citations, markdown, emojis before speaking."""
    # Remove citation tags
    text = re.sub(r'\[S\d+\]|\[W\d+\]', '', text)
    # Remove markdown
    text = re.sub(r'\*\*|\*|#+', '', text)
    # Remove emoji lines (📄 Sources:, 🌐 Web:)
    text = re.sub(r'[📄🌐]\s.*?(?=\n|$)', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    # Limit length (gTTS and ElevenLabs have limits)
    return text[:3000]


def _gtts(text: str, lang: str = "en") -> str | None:
    """Generate speech with gTTS. Returns base64 MP3 or None."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        print(f"[speech] gTTS failed: {e}")
        return None


def _elevenlabs(text: str) -> str | None:
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
        print(f"[speech] ElevenLabs failed: {e}")
        return None