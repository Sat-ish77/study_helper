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

from services.tts_service import LANGUAGE_CODES, clean_for_tts, gtts_synthesize, elevenlabs_synthesize


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
    text = clean_for_tts(body.text)
    lang_code = LANGUAGE_CODES.get(body.language, "en")

    # Try gTTS first
    audio_b64 = gtts_synthesize(text, lang_code)
    engine_used = "gtts"

    # Fallback to ElevenLabs if gTTS fails
    if not audio_b64:
        audio_b64 = elevenlabs_synthesize(text)
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