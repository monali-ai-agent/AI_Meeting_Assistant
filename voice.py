"""Voice helpers: speech-to-text (mic audio -> text) and text-to-speech (text -> mp3 / browser speech)."""
import io
import json
import re


def transcribe_groq(client, wav_bytes: bytes, lang_code="en"):
    """Speech-to-text with Groq Whisper (same API key as the LLM). Returns (text, error)."""
    try:
        r = client.audio.transcriptions.create(file=("audio.wav", wav_bytes), model="whisper-large-v3-turbo", language=lang_code)
        text = (getattr(r, "text", None) or (r if isinstance(r, str) else "")).strip()
        return (text, None) if text else ("", "I couldn't hear anything. Please try again.")
    except Exception as e:
        return "", f"Groq transcription failed: {e}"


def transcribe(wav_bytes: bytes, language="en-IN"):
    """Returns (text, error). Uses Google's free web speech API via SpeechRecognition (needs internet)."""
    try:
        import speech_recognition as sr
    except ImportError:
        return "", "Install SpeechRecognition: pip install SpeechRecognition"
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=language), None
    except sr.UnknownValueError:
        return "", "I couldn't understand the audio. Please try again."
    except sr.RequestError as e:
        return "", f"Speech service unavailable: {e}"
    except Exception as e:
        return "", f"Audio error: {e}"


def speakable(text: str, limit=650) -> str:
    """Strip markdown and trim to a voice-friendly length."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"[#*_`>|]+", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[-]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return (cut[:end + 1] if end > 200 else cut) + " ... See the chat for full details."


def tts_mp3(text: str, lang="en"):
    """Returns (mp3_bytes | None, error). Uses gTTS (needs internet)."""
    try:
        from gtts import gTTS
    except ImportError:
        return None, "Install gTTS: pip install gTTS"
    try:
        buf = io.BytesIO()
        gTTS(text=speakable(text), lang=lang).write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, f"Text-to-speech failed: {e}"


def browser_speech_html(text: str, lang="en-IN") -> str:
    """HTML/JS snippet that speaks via the browser's built-in SpeechSynthesis (works offline)."""
    return f"""<script>
    const u = new SpeechSynthesisUtterance({json.dumps(speakable(text))});
    u.lang = {json.dumps(lang)}; u.rate = 1.0;
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
    </script><span style="font:13px sans-serif;color:#6B7C88">🔊 speaking…</span>"""
