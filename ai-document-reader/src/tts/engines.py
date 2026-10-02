"""Text-to-speech engines. No TTS model is trained in this project.

| Engine        | Languages                  | Speed control        | Needs internet |
|---------------|----------------------------|----------------------|----------------|
| gtts          | English, French            | normal / slow        | yes            |
| pyttsx3       | voices installed on the OS | words per minute     | no             |
| mms           | Kinyarwanda (pretrained    | speaking_rate        | first download |
|               | facebook/mms-tts-kin)      |                      |                |
| gtts_swahili  | APPROXIMATION for          | normal / slow        | yes            |
|               | Kinyarwanda (Swahili voice)|                      |                |

Kinyarwanda TTS support is LIMITED: Google TTS has no Kinyarwanda voice and
standard operating systems ship none. The best option is Meta's pretrained MMS
model (CC-BY-NC 4.0), which requires `transformers` + `torch`. The Swahili
fallback is a related Bantu language; pronunciation will be imperfect, and the
app labels it clearly as an approximation.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GTTS_LANG = {"english": "en", "french": "fr"}
MMS_MODELS = {"kinyarwanda": "facebook/mms-tts-kin"}


class TTSUnavailableError(RuntimeError):
    pass


@dataclass
class SpeechResult:
    audio: bytes
    mime: str
    engine: str
    note: str = ""


def engines_for(language: str) -> list[str]:
    """Engines to offer for a language, best first."""
    if language in GTTS_LANG:
        return ["gtts", "pyttsx3"]
    if language == "kinyarwanda":
        return ["mms", "gtts_swahili"]
    return ["pyttsx3"]


ENGINE_LABELS = {
    "gtts": "Google TTS (gTTS, online)",
    "pyttsx3": "Offline system voice (pyttsx3)",
    "mms": "Meta MMS-TTS Kinyarwanda (pretrained, facebook/mms-tts-kin)",
    "gtts_swahili": "Approximation: Swahili voice (NOT a Kinyarwanda voice)",
}


def synthesize(text: str, language: str, engine: str, speed: float = 1.0) -> SpeechResult:
    """speed: 1.0 = normal. gTTS only supports normal or slow (speed < 1)."""
    text = text.strip()
    if not text:
        raise TTSUnavailableError("Nothing to read: the text is empty.")
    if engine == "gtts":
        if language not in GTTS_LANG:
            raise TTSUnavailableError(f"gTTS has no voice for {language}.")
        return _gtts(text, GTTS_LANG[language], speed, engine)
    if engine == "gtts_swahili":
        result = _gtts(text, "sw", speed, engine)
        result.note = ("Read with a Swahili voice as an approximation. This is not a "
                       "Kinyarwanda voice; pronunciation will be imperfect.")
        return result
    if engine == "pyttsx3":
        return _pyttsx3(text, language, speed)
    if engine == "mms":
        return _mms(text, language, speed)
    raise TTSUnavailableError(f"Unknown TTS engine '{engine}'.")


def _gtts(text: str, lang: str, speed: float, engine: str) -> SpeechResult:
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise TTSUnavailableError("gTTS is not installed (pip install gTTS).") from exc
    buffer = io.BytesIO()
    try:
        gTTS(text, lang=lang, slow=speed < 1.0).write_to_fp(buffer)
    except Exception as exc:
        raise TTSUnavailableError(f"gTTS failed (internet connection required): {exc}") from exc
    note = "gTTS supports only normal or slow speed." if speed > 1.0 else ""
    return SpeechResult(buffer.getvalue(), "audio/mp3", engine, note)


# pyttsx3 uses OS speech APIs (SAPI5 on Windows) that misbehave inside
# Streamlit's worker threads, so it runs in a short-lived subprocess.
_PYTTSX3_SCRIPT = r"""
import sys, pyttsx3
text_path, out_path, rate, lang = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
engine = pyttsx3.init()
for voice in engine.getProperty("voices"):
    langs = " ".join(str(l) for l in (voice.languages or [])) + " " + voice.name
    if lang and lang.lower() in langs.lower():
        engine.setProperty("voice", voice.id)
        print("VOICE:" + voice.name)
        break
engine.setProperty("rate", rate)
engine.save_to_file(open(text_path, encoding="utf-8").read(), out_path)
engine.runAndWait()
"""
_PYTTSX3_LANG_HINT = {"english": "en", "french": "fr", "kinyarwanda": "rw"}


def _pyttsx3(text: str, language: str, speed: float) -> SpeechResult:
    with tempfile.TemporaryDirectory() as tmp:
        text_path, out_path = Path(tmp) / "text.txt", Path(tmp) / "speech.wav"
        text_path.write_text(text, encoding="utf-8")
        rate = int(180 * speed)
        proc = subprocess.run(
            [sys.executable, "-c", _PYTTSX3_SCRIPT, str(text_path), str(out_path), str(rate),
             _PYTTSX3_LANG_HINT.get(language, "")],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise TTSUnavailableError(f"pyttsx3 failed: {proc.stderr.strip()[-300:]}")
        voice = next((l[6:] for l in proc.stdout.splitlines() if l.startswith("VOICE:")), None)
        note = (f"Voice: {voice}" if voice else
                f"No {language} voice is installed on this computer; the default system voice was used.")
        return SpeechResult(out_path.read_bytes(), "audio/wav", "pyttsx3", note)


@lru_cache(maxsize=2)
def _load_mms(model_name: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise TTSUnavailableError("Kinyarwanda TTS needs `transformers` and `torch`.") from exc
    try:
        return VitsModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise TTSUnavailableError(
            f"Could not load {model_name} (Hugging Face download required): {exc}"
        ) from exc


def _mms(text: str, language: str, speed: float) -> SpeechResult:
    if language not in MMS_MODELS:
        raise TTSUnavailableError(f"No MMS model configured for {language}.")
    import numpy as np
    import torch
    import wave

    model, tokenizer = _load_mms(MMS_MODELS[language])
    model.speaking_rate = speed
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].numpy()
    pcm = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(model.config.sampling_rate)
        wav.writeframes(pcm.tobytes())
    return SpeechResult(buffer.getvalue(), "audio/wav", "mms",
                        f"Pretrained model {MMS_MODELS[language]} (Meta MMS, CC-BY-NC 4.0).")
