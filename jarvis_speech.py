"""Local speech adapters for Jarvis.

The implementation follows the small, lazy-loaded backend pattern used by
OpenJarvis's Faster-Whisper and Kokoro adapters, without depending on the
OpenJarvis application or its server/configuration system.

STT: Faster-Whisper (local CTranslate2 inference)
TTS: Kokoro (local neural TTS), with Windows SAPI5 as a fallback
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
from typing import Any


SAMPLE_RATE = int(os.environ.get("JARVIS_SAMPLE_RATE", "16000"))
STT_MODEL_SIZE = os.environ.get("JARVIS_STT_MODEL", "small.en").strip() or "small.en"
STT_DEVICE = os.environ.get("JARVIS_STT_DEVICE", "cpu").strip().lower() or "cpu"
STT_COMPUTE_TYPE = os.environ.get("JARVIS_STT_COMPUTE_TYPE", "int8").strip().lower() or "int8"
STT_LANGUAGE = os.environ.get("JARVIS_STT_LANGUAGE", "en").strip() or None
TTS_BACKEND = os.environ.get("JARVIS_TTS_BACKEND", "kokoro").strip().lower() or "kokoro"
TTS_VOICE = os.environ.get("JARVIS_TTS_VOICE", "af_heart").strip() or "af_heart"
TTS_DEVICE = os.environ.get("JARVIS_TTS_DEVICE", "cpu").strip().lower() or "cpu"
TTS_SPEED = float(os.environ.get("JARVIS_TTS_SPEED", "1.0"))

_STT_MODEL: Any = None
_STT_LOCK = threading.RLock()
_TTS_LOCK = threading.RLock()
_TTS_ENGINE: Any = None
_KOKORO_MODEL: Any = None
_KOKORO_PIPELINES: dict[str, Any] = {}
_KOKORO_DISABLED = False
_SAPI_DISABLED = False


class SpeechBackendError(RuntimeError):
    """Raised when the requested local speech backend cannot be used."""


def _resolve_stt_compute_type(device: str, requested: str) -> str:
    """Choose a supported CTranslate2 compute type, as OpenJarvis does."""
    try:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types(device))
    except Exception:
        # The model constructor will report the authoritative error if the
        # installed CTranslate2 build cannot report supported compute types.
        return requested

    if requested in supported:
        return requested

    preferred = (
        ("int8", "float32", "int8_float32", "int16")
        if requested == "float16"
        else ("float32", "int8", "int8_float32", "int16")
    )
    fallback = next((item for item in preferred if item in supported), None)
    if fallback:
        print(
            f"STT compute type {requested!r} is not supported on {device!r}; "
            f"using {fallback!r}.",
            flush=True,
        )
        return fallback
    return requested


def _get_stt_model():
    """Load Faster-Whisper on first use; reuse it for every later request."""
    global _STT_MODEL
    with _STT_LOCK:
        if _STT_MODEL is not None:
            return _STT_MODEL

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechBackendError(
                "Faster-Whisper is missing. Run: "
                ".venv\\Scripts\\python.exe -m pip install -r requirements-voice.txt"
            ) from exc

        compute_type = _resolve_stt_compute_type(STT_DEVICE, STT_COMPUTE_TYPE)
        print(
            f"Loading local STT model {STT_MODEL_SIZE!r} "
            f"(device={STT_DEVICE}, compute_type={compute_type})...",
            flush=True,
        )
        try:
            _STT_MODEL = WhisperModel(
                STT_MODEL_SIZE,
                device=STT_DEVICE,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise SpeechBackendError(
                f"Could not load Faster-Whisper ({STT_MODEL_SIZE}, "
                f"{STT_DEVICE}/{compute_type}): {type(exc).__name__}: {exc}"
            ) from exc

        print("Faster-Whisper is ready.", flush=True)
        return _STT_MODEL


def _preprocess_audio(samples) -> "np.ndarray":
    """Normalize amplitude and apply pre-emphasis for clearer STT input.

    Two-step pipeline:
    1. Amplitude normalization — scales the peak to 90 % of full scale so that
       quiet microphone captures are as loud as Whisper expects.  Recordings
       whose peak is already fine (> 0.5) are left alone to avoid clipping.
    2. Pre-emphasis filter — y[n] = x[n] - 0.97 * x[n-1].  This boosts
       high-frequency consonants (s, t, k, p …) that Whisper uses heavily for
       word boundaries and that are disproportionately attenuated by low-end
       microphones.
    """
    import numpy as np

    peak = float(np.max(np.abs(samples)))
    if 0.005 < peak < 0.5:          # only normalize genuinely quiet recordings
        samples = (samples / peak * 0.90).astype(np.float32)
    # pre-emphasis
    out = samples.copy()
    out[1:] = samples[1:] - 0.97 * samples[:-1]
    return out.astype(np.float32)


def transcribe_audio(audio, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a float audio array using local Faster-Whisper.

    A temporary WAV is used instead of keeping an open NamedTemporaryFile
    handle while PyAV opens it; this avoids Windows file-sharing errors.
    """
    import numpy as np
    import soundfile as sf

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return ""

    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(np.abs(samples)))
    if rms < 0.0018 or peak < 0.008:
        print(
            f"Ignoring near-silent audio (RMS={rms:.4f}, peak={peak:.4f}).",
            flush=True,
        )
        return ""

    # Pre-process: normalize + pre-emphasis before handing to Whisper.
    samples = _preprocess_audio(samples)

    path = None
    try:
        # OpenJarvis's adapter writes bytes to a temporary file before calling
        # Faster-Whisper. Here we do the same with the captured float samples.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_16")

        model = _get_stt_model()
        kwargs: dict[str, Any] = {
            "beam_size": 5,
            "best_of": 5,
            # patience > 1.0 lets beam search explore more candidates before
            # committing, which helps with short or unclear utterances.
            "patience": 1.5,
            "suppress_blank": True,
            "condition_on_previous_text": False,
            # Temperature ladder: greedy first, then higher-entropy fallbacks.
            "temperature": [0.0, 0.2, 0.4],
            "no_speech_threshold": 0.45,
            "log_prob_threshold": -0.8,
            "compression_ratio_threshold": 2.4,
            "word_timestamps": False,
            "initial_prompt": (
                "Hey Jarvis, close all Chrome tabs. Set volume to fifty percent. "
                "Open Notepad. Open YouTube. Take a screenshot. Mute. Unmute. "
                "Increase volume. Decrease brightness. Lock the screen. "
                "Shutdown the computer. Restart the computer. Play music. Pause. "
                "Adjust my volume. What time is it? Open Chrome."
            ),
        }
        if STT_LANGUAGE:
            kwargs["language"] = STT_LANGUAGE

        # VAD reduces silence-related hallucinations. If the optional VAD
        # runtime is unavailable, retry once without VAD rather than failing
        # the entire voice turn.
        try:
            segments_iter, info = model.transcribe(
                path,
                vad_filter=True,
                vad_parameters={
                    # 250 ms is short enough to catch commands like "Mute".
                    "min_silence_duration_ms": 250,
                    "speech_pad_ms": 200,
                    "min_speech_duration_ms": 100,
                },
                **kwargs,
            )
            segments = list(segments_iter)
        except (ImportError, RuntimeError) as exc:
            message = str(exc).lower()
            if not any(token in message for token in ("onnx", "vad", "silero")):
                raise
            print(f"Whisper VAD unavailable; retrying without VAD: {exc}", flush=True)
            segments_iter, info = model.transcribe(path, vad_filter=False, **kwargs)
            segments = list(segments_iter)

        accepted = []
        for segment in segments:
            piece = str(getattr(segment, "text", "")).strip()
            if not piece:
                continue
            no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
            # Tighter hallucination filter: drop if no-speech prob is high OR
            # log-probability is very low (garbled / noise-only segment).
            if no_speech > 0.65 and avg_logprob < -1.0:
                print(f"Ignoring likely speech hallucination: {piece!r}", flush=True)
                continue
            if avg_logprob < -1.5:
                print(f"Ignoring very low-confidence segment: {piece!r}", flush=True)
                continue
            accepted.append(piece)

        transcript = " ".join(accepted).strip()
        if not transcript:
            print("No reliable speech was recognized.", flush=True)
        else:
            detected_language = getattr(info, "language", None)
            if detected_language:
                print(f"Recognized language: {detected_language}", flush=True)
            transcript = _apply_stt_corrections(transcript)
        return transcript
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as exc:
                print(f"Warning: couldn't remove temporary audio file: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Post-processing: fuzzy correction for common Whisper mishearings
# ---------------------------------------------------------------------------
# Each entry is (compiled_regex, replacement_string).
# Rules are applied SEQUENTIALLY — later rules operate on already-corrected text.
# Multiple rules CAN fire on the same input string.
# Patterns use (?i) inline flag for case-insensitivity to avoid re-compiling.
_STT_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # ── Stage 1: phonetic-word substitutions ──────────────────────────────────
    # Fix phonetic word-level confusions BEFORE running phrase rules below.
    # These target specific mishearings observed from this user's microphone.
    (re.compile(r"(?i)\bover\s*run\b"),          "open"),        # "open" → "overrun"
    (re.compile(r"(?i)\bkrome\b"),               "chrome"),
    (re.compile(r"(?i)\bklose\b"),               "close"),
    (re.compile(r"(?i)\bset\s*point\b"),         "set"),         # "setpoint quality" → "set volume"
    (re.compile(r"(?i)\bquality\b(?=\s+to)"),   "volume"),      # "setpoint quality to 50%"
    # "taps" / "tabs" / "concepts" are all mishearings of "tabs"
    (re.compile(r"(?i)\b(\d+\s+)?taps?\b"),     "tabs"),
    (re.compile(r"(?i)\bconcepts?\b"),           "tabs"),
    (re.compile(r"(?i)\bkill\s+all\b"),          "close all"),

    # ── Stage 2: Volume / brightness controls ─────────────────────────────────
    # (.+) captures multi-word values like "fifty percent" or "50%".
    (re.compile(r"(?i)\b(set|adjust|change|put|bring)\s+(the\s+)?(volume|sound|audio)\s+(to|at)\s+(.+)"), r"set volume to \5"),
    (re.compile(r"(?i)\b(set|adjust|change)\s+(the\s+)?(brightness|screen)\s+(to|at)\s+(.+)"), r"set brightness to \5"),
    (re.compile(r"(?i)\b(increase|raise|turn\s+up)\s+(the\s+)?(volume|sound|audio)"), "increase volume"),
    (re.compile(r"(?i)\b(decrease|lower|turn\s+down|reduce|quieter)\s+(the\s+)?(volume|sound|audio)"), "decrease volume"),

    # ── Stage 3: Tab / browser management ────────────────────────────────────
    # Broad patterns first so "close all my chrome tabs" always hits this rule.
    (re.compile(r"(?i)\b(close|shut|kill)\s+(all\s+)?(my\s+)?(chrome|browser|google)\s+(tabs?|windows?)"), "close all Chrome tabs"),
    (re.compile(r"(?i)\b(open|launch|start)\s+(a?\s*new\s+)?(chrome|browser)\s+(tab|window)"), "open new Chrome tab"),
    (re.compile(r"(?i)\b(close|shut|kill)\s+(the\s+)?(current|active|this)\s+(tab|window)"), "close current tab"),

    # ── Stage 4: App open / close ─────────────────────────────────────────────
    (re.compile(r"(?i)\b(open|launch|start)\s+(up\s+)?(notepad|note\s*pad)"), "open Notepad"),
    (re.compile(r"(?i)\b(open|launch|start)\s+(up\s+)?(chrome|google\s+chrome)"), "open Chrome"),
    (re.compile(r"(?i)\b(open|launch|start)\s+(up\s+)?(youtube)"), "open YouTube"),
    (re.compile(r"(?i)\b(close|shut|exit|quit)\s+(the\s+)?(app|application|program|window)"), "close the application"),

    # ── Stage 5: Media playback ───────────────────────────────────────────────
    (re.compile(r"(?i)\b(play|start)\s+(some\s+)?(music|song|audio)"), "play music"),
    (re.compile(r"(?i)\b(pause|stop\s+playing|stop\s+music)\b"), "pause"),

    # ── Stage 6: Noise / hallucination suppression ────────────────────────────
    # Whisper sometimes emits repeated words / stock phrases for silence.
    (re.compile(r"(?i)(\bexplain\s+me\.?\s*){2,}"), "explain to me "),
    (re.compile(r"(?i)^\s*(thanks\s+for\s+watching|subscribe|like\s+and\s+subscribe)\.?\s*$"), ""),
    (re.compile(r"(?i)^\s*(you\.?|hmm\.?|uh\.?|um\.?)\s*$"), ""),
]


def _apply_stt_corrections(text: str) -> str:
    """Apply a list of regex corrections to fix common Whisper mishearings."""
    corrected = text
    for pattern, replacement in _STT_CORRECTIONS:
        new_text = pattern.sub(replacement, corrected)
        if new_text != corrected:
            print(f"STT correction: {corrected!r} -> {new_text!r}", flush=True)
            corrected = new_text
    return corrected


def _get_kokoro_pipeline(lang_code: str = "a"):
    """Lazily initialise one Kokoro model and cache a pipeline per language."""
    global _KOKORO_MODEL
    with _TTS_LOCK:
        cached = _KOKORO_PIPELINES.get(lang_code)
        if cached is not None:
            return cached

        try:
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise SpeechBackendError(
                "Kokoro is not installed. Run: "
                ".venv\\Scripts\\python.exe -m pip install -r requirements-voice.txt"
            ) from exc

        try:
            if _KOKORO_MODEL is None:
                print(
                    f"Loading local Kokoro TTS (device={TTS_DEVICE}). "
                    "The first run may download model weights.",
                    flush=True,
                )
                _KOKORO_MODEL = KModel().to(TTS_DEVICE).eval()
            pipeline = KPipeline(lang_code=lang_code, model=_KOKORO_MODEL)
        except Exception as exc:
            raise SpeechBackendError(
                f"Kokoro could not initialise: {type(exc).__name__}: {exc}. "
                "If phonemizer is the issue on Windows, install eSpeak NG and retry."
            ) from exc

        _KOKORO_PIPELINES[lang_code] = pipeline
        print("Kokoro is ready.", flush=True)
        return pipeline


def _text_chunks(text: str, limit: int = 350) -> list[str]:
    """Split very long responses at sentence/word boundaries for stable TTS."""
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return []
    sentences = re.split(r"(?<=[.!?।])\s+", clean)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        remaining = sentence.strip()
        while len(remaining) > limit:
            cut = remaining.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            piece, remaining = remaining[:cut].strip(), remaining[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            if piece:
                chunks.append(piece)
        if remaining:
            candidate = f"{current} {remaining}".strip()
            if current and len(candidate) > limit:
                chunks.append(current)
                current = remaining
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _as_float_audio(audio):
    """Convert Kokoro output (NumPy or a tensor) to a flat float array."""
    import numpy as np

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _get_sapi_engine():
    """Return the cached SAPI5 engine. Must be called while _TTS_LOCK is held."""
    global _TTS_ENGINE, _SAPI_DISABLED
    if _SAPI_DISABLED:
        raise SpeechBackendError("Windows SAPI5 fallback was disabled after an error.")
    if _TTS_ENGINE is None:
        try:
            import pyttsx3
        except ImportError as exc:
            _SAPI_DISABLED = True
            raise SpeechBackendError("pyttsx3 is not installed for Windows SAPI5 fallback.") from exc
        try:
            try:
                engine = pyttsx3.init("sapi5")
            except Exception:
                engine = pyttsx3.init()
            engine.setProperty("rate", int(os.environ.get("JARVIS_SAPI_RATE", "175")))
            engine.setProperty("volume", 1.0)
            _TTS_ENGINE = engine
        except Exception as exc:
            _SAPI_DISABLED = True
            raise SpeechBackendError(f"Could not initialise Windows SAPI5: {exc}") from exc
    return _TTS_ENGINE


def _speak_sapi(text: str) -> None:
    engine = _get_sapi_engine()
    engine.stop()
    engine.say(text)
    engine.runAndWait()


def _speak_kokoro(text: str) -> None:
    import numpy as np
    import sounddevice as sd

    # Kokoro supports multi-sentence input itself; our outer chunks only cap
    # unusually long responses. Concatenate all generated audio before playback
    # so the listener does not hear an artificial stop between sentences.
    voice_prefix = TTS_VOICE[:1]
    lang_code = voice_prefix if voice_prefix in set("abzjfiphe") else "a"
    pipeline = _get_kokoro_pipeline(lang_code)
    samples = []
    for chunk in _text_chunks(text):
        for _graphemes, _phonemes, audio in pipeline(chunk, voice=TTS_VOICE, speed=TTS_SPEED):
            values = _as_float_audio(audio)
            if values.size:
                samples.append(values)
    if not samples:
        raise SpeechBackendError("Kokoro produced no audio samples.")
    audio_out = np.concatenate(samples)
    if not np.isfinite(audio_out).all():
        raise SpeechBackendError("Kokoro produced non-finite audio values.")
    sd.stop()
    sd.play(audio_out, samplerate=24000, blocking=True)


def speak(text: str) -> bool:
    """Speak locally; if Kokoro fails, permanently fall back to SAPI5 this run."""
    global _KOKORO_DISABLED
    clean = str(text or "").strip()
    if not clean:
        print("TTS: No text received.", flush=True)
        return False

    print(f"JARVIS SPEAKING: {clean}", flush=True)
    backend = TTS_BACKEND
    if backend not in {"kokoro", "sapi", "sapi5"}:
        print(f"Unknown JARVIS_TTS_BACKEND={backend!r}; using Kokoro with SAPI5 fallback.", flush=True)
        backend = "kokoro"

    with _TTS_LOCK:
        if backend in {"sapi", "sapi5"} or _KOKORO_DISABLED:
            try:
                _speak_sapi(clean)
                return True
            except Exception as exc:
                print(f"SAPI5 TTS error: {type(exc).__name__}: {exc}", flush=True)
                return False

        try:
            _speak_kokoro(clean)
            return True
        except Exception as exc:
            _KOKORO_DISABLED = True
            print(
                f"Kokoro TTS failed ({type(exc).__name__}: {exc}); "
                "switching to Windows SAPI5 for this and later responses.",
                flush=True,
            )
            try:
                try:
                    import sounddevice as sd
                    sd.stop()
                except Exception:
                    pass
                _speak_sapi(clean)
                return True
            except Exception as fallback_exc:
                print(
                    f"SAPI5 fallback also failed: {type(fallback_exc).__name__}: {fallback_exc}",
                    flush=True,
                )
                return False


def backend_status() -> str:
    """Report configured backends without loading/downloading model weights."""
    return (
        f"STT: Faster-Whisper {STT_MODEL_SIZE} ({STT_DEVICE}/{STT_COMPUTE_TYPE}); "
        f"TTS: {TTS_BACKEND} (voice={TTS_VOICE}, device={TTS_DEVICE}, "
        "SAPI5 fallback enabled)"
    )
