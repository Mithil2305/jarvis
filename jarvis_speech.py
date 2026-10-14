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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = int(os.environ.get("JARVIS_SAMPLE_RATE", "16000"))
STT_MODEL_SIZE = os.environ.get("JARVIS_STT_MODEL", "large-v3").strip() or "large-v3"
STT_HOTWORDS = os.environ.get(
    "JARVIS_STT_HOTWORDS",
    "Jarvis, Antigravity, Chrome, YouTube, Notepad, Calculator, Spotify",
).strip()
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
    """Load and reuse the configured high-accuracy Faster-Whisper model."""
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


def _preprocess_audio(samples) -> np.ndarray:
    """Pass the natural microphone waveform to Whisper without distortion.

    Whisper's encoder expects speech with its original spectral balance. A
    near-unit pre-emphasis filter disproportionately boosts high frequencies
    and can push samples outside the valid PCM range. Peak-based normalization
    also made gain depend on occasional transients rather than speech level.
    """
    import numpy as np

    return np.asarray(samples, dtype=np.float32)


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

    # Preserve the microphone waveform before handing it to Whisper.
    samples = _preprocess_audio(samples)

    path = None
    try:
        # OpenJarvis's adapter writes bytes to a temporary file before calling
        # Faster-Whisper. Here we do the same with the captured float samples.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_16")
        return transcribe_file(path)
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError as exc:
                print(f"Warning: couldn't remove temporary audio file: {exc}", flush=True)


def transcribe_file(path: str) -> str:
    """Transcribe an audio file from disk (e.g. WAV, OGG, MP3) using Faster-Whisper."""
    if not path or not os.path.exists(path):
        return ""

    model = _get_stt_model()
    kwargs: dict[str, Any] = {
        # Spend more search effort on each final transcript. The beam
        # search helps resolve ambiguous words; best_of applies to the
        # non-zero-temperature fallback passes.
        "beam_size": 8,
        "best_of": 8,
        "patience": 2.0,
        "suppress_blank": True,
        "condition_on_previous_text": False,
        # Temperature ladder: greedy first, then higher-entropy fallbacks.
        "temperature": [0.0, 0.2, 0.4],
        # Avoid treating quiet but valid speech as silence or forcing
        # fallback decoding too early.
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        "word_timestamps": False,
        # Keep prompt context short; a long command list can bias decoding
        # toward words that were never spoken.
        "initial_prompt": "Jarvis.",
    }
    if STT_HOTWORDS:
        kwargs["hotwords"] = STT_HOTWORDS
    if STT_LANGUAGE:
        kwargs["language"] = STT_LANGUAGE

    # Keep VAD's speech boundaries padded so quiet word beginnings and
    # endings survive noise filtering.
    try:
        segments_iter, info = model.transcribe(
            path,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.35,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 450,
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
        # Reject only segments that look like silence and are also weakly
        # decoded. A low log probability alone is not enough to discard
        # quiet accented speech.
        if no_speech > 0.7 and avg_logprob < -1.2:
            print(f"Ignoring likely speech hallucination: {piece!r}", flush=True)
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


def generate_speech_audio(text: str) -> tuple[Any, int]:
    """Generate audio waveform using Kokoro. Returns (np.ndarray of float32, sample_rate)."""
    import numpy as np

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
    return audio_out, 24000


def synthesize_to_file(text: str, output_path: str) -> str:
    """Synthesize a Telegram-compatible OGG/Opus voice message."""
    global _KOKORO_DISABLED

    clean = str(text or "").strip()
    if not clean:
        raise SpeechBackendError("Cannot synthesize an empty response.")

    backend = TTS_BACKEND
    if backend not in {"kokoro", "sapi", "sapi5"}:
        print(f"Unknown JARVIS_TTS_BACKEND={backend!r}; using Kokoro with SAPI5 fallback.", flush=True)
        backend = "kokoro"

    with _TTS_LOCK:
        if backend in {"sapi", "sapi5"} or _KOKORO_DISABLED:
            audio_out, rate = _generate_sapi_audio(clean)
        else:
            try:
                audio_out, rate = generate_speech_audio(clean)
            except Exception as exc:
                _KOKORO_DISABLED = True
                print(
                    f"Kokoro file synthesis failed ({type(exc).__name__}: {exc}); "
                    "switching to Windows SAPI5 for this and later responses.",
                    flush=True,
                )
                audio_out, rate = _generate_sapi_audio(clean)
        audio_out = _resample_mono(audio_out, rate)
        _write_ogg_opus(audio_out, output_path)
    return output_path


def _speak_kokoro(text: str) -> None:
    import sounddevice as sd

    # Kokoro supports multi-sentence input itself; our outer chunks only cap
    # unusually long responses. Concatenate all generated audio before playback
    # so the listener does not hear an artificial stop between sentences.
    audio_out, rate = generate_speech_audio(text)
    sd.stop()
    sd.play(audio_out, samplerate=rate, blocking=True)



def _resample_mono(audio, source_rate: int, target_rate: int = 24000):
    """Convert a waveform to mono float32 at the rate used for Telegram audio."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    else:
        samples = samples.reshape(-1)
    if samples.size == 0 or source_rate <= 0:
        raise SpeechBackendError("The speech backend produced an empty audio clip.")
    if source_rate != target_rate:
        output_size = max(1, round(samples.size * target_rate / source_rate))
        old_positions = np.arange(samples.size, dtype=np.float64)
        new_positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
        samples = np.interp(new_positions, old_positions, samples).astype(np.float32)
    if not np.isfinite(samples).all():
        raise SpeechBackendError("The speech backend produced non-finite audio values.")
    return np.clip(samples, -1.0, 1.0)


def _generate_sapi_audio(text: str) -> tuple[Any, int]:
    """Render SAPI speech to a temporary WAV and return a mono waveform."""
    import soundfile as sf

    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        with _TTS_LOCK:
            engine = _get_sapi_engine()
            engine.stop()
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        return _resample_mono(samples, int(sample_rate)), 24000
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _write_ogg_opus(audio, output_path: str, sample_rate: int = 24000) -> None:
    """Write mono Opus frames in an OGG container for Telegram voice messages."""
    import av
    import numpy as np
    from fractions import Fraction

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise SpeechBackendError("Cannot encode an empty voice message.")

    frame_size = 960  # 40 ms at 24 kHz, a supported Opus frame duration.
    with av.open(output_path, mode="w", format="ogg") as container:
        stream = container.add_stream("libopus", rate=sample_rate)
        stream.codec_context.layout = "mono"
        stream.codec_context.bit_rate = 32000
        for start in range(0, samples.size, frame_size):
            frame_samples = np.zeros(frame_size, dtype=np.float32)
            chunk = samples[start : start + frame_size]
            frame_samples[: chunk.size] = chunk
            frame = av.AudioFrame.from_ndarray(
                frame_samples.reshape(1, -1), format="flt", layout="mono"
            )
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


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
