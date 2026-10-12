import os
import queue
import re
import tempfile
import time
import gc
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import pyttsx3
import torch
from faster_whisper import WhisperModel
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# ---------------- CONFIGURATION ----------------

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

MIC_DEVICE = 1
SAMPLE_RATE = 16000

FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

CALIBRATION_SECONDS = 1.0
SILENCE_SECONDS = 1.15
MAX_SPEECH_SECONDS = 15
MIN_SPEECH_SECONDS = 0.35
PREROLL_FRAMES = 15

PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"
# Default to instant, smooth Windows TTS. Set JARVIS_TTS_MODE=parler to use
# higher-quality Indic Parler-TTS (currently slower on this laptop).
TTS_MODE = os.environ.get("JARVIS_TTS_MODE", "sapi").strip().lower()
if TTS_MODE not in {"sapi", "parler"}:
    print(f"Unknown JARVIS_TTS_MODE={TTS_MODE!r}; using SAPI5.", flush=True)
    TTS_MODE = "sapi"
# CPU usage can be tuned from PowerShell with $env:JARVIS_TTS_THREADS = "4"
TORCH_THREADS = max(1, int(os.environ.get("JARVIS_TTS_THREADS", "4")))
try:
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SYSTEM_PROMPT = """
You are Jarvis, a helpful personal desktop voice assistant.
Answer in natural spoken English, normally in 1 to 3 concise sentences.
Use plain text without Markdown, tables, bullet lists, code fences, or LaTeX unless the user specifically asks for a written or technical format.
For technical topics, explain clearly and briefly, and speak symbols as words when practical.
Remember the recent conversation.
Never claim to perform actions you cannot actually execute.
"""


# ---------------- INITIALIZE MODELS ----------------

print("Jarvis v5 — tuned STT / adaptive microphone thresholds", flush=True)
print(f"TTS mode: {TTS_MODE} (set JARVIS_TTS_MODE=parler for Indic Parler)", flush=True)
print("Loading local Whisper model (base.en, CPU int8)...", flush=True)
print(
    f"PyTorch: {torch.__version__} | "
    f"CUDA available: {torch.cuda.is_available()} | "
    f"CUDA runtime: {torch.version.cuda}",
    flush=True,
)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

whisper = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
)

# Parler is loaded lazily the first time speak() is called. This avoids
# holding its memory while Jarvis starts and listens.
parler_model = None
parler_tokenizer = None
description_tokenizer = None
parler_sample_rate = None
parler_disabled = False
PARLER_DEVICE = "cpu"
PARLER_DTYPE = torch.float32

# Keep a single Windows SAPI5 engine as a fallback.
fallback_tts = pyttsx3.init("sapi5")
fallback_tts.setProperty("rate", 175)
fallback_tts.setProperty("volume", 1.0)

messages = []

VOICE_DESCRIPTION = (
    "A natural Indian English speaker speaks in a warm, calm, conversational tone "
    "at a normal pace with smooth phrasing, steady volume, and clear pronunciation. "
    "The recording is close, dry, and very clear, with no background noise."
)


# ---------------- INDIC PARLER TEXT TO SPEECH ----------------

def load_parler():
    """Load Indic Parler-TTS once, preferring CUDA FP16 when available."""
    global parler_model, parler_tokenizer
    global description_tokenizer, parler_sample_rate
    global parler_disabled, PARLER_DEVICE, PARLER_DTYPE

    if parler_model is not None:
        return
    if parler_disabled:
        raise RuntimeError("Indic Parler-TTS was disabled after an earlier error.")

    cuda_available = torch.cuda.is_available()
    target_device = "cuda:0" if cuda_available else "cpu"
    target_dtype = torch.float16 if cuda_available else torch.float32

    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(
                f"CUDA detected: {gpu_name} | "
                f"free VRAM: {free_mem / (1024 ** 3):.2f} GB / "
                f"{total_mem / (1024 ** 3):.2f} GB",
                flush=True,
            )
        except Exception:
            print("CUDA detected; loading Parler-TTS on the GPU.", flush=True)
    else:
        print(
            "CUDA is unavailable in this PyTorch environment; "
            "Parler-TTS will use the CPU.",
            flush=True,
        )

    print(
        f"Loading Indic Parler-TTS on {target_device} "
        f"({str(target_dtype).replace('torch.', '')}). First load may take a while...",
        flush=True,
    )

    try:
        try:
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                PARLER_MODEL_ID,
                torch_dtype=target_dtype,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to(target_device)
        except Exception as first_error:
            if target_device == "cpu":
                raise
            print(
                "GPU model loading failed; trying CPU fallback. "
                f"Reason: {type(first_error).__name__}: {first_error}",
                flush=True,
            )
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            target_device = "cpu"
            target_dtype = torch.float32
            model = ParlerTTSForConditionalGeneration.from_pretrained(
                PARLER_MODEL_ID,
                torch_dtype=target_dtype,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).to(target_device)

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_ID)
        desc_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path
        )

        parler_model = model
        parler_tokenizer = tokenizer
        description_tokenizer = desc_tokenizer
        parler_sample_rate = model.config.sampling_rate
        PARLER_DEVICE = target_device
        PARLER_DTYPE = target_dtype

        print(
            f"Indic Parler-TTS ready on {PARLER_DEVICE} "
            f"(sample rate: {parler_sample_rate} Hz).",
            flush=True,
        )
    except Exception:
        parler_disabled = True
        raise

def _make_parler_inputs(text):
    """Tokenize the voice description and the text with their respective tokenizers."""
    description_inputs = description_tokenizer(
        VOICE_DESCRIPTION,
        return_tensors="pt",
    ).to(PARLER_DEVICE)
    text_inputs = parler_tokenizer(
        text,
        return_tensors="pt",
    ).to(PARLER_DEVICE)
    return description_inputs, text_inputs


def generate_parler_audio(text):
    """Non-streaming fallback for installations without the streaming API."""
    load_parler()
    description_inputs, text_inputs = _make_parler_inputs(text)

    with torch.inference_mode():
        output = parler_model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=text_inputs.input_ids,
            prompt_attention_mask=text_inputs.attention_mask,
        )

    audio = output.detach().cpu().numpy().squeeze()
    if audio.size == 0:
        raise RuntimeError("Indic Parler-TTS generated empty audio.")
    if not np.isfinite(audio).all():
        raise RuntimeError("Indic Parler-TTS generated invalid audio values.")
    return audio


def speak_with_fallback(text):
    """Speak through the already-initialized Windows SAPI5 engine."""
    fallback_tts.say(text)
    fallback_tts.runAndWait()


def speak(text):
    """
    Speak immediately with SAPI5 by default, or use complete Parler generation
    when JARVIS_TTS_MODE=parler. Complete-clip Parler playback avoids chunk gaps
    but may have high latency on this laptop.
    """
    global parler_disabled

    text = str(text).strip()
    if not text:
        print("TTS: No text received.", flush=True)
        return

    print(f"JARVIS SPEAKING: {text}", flush=True)

    if TTS_MODE == "sapi" or parler_disabled:
        try:
            speak_with_fallback(text)
        except Exception as exc:
            print(f"SAPI5 TTS ERROR: {type(exc).__name__}: {exc}", flush=True)
        return

    try:
        print("Generating complete Parler speech (continuous playback)...", flush=True)
        generation_start = time.perf_counter()
        audio = generate_parler_audio(text)
        generation_seconds = time.perf_counter() - generation_start

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("Parler-TTS returned empty or invalid audio.")

        audio_duration = len(audio) / float(parler_sample_rate)
        print(
            f"Speech generated in {generation_seconds:.1f}s "
            f"({audio_duration:.1f}s of audio). Playing continuously...",
            flush=True,
        )
        sd.play(audio, samplerate=parler_sample_rate, blocking=True)
        print("Speech playback completed.", flush=True)

    except Exception as exc:
        print(
            f"Indic Parler-TTS error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        try:
            sd.stop()
        except Exception:
            pass
        try:
            if PARLER_DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Use SAPI5 for future utterances instead of repeatedly retrying a
        # failed Parler inference.
        parler_disabled = True
        try:
            speak_with_fallback(text)
        except Exception as fallback_exc:
            print(
                f"SAPI5 fallback error: {type(fallback_exc).__name__}: {fallback_exc}",
                flush=True,
            )


# ---------------- AUDIO UTILITIES ----------------

def rms(frame):
    """Calculate the volume of a microphone audio frame."""
    return float(np.sqrt(np.mean(np.square(frame))))


def record_until_silence():
    """
    Adaptive energy-based recording with separate start/stop thresholds.
    The higher start threshold avoids noise triggers; the lower stop threshold
    preserves quieter syllables once speech has begun.
    """
    audio_queue = queue.Queue(maxsize=250)
    pre_roll = deque(maxlen=PREROLL_FRAMES)

    def audio_callback(indata, frames, time_info, status):
        if status and ("overflow" in str(status).lower() or "underflow" in str(status).lower()):
            print(f"Audio status: {status}", flush=True)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    print("\nCalibrating microphone noise level...", flush=True)

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        latency="high",
        callback=audio_callback,
    ):
        calibration_levels = []
        for _ in range(int(CALIBRATION_SECONDS * 1000 / FRAME_MS)):
            frame = audio_queue.get(timeout=3)
            calibration_levels.append(rms(frame))

        # Median calibration is less affected by a single clap/click than
        # the previous upper-quartile estimate.
        noise_level = float(np.median(calibration_levels))

        # Two thresholds (hysteresis): harder to start accidentally, easier
        # to keep recording low-volume syllables after speech starts.
        start_threshold = min(max(noise_level * 2.7, 0.0055), 0.014)
        stop_threshold = max(noise_level * 1.45, 0.0035)
        stop_threshold = min(stop_threshold, start_threshold * 0.75)

        # Throw away stale calibration frames so they don't trigger recording.
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        print(
            f"Noise RMS: {noise_level:.4f} | "
            f"Start threshold: {start_threshold:.4f} | "
            f"Stop threshold: {stop_threshold:.4f}",
            flush=True,
        )
        print("Listening... Speak clearly at normal volume.", flush=True)

        speech_started = False
        loud_run = 0
        silent_frames = 0
        speech_frames = []

        start_confirm_frames = max(3, int(120 / FRAME_MS))
        silence_limit = int(SILENCE_SECONDS * 1000 / FRAME_MS)
        max_frames = int(MAX_SPEECH_SECONDS * 1000 / FRAME_MS)

        while True:
            frame = audio_queue.get(timeout=15)
            level = rms(frame)
            pre_roll.append(frame)

            if not speech_started:
                if level >= start_threshold:
                    loud_run += 1
                else:
                    loud_run = 0

                if loud_run >= start_confirm_frames:
                    speech_started = True
                    speech_frames = list(pre_roll)
                    print("Speech detected.", flush=True)
                continue

            speech_frames.append(frame)

            if level < stop_threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= silence_limit:
                break

            if len(speech_frames) >= max_frames:
                print("Maximum recording duration reached.", flush=True)
                break

        recorded = np.concatenate(speech_frames).astype(np.float32, copy=False)
        duration = len(recorded) / SAMPLE_RATE
        frame_levels = [rms(frame) for frame in speech_frames]
        active_frames = sum(level >= stop_threshold for level in frame_levels)
        active_duration = active_frames * FRAME_MS / 1000.0
        rms_level = float(np.sqrt(np.mean(recorded ** 2)))
        peak_level = float(np.max(np.abs(recorded)))

        print(
            f"Captured {duration:.1f}s | activity: {active_duration:.1f}s | "
            f"RMS: {rms_level:.4f} | peak: {peak_level:.4f}",
            flush=True,
        )

        # Avoid sending a near-silent clip to Whisper, but don't demand that
        # every spoken frame exceed the start threshold.
        if duration < MIN_SPEECH_SECONDS or active_duration < 0.22:
            print("Ignoring very short/noisy audio; please try again.", flush=True)
            return None
        if rms_level < 0.0018 or peak_level < 0.008:
            print("Ignoring a near-silent recording.", flush=True)
            return None

        return recorded


# ---------------- SPEECH TO TEXT ----------------

def transcribe(audio):
    """Transcribe English using the previous baseline model and a concise vocabulary hint."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return ""

    rms_level = float(np.sqrt(np.mean(audio ** 2)))
    peak_level = float(np.max(np.abs(audio)))
    if rms_level < 0.0018 or peak_level < 0.008:
        print(
            f"Audio too quiet (RMS={rms_level:.4f}, peak={peak_level:.4f}); ignoring.",
            flush=True,
        )
        return ""

    segments_generator, info = whisper.transcribe(
        audio,
        language="en",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
            "min_speech_duration_ms": 150,
        },
        condition_on_previous_text=False,
        initial_prompt=(
            "Jarvis, recursion, programming, Python, computer science, "
            "artificial intelligence, equations, and spoken desktop-assistant commands."
        ),
        temperature=0.0,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    segments = list(segments_generator)
    if not segments:
        print("VAD found no clear speech; ignoring this recording.", flush=True)
        return ""

    accepted = []
    for segment in segments:
        piece = segment.text.strip()
        if not piece:
            continue

        no_speech = float(getattr(segment, "no_speech_prob", 0.0))
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0))

        # Reject only strong no-speech evidence; avoid the previous very strict
        # combined cutoff that could discard short, quiet speech.
        if no_speech > 0.80 and avg_logprob < -1.2:
            print(
                f"Ignoring likely hallucination: {piece!r} "
                f"(no_speech={no_speech:.2f}, logprob={avg_logprob:.2f})",
                flush=True,
            )
            continue
        accepted.append(piece)

    transcript = " ".join(accepted).strip()
    if not transcript:
        print("No reliable speech recognized; ignoring this recording.", flush=True)
    return transcript


# ---------------- GEMMA CHAT ----------------

def ask_gemma(text):
    messages.append({"role": "user", "content": text})

    payload = {
        "model": "gemma-4",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages[-10:],
        ],
        "temperature": 0.4,
        "max_tokens": 110,
        "stream": False,
    }

    response = requests.post(
        LLM_URL,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()

    answer = (
        response.json()["choices"][0]["message"]["content"] or ""
    ).strip()

    if not answer:
        messages.pop()
        raise RuntimeError("Gemma returned an empty response.")

    messages.append({"role": "assistant", "content": answer})
    return answer


# ---------------- MAIN LOOP ----------------

def main():
    speak("Jarvis is online. I am listening.")

    print(
        "\nSpeak whenever you are ready. "
        "Say 'stop jarvis' to exit.",
        flush=True,
    )

    while True:
        try:
            audio = record_until_silence()
            if audio is None:
                continue

            print("Transcribing...", flush=True)
            text = transcribe(audio)

            if not text:
                print("No speech recognized.", flush=True)
                continue

            print(f"\nYOU: {text}", flush=True)

            if text.lower().strip() in {
                "exit",
                "quit",
                "stop jarvis",
                "shutdown jarvis",
            }:
                speak("Goodbye.")
                break

            print("Thinking...", flush=True)
            answer = ask_gemma(text)
            speak(answer)

        except KeyboardInterrupt:
            print("\nJarvis stopped.", flush=True)
            break

        except queue.Empty:
            print(
                "No microphone audio received. Check the microphone device.",
                flush=True,
            )

        except requests.RequestException as exc:
            print(f"LLM connection error: {exc}", flush=True)
            time.sleep(1)

        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
