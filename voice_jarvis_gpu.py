import os
import queue
import re
import tempfile
import time
import gc
from collections import deque
from threading import Thread

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import pyttsx3
import torch
from faster_whisper import WhisperModel
from parler_tts import ParlerTTSForConditionalGeneration
try:
    from parler_tts import ParlerTTSStreamer
except ImportError:
    ParlerTTSStreamer = None
from transformers import AutoTokenizer


# ---------------- CONFIGURATION ----------------

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

MIC_DEVICE = 1
SAMPLE_RATE = 16000

FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

CALIBRATION_SECONDS = 1.2
SILENCE_SECONDS = 1.0
MAX_SPEECH_SECONDS = 15
MIN_SPEECH_SECONDS = 0.25
PREROLL_FRAMES = 10

PARLER_MODEL_ID = "ai4bharat/indic-parler-tts"
# CPU usage can be tuned from PowerShell with $env:JARVIS_TTS_THREADS = "4"
TORCH_THREADS = max(1, int(os.environ.get("JARVIS_TTS_THREADS", "4")))
try:
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SYSTEM_PROMPT = """
You are Jarvis, a helpful personal desktop assistant.
Speak naturally and keep answers concise.
Remember the recent conversation.
Never claim to perform actions you cannot actually execute.
"""


# ---------------- INITIALIZE MODELS ----------------

print("Loading local Whisper model...", flush=True)
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
    "A natural, conversational Indian English speaker speaks clearly "
    "at a slightly fast but clear pace with a warm, friendly tone. The audio is clear "
    "and close, with no background noise."
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
                attn_implementation="sdpa",
            ).to(target_device)
        except Exception as first_error:
            if target_device == "cpu":
                raise
            print(
                "GPU/SDPA loading failed; trying CPU fallback. "
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


def stream_parler_audio(text):
    """
    Generate audio in a worker thread and play chunks as soon as they are ready.
    Returns True if at least one non-empty audio chunk was played.
    """
    load_parler()
    if ParlerTTSStreamer is None:
        audio = generate_parler_audio(text)
        sd.play(audio, samplerate=parler_sample_rate, blocking=True)
        return True

    description_inputs, text_inputs = _make_parler_inputs(text)
    frame_rate = float(parler_model.audio_encoder.config.frame_rate)
    # About 0.4 seconds of speech per chunk. Smaller values lower startup
    # latency but can increase decode overhead on a CPU.
    play_steps = max(16, int(frame_rate * 0.4))
    streamer = ParlerTTSStreamer(
        parler_model,
        device=PARLER_DEVICE,
        play_steps=play_steps,
        timeout=90.0,
    )
    worker_errors = queue.Queue(maxsize=1)

    generation_kwargs = {
        "input_ids": description_inputs.input_ids,
        "attention_mask": description_inputs.attention_mask,
        "prompt_input_ids": text_inputs.input_ids,
        "prompt_attention_mask": text_inputs.attention_mask,
        "streamer": streamer,
        "do_sample": True,
        "temperature": 1.0,
        "min_new_tokens": 10,
    }

    def generation_worker():
        try:
            with torch.inference_mode():
                parler_model.generate(**generation_kwargs)
        except BaseException as exc:
            try:
                worker_errors.put_nowait(exc)
            except queue.Full:
                pass
            # Wake the consumer if generation fails before streamer.end().
            try:
                streamer.on_finalized_audio(
                    np.zeros(0, dtype=np.float32),
                    stream_end=True,
                )
            except Exception:
                pass

    worker = Thread(target=generation_worker, name="parler-tts-generation", daemon=True)
    worker.start()

    played_audio = False
    try:
        for audio_chunk in streamer:
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32).squeeze()
            if audio_chunk.size == 0:
                break
            if not np.isfinite(audio_chunk).all():
                raise RuntimeError("Indic Parler-TTS streamed invalid audio values.")

            # Blocking playback is deliberate: it plays chunks in sequence.
            # The generation thread continues working while this chunk plays.
            sd.play(audio_chunk, samplerate=parler_sample_rate, blocking=True)
            played_audio = True
    finally:
        try:
            sd.stop()
        except Exception:
            pass

    worker.join(timeout=2.0)
    if not worker_errors.empty():
        raise RuntimeError(f"Parler streaming generation failed: {worker_errors.get()}")
    if worker.is_alive():
        print("Warning: TTS generation thread is still finishing.", flush=True)
    if not played_audio:
        raise RuntimeError("Parler streaming produced no playable audio.")
    return True

def speak_with_fallback(text):
    """Speak using Windows SAPI5 if Parler is unavailable."""
    print("Using Windows SAPI5 fallback.", flush=True)
    fallback_tts.say(text)
    fallback_tts.runAndWait()


def speak(text):
    global parler_disabled

    text = str(text).strip()
    if not text:
        print("TTS: No text received.", flush=True)
        return

    print(f"JARVIS SPEAKING: {text}", flush=True)

    if parler_disabled:
        speak_with_fallback(text)
        return

    try:
        print("Generating and streaming speech...", flush=True)
        stream_parler_audio(text)
        print("Speech playback completed.", flush=True)
    except Exception as exc:
        print(
            f"Indic Parler-TTS streaming error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        try:
            sd.stop()
        except Exception:
            pass
        # Streaming may have partially played before an error; use the simple
        # fallback voice to avoid restarting model loading or hanging Jarvis.
        parler_disabled = True
        speak_with_fallback(text)


# ---------------- AUDIO UTILITIES ----------------

def rms(frame):
    """Calculate the volume of a microphone audio frame."""
    return float(np.sqrt(np.mean(np.square(frame))))


def record_until_silence():
    """
    Listen continuously until speech is detected.
    Stop after sufficient silence or the maximum duration.
    """
    audio_queue = queue.Queue(maxsize=200)
    pre_roll = deque(maxlen=PREROLL_FRAMES)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", flush=True)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Never block PortAudio's callback thread.
            pass

    print("\nCalibrating microphone noise level...", flush=True)

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=audio_callback,
    ):
        calibration_levels = []

        for _ in range(int(CALIBRATION_SECONDS * 1000 / FRAME_MS)):
            frame = audio_queue.get(timeout=3)
            calibration_levels.append(rms(frame))

        noise_level = float(np.median(calibration_levels))
        threshold = max(noise_level * 2.9, 0.005)

        print(
            f"Noise level: {noise_level:.4f} | "
            f"Speech threshold: {threshold:.4f}",
            flush=True,
        )
        print("Listening... Speak naturally.", flush=True)

        speech_started = False
        consecutive_loud_frames = 0
        silent_frames = 0
        speech_frames = []

        silence_limit = int(SILENCE_SECONDS * 1000 / FRAME_MS)
        max_frames = int(MAX_SPEECH_SECONDS * 1000 / FRAME_MS)

        while True:
            frame = audio_queue.get(timeout=10)
            level = rms(frame)
            pre_roll.append(frame)

            if not speech_started:
                if level > threshold:
                    consecutive_loud_frames += 1
                else:
                    consecutive_loud_frames = 0

                if consecutive_loud_frames >= 2:
                    speech_started = True
                    speech_frames = list(pre_roll)
                    print("Speech detected.", flush=True)
                continue

            speech_frames.append(frame)

            if level < threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= silence_limit:
                break

            if len(speech_frames) >= max_frames:
                print("Maximum recording duration reached.", flush=True)
                break

        duration = len(speech_frames) * FRAME_MS / 1000
        if duration < MIN_SPEECH_SECONDS:
            print("Recording too short. Try again.", flush=True)
            return None

        print(f"Captured {duration:.1f} seconds of audio.", flush=True)
        return np.concatenate(speech_frames)


# ---------------- SPEECH TO TEXT ----------------

def transcribe(audio):
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            path = file.name

        sf.write(path, audio, SAMPLE_RATE)

        segments, info = whisper.transcribe(
            path,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=(
                "Jarvis, programming, recursion, equations, Python, "
                "computer science."
            ),
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments
        ).strip()
        return transcript
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


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
        "max_tokens": 200,
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
