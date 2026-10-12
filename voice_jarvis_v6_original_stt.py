
import os
import queue
import tempfile
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import pyttsx3
from faster_whisper import WhisperModel


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

SYSTEM_PROMPT = """
You are Jarvis, a helpful personal desktop assistant.
Speak naturally and keep answers concise.
Remember the recent conversation.
Never claim to perform actions you cannot actually execute.
"""


# ---------------- INITIALIZE MODELS ----------------

print("Jarvis v6 — restored original STT and microphone logic", flush=True)
print("Loading local Whisper model (base.en, CPU int8)...", flush=True)

whisper = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
)


messages = []



# ---------------- TEXT TO SPEECH ----------------

tts_engine = pyttsx3.init("sapi5")
tts_engine.setProperty("rate", 175)
tts_engine.setProperty("volume", 1.0)

def speak(text):
    text = str(text).strip()

    if not text:
        print("TTS: No text received.", flush=True)
        return

    print(f"JARVIS SPEAKING: {text}", flush=True)

    try:
        tts_engine.stop()
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS ERROR: {e}", flush=True)



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
            # Avoid blocking the audio callback.
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

        for _ in range(
            int(CALIBRATION_SECONDS * 1000 / FRAME_MS)
        ):
            frame = audio_queue.get(timeout=3)
            calibration_levels.append(rms(frame))

        noise_level = float(np.median(calibration_levels))

        # Adaptive threshold based on ambient noise.
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

        silence_limit = int(
            SILENCE_SECONDS * 1000 / FRAME_MS
        )
        max_frames = int(
            MAX_SPEECH_SECONDS * 1000 / FRAME_MS
        )

        while True:
            frame = audio_queue.get(timeout=10)
            level = rms(frame)

            # Preserve a small amount of audio before speech starts.
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

        print(
            f"Captured {duration:.1f} seconds of audio.",
            flush=True,
        )

        return np.concatenate(speech_frames)


# ---------------- SPEECH TO TEXT ----------------

def transcribe(audio):
    path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as file:
            path = file.name

        # Save recorded audio as a WAV file
        sf.write(path, audio, SAMPLE_RATE)

        # Improve transcription accuracy
        segments, info = whisper.transcribe(
            path,
            language="en",
            beam_size=5,
            best_of=5,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=(
                "Jarvis, programming, recursion, "
                "equations, Python, computer science."
            ),
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return transcript

    finally:
        if path and os.path.exists(path):
            os.remove(path)



# ---------------- GEMMA CHAT ----------------

def ask_gemma(text):
    messages.append({
        "role": "user",
        "content": text,
    })

    payload = {
        "model": "gemma-4",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
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
        response.json()["choices"][0]["message"]["content"]
        or ""
    ).strip()

    if not answer:
        messages.pop()
        raise RuntimeError("Gemma returned an empty response.")

    messages.append({
        "role": "assistant",
        "content": answer,
    })

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
                "No microphone audio received. "
                "Check the microphone device.",
                flush=True,
            )

        except requests.RequestException as exc:
            print(f"LLM connection error: {exc}", flush=True)
            time.sleep(1)

        except Exception as exc:
            print(
                f"Error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(0.5)


if __name__ == "__main__":
    main()
