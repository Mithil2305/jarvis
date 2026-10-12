"""Jarvis voice assistant: microphone -> local STT -> local Gemma -> local TTS."""
from __future__ import annotations

import os
import queue
import re
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd

from jarvis_speech import backend_status, speak, transcribe_audio
from jarvis_actions import handle_command, PendingAction


# ---------------- CONFIGURATION ----------------
LLM_URL = os.environ.get("JARVIS_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
SAMPLE_RATE = int(os.environ.get("JARVIS_SAMPLE_RATE", "16000"))
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
CALIBRATION_SECONDS = float(os.environ.get("JARVIS_CALIBRATION_SECONDS", "1.0"))
SILENCE_SECONDS = float(os.environ.get("JARVIS_SILENCE_SECONDS", "1.15"))
MAX_SPEECH_SECONDS = float(os.environ.get("JARVIS_MAX_SPEECH_SECONDS", "20"))
MIN_SPEECH_SECONDS = float(os.environ.get("JARVIS_MIN_SPEECH_SECONDS", "0.35"))
PREROLL_FRAMES = 15

MIC_DEVICE_RAW = os.environ.get("JARVIS_MIC_DEVICE", "").strip().lower()
if MIC_DEVICE_RAW in {"", "default", "auto", "none"}:
    MIC_DEVICE = None
else:
    try:
        MIC_DEVICE = int(MIC_DEVICE_RAW)
    except ValueError:
        MIC_DEVICE = MIC_DEVICE_RAW

SYSTEM_PROMPT = """
You are Jarvis, a helpful personal desktop voice assistant running on Windows.
You CAN execute real desktop actions: open/close apps, control volume, take screenshots,
open websites, control media playback, lock the screen, and more.
For commands you can execute, do so — your action layer handles them automatically.
For everything else, answer in natural spoken English, normally in one to three concise sentences.
Use plain text without Markdown, tables, bullet lists, code fences, or LaTeX unless
the user specifically asks for a written or technical format.
Remember the recent conversation.
""".strip()

messages: list[dict[str, str]] = []


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation helper
# ─────────────────────────────────────────────────────────────────────────────

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go",
    "do it", "proceed", "confirm", "confirmed", "correct", "right",
    "affirmative", "definitely", "absolutely", "please", "fine",
}


def _is_affirmative(text: str) -> bool:
    """Return True when the user's reply is a clear yes/confirmation."""
    # Strip all punctuation and check for any affirmative word.
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return bool(set(clean.split()) & _AFFIRMATIVE)


def rms(frame) -> float:
    """Return root-mean-square energy for one microphone frame."""
    return float(np.sqrt(np.mean(np.square(frame))))


def _record_from_device(device):
    """Record one utterance using adaptive, hysteresis-based energy detection."""
    audio_queue: queue.Queue = queue.Queue(maxsize=250)
    pre_roll = deque(maxlen=PREROLL_FRAMES)

    def audio_callback(indata, frames, time_info, status):
        if status and ("overflow" in str(status).lower() or "underflow" in str(status).lower()):
            print(f"Microphone status: {status}", flush=True)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Never block the real-time PortAudio callback.
            pass

    print("\nCalibrating microphone noise level...", flush=True)
    with sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        latency="high",
        callback=audio_callback,
    ):
        calibration_levels = []
        for _ in range(max(1, int(CALIBRATION_SECONDS * 1000 / FRAME_MS))):
            calibration_levels.append(rms(audio_queue.get(timeout=5)))

        noise_level = float(np.median(calibration_levels))
        # Distinct start/stop thresholds prevent brief noise from starting a
        # recording while preserving softer syllables after speech begins.
        start_threshold = min(max(noise_level * 2.7, 0.0055), 0.014)
        stop_threshold = max(noise_level * 1.45, 0.0035)
        stop_threshold = min(stop_threshold, start_threshold * 0.75)

        # Discard frames from the calibration interval before waiting for speech.
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        print(
            f"Noise RMS: {noise_level:.4f} | start: {start_threshold:.4f} | "
            f"stop: {stop_threshold:.4f}",
            flush=True,
        )
        print("Listening... Speak naturally.", flush=True)

        speech_started = False
        loud_run = 0
        silent_frames = 0
        speech_frames = []
        start_confirm_frames = max(3, int(120 / FRAME_MS))
        silence_limit = max(1, int(SILENCE_SECONDS * 1000 / FRAME_MS))
        max_frames = max(1, int(MAX_SPEECH_SECONDS * 1000 / FRAME_MS))

        while True:
            frame = audio_queue.get(timeout=15)
            level = rms(frame)
            pre_roll.append(frame)

            if not speech_started:
                loud_run = loud_run + 1 if level >= start_threshold else 0
                if loud_run >= start_confirm_frames:
                    speech_started = True
                    speech_frames = list(pre_roll)
                    print("Speech detected.", flush=True)
                continue

            speech_frames.append(frame)
            silent_frames = silent_frames + 1 if level < stop_threshold else 0

            if silent_frames >= silence_limit:
                break
            if len(speech_frames) >= max_frames:
                print("Maximum recording duration reached.", flush=True)
                break

        if not speech_frames:
            return None

        audio = np.concatenate(speech_frames).astype(np.float32, copy=False)
        duration = len(audio) / SAMPLE_RATE
        levels = [rms(frame) for frame in speech_frames]
        active_duration = sum(level >= stop_threshold for level in levels) * FRAME_MS / 1000.0
        rms_level = float(np.sqrt(np.mean(np.square(audio))))
        peak_level = float(np.max(np.abs(audio)))

        print(
            f"Captured {duration:.1f}s | active: {active_duration:.1f}s | "
            f"RMS: {rms_level:.4f} | peak: {peak_level:.4f}",
            flush=True,
        )
        if duration < MIN_SPEECH_SECONDS or active_duration < 0.22:
            print("Ignoring a short/noisy clip. Please try again.", flush=True)
            return None
        if rms_level < 0.0018 or peak_level < 0.008:
            print("Ignoring a near-silent clip.", flush=True)
            return None
        return audio


def record_until_silence():
    """Use the configured mic; retry the Windows default input if index fails."""
    try:
        return _record_from_device(MIC_DEVICE)
    except sd.PortAudioError as exc:
        if MIC_DEVICE is not None:
            print(
                f"Microphone {MIC_DEVICE!r} could not be opened ({exc}). "
                "Retrying the Windows default input device.",
                flush=True,
            )
            return _record_from_device(None)
        raise


def ask_gemma(text: str) -> str:
    """Send the conversation to the existing local llama.cpp endpoint."""
    recent = [*messages, {"role": "user", "content": text}][-10:]
    payload = {
        "model": os.environ.get("JARVIS_MODEL_NAME", "gemma-4"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *recent],
        "temperature": 0.4,
        "max_tokens": int(os.environ.get("JARVIS_MAX_TOKENS", "160")),
        "stream": False,
    }
    response = requests.post(LLM_URL, json=payload, timeout=180)
    response.raise_for_status()

    try:
        answer = (response.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The local LLM returned an unexpected response format.") from exc
    if not answer:
        raise RuntimeError("The local Gemma model returned an empty response.")

    # Commit the turn only after a valid answer is received, preventing a failed
    # HTTP request from leaving unmatched user messages in the history.
    messages.extend(({"role": "user", "content": text}, {"role": "assistant", "content": answer}))
    del messages[:-20]
    return answer


def main():
    print("Jarvis — local voice pipeline", flush=True)
    print(backend_status(), flush=True)
    print(f"LLM endpoint: {LLM_URL}", flush=True)
    if not LLM_URL.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        print("WARNING: the configured LLM endpoint is not loopback/local.", flush=True)

    speak("Jarvis is online. I am listening.")
    print("Speak when ready. Say 'stop Jarvis' to exit. Press Ctrl+C to quit.", flush=True)

    while True:
        try:
            audio = record_until_silence()
            if audio is None:
                continue

            print("Transcribing with local Faster-Whisper...", flush=True)
            text = transcribe_audio(audio, SAMPLE_RATE)
            if not text:
                continue
            print(f"\nYOU: {text}", flush=True)

            # Strip all trailing/leading punctuation so "stop Jarvis," and
            # "stop Jarvis?" are caught along with the plain form.
            normalized = re.sub(r"[^\w\s]", "", text).lower().strip()
            if normalized in {
                "exit", "quit", "stop jarvis", "shutdown jarvis", "goodbye jarvis",
            }:
                speak("Goodbye.")
                break

            # ── Command intercept ──────────────────────────────────────────
            # Try to execute as a desktop action first.  Only fall through to
            # the LLM when no action pattern matches.
            action_result = handle_command(text)

            if isinstance(action_result, PendingAction):
                # Destructive command — ask for verbal confirmation.
                speak(action_result.confirmation_prompt)
                print("[confirm] Waiting for yes/no...", flush=True)
                try:
                    confirm_audio = record_until_silence()
                    if confirm_audio is not None:
                        confirm_text = transcribe_audio(confirm_audio, SAMPLE_RATE)
                        print(f"[confirm] Heard: {confirm_text!r}", flush=True)
                        if _is_affirmative(confirm_text):
                            final = action_result.execute()
                            speak(final)
                        else:
                            speak("Okay, action cancelled.")
                    else:
                        speak("No response received. Action cancelled.")
                except Exception as confirm_exc:
                    print(f"[confirm] Error during confirmation: {confirm_exc}", flush=True)
                    speak("I could not hear your confirmation. Action cancelled.")
                continue

            if action_result is not None:
                speak(action_result)
                continue

            print("Thinking with local Gemma...", flush=True)
            answer = ask_gemma(text)
            speak(answer)

        except KeyboardInterrupt:
            print("\nJarvis stopped.", flush=True)
            break
        except queue.Empty:
            print("No microphone audio received. Check the input device/permissions.", flush=True)
        except requests.RequestException as exc:
            print(f"Local LLM connection error: {exc}", flush=True)
            print("Confirm llama-server is running at the configured endpoint.", flush=True)
            time.sleep(1)
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
