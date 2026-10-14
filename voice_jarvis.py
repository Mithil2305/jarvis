"""Jarvis voice assistant: microphone -> local STT -> local Gemma -> local TTS."""
from __future__ import annotations

import json
import os
import queue
import re
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd

import threading
from typing import Optional

from jarvis_speech import backend_status, speak, transcribe_audio
from jarvis_actions import handle_command, handle_structured_command, PendingAction
from jarvis_indicator import JarvisIndicator, get_indicator, set_global_indicator

_SHUTDOWN_EVENT = threading.Event()
_INDICATOR: Optional[JarvisIndicator] = None



# ---------------- CONFIGURATION ----------------
LLM_URL = os.environ.get("JARVIS_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
SAMPLE_RATE = int(os.environ.get("JARVIS_SAMPLE_RATE", "16000"))
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
CALIBRATION_SECONDS = float(os.environ.get("JARVIS_CALIBRATION_SECONDS", "0.5"))
SILENCE_SECONDS = float(os.environ.get("JARVIS_SILENCE_SECONDS", "0.55"))
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
Interpret the user's latest request using the conversation history. Return exactly one JSON object,
with no Markdown or text outside the JSON. Use one of these forms:
{"type":"action","intent":"...","target":null,"url":null,"value":null,"direction":null,"confidence":0.0}
{"type":"response","text":"..."}
{"type":"clarify","text":"..."}

Use an action only for a clear, direct request. Allowed intents are: open_app, open_website,
open_new_tab, close_app, close_chrome_tabs, close_current_tab, set_volume, adjust_volume,
set_brightness, adjust_brightness, mute, unmute, play_pause, stop_media, next_track,
previous_track, take_screenshot, lock_screen, sleep_pc, shutdown_pc, restart_pc, cancel_shutdown.
For open_app/open_website/close_app, provide target. For open_new_tab, provide a complete
http/https url or a known website name as target. For set_volume/set_brightness, provide a
numeric value from 0 to 100. For adjust_volume/adjust_brightness, direction must be increase
or decrease. Include a confidence number from 0 to 1 for every action; use clarify when the
intent or required target is ambiguous. Do not infer an action from quoted text or a question
that merely discusses an action, or from a request that negates/cancels the action. Never return
shell commands or claim an action was completed.

Examples:
"Could you close Antigravity?" -> close_app with target "Antigravity".
"Turn the volume down a little." -> adjust_volume with direction "decrease".
"What happens when I close all Chrome tabs?" -> {"type":"response","text":"Explain what closing the tabs does."}

For ordinary questions or conversation, use type=response with a natural, concise spoken-English
answer in text. Remember the recent conversation.
""".strip()

messages: list[dict[str, str]] = []


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation helper
# ─────────────────────────────────────────────────────────────────────────────

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go",
    "do it", "proceed", "confirm", "confirmed", "correct", "right",
    "affirmative", "definitely", "absolutely", "fine",
}

_DISCUSSION_OR_NEGATION = re.compile(
    r"^\s*(?:what|why|how|when|where|who|which)\b|"
    r"^\s*(?:explain|describe|define|tell\s+me\b)\b|"
    r"\b(?:how\s+to|what\s+does|what\s+if)\b|"
    r"\b(?:do\s+not|don['’]?t|doesn['’]?t|didn['’]?t|shouldn['’]?t|"
    r"wouldn['’]?t|won['’]?t|can['’]?t|cannot|never|without|avoid|not)\b",
    re.IGNORECASE,
)


def _is_affirmative(text: str) -> bool:
    """Return True when the user's reply is a clear yes/confirmation."""
    # Reject any explicit negation before looking for affirmative words, so
    # phrases such as "no, don't" cannot accidentally approve an action.
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    words = clean.split()
    if set(words) & {"no", "not", "never", "cancel", "dont", "don't", "nope"}:
        return False
    if clean in _AFFIRMATIVE:
        return True
    single_word_answers = {answer for answer in _AFFIRMATIVE if " " not in answer}
    return bool(set(words) & single_word_answers)


def rms(frame) -> float:
    """Return root-mean-square energy for one microphone frame."""
    return float(np.sqrt(np.mean(np.square(frame))))


_CALIBRATED_THRESHOLDS: dict[object, tuple[float, float, float]] = {}


def calibrate_device(device, seconds: float = CALIBRATION_SECONDS) -> tuple[float, float, float]:
    """Measure ambient room noise once to determine speech detection thresholds."""
    key = str(device)
    if key in _CALIBRATED_THRESHOLDS:
        return _CALIBRATED_THRESHOLDS[key]

    audio_queue: queue.Queue = queue.Queue(maxsize=250)

    def audio_callback(indata, frames, time_info, status):
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
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
        for _ in range(max(1, int(seconds * 1000 / FRAME_MS))):
            calibration_levels.append(rms(audio_queue.get(timeout=5)))

        noise_level = float(np.median(calibration_levels))
        start_threshold = min(max(noise_level * 2.0, 0.0035), 0.012)
        stop_threshold = max(noise_level * 1.25, 0.002)
        stop_threshold = min(stop_threshold, start_threshold * 0.8)

    _CALIBRATED_THRESHOLDS[key] = (noise_level, start_threshold, stop_threshold)
    print(
        f"Calibrated noise RMS: {noise_level:.4f} | start: {start_threshold:.4f} | "
        f"stop: {stop_threshold:.4f}",
        flush=True,
    )
    return noise_level, start_threshold, stop_threshold


def _record_from_device(device):
    """Record one utterance using cached calibration and adaptive energy detection."""
    if _SHUTDOWN_EVENT.is_set():
        return None

    key = str(device)
    if key not in _CALIBRATED_THRESHOLDS:
        noise_level, start_threshold, stop_threshold = calibrate_device(device)
    else:
        noise_level, start_threshold, stop_threshold = _CALIBRATED_THRESHOLDS[key]

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

    with sd.InputStream(
        device=device,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        latency="high",
        callback=audio_callback,
    ):
        # Discard any residual frames from previous playback or stream startup
        while True:
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        print("Listening... Speak naturally.", flush=True)

        speech_started = False
        loud_run = 0
        silent_frames = 0
        speech_frames = []
        start_confirm_frames = max(3, int(120 / FRAME_MS))
        silence_limit = max(1, int(SILENCE_SECONDS * 1000 / FRAME_MS))
        max_frames = max(1, int(MAX_SPEECH_SECONDS * 1000 / FRAME_MS))

        while not _SHUTDOWN_EVENT.is_set():
            try:
                frame = audio_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            level = rms(frame)
            pre_roll.append(frame)

            if not speech_started:
                # Smoothly track ambient noise floor during silence without blocking
                if level < start_threshold:
                    noise_level = 0.98 * noise_level + 0.02 * level
                    start_threshold = min(max(noise_level * 2.0, 0.0035), 0.012)
                    stop_threshold = max(noise_level * 1.25, 0.002)
                    stop_threshold = min(stop_threshold, start_threshold * 0.8)
                    _CALIBRATED_THRESHOLDS[key] = (noise_level, start_threshold, stop_threshold)

                loud_run = loud_run + 1 if level >= start_threshold else 0
                if loud_run >= start_confirm_frames:
                    speech_started = True
                    speech_frames = list(pre_roll)
                    print("Speech detected.", flush=True)
                    if _INDICATOR:
                        _INDICATOR.set_state("hearing", "Speech detected...")
                continue

            speech_frames.append(frame)
            silent_frames = silent_frames + 1 if level < stop_threshold else 0

            if silent_frames >= silence_limit:
                break
            if len(speech_frames) >= max_frames:
                print("Maximum recording duration reached.", flush=True)
                break

        if not speech_frames or _SHUTDOWN_EVENT.is_set():
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


_ALLOWED_LLM_INTENTS = {
    "open_app", "open_website", "open_new_tab", "close_app",
    "close_chrome_tabs", "close_current_tab", "set_volume", "adjust_volume",
    "set_brightness", "adjust_brightness", "mute", "unmute", "play_pause",
    "stop_media", "next_track", "previous_track", "take_screenshot",
    "lock_screen", "sleep_pc", "shutdown_pc", "restart_pc", "cancel_shutdown",
}


def _parse_gemma_route(content: str) -> dict[str, object]:
    """Parse and constrain the local model's JSON route before it reaches actions."""
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return {
            "type": "clarify",
            "text": "I had trouble understanding that. Could you say it another way?",
        }
    try:
        result = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {
            "type": "clarify",
            "text": "I had trouble understanding that. Could you say it another way?",
        }
    if not isinstance(result, dict):
        return {"type": "clarify", "text": "Could you rephrase that request?"}

    route_type = result.get("type")
    if route_type == "action":
        intent = result.get("intent")
        confidence = result.get("confidence")
        if (
            not isinstance(intent, str)
            or intent not in _ALLOWED_LLM_INTENTS
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            return {
                "type": "clarify",
                "text": "I couldn't safely interpret that command. Could you rephrase it?",
            }
        return result

    if route_type in {"response", "clarify"}:
        text = result.get("text")
        if isinstance(text, str) and text.strip():
            return {"type": route_type, "text": text.strip()}
    return {"type": "clarify", "text": "Could you rephrase that request?"}


def ask_gemma(text: str) -> dict[str, object]:
    """Ask the local LLM for either a safe action intent or a spoken response."""
    recent = [*messages, {"role": "user", "content": text}][-10:]
    payload = {
        "model": os.environ.get("JARVIS_MODEL_NAME", "gemma-4"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *recent],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("JARVIS_MAX_TOKENS", "160")),
        "stream": False,
    }
    response = requests.post(LLM_URL, json=payload, timeout=180)
    response.raise_for_status()

    try:
        content = (response.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The local LLM returned an unexpected response format.") from exc
    if not content:
        raise RuntimeError("The local Gemma model returned an empty response.")
    return _parse_gemma_route(content)


def _remember_turn(user_text: str, assistant_text: str) -> None:
    """Keep a compact history so follow-up requests can resolve references."""
    messages.extend((
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ))
    del messages[:-20]


def _complete_action(action_result: str | PendingAction) -> str:
    """Resolve a pending action only after a clear spoken confirmation."""
    if not isinstance(action_result, PendingAction):
        return action_result

    speak(action_result.confirmation_prompt)
    print("[confirm] Waiting for yes/no...", flush=True)
    try:
        confirm_audio = record_until_silence()
        if confirm_audio is None:
            return "No response received. Action cancelled."
        confirm_text = transcribe_audio(confirm_audio, SAMPLE_RATE)
        print(f"[confirm] Heard: {confirm_text!r}", flush=True)
        if _is_affirmative(confirm_text):
            return action_result.execute()
        return "Okay, action cancelled."
    except Exception as exc:
        print(f"[confirm] Error during confirmation: {exc}", flush=True)
        return "I could not hear your confirmation. Action cancelled."


def _voice_assistant_loop(indicator: Optional[JarvisIndicator] = None) -> None:
    print("Jarvis — local voice pipeline (real-time mode)", flush=True)
    print(backend_status(), flush=True)
    print(f"LLM endpoint: {LLM_URL}", flush=True)
    if not LLM_URL.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        print("WARNING: the configured LLM endpoint is not loopback/local.", flush=True)

    if indicator:
        indicator.set_state("starting", "Warming up models...")

    print("Warming up local speech models...", flush=True)
    from jarvis_speech import _get_stt_model
    _get_stt_model()
    try:
        calibrate_device(MIC_DEVICE)
    except Exception as exc:
        print(f"Initial mic calibration note: {exc}", flush=True)

    speak("Jarvis is online. I am listening.")
    print("Speak when ready. Say 'stop Jarvis' to exit. Press Ctrl+C to quit.", flush=True)

    while not _SHUTDOWN_EVENT.is_set():
        try:
            if indicator:
                indicator.set_state("listening", "Listening... Speak naturally")

            audio = record_until_silence()
            if _SHUTDOWN_EVENT.is_set():
                break
            if audio is None:
                continue

            if indicator:
                indicator.set_state("transcribing", "Transcribing speech...")

            print("Transcribing with local Faster-Whisper...", flush=True)
            text = transcribe_audio(audio, SAMPLE_RATE)
            if not text:
                continue
            print(f"\nYOU: {text}", flush=True)

            normalized = re.sub(r"[^\w\s]", "", text).lower().strip()
            if normalized in {
                "exit", "quit", "stop jarvis", "shutdown jarvis", "goodbye jarvis",
            }:
                if indicator:
                    indicator.set_state("stopping", "Shutting down...")
                speak("Goodbye.")
                _SHUTDOWN_EVENT.set()
                if indicator:
                    indicator.request_shutdown()
                break

            action_result = (
                None
                if _DISCUSSION_OR_NEGATION.search(text)
                else handle_command(text)
            )
            if action_result is not None:
                if indicator:
                    indicator.set_state("executing", "Executing command...")
                answer = _complete_action(action_result)
                if indicator:
                    indicator.set_state("speaking", "Speaking response...")
                speak(answer)
                _remember_turn(text, answer)
                continue

            if indicator:
                indicator.set_state("thinking", "Interpreting...")
            print("Interpreting with local Gemma...", flush=True)
            route = ask_gemma(text)
            route_type = route.get("type")
            if route_type == "action":
                intent_name = str(route.get("intent", "action"))
                if indicator:
                    indicator.set_state("executing", f"Executing {intent_name}...")
                print(
                    f"[router] Intent: {route.get('intent')} | "
                    f"confidence: {route.get('confidence')}",
                    flush=True,
                )
                action_result = handle_structured_command(route)
                if action_result is None:
                    answer = "I couldn't safely map that request to an action. Could you rephrase it?"
                else:
                    answer = _complete_action(action_result)
            else:
                answer = str(route.get("text", "Could you rephrase that request?"))
            if indicator:
                indicator.set_state("speaking", "Speaking response...")
            speak(answer)
            _remember_turn(text, answer)

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

    if indicator:
        indicator.close()


def main():
    global _INDICATOR
    if os.environ.get("JARVIS_NO_GUI", "false").lower() in {"1", "true", "yes"}:
        _voice_assistant_loop(None)
        return

    def on_gui_shutdown():
        _SHUTDOWN_EVENT.set()

    _INDICATOR = JarvisIndicator(on_shutdown=on_gui_shutdown)
    set_global_indicator(_INDICATOR)

    worker = threading.Thread(
        target=_voice_assistant_loop,
        args=(_INDICATOR,),
        daemon=True,
    )
    worker.start()

    try:
        _INDICATOR.run_mainloop()
    except KeyboardInterrupt:
        _SHUTDOWN_EVENT.set()
        _INDICATOR.request_shutdown()


if __name__ == "__main__":
    main()
