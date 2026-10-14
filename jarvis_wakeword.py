"""Wait for a hotkey and wake phrase, then start the local Jarvis services."""
from __future__ import annotations

import ctypes
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
WAKE_MODEL_DIR = ROOT / "data" / "wakeword_models"
MODEL_START_BAT = ROOT / "start_model.bat"
PYTHON = Path(sys.executable)
SAMPLE_RATE = 16_000
BLOCK_SAMPLES = 1_280  # openWakeWord's recommended 80 ms frame
_MODEL_PROCESS: subprocess.Popen | None = None
_VOICE_PROCESS: subprocess.Popen | None = None
_TELEGRAM_PROCESS: subprocess.Popen | None = None
_SHUTDOWN_REQUEST = LOG_DIR / "jarvis_shutdown.request"


def _load_env_file() -> None:
    """Load simple .env values without replacing variables already set by Windows."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logging.warning("Could not read .env: %s", exc)


def _configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            LOG_DIR / "jarvis_wakeword.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    ]
    if sys.stdout and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _play_chime(tones: list[tuple[int, int]]) -> None:
    """Play a short audible tone sequence in the background for feedback."""
    def _run() -> None:
        try:
            import winsound
            for freq, dur in tones:
                winsound.Beep(freq, dur)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def _speak_async(text: str) -> None:
    """Announce brief status updates using Windows SAPI without blocking."""
    def _run() -> None:
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            try:
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Speak(text, 1)  # 1 = SVSFlagsAsync
            finally:
                pythoncom.CoUninitialize()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def _wake_key() -> tuple[str, int]:
    key = os.environ.get("JARVIS_WAKE_KEY", "F9").strip().upper()
    match = re.fullmatch(r"F(\d{1,2})", key)
    if match and 1 <= int(match.group(1)) <= 24:
        return key, 0x70 + int(match.group(1)) - 1
    extra_keys = {"PAUSE": 0x13, "SCROLL_LOCK": 0x91}
    if key in extra_keys:
        return key, extra_keys[key]
    raise ValueError("JARVIS_WAKE_KEY must be F1-F24, PAUSE, or SCROLL_LOCK")


def _wait_for_hotkey(vk_code: int) -> None:
    was_down = False
    while True:
        is_down = bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
        if is_down and not was_down:
            return
        was_down = is_down
        time.sleep(0.05)


def _download_model(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    logging.info("Downloading wake-word resource %s", destination.name)
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with partial_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        output.write(chunk)
        if partial_path.stat().st_size == 0:
            raise OSError("downloaded file was empty")
        partial_path.replace(destination)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def _load_wake_model():
    try:
        import openwakeword
        from openwakeword.model import Model
    except ImportError as exc:
        raise RuntimeError(
            "Wake-word dependencies are missing. Run "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements-voice.txt'."
        ) from exc

    feature_urls = {
        "melspectrogram.onnx": openwakeword.FEATURE_MODELS["melspectrogram"]["download_url"].replace(".tflite", ".onnx"),
        "embedding_model.onnx": openwakeword.FEATURE_MODELS["embedding"]["download_url"].replace(".tflite", ".onnx"),
    }
    for filename, url in feature_urls.items():
        _download_model(url, WAKE_MODEL_DIR / filename)

    wake_url = openwakeword.MODELS["hey_jarvis"]["download_url"].replace(".tflite", ".onnx")
    wake_model_path = WAKE_MODEL_DIR / "hey_jarvis_v0.1.onnx"
    _download_model(wake_url, wake_model_path)

    logging.info("Loading the Hey Jarvis wake-word model")
    return Model(
        wakeword_models=[str(wake_model_path)],
        inference_framework="onnx",
        melspec_model_path=str(WAKE_MODEL_DIR / "melspectrogram.onnx"),
        embedding_model_path=str(WAKE_MODEL_DIR / "embedding_model.onnx"),
    )


def _microphone_device():
    raw = os.environ.get("JARVIS_MIC_DEVICE", "").strip().lower()
    if not raw or raw in {"default", "auto", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def _listen_for_wake_word(wake_model, vk_code: int | None = None) -> bool:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is missing; reinstall requirements-voice.txt") from exc

    try:
        timeout_seconds = max(1.0, float(os.environ.get("JARVIS_WAKE_TIMEOUT_SECONDS", "18")))
        score_threshold = float(os.environ.get("JARVIS_WAKE_SCORE_THRESHOLD", "0.25"))
    except ValueError as exc:
        raise ValueError("Wake timeout and score threshold must be numeric") from exc
    if not 0.0 < score_threshold <= 1.0:
        raise ValueError("JARVIS_WAKE_SCORE_THRESHOLD must be greater than 0 and at most 1")

    # Reset wake model feature buffers from any prior listening session
    if hasattr(wake_model, "reset"):
        try:
            wake_model.reset()
        except Exception:
            pass

    device = _microphone_device()
    stream_args = {
        "samplerate": SAMPLE_RATE,
        "blocksize": BLOCK_SAMPLES,
        "channels": 1,
        "dtype": "int16",
    }
    stream = None
    try:
        stream = sd.InputStream(device=device, **stream_args)
        stream.start()
    except Exception as exc:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
        if device is None:
            raise
        logging.warning("Could not open microphone %r (%s); retrying Windows default input", device, exc)
        stream = sd.InputStream(device=None, **stream_args)
        stream.start()

    # Audible cue indicating microphone is open and listening
    _play_chime([(880, 80), (1320, 120)])

    logging.info("Microphone open for up to %.0f seconds; say 'Hey Jarvis'", timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    peak_score = 0.0
    peak_audio_rms = 0.0
    try:
        while time.monotonic() < deadline:
            # Check if user double-pressed the hotkey to wake immediately without voice
            if vk_code is not None and bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000):
                logging.info("Hotkey pressed again during listening window; manual wake triggered")
                _play_chime([(1320, 80), (1760, 140)])
                _speak_async("Starting Jarvis. Loading the model.")
                return True

            audio, overflowed = stream.read(BLOCK_SAMPLES)
            if overflowed:
                logging.warning("Microphone input overflowed while listening for wake word")
            audio_frame = np.asarray(audio, dtype=np.int16).reshape(-1)
            frame_rms = float(np.sqrt(np.mean(audio_frame.astype(np.float32) ** 2)))
            peak_audio_rms = max(peak_audio_rms, frame_rms)
            scores = wake_model.predict(audio_frame)
            score = float(scores.get("hey_jarvis_v0.1", 0.0))
            peak_score = max(peak_score, score)
            if score >= score_threshold:
                logging.info("Wake phrase detected (score %.3f)", score)
                _play_chime([(1320, 80), (1760, 140)])
                _speak_async("Starting Jarvis. Loading the model.")
                return True
    finally:
        try:
            stream.stop()
        finally:
            stream.close()

    # Soft low tone indicating listening window ended without detection
    _play_chime([(440, 120)])
    logging.info(
        "No wake phrase detected (peak score %.3f / threshold %.2f; peak frame RMS %.1f); waiting for %s again",
        peak_score,
        score_threshold,
        peak_audio_rms,
        os.environ.get("JARVIS_WAKE_KEY", "F9"),
    )
    return False


def _llm_health_url() -> str:
    llm_url = os.environ.get(
        "JARVIS_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions"
    )
    parsed = urlsplit(llm_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("JARVIS_LLM_URL must be a complete http(s) URL")
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def _model_is_ready(health_url: str) -> bool:
    try:
        response = requests.get(health_url, timeout=1.0)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _launch_model() -> bool:
    global _MODEL_PROCESS
    health_url = _llm_health_url()
    if _model_is_ready(health_url):
        logging.info("Local model server is already healthy")
        return True
    if not MODEL_START_BAT.is_file():
        logging.error("Model launcher is missing: %s", MODEL_START_BAT)
        return False

    if _MODEL_PROCESS is None or _MODEL_PROCESS.poll() is not None:
        environment = os.environ.copy()
        environment["JARVIS_OPEN_MODEL_BROWSER"] = "false"
        environment["JARVIS_NO_PAUSE"] = "1"
        model_log = (LOG_DIR / "llama_server.log").open("ab")
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", str(MODEL_START_BAT)]
        logging.info("Starting llama.cpp with the existing model configuration")
        try:
            _MODEL_PROCESS = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=model_log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            logging.exception("Could not start start_model.bat")
            return False
        finally:
            model_log.close()
    process = _MODEL_PROCESS

    try:
        wait_seconds = max(10.0, float(os.environ.get("JARVIS_MODEL_START_TIMEOUT_SECONDS", "300")))
    except ValueError:
        wait_seconds = 300.0
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if _model_is_ready(health_url):
            logging.info("Local model server is healthy")
            _speak_async("Model ready. Launching Jarvis.")
            return True
        if process.poll() is not None:
            logging.error("Model launcher exited with code %s; see logs/llama_server.log", process.returncode)
            _speak_async("Model server failed to start. Check the logs.")
            return False
        time.sleep(2.0)
    logging.error("Model server did not become healthy within %.0f seconds", wait_seconds)
    _speak_async("Model server timed out.")
    return False


def _telegram_token_is_configured() -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        env_path = ROOT / ".env"
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("TELEGRAM_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("\"'")
                    break
        except OSError:
            pass
    return bool(token and token.lower() not in {"your_bot_token_here", "changeme"})


def _launch_python_service(script_name: str, log_name: str) -> subprocess.Popen | None:
    script_path = ROOT / script_name
    if not script_path.is_file():
        logging.error("Service script is missing: %s", script_path)
        return None
    log_file = (LOG_DIR / log_name).open("ab")
    try:
        process = subprocess.Popen(
            [str(PYTHON), str(script_path)],
            cwd=ROOT,
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        logging.exception("Could not start %s", script_name)
        return None
    finally:
        log_file.close()
    logging.info("Started %s (pid %s); output: logs/%s", script_name, process.pid, log_name)
    return process


def _start_services() -> bool:
    global _VOICE_PROCESS, _TELEGRAM_PROCESS
    if not _launch_model():
        logging.error("Jarvis services were not launched because the model server is unavailable")
        return False

    if _telegram_token_is_configured():
        if _TELEGRAM_PROCESS is None or _TELEGRAM_PROCESS.poll() is not None:
            _TELEGRAM_PROCESS = _launch_python_service("telegram_jarvis.py", "telegram_jarvis.log")
        if _TELEGRAM_PROCESS is None:
            logging.error("Could not start Telegram; press the wake key to retry")
            return False
    else:
        logging.warning("Telegram was not started: TELEGRAM_BOT_TOKEN is not configured in .env")

    if _VOICE_PROCESS is None or _VOICE_PROCESS.poll() is not None:
        _VOICE_PROCESS = _launch_python_service("voice_jarvis.py", "voice_jarvis.log")
    if _VOICE_PROCESS is None:
        logging.error("Could not start the desktop voice assistant; press the wake key to retry")
        return False
    logging.info("Jarvis startup sequence finished")
    return True


def _stop_process_tree(process: subprocess.Popen | None, name: str) -> None:
    """Stop a service and any child processes it launched."""
    if process is None or process.poll() is not None:
        return
    try:
        result = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            logging.warning("Could not stop %s process tree: %s", name, result.stderr.strip())
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        logging.exception("Could not stop %s (pid %s)", name, process.pid)


def _shutdown_services() -> None:
    """Stop the Jarvis services started by this supervisor."""
    logging.info("Stopping Jarvis services")
    _stop_process_tree(_TELEGRAM_PROCESS, "Telegram")
    _stop_process_tree(_VOICE_PROCESS, "voice assistant")
    _stop_process_tree(_MODEL_PROCESS, "model server")


def main() -> int:
    if sys.platform != "win32":
        print("The Jarvis wake-word launcher currently requires Windows.", file=sys.stderr)
        return 1

    _configure_logging()
    _load_env_file()
    try:
        _SHUTDOWN_REQUEST.unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not clear an old Jarvis shutdown request")
    try:
        key_name, vk_code = _wake_key()
        wake_model = None
        logging.info("Wake listener started; press %s, then say 'Hey Jarvis'", key_name)
        try:
            # Prepare model files and ONNX sessions at sign-in. The microphone
            # remains closed until the user presses the configured hotkey.
            wake_model = _load_wake_model()
            logging.info("Wake detector ready; press %s, then speak (or double-press %s)", key_name, key_name)
        except Exception:
            logging.exception("Could not prepare the wake model at sign-in; it will retry after the hotkey")
        while True:
            _wait_for_hotkey(vk_code)
            logging.info("Hotkey %s pressed", key_name)
            # Wait for key release so the initial press is not interpreted as double-press
            while bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000):
                time.sleep(0.02)
            try:
                if wake_model is None:
                    wake_model = _load_wake_model()
                if _listen_for_wake_word(wake_model, vk_code):
                    if _start_services():
                        logging.info("Jarvis supervisor is monitoring services")
                        while not _SHUTDOWN_REQUEST.exists():
                            services = (_TELEGRAM_PROCESS, _VOICE_PROCESS, _MODEL_PROCESS)
                            if all(process is None or process.poll() is not None for process in services):
                                logging.info("All Jarvis services have exited")
                                return 0
                            time.sleep(0.25)
                        _shutdown_services()
                        try:
                            _SHUTDOWN_REQUEST.unlink(missing_ok=True)
                        except OSError:
                            logging.warning("Could not remove the Jarvis shutdown request file")
                        logging.info("Jarvis shutdown completed")
                        return 0
            except Exception:
                logging.exception("Wake attempt failed; press %s to retry", key_name)
    except Exception:
        logging.exception("Wake listener stopped because of a configuration or startup error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
