# Jarvis — local voice assistant

## Applying this update to your existing Windows folder

Extract this ZIP directly into `C:\Users\Mithilesh\Jarvis` and allow `voice_jarvis.py` to be replaced. The archive contains source files only: it does **not** include `.venv`, local model weights, or the `llama-server.exe` binary. Do not delete those existing folders/files.

This repository's primary entry point is `voice_jarvis.py`. It keeps the existing local Gemma model served by `llama.cpp` and does not require the OpenJarvis application, OpenJarvis server, or an OpenJarvis configuration directory.

## Speech pipeline

- **Microphone input:** `sounddevice`, with adaptive noise calibration, a short pre-roll buffer, separate speech-start/speech-stop thresholds, silence-based end detection, and near-silence rejection.
- **Speech-to-text:** local Faster-Whisper (`large-v3`, CPU `int8` by default) performs the only transcription pass, with a wider beam search and optional app-name hotwords. Jarvis avoids a competing live-preview model while recording, preserves the microphone waveform, applies padded VAD, and cleans up temporary WAV files safely on Windows. The model loads lazily and is reused.
- **Command understanding:** exact action patterns run locally first. Unmatched phrasing goes to the local LLM, which returns either a typed allowlisted action, a spoken answer, or a clarification. The action layer validates targets and values; closing apps, tabs, sleeping, shutting down, and restarting still require spoken confirmation.
- **Text-to-speech:** local Kokoro neural speech (`af_heart` by default), loaded lazily and reused. If Kokoro fails, Jarvis disables it for the rest of that run and falls back to Windows SAPI5. Set `JARVIS_TTS_BACKEND=sapi` to use only the built-in Windows voice.
- **LLM:** existing local OpenAI-compatible llama.cpp endpoint at `http://127.0.0.1:8080/v1/chat/completions`. It must be running for fallback command understanding and conversational replies. No cloud LLM or cloud TTS endpoint is configured.

The first Faster-Whisper/Kokoro run may download model weights if they are not already cached. Inference is local. Kokoro's English phonemizer can need eSpeak NG on Windows; if Kokoro cannot initialise, the script falls back to SAPI5 and prints the reason.

## Windows setup (PowerShell)

Run these commands from the project directory (`C:\Users\Mithilesh\Jarvis`). This uses your existing `.venv`; it does not create a new environment.

```powershell
# Install/update only the speech dependencies used by the unified entry point.
.\.venv\Scripts\python.exe -m pip install -r .\requirements-voice.txt

# Start Jarvis (make sure your existing llama-server is already running on port 8080).
.\.venv\Scripts\python.exe .\voice_jarvis.py
```

If you want immediate Windows speech instead of Kokoro neural speech:

```powershell
$env:JARVIS_TTS_BACKEND = "sapi"
.\.venv\Scripts\python.exe .\voice_jarvis.py
```

To return to Kokoro in a new PowerShell session, use the default or run `$env:JARVIS_TTS_BACKEND = "kokoro"`.

## Start Jarvis by voice after Windows sign-in

Install the voice dependencies, then register the wake listener in the current Windows user's Startup folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-voice.txt
powershell -ExecutionPolicy Bypass -File .\install_jarvis_startup.ps1
```

After signing in, press **F9** and say **“Hey Jarvis”** within 12 seconds. On first sign-in, the listener downloads its wake and feature models into `data/wakeword_models` and prepares the wake detector; it keeps the microphone closed until the hotkey is pressed. A successful wake starts the existing llama.cpp model, then `voice_jarvis.py` and the Telegram bot. The local model can take several minutes to load; the default wait is five minutes.

Set `JARVIS_WAKE_KEY` in `.env` to `F1`–`F24`, `PAUSE`, or `SCROLL_LOCK` to change the key. Set `JARVIS_WAKE_TIMEOUT_SECONDS` there to change how long it listens after the key press. Logs are written to `logs/jarvis_wakeword.log`, `logs/llama_server.log`, `logs/voice_jarvis.log`, and `logs/telegram_jarvis.log`. Telegram starts only when `.env` has a configured `TELEGRAM_BOT_TOKEN`.

Remove the automatic start with `powershell -ExecutionPolicy Bypass -File .\install_jarvis_startup.ps1 -Remove`.

## Optional settings

Defaults are chosen to suit the existing laptop setup. Set these in the same PowerShell window before launching Jarvis:

```powershell
$env:JARVIS_MIC_DEVICE = "1"             # Existing microphone index; use "default" to use Windows default
$env:JARVIS_STT_MODEL = "large-v3"        # Highest-accuracy Whisper model
$env:JARVIS_STT_HOTWORDS = "Jarvis, Antigravity, Chrome, YouTube, Notepad" # Add names you use
$env:JARVIS_STT_DEVICE = "cpu"
$env:JARVIS_STT_COMPUTE_TYPE = "int8"
$env:JARVIS_TTS_BACKEND = "kokoro"       # kokoro or sapi
$env:JARVIS_TTS_VOICE = "af_heart"       # Kokoro voice; e.g. af_bella, am_adam, bm_george
$env:JARVIS_TTS_DEVICE = "cpu"           # Keeps speech synthesis off the 4 GB GPU by default
$env:JARVIS_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
```

If microphone index `1` fails to open, Jarvis automatically retries the Windows default input device. Say **“stop Jarvis”** or press **Ctrl+C** to exit.

## Remove the separate OpenJarvis installation (optional)

The source ZIP you supplied did not contain OpenJarvis application/config files, so this is a separate cleanup for the Windows installation created earlier. From PowerShell, run the included script and confirm the listed paths:

```powershell
powershell -ExecutionPolicy Bypass -File .\remove_openjarvis.ps1
```

It removes only the OpenJarvis checkout/config and the three temporary installer scripts if present. It deliberately keeps shared Git, `uv`, Rust, Python, and the original Jarvis project, model files, and llama.cpp server.

## Local model/server requirement

Start the existing `llama.cpp` server with your current Gemma GGUF and mmproj exactly as you already do. Jarvis expects the OpenAI-compatible chat-completions route on port `8080`. This ZIP does not include model weights or the `llama-server.exe` binary; keep using the ones already on your machine.

## Remote Mobile Access via Telegram Bot

You can access and control Jarvis directly from your mobile phone via Telegram with full voice note support and PC remote control.

### 1. Setup Instructions

1. **Create your Bot Token:**
   - Open Telegram on your phone or PC and search for `@BotFather`.
   - Send `/newbot`, name your bot (e.g., `MyPersonalJarvisBot`), and copy the HTTP API token.
2. **Find your Telegram User ID:**
   - Open Telegram and message `@userinfobot` to get your numeric Id (e.g. `123456789`).
   - *(Alternatively, if you run the bot without setting this, it will automatically reject access and print your ID in the chat!)*
3. **Configure `.env`:**
   - Copy `.env.example` to `.env`:
     ```powershell
     Copy-Item .env.example .env
     ```
   - Open `.env` and fill in:
     ```ini
     TELEGRAM_BOT_TOKEN="your_token_from_botfather"
     TELEGRAM_ALLOWED_USER_IDS="your_telegram_numeric_id"
     TELEGRAM_VOICE_MODE="adaptive"
     ```

### 2. Running the Telegram Bot

Run from PowerShell:
```powershell
.\.venv\Scripts\python.exe .\telegram_jarvis.py
```
Or double-click `start_telegram_jarvis.bat`.

### 3. Mobile Capabilities
- **Voice Notes:** Hold the Telegram mic button, speak any question or command. Jarvis transcribes with Faster-Whisper, runs Gemma, and replies with an authentic Kokoro neural voice note.
- **Remote Screenshots:** Ask *"Take a screenshot"* or send `/screenshot` — Jarvis captures the Windows screen and sends the photo directly into your chat.
- **PC Control:** Control volume (*"set volume to 60%"*), mute, media playback (*"play pause"*), or lock screen (`/lock`).
- **Interactive Confirmations:** Destructive actions (like PC shutdown) present interactive `[ Confirm ]` / `[ Cancel ]` buttons in Telegram.

## Files

- `voice_jarvis.py` — main desktop assistant loop, physical microphone capture, and local LLM connection.
- `telegram_jarvis.py` — remote Telegram assistant with voice note input/output and remote PC controls.
- `start_telegram_jarvis.bat` — one-click Windows launcher for the Telegram service.
- `jarvis_wakeword.py` and `install_jarvis_startup.ps1` — hotkey-gated wake listener and Windows sign-in registration.
- `jarvis_speech.py` — lazy Faster-Whisper adapter and Kokoro/SAPI5 speech output (both speaker and file generation).
- `jarvis_actions.py` — desktop action executor (volume, screenshot tracking, system keys).
- `requirements-voice.txt` — dependency list including Faster-Whisper, Kokoro, and `python-telegram-bot`.
- `remove_openjarvis.ps1` — optional cleanup of the separate OpenJarvis installation/config.
- Older `voice_jarvis_*` files remain as reference/rollback versions; use `voice_jarvis.py` for the integrated pipeline.

The STT/TTS adapter design is inspired by the lazy local-backend approach in [OpenJarvis's Faster-Whisper and Kokoro adapters](https://github.com/open-jarvis/OpenJarvis/tree/main/src/openjarvis/speech). The Jarvis runtime does not import or invoke OpenJarvis.

