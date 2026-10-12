# Jarvis — local voice assistant

## Applying this update to your existing Windows folder

Extract this ZIP directly into `C:\Users\Mithilesh\Jarvis` and allow `voice_jarvis.py` to be replaced. The archive contains source files only: it does **not** include `.venv`, local model weights, or the `llama-server.exe` binary. Do not delete those existing folders/files.

This repository's primary entry point is `voice_jarvis.py`. It keeps the existing local Gemma model served by `llama.cpp` and does not require the OpenJarvis application, OpenJarvis server, or an OpenJarvis configuration directory.

## Speech pipeline

- **Microphone input:** `sounddevice`, with adaptive noise calibration, a short pre-roll buffer, separate speech-start/speech-stop thresholds, silence-based end detection, and near-silence rejection.
- **Speech-to-text:** local Faster-Whisper (`base.en`, CPU `int8` by default). The model is loaded lazily and reused. The adapter chooses a supported CTranslate2 compute type where the installed build exposes that information, uses VAD to suppress silence-related hallucinations, and cleans up temporary WAV files safely on Windows. Dependencies constrain PyAV to a compatibility range.
- **Text-to-speech:** local Kokoro neural speech (`af_heart` by default), loaded lazily and reused. If Kokoro fails, Jarvis disables it for the rest of that run and falls back to Windows SAPI5. Set `JARVIS_TTS_BACKEND=sapi` to use only the built-in Windows voice.
- **LLM:** existing local OpenAI-compatible llama.cpp endpoint at `http://127.0.0.1:8080/v1/chat/completions`. No cloud LLM or cloud TTS endpoint is configured.

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

## Optional settings

Defaults are chosen to suit the existing laptop setup. Set these in the same PowerShell window before launching Jarvis:

```powershell
$env:JARVIS_MIC_DEVICE = "1"             # Existing microphone index; use "default" to use Windows default
$env:JARVIS_STT_MODEL = "base.en"         # e.g. tiny.en, base.en, small.en
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

## Files

- `voice_jarvis.py` — main assistant loop, microphone capture, and local LLM connection.
- `jarvis_speech.py` — lazy Faster-Whisper adapter and Kokoro/SAPI5 speech output.
- `requirements-voice.txt` — minimal dependency list for this entry point.
- `remove_openjarvis.ps1` — optional cleanup of the separate OpenJarvis installation/config.
- Older `voice_jarvis_*` files remain as reference/rollback versions; use `voice_jarvis.py` for the integrated pipeline.

The STT/TTS adapter design is inspired by the lazy local-backend approach in [OpenJarvis's Faster-Whisper and Kokoro adapters](https://github.com/open-jarvis/OpenJarvis/tree/main/src/openjarvis/speech). The Jarvis runtime does not import or invoke OpenJarvis.
