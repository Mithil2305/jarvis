"""telegram_jarvis.py — Remote Telegram interface for Jarvis voice assistant.

Features:
- Full voice-note support: speak to Jarvis from anywhere via Telegram voice notes.
- Speech-to-text with local Faster-Whisper.
- Text-to-speech with local Kokoro neural voice (sent back as playable voice notes).
- Desktop automation via jarvis_actions.py (volume, apps, websites, screen lock, media).
- Real-time screenshot capture and delivery to mobile.
- Interactive inline buttons for destructive action confirmations.
- Strict Telegram user ID authorization whitelist.
- Local Gemma LLM integration via llama.cpp.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


def _load_env_file(path: str = ".env") -> None:
    """Load simple key=value pairs from a local .env file if present."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        print(f"[config] Warning loading .env: {exc}", flush=True)


_load_env_file()

import requests
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from jarvis_actions import (
    PendingAction,
    get_last_screenshot,
    handle_command,
    handle_structured_command,
)
from jarvis_speech import (
    backend_status,
    synthesize_to_file,
    transcribe_file,
)

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Configuration
# ─────────────────────────────────────────────────────────────────────────────

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
SHUTDOWN_REQUEST = Path(__file__).resolve().parent / "logs" / "jarvis_shutdown.request"
LLM_URL = os.environ.get("JARVIS_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
MODEL_NAME = os.environ.get("JARVIS_MODEL_NAME", "gemma-4")
VOICE_MODE = os.environ.get("TELEGRAM_VOICE_MODE", "adaptive").strip().lower()  # adaptive | always | never
LOCAL_PLAYBACK = os.environ.get("JARVIS_LOCAL_SPEAKER_PLAYBACK", "false").strip().lower() == "true"

# Parse allowed user IDs: "1234567, 9876543"
_RAW_ALLOWED_USERS = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
ALLOWED_USERS: set[int] = set()
if _RAW_ALLOWED_USERS:
    for part in _RAW_ALLOWED_USERS.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            ALLOWED_USERS.add(int(part))

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

_ALLOWED_LLM_INTENTS = {
    "open_app", "open_website", "open_new_tab", "close_app",
    "close_chrome_tabs", "close_current_tab", "set_volume", "adjust_volume",
    "set_brightness", "adjust_brightness", "mute", "unmute", "play_pause",
    "stop_media", "next_track", "previous_track", "take_screenshot",
    "lock_screen", "sleep_pc", "shutdown_pc", "restart_pc", "cancel_shutdown",
}

_DISCUSSION_OR_NEGATION = re.compile(
    r"\b(what\s+is|what\s+are|what\s+does|how\s+to|why|explain|mean|"
    r"don't|do\s+not|never|should\s+i|can\s+you\s+explain)\b",
    re.IGNORECASE,
)

# In-memory storage, isolated to each private chat.
_CONVERSATION_HISTORY: dict[int, list[dict[str, str]]] = {}
_PENDING_ACTIONS: dict[str, tuple[PendingAction, float, int, int]] = {}
# action_id -> (action, timestamp, owner_user_id, owner_chat_id)

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_jarvis")


# ─────────────────────────────────────────────────────────────────────────────
# Authorization Check
# ─────────────────────────────────────────────────────────────────────────────

def _is_authorized(update: Update) -> bool:
    """Allow only whitelisted users in private chats."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat or chat.type != "private":
        return False
    if not ALLOWED_USERS:
        # If no user IDs are configured yet, inform the user with their ID
        return False
    return user.id in ALLOWED_USERS


async def _handle_unauthorized(update: Update) -> None:
    """Politely reject unauthorized interactions and provide the user's ID for setup."""
    chat = update.effective_chat
    if chat and chat.type != "private":
        if update.callback_query:
            await update.callback_query.answer(
                "Please use Jarvis in a private chat.", show_alert=True
            )
        return

    user = update.effective_user
    uid = user.id if user else "Unknown"
    username = f"@{user.username}" if (user and user.username) else "unknown"
    logger.warning("Rejected unauthorized access from %s (User ID: %s)", username, uid)

    msg = (
        "🔒 **Jarvis Security Guard**\n\n"
        "Access to this desktop assistant is restricted.\n\n"
        f"Your Telegram User ID is: `{uid}`\n\n"
        "To authorize this account, add this ID to `TELEGRAM_ALLOWED_USER_IDS` "
        "in your Jarvis `.env` file on your PC and restart the bot."
    )
    if update.message:
        await update.message.reply_text(msg, parse_mode="Markdown")
    elif update.callback_query:
        await update.callback_query.answer("Unauthorized.", show_alert=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Integration (llama.cpp)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gemma_route(content: str) -> dict[str, Any]:
    """Parse and constrain the local Gemma route JSON."""
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return {"type": "response", "text": raw}
    try:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            route_type = parsed.get("type")
            if route_type == "action":
                intent = parsed.get("intent")
                if intent in _ALLOWED_LLM_INTENTS:
                    return parsed
            if route_type in {"response", "clarify"}:
                text = parsed.get("text")
                if isinstance(text, str) and text.strip():
                    return {"type": route_type, "text": text.strip()}
    except Exception:
        pass
    return {"type": "response", "text": raw}


def _query_local_gemma(prompt: str, history: list[dict[str, str]]) -> dict[str, Any]:
    """Send user prompt to local llama.cpp endpoint and return parsed route."""
    recent = [*history, {"role": "user", "content": prompt}][-10:]
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *recent],
        "temperature": 0.1,
        "max_tokens": int(os.environ.get("JARVIS_MAX_TOKENS", "180")),
        "stream": False,
    }
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        if not content:
            return {"type": "response", "text": "I received an empty response from Gemma."}
        return _parse_gemma_route(content)
    except requests.exceptions.ConnectionError:
        return {
            "type": "response",
            "text": (
                "⚠️ Could not connect to the local Gemma LLM server at "
                f"{LLM_URL}.\nPlease make sure `start_model.bat` is running on your PC."
            ),
        }
    except Exception as exc:
        logger.error("LLM request failed: %s", exc)
        return {"type": "response", "text": f"Error querying local model: {exc}"}


def _check_llm_online() -> bool:
    """Check health first, then verify the chat endpoint with an HTTP request."""
    parts = urlsplit(LLM_URL)
    health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
    try:
        response = requests.get(health_url, timeout=3)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        try:
            response = requests.post(
                LLM_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=3,
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False


def _remember_turn(chat_id: int, user_text: str, reply_text: str) -> None:
    """Store an exchange in the history for one private chat."""
    history = _CONVERSATION_HISTORY.setdefault(chat_id, [])
    history.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply_text},
    ])
    del history[:-20]


def _prune_pending_actions(now: Optional[float] = None) -> None:
    """Discard confirmations that can no longer be executed."""
    current_time = time.time() if now is None else now
    expired = [
        action_id
        for action_id, (_, created_at, _, _) in _PENDING_ACTIONS.items()
        if current_time - created_at > 120
    ]
    for action_id in expired:
        _PENDING_ACTIONS.pop(action_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Response Delivery
# ─────────────────────────────────────────────────────────────────────────────

async def _send_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    reply_text: str,
    as_voice: bool,
    photo_path: Optional[str] = None,
) -> None:
    """Send reply back via Telegram with optional voice note and/or photo attachment."""
    if not update.message:
        return

    chat_id = update.message.chat_id

    # 1. If a screenshot photo was captured, send it first
    if photo_path and os.path.exists(photo_path):
        try:
            with open(photo_path, "rb") as photo_file:
                await update.message.reply_photo(
                    photo=photo_file,
                    caption=reply_text,
                )
            return
        except Exception as exc:
            logger.error("Failed to send screenshot photo: %s", exc)

    # 2. Determine whether to send a voice note
    send_voice = False
    if VOICE_MODE == "always":
        send_voice = True
    elif VOICE_MODE == "never":
        send_voice = False
    else:  # "adaptive"
        send_voice = as_voice

    if send_voice and reply_text:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        tmp_ogg = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                tmp_ogg = tmp.name

            # Run synthesis off-thread
            await asyncio.to_thread(synthesize_to_file, reply_text, tmp_ogg)

            with open(tmp_ogg, "rb") as voice_file:
                await update.message.reply_voice(
                    voice=voice_file,
                    caption=reply_text,
                )
            return
        except Exception as exc:
            logger.warning("Voice synthesis/sending failed, falling back to text: %s", exc)
        finally:
            if tmp_ogg and os.path.exists(tmp_ogg):
                try:
                    os.unlink(tmp_ogg)
                except OSError:
                    pass

    # 3. Default: send clean text message
    await update.message.reply_text(reply_text)


# ─────────────────────────────────────────────────────────────────────────────
# Core Intent & Command Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

async def _process_user_intent(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    from_voice: bool,
) -> None:
    """Process transcribed voice or typed text through actions & LLM."""
    clean_text = text.strip()
    if not clean_text:
        return
    chat_id = update.effective_chat.id

    logger.info("Processing [%s]: %s", "VOICE" if from_voice else "TEXT", clean_text)

    # 1. Direct command match (regex table)
    action_result = None
    if not _DISCUSSION_OR_NEGATION.search(clean_text):
        action_result = handle_command(clean_text)

    # 2. Check for PendingAction (destructive actions requiring confirmation)
    if isinstance(action_result, PendingAction):
        await _prompt_pending_action(update, action_result)
        return

    # 3. If direct action succeeded
    if action_result is not None:
        is_screenshot_request = bool(
            re.search(r"\b(?:take|capture)\s+(?:a\s+)?screenshot\b", clean_text, re.IGNORECASE)
        )
        photo = get_last_screenshot() if is_screenshot_request else None
        _remember_turn(chat_id, clean_text, action_result)
        await _send_response(update, context, action_result, as_voice=from_voice, photo_path=photo)
        return

    # 4. Fall through to local Gemma LLM
    await context.bot.send_chat_action(
        chat_id=update.message.chat_id,
        action=ChatAction.RECORD_VOICE if from_voice else ChatAction.TYPING,
    )
    history_snapshot = list(_CONVERSATION_HISTORY.get(chat_id, []))
    route = await asyncio.to_thread(_query_local_gemma, clean_text, history_snapshot)
    route_type = route.get("type")

    if route_type == "action":
        action_obj = handle_structured_command(route)
        if isinstance(action_obj, PendingAction):
            await _prompt_pending_action(update, action_obj)
            return

        answer = action_obj if action_obj is not None else "I could not safely execute that action."
        photo = get_last_screenshot() if route.get("intent") == "take_screenshot" else None
    else:
        answer = str(route.get("text", "Could you rephrase that?"))
        photo = None

    _remember_turn(chat_id, clean_text, answer)
    await _send_response(update, context, answer, as_voice=from_voice, photo_path=photo)


async def _prompt_pending_action(update: Update, action: PendingAction) -> None:
    """Present interactive inline buttons for sensitive/destructive operations."""
    _prune_pending_actions()
    action_id = str(uuid.uuid4())[:8]
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat or chat.type != "private":
        return
    _PENDING_ACTIONS[action_id] = (action, time.time(), user.id, chat.id)

    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm & Execute", callback_data=f"confirm:{action_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"cancel:{action_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    msg = f"⚠️ **Confirmation Required**\n\n{action.confirmation_prompt}"
    if update.message:
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)


# ─────────────────────────────────────────────────────────────────────────────
# Telegram Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start and /help commands."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    status_str = backend_status()
    welcome_text = (
        "🤖 **Jarvis Remote Interface Online**\n\n"
        "You can control your Windows PC and talk to Jarvis remotely:\n"
        "• 🎙️ **Voice Notes:** Hold the mic button, speak, and Jarvis will reply with voice.\n"
        "• 💬 **Text Messages:** Type any question, command, or conversation.\n"
        "• 📸 **Screenshots:** Ask 'Take a screenshot' or send `/screenshot`.\n"
        "• 🔒 **Lock Workstation:** Send `/lock` to lock your PC immediately.\n"
        "• ⏻ **Shut Down Jarvis:** Send `/off` to stop Jarvis and its managed services.\n"
        "• 🔊 **Media & Volume:** 'Set volume to 50%', 'Mute', 'Play pause'.\n\n"
        f"**Engine Status:**\n`{status_str}`\n"
        f"**Local LLM Endpoint:** `{LLM_URL}`\n"
        f"**Voice Reply Mode:** `{VOICE_MODE}`"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    llm_online = await asyncio.to_thread(_check_llm_online)
    _prune_pending_actions()
    history = _CONVERSATION_HISTORY.get(update.effective_chat.id, [])

    status_report = (
        "📊 **Jarvis System Diagnostics**\n\n"
        f"• **Speech Engine:** `{backend_status()}`\n"
        f"• **Local LLM Server:** {'🟢 Online' if llm_online else '🔴 Offline / Unreachable'}\n"
        f"• **Voice Mode:** `{VOICE_MODE}`\n"
        f"• **Active Session History:** `{len(history) // 2} turns`\n"
        f"• **Pending Confirmations:** `{len(_PENDING_ACTIONS)}`"
    )
    await update.message.reply_text(status_report, parse_mode="Markdown")


async def handle_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Request a full Jarvis shutdown through the supervising launcher."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    await update.message.reply_text("Shutting down Jarvis and its voice, Telegram, and model services.")
    try:
        from jarvis_indicator import shutdown_jarvis_system
        shutdown_jarvis_system(caller_pid=os.getpid())
    except Exception as exc:
        logger.exception("Could not stop all Jarvis services: %s", exc)

    # run_polling() handles this signal and closes the Telegram application cleanly.
    context.application.stop_running()


async def handle_screenshot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /screenshot command."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    res = handle_command("take a screenshot")
    photo = get_last_screenshot()
    await _send_response(update, context, res or "Screenshot captured.", as_voice=False, photo_path=photo)


async def handle_lock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lock command."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    res = handle_command("lock screen")
    await _send_response(update, context, res or "Locking the screen.", as_voice=False)


async def handle_clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command to reset conversation memory."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    _CONVERSATION_HISTORY.pop(update.effective_chat.id, None)
    await update.message.reply_text("🧹 Conversation history cleared.")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram voice notes or audio recordings."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    await context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)

    tmp_voice_path = None
    try:
        # Download voice note to temporary file
        voice_file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_voice_path = tmp.name

        await voice_file.download_to_drive(custom_path=tmp_voice_path)

        # Transcribe with Faster-Whisper
        transcript = await asyncio.to_thread(transcribe_file, tmp_voice_path)

        if not transcript:
            await update.message.reply_text("I heard your voice note, but could not detect clear speech. Please try speaking closer to your phone.")
            return

        # Process recognized command
        await _process_user_intent(update, context, transcript, from_voice=True)

    except Exception as exc:
        logger.error("Failed to process voice note: %s", exc)
        await update.message.reply_text(f"Error processing voice note: {exc}")
    finally:
        if tmp_voice_path and os.path.exists(tmp_voice_path):
            try:
                os.unlink(tmp_voice_path)
            except OSError:
                pass


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle typed text messages."""
    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    if not update.message or not update.message.text:
        return

    await _process_user_intent(update, context, update.message.text, from_voice=False)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle interactive inline keyboard confirmation clicks."""
    query = update.callback_query
    if not query or not query.data:
        return

    if not _is_authorized(update):
        await _handle_unauthorized(update)
        return

    data = query.data
    if ":" not in data:
        await query.answer("This confirmation is invalid.", show_alert=True)
        return

    action_type, action_id = data.split(":", 1)
    if action_type not in {"confirm", "cancel"}:
        await query.answer("This confirmation is invalid.", show_alert=True)
        return

    _prune_pending_actions()
    stored = _PENDING_ACTIONS.get(action_id)

    if not stored:
        await query.answer()
        await query.edit_message_text("⚠️ This action expired or was already handled.")
        return

    pending_action, created_at, owner_user_id, owner_chat_id = stored
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat or user.id != owner_user_id or chat.id != owner_chat_id:
        await query.answer("This confirmation belongs to another chat.", show_alert=True)
        return

    _PENDING_ACTIONS.pop(action_id, None)
    if time.time() - created_at > 120:
        await query.answer()
        await query.edit_message_text("⌛ This action timed out.")
        return

    await query.answer()

    if action_type == "confirm":
        try:
            result_msg = pending_action.execute()
            await query.edit_message_text(f"✅ **Action Confirmed:** {result_msg}", parse_mode="Markdown")
        except Exception as exc:
            logger.error("Error executing confirmed action: %s", exc)
            await query.edit_message_text(f"❌ Error executing action: {exc}")
    elif action_type == "cancel":
        await query.edit_message_text("❌ Action cancelled.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Start the Jarvis Telegram Bot."""
    if not BOT_TOKEN:
        print("\n" + "=" * 65)
        print(" ERROR: TELEGRAM_BOT_TOKEN is missing!")
        print("=" * 65)
        print("To enable remote access from your phone:")
        print("1. Open Telegram and search for @BotFather")
        print("2. Send /newbot and follow instructions to get your HTTP API token")
        print("3. Create a '.env' file in this folder with:")
        print("   TELEGRAM_BOT_TOKEN=your_token_here")
        print("   TELEGRAM_ALLOWED_USER_IDS=your_telegram_id")
        print("4. Restart this script.")
        print("=" * 65 + "\n")
        return

    print("=" * 60)
    print(" Jarvis Telegram Remote Interface Starting...")
    print(f" Voice Mode: {VOICE_MODE}")
    print(f" LLM Endpoint: {LLM_URL}")
    print(f" Speech Engine: {backend_status()}")
    if ALLOWED_USERS:
        print(f" Authorized Users: {len(ALLOWED_USERS)} ID(s)")
    else:
        print(" ⚠️ WARNING: TELEGRAM_ALLOWED_USER_IDS is empty.")
        print(" The bot will reject requests and display the sender's User ID for setup.")
    print("=" * 60)

    application: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler(["start", "help"], handle_start))
    application.add_handler(CommandHandler("status", handle_status))
    application.add_handler(CommandHandler("off", handle_off))
    application.add_handler(CommandHandler(["screenshot", "screen"], handle_screenshot_cmd))
    application.add_handler(CommandHandler("lock", handle_lock_cmd))
    application.add_handler(CommandHandler(["clear", "reset"], handle_clear_cmd))

    # Messages
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Interactive Inline Buttons
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    print("\nJarvis is now listening on Telegram. Press Ctrl+C to stop.\n")
    application.run_polling()


if __name__ == "__main__":
    main()
