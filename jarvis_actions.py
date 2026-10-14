"""jarvis_actions.py — Desktop action executor for Jarvis.

Maps recognised voice commands to real Windows system actions.
Each public handler returns a short spoken-English confirmation string,
or raises RuntimeError with a user-friendly message on failure.

The main entry point is ``handle_command(text)``.  It returns a
confirmation string when a command is matched and executed, or ``None``
when the text doesn't match any known action (so the caller can fall
through to the LLM).
"""
from __future__ import annotations

import os
import math
import re
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from typing import Any, Callable, Optional, Union
from urllib.parse import urlsplit


# ──────────────────────────────────────────────────────────────────────────────
# PendingAction: destructive commands that require verbal confirmation
# ──────────────────────────────────────────────────────────────────────────────

class PendingAction:
    """Encapsulates a destructive action that requires verbal YES/NO confirmation.

    Usage inside a handler::

        return PendingAction(
            "Are you sure you want to close all Chrome tabs? Say yes to confirm.",
            close_chrome_tabs,
        )

    The caller (voice_jarvis.py) will speak ``confirmation_prompt``, listen for
    the user's reply, and call ``execute()`` only if the reply is affirmative.
    """
    __slots__ = ("confirmation_prompt", "_callback")

    def __init__(self, confirmation_prompt: str, callback: Callable[[], str]):
        self.confirmation_prompt = confirmation_prompt
        self._callback = callback

    def execute(self) -> str:
        """Run the guarded action and return a spoken confirmation."""
        return self._callback()

    def __repr__(self) -> str:
        return f"PendingAction({self.confirmation_prompt!r})"


class UnsupportedApplicationError(RuntimeError):
    """Raised when the local app map cannot resolve an application name."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_percent(text: str) -> Optional[int]:
    """Extract a 0-100 integer from text like '50%', '50', 'fifty percent'."""
    # Numeric first
    m = re.search(r"\b(\d{1,3})\s*%?", text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    # English words
    word_map = {
        "zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        "hundred": 100, "full": 100, "maximum": 100, "max": 100,
        "half": 50, "minimum": 0, "min": 0,
    }
    lower = text.lower()
    for word, val in word_map.items():
        if re.search(rf"\b{word}\b", lower):
            return val
    return None


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output, suppressing console windows."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        **kwargs,
    )


def _powershell(script: str) -> subprocess.CompletedProcess:
    """Run a short inline PowerShell script."""
    return _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Volume
# ─────────────────────────────────────────────────────────────────────────────

def _get_volume_interface():
    """Return the Windows IAudioEndpointVolume COM interface via pycaw."""
    from pycaw.pycaw import AudioUtilities  # type: ignore
    device = AudioUtilities.GetSpeakers()
    return device.EndpointVolume


def set_volume(percent: int) -> str:
    """Set master system volume to *percent* (0–100)."""
    percent = max(0, min(100, int(percent)))
    try:
        vol = _get_volume_interface()
        vol.SetMasterVolumeLevelScalar(percent / 100.0, None)
        return f"Volume set to {percent} percent."
    except Exception as primary_err:
        print(f"[actions] pycaw volume failed: {primary_err}; trying nircmd.", flush=True)
        nircmd = shutil.which("nircmd")
        if nircmd:
            scalar = int(percent * 65535 / 100)
            result = _run([nircmd, "setsysvolume", str(scalar)])
            if result.returncode == 0:
                return f"Volume set to {percent} percent."
        raise RuntimeError(
            "Could not set volume. Install pycaw: pip install pycaw comtypes"
        ) from primary_err


def get_volume() -> int:
    """Return the current master volume as a 0-100 integer."""
    from pycaw.pycaw import AudioUtilities  # type: ignore
    device = AudioUtilities.GetSpeakers()
    return round(device.volume_percent)


def change_volume(delta: int) -> str:
    """Increase or decrease volume by *delta* percentage points."""
    try:
        current = get_volume()
        new_level = max(0, min(100, current + delta))
        return set_volume(new_level)
    except Exception:
        # Fallback: use media keys
        import pyautogui  # type: ignore
        key = "volumeup" if delta > 0 else "volumedown"
        steps = max(1, abs(delta) // 2)  # each key ≈ 2%
        for _ in range(steps):
            pyautogui.press(key)
        direction = "increased" if delta > 0 else "decreased"
        return f"Volume {direction}."


def mute_volume() -> str:
    try:
        vol = _get_volume_interface()
        vol.SetMute(1, None)
        return "System audio muted."
    except Exception:
        import pyautogui  # type: ignore
        pyautogui.press("volumemute")
        return "Muted."


def unmute_volume() -> str:
    try:
        vol = _get_volume_interface()
        vol.SetMute(0, None)
        return "Unmuted."
    except Exception:
        import pyautogui  # type: ignore
        pyautogui.press("volumemute")
        return "Unmuted."


# ─────────────────────────────────────────────────────────────────────────────
# Brightness (Windows WMI)
# ─────────────────────────────────────────────────────────────────────────────

def set_brightness(percent: int) -> str:
    percent = max(0, min(100, int(percent)))
    script = (
        f"$brightness = {percent}; "
        r"$monitor = Get-WmiObject -Namespace root\wmi -Class WmiMonitorBrightnessMethods; "
        r"$monitor.WmiSetBrightness(1, $brightness)"
    )
    result = _powershell(script)
    if result.returncode == 0:
        return f"Brightness set to {percent} percent."
    raise RuntimeError(f"Could not set brightness: {result.stderr.strip()}")


# ─────────────────────────────────────────────────────────────────────────────
# Chrome / browser
# ─────────────────────────────────────────────────────────────────────────────

def _find_chrome() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    found = shutil.which("chrome")
    return found


def close_chrome_tabs() -> str:
    """Close all Chrome windows (gracefully)."""
    result = _run(["taskkill", "/im", "chrome.exe"])
    if result.returncode in (0, 128):   # 128 = process not found (already closed)
        return "All Chrome tabs closed."
    raise RuntimeError(f"Could not close Chrome: {result.stderr.strip()}")


def open_chrome(url: str = "") -> str:
    chrome = _find_chrome()
    if chrome:
        args = [chrome]
        if url:
            args.append(url)
        subprocess.Popen(args)
        return "Opening Chrome." if not url else f"Opening {url} in Chrome."
    # Fallback: use webbrowser
    webbrowser.open(url or "https://www.google.com")
    return "Opening Chrome."


def open_chrome_tab(url: str = "https://www.google.com") -> str:
    chrome = _find_chrome()
    if chrome:
        subprocess.Popen([chrome, "--new-tab", url])
    else:
        webbrowser.open_new_tab(url)
    return "Opening a new Chrome tab."


def close_current_tab() -> str:
    """Send Ctrl+W to whatever window is active."""
    import pyautogui  # type: ignore
    pyautogui.hotkey("ctrl", "w")
    return "Current tab closed."


# ─────────────────────────────────────────────────────────────────────────────
# Applications
# ─────────────────────────────────────────────────────────────────────────────

_APP_MAP: dict[str, str] = {
    "notepad":     "notepad.exe",
    "calculator":  "calc.exe",
    "explorer":    "explorer.exe",
    "paint":       "mspaint.exe",
    "word":        "winword.exe",
    "excel":       "excel.exe",
    "powerpoint":  "powerpnt.exe",
    "task manager":"taskmgr.exe",
    "cmd":         "cmd.exe",
    "terminal":    "wt.exe",          # Windows Terminal
    "settings":    "ms-settings:",
    "spotify":     "spotify.exe",
    "discord":     "discord.exe",
    "vlc":         "vlc.exe",
    "vscode":      "code.exe",
    "vs code":     "code.exe",
}

_URL_MAP: dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "google":     "https://www.google.com",
    "github":     "https://www.github.com",
    "gmail":      "https://mail.google.com",
    "maps":       "https://maps.google.com",
    "reddit":     "https://www.reddit.com",
    "twitter":    "https://www.twitter.com",
    "x":          "https://www.x.com",
    "netflix":    "https://www.netflix.com",
    "chatgpt":    "https://chat.openai.com",
}


def open_app(name: str) -> str:
    lower = name.lower().strip()
    # Chrome has a dedicated launcher that searches the common Windows install
    # paths; it is not usually on PATH like the other executables below.
    if re.fullmatch(r"(?:google\s+)?chrome(?:\s+browser)?", lower):
        return open_chrome()

    # Website?
    for keyword, url in _URL_MAP.items():
        if keyword in lower:
            webbrowser.open(url)
            return f"Opening {keyword.title()} in your browser."
    # Executable app
    exe = _APP_MAP.get(lower)
    if exe is None:
        # Try direct shutil.which lookup
        exe = shutil.which(lower) or shutil.which(lower + ".exe")
    if exe:
        if exe.startswith("ms-"):
            os.startfile(exe)
        else:
            subprocess.Popen([exe], shell=True)
        return f"Opening {name}."
    raise UnsupportedApplicationError(f"I don't know how to open '{name}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Media keys
# ─────────────────────────────────────────────────────────────────────────────

def media_play_pause() -> str:
    import pyautogui  # type: ignore
    pyautogui.press("playpause")
    return "Play or pause toggled."


def media_next() -> str:
    import pyautogui  # type: ignore
    pyautogui.press("nexttrack")
    return "Skipped to the next track."


def media_previous() -> str:
    import pyautogui  # type: ignore
    pyautogui.press("prevtrack")
    return "Going back to the previous track."


def media_stop() -> str:
    import pyautogui  # type: ignore
    pyautogui.press("stop")
    return "Media stopped."


# ─────────────────────────────────────────────────────────────────────────────
# System
# ─────────────────────────────────────────────────────────────────────────────

_LAST_SCREENSHOT_PATH: Optional[str] = None


def get_last_screenshot() -> Optional[str]:
    """Return the filesystem path of the most recently captured screenshot."""
    return _LAST_SCREENSHOT_PATH


def take_screenshot() -> str:
    global _LAST_SCREENSHOT_PATH
    _LAST_SCREENSHOT_PATH = None
    try:
        import pyautogui  # type: ignore
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = os.path.join(desktop, f"screenshot_{ts}.png")
        pyautogui.screenshot(path)
        _LAST_SCREENSHOT_PATH = path
        return f"Screenshot saved to your Desktop as screenshot_{ts}.png."
    except ImportError:
        import pyautogui  # type: ignore
        pyautogui.hotkey("win", "shift", "s")
        return "Screenshot tool opened."



def lock_screen() -> str:
    _run(["rundll32.exe", "user32.dll,LockWorkStation"])
    return "Locking the screen."


def shutdown_pc() -> str:
    _run(["shutdown", "/s", "/t", "10", "/c", "Jarvis shutdown initiated."])
    return "Shutting down in 10 seconds. Say 'cancel shutdown' to abort."


def cancel_shutdown() -> str:
    _run(["shutdown", "/a"])
    return "Shutdown cancelled."


def restart_pc() -> str:
    _run(["shutdown", "/r", "/t", "10", "/c", "Jarvis restart initiated."])
    return "Restarting in 10 seconds."


def sleep_pc() -> str:
    _run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
    return "Putting the computer to sleep."


# ─────────────────────────────────────────────────────────────────────────────
# Command dispatch table
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (compiled_pattern, handler_fn)
# handler_fn(match, full_text) -> str
# Patterns are tested in order; first match wins.

CommandHandler = Callable[[re.Match, str], str]
_DISPATCH: list[tuple[re.Pattern, CommandHandler]] = []


def _register(pattern: str):
    """Decorator to register a handler with a compiled regex."""
    def decorator(fn: CommandHandler) -> CommandHandler:
        _DISPATCH.append((re.compile(pattern, re.IGNORECASE), fn))
        return fn
    return decorator


# ── Volume ────────────────────────────────────────────────────────────────────

@_register(r"\bset\s+(the\s+)?volume\s+to\s+(.+)")
def _h_set_volume(m: re.Match, text: str) -> str:
    pct = _parse_percent(m.group(2))
    if pct is None:
        raise RuntimeError("I didn't catch the volume level. Please say a number like 'set volume to 50 percent'.")
    return set_volume(pct)


@_register(r"\b(increase|raise|turn\s+up)\s+(the\s+)?(volume|sound|audio)\b")
def _h_vol_up(m: re.Match, text: str) -> str:
    return change_volume(+20)


@_register(r"\b(decrease|lower|turn\s+down|reduce|quieter)\s+(the\s+)?(volume|sound|audio)\b")
def _h_vol_down(m: re.Match, text: str) -> str:
    return change_volume(-20)


@_register(r"\badjust\s+(the\s+)?(volume|sound)\b")
def _h_adj_volume(m: re.Match, text: str) -> str:
    # "Adjust volume" with no direction → ask for clarification via LLM
    return None  # type: ignore  # fall through to LLM


@_register(r"\b(mute|silence|unmute)\s+(?:the\s+)?(?:mic|microphone)\b")
def _h_mute_microphone(m: re.Match, text: str) -> str:
    return "I can mute system audio, but I can't mute the microphone."


@_register(r"\b(mute|silence)(?!\s+(?:the\s+)?(?:mic|microphone)\b)(?:\s+(the\s+)?(volume|sound|audio))?\b")
def _h_mute(m: re.Match, text: str) -> str:
    return mute_volume()


@_register(r"\bunmute\b")
def _h_unmute(m: re.Match, text: str) -> str:
    return unmute_volume()


@_register(r"\bvolume\s+(to\s+)?(\d{1,3})\s*%?")
def _h_volume_shorthand(m: re.Match, text: str) -> str:
    return set_volume(int(m.group(2)))


# ── Brightness ────────────────────────────────────────────────────────────────

@_register(r"\bset\s+(the\s+)?brightness\s+to\s+(.+)")
def _h_set_brightness(m: re.Match, text: str) -> str:
    pct = _parse_percent(m.group(2))
    if pct is None:
        raise RuntimeError("Please say a number, like 'set brightness to 70 percent'.")
    return set_brightness(pct)


@_register(r"\b(increase|raise|turn\s+up)\s+(the\s+)?brightness\b")
def _h_bright_up(m: re.Match, text: str) -> str:
    return set_brightness(80)


@_register(r"\b(decrease|lower|dim|reduce)\s+(the\s+)?brightness\b")
def _h_bright_down(m: re.Match, text: str) -> str:
    return set_brightness(30)


# ── Chrome / browser ─────────────────────────────────────────────────────────

@_register(r"\bclose\s+(all\s+)?(my\s+)?(chrome|browser|google)\s+(tabs?|windows?)\b")
def _h_close_chrome(m: re.Match, text: str) -> Union[str, PendingAction]:
    return PendingAction(
        "Are you sure you want to close all Chrome tabs? Say yes to confirm.",
        close_chrome_tabs,
    )


@_register(r"\b(open|launch|start)\s+(a?\s*new\s+)?(chrome|browser)\s+(tab|window)\b")
def _h_new_tab(m: re.Match, text: str) -> str:
    return open_chrome_tab()


@_register(r"\bclose\s+(the\s+)?(current|active|this)\s+(tab|window)\b")
def _h_close_tab(m: re.Match, text: str) -> Union[str, PendingAction]:
    return PendingAction(
        "Close the current tab? Say yes to confirm.",
        close_current_tab,
    )


# ── Open app / website ────────────────────────────────────────────────────────

@_register(r"\b(open|launch|start)\s+(up\s+)?(.+)")
def _h_open(m: re.Match, text: str) -> str:
    target = m.group(3).strip().rstrip(".")
    return open_app(target)


# ── Media keys ────────────────────────────────────────────────────────────────

@_register(r"\b(play|resume)\b(?!\s+music\s+on\b)")
def _h_play(m: re.Match, text: str) -> str:
    return media_play_pause()


@_register(r"\bpause\b")
def _h_pause(m: re.Match, text: str) -> str:
    return media_play_pause()


@_register(r"\b(next|skip)\s+(track|song|music)\b")
def _h_next(m: re.Match, text: str) -> str:
    return media_next()


@_register(r"\b(previous|prev|back|last)\s+(track|song|music)\b")
def _h_prev(m: re.Match, text: str) -> str:
    return media_previous()


@_register(r"\bstop\s+(music|the\s+music|playing|playback)\b")
def _h_stop_media(m: re.Match, text: str) -> str:
    return media_stop()


# ── System ────────────────────────────────────────────────────────────────────

@_register(r"\b(take|capture)\s+(a\s+)?screenshot\b")
def _h_screenshot(m: re.Match, text: str) -> str:
    return take_screenshot()


@_register(r"\block\s+(the\s+)?(screen|computer|pc|desktop)\b")
def _h_lock(m: re.Match, text: str) -> str:
    return lock_screen()


@_register(r"\bsleep\s+(the\s+)?(computer|pc|desktop)?\b")
def _h_sleep(m: re.Match, text: str) -> Union[str, PendingAction]:
    return PendingAction(
        "Put the computer to sleep? Say yes to confirm.",
        sleep_pc,
    )


@_register(r"\bshutdown\s+(the\s+)?(computer|pc|system)\b")
def _h_shutdown(m: re.Match, text: str) -> Union[str, PendingAction]:
    return PendingAction(
        "Shut down the computer? This will close all open programs. Say yes to confirm.",
        shutdown_pc,
    )


@_register(r"\bcancel\s+shutdown\b")
def _h_cancel_shutdown(m: re.Match, text: str) -> str:
    return cancel_shutdown()


@_register(r"\brestart\s+(the\s+)?(computer|pc|system)\b")
def _h_restart(m: re.Match, text: str) -> Union[str, PendingAction]:
    return PendingAction(
        "Restart the computer? Say yes to confirm.",
        restart_pc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def handle_command(text: str) -> Optional[Union[str, PendingAction]]:
    """Try to match *text* against the dispatch table and execute the action.

    Returns:
        ``str``            — a spoken confirmation; command was executed.
        ``PendingAction``  — a destructive command waiting for verbal YES/NO.
        ``None``           — no pattern matched; caller should fall through to LLM.

    Never raises — all exceptions are caught and returned as spoken error strings.
    """
    command_text = (text or "").strip()
    # Direct dispatch is for requests phrased as commands. Embedded commands in
    # reported or quoted speech should go through the intent router instead.
    command_text = re.sub(
        r"^(?:(?:hey\s+)?jarvis[,\s:-]+|please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+|will\s+you\s+)+",
        "",
        command_text,
        flags=re.IGNORECASE,
    )
    command_text = re.sub(
        r"\s+(?:please|now|for me|thanks|thank you)[.!?]*$",
        "",
        command_text,
        flags=re.IGNORECASE,
    ).strip(" \t\r\n.,!?;")
    if not command_text:
        return None

    for pattern, handler in _DISPATCH:
        m = pattern.match(command_text)
        if m:
            try:
                result = handler(m, text)
                if result is None:
                    # Handler explicitly opted out (e.g. ambiguous command).
                    return None
                if isinstance(result, PendingAction):
                    print(f"[actions] Confirmation required: {result.confirmation_prompt!r}", flush=True)
                else:
                    print(f"[actions] Executed '{pattern.pattern}' -> {result!r}", flush=True)
                return result
            except UnsupportedApplicationError as exc:
                # Let the LLM router resolve a spoken app alias before giving
                # up; only known action functions can be called afterward.
                print(f"[actions] Unresolved application target: {exc}", flush=True)
                return None
            except RuntimeError as exc:
                # User-friendly error already composed in the handler.
                print(f"[actions] Command error: {exc}", flush=True)
                return str(exc)
            except Exception as exc:
                print(f"[actions] Unexpected error: {type(exc).__name__}: {exc}", flush=True)
                return f"I encountered an error trying to do that: {exc}"
    return None


def _structured_percent(value: Any, label: str) -> int:
    """Validate a numeric percentage returned by the LLM router."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"I couldn't determine the {label} level.")
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise RuntimeError(f"The {label} level must be between 0 and 100 percent.")
    return round(value)


def _close_app_process(target: str) -> str:
    """Close one named Windows process, without shell interpretation."""
    target = target.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}", target):
        raise RuntimeError("I couldn't safely identify that application.")

    lower = target.lower()
    if lower in {"chrome", "google chrome"}:
        image_name = "chrome.exe"
    else:
        mapped_exe = next(
            (
                exe
                for app, exe in _APP_MAP.items()
                if app == lower and not exe.lower().startswith("ms-")
            ),
            None,
        )
        if mapped_exe:
            image_name = os.path.basename(mapped_exe)
        else:
            stem = re.sub(r"\s+", "", target)
            image_name = stem if stem.lower().endswith(".exe") else f"{stem}.exe"

    result = _run(["taskkill", "/im", image_name])
    if result.returncode == 0:
        return f"Closed {target}."
    detail = (result.stderr or result.stdout).strip()
    raise RuntimeError(detail or f"Could not close {target}.")


def _validated_http_url(value: Any) -> Optional[str]:
    """Accept only complete HTTP(S) URLs from the structured router."""
    if not isinstance(value, str) or not value.strip():
        return None
    url = value.strip()
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("I can only open a complete HTTP or HTTPS web address.")
    return url


def handle_structured_command(command: dict[str, Any]) -> Optional[Union[str, PendingAction]]:
    """Validate and execute one allowlisted action produced by the LLM router.

    This deliberately accepts a small, typed action schema instead of arbitrary
    shell commands or free-form instructions. Destructive actions still return
    PendingAction so the caller must obtain verbal confirmation.
    """
    if not isinstance(command, dict):
        return None

    intent = command.get("intent")
    confidence = command.get("confidence")
    if (
        not isinstance(intent, str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        return "I couldn't safely understand that command. Could you say it another way?"
    if confidence < 0.75:
        return "I'm not sure I understood that command. Could you rephrase it?"

    target = command.get("target")
    if target is not None and not isinstance(target, str):
        return "I couldn't identify the requested target. Could you rephrase it?"
    target = target.strip() if target else ""
    direction = command.get("direction")
    if direction is not None and not isinstance(direction, str):
        return "I couldn't determine the direction for that command."
    direction = direction.lower().strip() if direction else ""

    try:
        if intent == "open_app":
            if not target:
                raise RuntimeError("Which application or website should I open?")
            result: Union[str, PendingAction] = open_app(target)
        elif intent == "open_website":
            if not target:
                raise RuntimeError("Which website should I open?")
            url = _validated_http_url(target)
            if url:
                webbrowser.open(url)
                result = f"Opening {url}."
            else:
                result = open_app(target)
        elif intent == "open_new_tab":
            url = _validated_http_url(command.get("url"))
            if not url and target:
                known_url = _URL_MAP.get(target.lower())
                if known_url:
                    url = known_url
                else:
                    raise RuntimeError("I couldn't identify a safe website address.")
            result = open_chrome_tab(url or "https://www.google.com")
        elif intent == "close_app":
            if not target:
                raise RuntimeError("Which application should I close?")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}", target):
                raise RuntimeError("I couldn't safely identify that application.")
            result = PendingAction(
                f"Close {target}? Unsaved work in that application may be lost. Say yes to confirm.",
                lambda target=target: _close_app_process(target),
            )
        elif intent == "close_chrome_tabs":
            result = PendingAction(
                "Are you sure you want to close all Chrome tabs? Say yes to confirm.",
                close_chrome_tabs,
            )
        elif intent == "close_current_tab":
            result = PendingAction(
                "Close the current browser tab? Say yes to confirm.",
                close_current_tab,
            )
        elif intent == "set_volume":
            result = set_volume(_structured_percent(command.get("value"), "volume"))
        elif intent == "adjust_volume":
            if direction not in {"increase", "decrease"}:
                raise RuntimeError("Should I increase or decrease the volume?")
            result = change_volume(20 if direction == "increase" else -20)
        elif intent == "set_brightness":
            result = set_brightness(_structured_percent(command.get("value"), "brightness"))
        elif intent == "adjust_brightness":
            if direction not in {"increase", "decrease"}:
                raise RuntimeError("Should I increase or decrease the brightness?")
            result = set_brightness(80 if direction == "increase" else 30)
        elif intent == "mute":
            result = mute_volume()
        elif intent == "unmute":
            result = unmute_volume()
        elif intent == "play_pause":
            result = media_play_pause()
        elif intent == "stop_media":
            result = media_stop()
        elif intent == "next_track":
            result = media_next()
        elif intent == "previous_track":
            result = media_previous()
        elif intent == "take_screenshot":
            result = take_screenshot()
        elif intent == "lock_screen":
            result = lock_screen()
        elif intent == "sleep_pc":
            result = PendingAction(
                "Put the computer to sleep? Say yes to confirm.", sleep_pc
            )
        elif intent == "shutdown_pc":
            result = PendingAction(
                "Shut down the computer? This will close open programs. Say yes to confirm.",
                shutdown_pc,
            )
        elif intent == "restart_pc":
            result = PendingAction(
                "Restart the computer? Say yes to confirm.", restart_pc
            )
        elif intent == "cancel_shutdown":
            result = cancel_shutdown()
        else:
            return "I can't safely perform that action yet."
    except RuntimeError as exc:
        print(f"[actions] Structured command error: {exc}", flush=True)
        return str(exc)
    except Exception as exc:
        print(
            f"[actions] Structured command error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return f"I encountered an error trying to do that: {exc}"

    if isinstance(result, PendingAction):
        print(f"[actions] Confirmation required: {result.confirmation_prompt!r}", flush=True)
    else:
        print(f"[actions] LLM intent {intent!r} -> {result!r}", flush=True)
    return result
