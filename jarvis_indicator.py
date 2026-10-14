"""jarvis_indicator.py — Visual status indicator and desktop HUD for Jarvis.

Displays a modern, floating desktop indicator whenever Jarvis is awake and
listening, with dynamic state animations and a clear 1-click Shut Down button.
"""
from __future__ import annotations

import ctypes
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False
    tk = None  # type: ignore

try:
    import win32gui
    import win32con
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
ICON_ICO = DATA_DIR / "jarvis.ico"
ICON_PNG = DATA_DIR / "jarvis.png"
SHUTDOWN_REQUEST = LOGS_DIR / "jarvis_shutdown.request"


def ensure_icon_assets() -> None:
    """Ensure data/jarvis.ico and data/jarvis.png exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ICON_ICO.is_file() and ICON_PNG.is_file():
        return
    try:
        from PIL import Image, ImageDraw
        size = 256
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Outer glow
        for r, alpha in [(124, 40), (120, 80), (116, 140)]:
            draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=(0, 229, 255, alpha), width=3)

        # Bright cyan outer ring
        draw.ellipse([20, 20, 236, 236], outline=(0, 229, 255, 255), width=8)

        # Mid ring
        draw.ellipse([54, 54, 202, 202], outline=(0, 176, 255, 220), width=6)

        # Arc reactor rays
        for i in range(12):
            angle = i * (math.pi / 6)
            x1 = 128 + 58 * math.cos(angle)
            y1 = 128 + 58 * math.sin(angle)
            x2 = 128 + 104 * math.cos(angle)
            y2 = 128 + 104 * math.sin(angle)
            draw.line([x1, y1, x2, y2], fill=(0, 229, 255, 230), width=5)

        # Center core
        draw.ellipse([82, 82, 174, 174], fill=(0, 229, 255, 220), outline=(255, 255, 255, 255), width=6)
        draw.ellipse([102, 102, 154, 154], fill=(255, 255, 255, 255))

        img.save(str(ICON_PNG), format="PNG")
        img.save(
            str(ICON_ICO),
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
    except Exception as exc:
        print(f"[indicator] Icon asset creation note: {exc}", flush=True)


def shutdown_jarvis_system(caller_pid: Optional[int] = None) -> dict[str, int]:
    """Gracefully terminate all active Jarvis services (model server, Telegram bot, wakeword)."""
    if caller_pid is None:
        caller_pid = os.getpid()

    report = {"model": 0, "telegram": 0, "wakeword": 0}
    print("\n[shutdown] Terminating all active Jarvis background services...", flush=True)

    # 1. Terminate llama-server.exe (frees 5GB RAM and GPU VRAM)
    try:
        res = subprocess.run(
            ["taskkill", "/F", "/IM", "llama-server.exe", "/T"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode == 0:
            report["model"] += 1
            print("[shutdown] Successfully stopped llama-server process tree.", flush=True)
    except Exception as exc:
        print(f"[shutdown] Note on llama-server taskkill: {exc}", flush=True)

    # 2. Terminate telegram_jarvis.py and jarvis_wakeword.py processes
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = p.info["pid"]
                if pid == caller_pid:
                    continue
                args = p.info.get("cmdline") or []
                name = (p.info.get("name") or "").lower()

                if "llama-server" in name:
                    p.kill()
                    report["model"] += 1
                elif any(arg.endswith("telegram_jarvis.py") or arg == "telegram_jarvis.py" for arg in args):
                    print(f"[shutdown] Terminating Telegram bot (PID {pid})...", flush=True)
                    p.terminate()
                    try:
                        p.wait(timeout=1.5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    report["telegram"] += 1
                elif any(arg.endswith("jarvis_wakeword.py") or arg == "jarvis_wakeword.py" for arg in args):
                    print(f"[shutdown] Terminating wakeword supervisor (PID {pid})...", flush=True)
                    p.terminate()
                    try:
                        p.wait(timeout=1.5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    report["wakeword"] += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as exc:
        print(f"[shutdown] Note on process scan: {exc}", flush=True)

    # 3. Touch shutdown request file for supervisor
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SHUTDOWN_REQUEST.write_text("shutdown requested from desktop HUD\n", encoding="utf-8")
    except OSError:
        pass

    return report


class JarvisIndicator:
    """Floating desktop HUD widget that shows Jarvis's active status and shutdown controls."""

    STATE_THEMES = {
        "starting": {
            "color": "#38bdf8",
            "title": "JARVIS",
            "status": "Starting up...",
            "sub": "Warming up speech models",
        },
        "listening": {
            "color": "#00e5ff",
            "title": "JARVIS",
            "status": "Listening... Speak naturally",
            "sub": "Microphone Active",
        },
        "hearing": {
            "color": "#ffb300",
            "title": "JARVIS",
            "status": "Speech detected...",
            "sub": "Recording voice",
        },
        "transcribing": {
            "color": "#00b0ff",
            "title": "JARVIS",
            "status": "Transcribing...",
            "sub": "Faster-Whisper (real-time)",
        },
        "thinking": {
            "color": "#b388ff",
            "title": "JARVIS",
            "status": "Interpreting...",
            "sub": "Local Gemma",
        },
        "executing": {
            "color": "#00e676",
            "title": "JARVIS",
            "status": "Executing command...",
            "sub": "Windows Desktop",
        },
        "speaking": {
            "color": "#00e676",
            "title": "JARVIS",
            "status": "Speaking...",
            "sub": "Voice Output",
        },
        "stopping": {
            "color": "#ef4444",
            "title": "JARVIS",
            "status": "Shutting down...",
            "sub": "Stopping services",
        },
    }

    def __init__(self, on_shutdown: Optional[Callable[[], None]] = None) -> None:
        self.on_shutdown = on_shutdown
        self._msg_queue: queue.Queue = queue.Queue()
        self._current_state = "starting"
        self._custom_status: Optional[str] = None
        self._is_running = True
        self._is_shutting_down = False
        self._anim_tick = 0
        self.root: Optional[tk.Tk] = None
        self._drag_start_x = 0
        self._drag_start_y = 0

        ensure_icon_assets()

    def _setup_window(self) -> None:
        if not _TK_AVAILABLE:
            return

        # Register distinct Windows AppUserModelID so taskbar groups it as Jarvis
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mithil.jarvis.desktop.assistant")
        except Exception:
            pass

        self.root = tk.Tk()
        self.root.title("Jarvis Voice Assistant")
        self.root.configure(bg="#0b0f17")

        # Set Window and Taskbar Icon
        if ICON_PNG.is_file():
            try:
                self._photo_icon = ImageTk.PhotoImage(Image.open(ICON_PNG))
                self.root.iconphoto(True, self._photo_icon)
            except Exception:
                pass
        if ICON_ICO.is_file():
            try:
                self.root.iconbitmap(str(ICON_ICO))
            except Exception:
                pass

        # Position at top-right of screen
        screen_w = self.root.winfo_screenwidth()
        win_w, win_h = 340, 72
        pos_x = max(20, screen_w - win_w - 30)
        pos_y = 35

        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # Make frameless window appear in Windows Taskbar
        if _WIN32_AVAILABLE:
            try:
                self.root.update_idletasks()
                hwnd = win32gui.GetParent(self.root.winfo_id())
                if hwnd:
                    style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                    style |= win32con.WS_EX_APPWINDOW
                    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
                    self.root.withdraw()
                    self.root.deiconify()
            except Exception as exc:
                print(f"[indicator] Win32 taskbar style note: {exc}", flush=True)

        # Outer border frame
        self._outer_frame = tk.Frame(self.root, bg="#1e293b", padx=1, pady=1)
        self._outer_frame.pack(fill="both", expand=True)

        # Inner container
        self._inner_frame = tk.Frame(self._outer_frame, bg="#0b0f17")
        self._inner_frame.pack(fill="both", expand=True, padx=1, pady=1)

        # Left: Canvas Orb
        self._canvas = tk.Canvas(
            self._inner_frame,
            width=50,
            height=50,
            bg="#0b0f17",
            highlightthickness=0,
            cursor="fleur",
        )
        self._canvas.pack(side="left", padx=(10, 6), pady=10)

        # Center: Text Info
        self._info_frame = tk.Frame(self._inner_frame, bg="#0b0f17", cursor="fleur")
        self._info_frame.pack(side="left", fill="both", expand=True, pady=12)

        self._title_lbl = tk.Label(
            self._info_frame,
            text="JARVIS",
            font=("Segoe UI", 10, "bold"),
            fg="#00e5ff",
            bg="#0b0f17",
        )
        self._title_lbl.pack(anchor="w")

        self._status_lbl = tk.Label(
            self._info_frame,
            text="Starting up...",
            font=("Segoe UI", 8),
            fg="#94a3b8",
            bg="#0b0f17",
        )
        self._status_lbl.pack(anchor="w", pady=(1, 0))

        # Right: Clear Shut Down Button
        self._shutdown_btn = tk.Button(
            self._inner_frame,
            text="⏻ Shut Down",
            font=("Segoe UI", 8, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#ef4444",
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2",
            command=self.request_shutdown,
        )
        self._shutdown_btn.pack(side="right", padx=(6, 12), pady=16)

        # Make entire widget draggable
        for widget in (self._inner_frame, self._outer_frame, self._info_frame, self._title_lbl, self._status_lbl, self._canvas):
            widget.bind("<Button-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        # Animation & queue poll loop
        self.root.after(40, self._ui_tick)

    def _on_drag_start(self, event) -> None:
        self._drag_start_x = event.x_root - self.root.winfo_x()
        self._drag_start_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        new_x = event.x_root - self._drag_start_x
        new_y = event.y_root - self._drag_start_y
        self.root.geometry(f"+{new_x}+{new_y}")

    def _draw_orb(self) -> None:
        if not self._canvas:
            return
        self._canvas.delete("all")
        theme = self.STATE_THEMES.get(self._current_state, self.STATE_THEMES["listening"])
        color = theme["color"]
        cx, cy = 25, 25

        t = self._anim_tick * 0.15
        pulse = math.sin(t)

        if self._current_state == "listening":
            # Gentle breathing ring + Arc Reactor core
            r_glow = 21 + 2.0 * pulse
            self._canvas.create_oval(cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow, outline=color, width=1)
            self._canvas.create_oval(cx - 16, cy - 16, cx + 16, cy + 16, outline=color, width=2)
            # Rotating arc segments
            angle_offset = (self._anim_tick * 2) % 360
            for i in range(4):
                start = angle_offset + i * 90
                self._canvas.create_arc(cx - 13, cy - 13, cx + 13, cy + 13, start=start, extent=55, outline=color, width=2, style="arc")
            # Center core
            r_core = 6 + 1.0 * pulse
            self._canvas.create_oval(cx - r_core, cy - r_core, cx + r_core, cy + r_core, fill=color, outline="#ffffff", width=1)

        elif self._current_state == "hearing":
            # Active pulsing waves for speech
            r_wave = 15 + ((self._anim_tick * 1.5) % 10)
            self._canvas.create_oval(cx - r_wave, cy - r_wave, cx + r_wave, cy + r_wave, outline=color, width=2)
            self._canvas.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, outline="#ffffff", width=2)
            self._canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=color, outline="#ffffff", width=1)

        elif self._current_state in {"transcribing", "thinking"}:
            # Orbiting dots for model processing
            self._canvas.create_oval(cx - 16, cy - 16, cx + 16, cy + 16, outline=color, width=1)
            for i in range(4):
                ang = t * 1.5 + i * (math.pi / 2)
                dx = cx + 12 * math.cos(ang)
                dy = cy + 12 * math.sin(ang)
                self._canvas.create_oval(dx - 3, dy - 3, dx + 3, dy + 3, fill=color, outline="#ffffff")
            self._canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#ffffff", outline=color, width=1)

        elif self._current_state in {"speaking", "executing"}:
            # Audio equalizer wave bars
            self._canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, outline=color, width=1)
            bar_offsets = [-9, -4, 1, 6, 11]
            for idx, ox in enumerate(bar_offsets):
                h = 5 + 9 * abs(math.sin(t * 1.8 + idx * 0.8))
                self._canvas.create_line(cx + ox, cy - h, cx + ox, cy + h, fill=color, width=2)

        elif self._current_state == "stopping":
            # Red warning circle
            self._canvas.create_oval(cx - 18, cy - 18, cx + 18, cy + 18, outline=color, width=2)
            self._canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, fill=color, outline="#ffffff", width=1)

    def _ui_tick(self) -> None:
        if not self._is_running or not self.root:
            return

        self._anim_tick += 1

        # Process queued messages
        try:
            while not self._msg_queue.empty():
                kind, state, text = self._msg_queue.get_nowait()
                if kind == "state":
                    self._current_state = state
                    self._custom_status = text
                    theme = self.STATE_THEMES.get(state, self.STATE_THEMES["listening"])
                    display_text = text or theme["status"]
                    self._title_lbl.config(fg=theme["color"])
                    self._status_lbl.config(text=display_text)
                elif kind == "close":
                    self._destroy_window()
                    return
        except Exception:
            pass

        self._draw_orb()
        self.root.after(45, self._ui_tick)

    def set_state(self, state: str, text: Optional[str] = None) -> None:
        """Thread-safe state update."""
        self._msg_queue.put(("state", state, text))

    def request_shutdown(self) -> None:
        """Initiate clean shutdown from user click or voice command."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self.set_state("stopping", "Shutting down all Jarvis services...")
        print("\n[HUD] Shut Down requested. Terminating model, Telegram bot, and voice pipeline...", flush=True)

        # 1. Terminate all background services (model server, Telegram bot, etc.)
        shutdown_jarvis_system(caller_pid=os.getpid())

        # 2. Trigger external callback
        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as exc:
                print(f"[HUD] Error in shutdown handler: {exc}", flush=True)

        # 3. Destroy window after short feedback pause
        if self.root:
            self.root.after(350, self._destroy_window)

    def _destroy_window(self) -> None:
        self._is_running = False
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None

    def close(self) -> None:
        """Thread-safe close."""
        self._msg_queue.put(("close", None, None))

    def run_mainloop(self) -> None:
        """Run the Tkinter event loop on the main thread."""
        if not _TK_AVAILABLE:
            print("[indicator] Tkinter unavailable; running in console mode.", flush=True)
            while self._is_running:
                time.sleep(0.5)
            return

        self._setup_window()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.request_shutdown()


_GLOBAL_INDICATOR: Optional[JarvisIndicator] = None


def get_indicator() -> Optional[JarvisIndicator]:
    """Return the active indicator instance if running."""
    return _GLOBAL_INDICATOR


def set_global_indicator(indicator: Optional[JarvisIndicator]) -> None:
    global _GLOBAL_INDICATOR
    _GLOBAL_INDICATOR = indicator
