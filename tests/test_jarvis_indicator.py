"""Unit tests for Jarvis visual indicator and shutdown control."""
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import jarvis_indicator


class JarvisIndicatorTests(unittest.TestCase):
    def test_ensure_icon_assets_creates_files(self):
        jarvis_indicator.ensure_icon_assets()
        self.assertTrue(jarvis_indicator.ICON_PNG.is_file())
        self.assertTrue(jarvis_indicator.ICON_ICO.is_file())

    def test_state_themes_exist(self):
        themes = jarvis_indicator.JarvisIndicator.STATE_THEMES
        for expected in ["listening", "hearing", "transcribing", "thinking", "executing", "speaking", "stopping"]:
            self.assertIn(expected, themes)
            self.assertIn("color", themes[expected])
            self.assertIn("status", themes[expected])

    def test_indicator_state_queueing(self):
        ind = jarvis_indicator.JarvisIndicator()
        ind.set_state("listening", "Listening test")
        self.assertFalse(ind._msg_queue.empty())
        item = ind._msg_queue.get_nowait()
        self.assertEqual(item, ("state", "listening", "Listening test"))

    def test_request_shutdown_writes_file_and_calls_callback(self):
        called = False

        def on_shutdown():
            nonlocal called
            called = True

        ind = jarvis_indicator.JarvisIndicator(on_shutdown=on_shutdown)
        if jarvis_indicator.SHUTDOWN_REQUEST.exists():
            jarvis_indicator.SHUTDOWN_REQUEST.unlink()

        try:
            ind.request_shutdown()
            self.assertTrue(called)
            self.assertTrue(jarvis_indicator.SHUTDOWN_REQUEST.is_file())
            content = jarvis_indicator.SHUTDOWN_REQUEST.read_text(encoding="utf-8")
            self.assertIn("shutdown requested", content)
        finally:
            if jarvis_indicator.SHUTDOWN_REQUEST.exists():
                jarvis_indicator.SHUTDOWN_REQUEST.unlink()


if __name__ == "__main__":
    unittest.main()
