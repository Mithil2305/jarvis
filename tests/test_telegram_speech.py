"""Tests for Telegram bot audio adapters and actions."""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import jarvis_actions
import jarvis_speech
import telegram_jarvis


class TelegramSpeechTests(unittest.TestCase):
    def test_synthesize_to_file_creates_audio_file(self):
        fake_audio = np.array([0.05, -0.05, 0.1, -0.1], dtype=np.float32)

        class FakePipeline:
            def __call__(self, text, *, voice, speed):
                yield text, "phonemes", fake_audio

        with patch.object(jarvis_speech, "_get_kokoro_pipeline", return_value=FakePipeline()):
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                out_path = tmp.name

            try:
                result_path = jarvis_speech.synthesize_to_file("Testing Telegram voice note", out_path)
                self.assertEqual(result_path, out_path)
                self.assertTrue(os.path.exists(out_path))
                self.assertGreater(os.path.getsize(out_path), 0)
            finally:
                if os.path.exists(out_path):
                    os.unlink(out_path)

    def test_transcribe_file_empty_or_missing_path(self):
        self.assertEqual(jarvis_speech.transcribe_file(""), "")
        self.assertEqual(jarvis_speech.transcribe_file("non_existent_file.ogg"), "")

    def test_get_last_screenshot_tracking(self):
        with patch("pyautogui.screenshot") as mock_shot:
            jarvis_actions._LAST_SCREENSHOT_PATH = None
            msg = jarvis_actions.take_screenshot()
            self.assertIn("Screenshot saved", msg)
            last_path = jarvis_actions.get_last_screenshot()
            self.assertIsNotNone(last_path)
            self.assertTrue(last_path.endswith(".png"))

    def test_telegram_authorization_check(self):
        mock_update = MagicMock()
        mock_update.effective_user.id = 12345

        # With empty allowed users, should reject
        with patch.object(telegram_jarvis, "ALLOWED_USERS", set()):
            self.assertFalse(telegram_jarvis._is_authorized(mock_update))

        # With user in allowed users, should accept
        with patch.object(telegram_jarvis, "ALLOWED_USERS", {12345}):
            self.assertTrue(telegram_jarvis._is_authorized(mock_update))

        # With other user, should reject
        with patch.object(telegram_jarvis, "ALLOWED_USERS", {99999}):
            self.assertFalse(telegram_jarvis._is_authorized(mock_update))


if __name__ == "__main__":
    unittest.main()
