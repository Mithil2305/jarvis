"""Lightweight tests for speech helper logic; no microphone or model download."""
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import jarvis_speech


class SpeechHelperTests(unittest.TestCase):
    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(jarvis_speech._text_chunks("  \n "), [])

    def test_short_text_is_preserved(self):
        self.assertEqual(jarvis_speech._text_chunks("Hello, Jarvis!"), ["Hello, Jarvis!"])

    def test_long_sentence_is_split_at_word_boundaries(self):
        text = "word " * 200
        chunks = jarvis_speech._text_chunks(text, limit=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_compute_type_keeps_supported_value(self):
        fake = types.SimpleNamespace(
            get_supported_compute_types=lambda device: ["int8", "float32"]
        )
        with patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(jarvis_speech._resolve_stt_compute_type("cpu", "int8"), "int8")

    def test_compute_type_falls_back_when_unsupported(self):
        fake = types.SimpleNamespace(
            get_supported_compute_types=lambda device: ["int8", "float32"]
        )
        with patch.dict(sys.modules, {"ctranslate2": fake}):
            self.assertEqual(jarvis_speech._resolve_stt_compute_type("cpu", "float16"), "int8")

    def test_status_does_not_load_models(self):
        self.assertIn("Faster-Whisper", jarvis_speech.backend_status())
        self.assertIsNone(jarvis_speech._STT_MODEL)
        self.assertEqual(jarvis_speech._KOKORO_PIPELINES, {})

    def test_kokoro_playback_uses_one_continuous_output_clip(self):
        played = []

        class FakePipeline:
            def __call__(self, text, *, voice, speed):
                yield text, "phonemes", np.array([0.1, 0.2], dtype=np.float32)

        fake_sounddevice = types.SimpleNamespace(
            stop=lambda: None,
            play=lambda audio, samplerate, blocking: played.append(
                (np.asarray(audio).copy(), samplerate, blocking)
            ),
        )
        previous_disabled = jarvis_speech._KOKORO_DISABLED
        try:
            jarvis_speech._KOKORO_DISABLED = False
            with patch.object(jarvis_speech, "_get_kokoro_pipeline", return_value=FakePipeline()), \
                 patch.object(jarvis_speech, "_text_chunks", return_value=["First sentence.", "Second sentence."]), \
                 patch.dict(sys.modules, {"sounddevice": fake_sounddevice}):
                self.assertTrue(jarvis_speech.speak("Two sentences."))
            self.assertEqual(len(played), 1)
            self.assertEqual(played[0][1:], (24000, True))
            self.assertEqual(played[0][0].size, 4)
        finally:
            jarvis_speech._KOKORO_DISABLED = previous_disabled

    def test_transcription_uses_and_cleans_up_a_closed_temp_wav(self):
        captured = {}

        class FakeSegment:
            text = " Open the browser. "
            no_speech_prob = 0.05
            avg_logprob = -0.2

        class FakeModel:
            def transcribe(self, path, **kwargs):
                captured["path"] = path
                captured["kwargs"] = kwargs
                self.assert_file_existed = os.path.isfile(path)
                return iter([FakeSegment()]), SimpleNamespace(language="en")

        previous_model = jarvis_speech._STT_MODEL
        jarvis_speech._STT_MODEL = FakeModel()
        audio = (np.sin(np.linspace(0, 30, 8000)) * 0.1).astype(np.float32)
        try:
            transcript = jarvis_speech.transcribe_audio(audio, sample_rate=16000)
            self.assertEqual(transcript, "Open the browser.")
            self.assertTrue(jarvis_speech._STT_MODEL.assert_file_existed)
            self.assertFalse(os.path.exists(captured["path"]))
            self.assertTrue(captured["kwargs"]["vad_filter"])
        finally:
            jarvis_speech._STT_MODEL = previous_model


if __name__ == "__main__":
    unittest.main()
