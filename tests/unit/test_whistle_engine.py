import unittest

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.whistle_engine import CactusWhistleBackend, WhistleEngine


class Audio:
    size = 1


class FakeWhistle:
    def __init__(self, weights=None):
        self.weights = weights
        self.calls = []

    def transcribe(self, audio, language=None, keywords=None):
        self.calls.append((language, keywords))
        return {"text": " hello world ", "language": language or "de"}


class WhistleEngineTests(unittest.TestCase):
    def make_engine(self, **engine_options):
        config = TranscriptionEngineConfig(model="tiny", engine_options=engine_options)
        backend = CactusWhistleBackend(config, model_cls=FakeWhistle)
        return WhistleEngine(config, backend=backend), backend.model

    def test_registered(self):
        self.assertIn("whistle", get_supported_transcription_engines())

    def test_transcribe_passes_supported_language_and_keywords(self):
        engine, model = self.make_engine(keywords=["Kokoro"])
        result = engine.transcribe(Audio(), language="en")
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.info.language, "en")
        self.assertEqual(model.calls, [("en", ["Kokoro"])])

    def test_unsupported_language_falls_back_to_detection(self):
        engine, model = self.make_engine()
        result = engine.transcribe(Audio(), language="ja")
        self.assertEqual(model.calls, [(None, None)])
        self.assertEqual(result.info.language, "de")

    def test_weights_from_engine_options_or_cact_model(self):
        _, model = self.make_engine(weights="tuned.cact")
        self.assertEqual(model.weights, "tuned.cact")
        config = TranscriptionEngineConfig(model="my.cact")
        self.assertEqual(CactusWhistleBackend(config, model_cls=FakeWhistle).model.weights, "my.cact")
        config = TranscriptionEngineConfig(model="tiny")
        self.assertIsNone(CactusWhistleBackend(config, model_cls=FakeWhistle).model.weights)


if __name__ == "__main__":
    unittest.main()
