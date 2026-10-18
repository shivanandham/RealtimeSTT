"""
Adapts Cactus Whistle (cactus-needle) models to the transcription engine interface.
"""

from importlib import import_module

from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


WHISTLE_LANGUAGES = ("en", "de", "fr", "es", "it", "nl", "pl")


class CactusWhistleBackend:
    """
    Wraps a needle.Whistle model instance.
    """

    def __init__(self, config, model_cls=None):
        """
        Loads the Whistle model, from a custom .cact file when one is configured.
        """
        engine_options = dict(config.engine_options or {})
        self.keywords = engine_options.get("keywords")
        weights = engine_options.get("weights")
        if weights is None and str(config.model or "").endswith(".cact"):
            weights = config.model
        model_cls = model_cls or self._load_model_class()
        self.model = model_cls(weights=weights)

    @staticmethod
    def _load_model_class():
        """
        Loads the optional needle Whistle class.
        """
        try:
            module = import_module("needle")
        except ModuleNotFoundError as exc:
            raise TranscriptionEngineError(
                "The 'whistle' transcription engine requires the optional "
                "'cactus-needle' package. Install it with 'pip install cactus-needle' "
                "or select a different transcription engine."
            ) from exc
        return module.Whistle

    def transcribe(self, audio, language=None):
        """
        Runs Whistle transcription and returns its result dict.
        """
        return self.model.transcribe(audio, language=language, keywords=self.keywords)


class WhistleEngine(BaseTranscriptionEngine):
    """
    Transcribes audio with Cactus Whistle.
    """

    engine_name = "whistle"

    def __init__(self, config, backend=None, backend_cls=None):
        """
        Initializes the Whistle engine backend.
        """
        super().__init__(config)
        self.backend = backend or (backend_cls or CactusWhistleBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes audio and returns normalized Whistle output.
        """
        audio = self._normalize_audio(audio)
        # Whistle detects the language itself when given one it does not support.
        whistle_language = language if language in WHISTLE_LANGUAGES else None
        result = self.backend.transcribe(audio, language=whistle_language)
        detected = result.get("language") or whistle_language
        return TranscriptionResult(
            text=(result.get("text") or "").strip(),
            info=TranscriptionInfo(
                language=detected,
                language_probability=1.0 if detected else 0.0,
            ),
        )
