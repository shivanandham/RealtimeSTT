"""
Creates transcription engine instances by configured backend name.
"""

import importlib.util
from importlib import import_module

from .base import TranscriptionEngineConfig, UnsupportedTranscriptionEngineError


ENGINE_CLASS_PATHS = {
    "faster_whisper": (".faster_whisper_engine", "FasterWhisperEngine"),
    "whisper_cpp": (".whisper_cpp_engine", "WhisperCppEngine"),
    "openai_whisper": (".openai_whisper_engine", "OpenAIWhisperEngine"),
    "openai_api": (".openai_api_engine", "OpenAIAPIEngine"),
    "parakeet": (".parakeet_engine", "ParakeetEngine"),
    "nvidia_parakeet": (".parakeet_engine", "ParakeetEngine"),
    "cohere_transcribe": (".cohere_transcribe_engine", "CohereTranscribeEngine"),
    "cohere": (".cohere_transcribe_engine", "CohereTranscribeEngine"),
    "granite_speech": (".granite_speech_engine", "GraniteSpeechEngine"),
    "granite": (".granite_speech_engine", "GraniteSpeechEngine"),
    "qwen3_asr": (".qwen3_asr_engine", "Qwen3ASREngine"),
    "qwen_asr": (".qwen3_asr_engine", "Qwen3ASREngine"),
    "omnilingual_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "omnilingual": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "meta_omnilingual_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "omni_asr": (".omnilingual_asr_engine", "OmnilingualASREngine"),
    "moonshine": (".moonshine_engine", "MoonshineEngine"),
    "moonshine_streaming": (".moonshine_engine", "MoonshineEngine"),
    "sherpa_onnx_parakeet": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "sherpa_parakeet": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "parakeet_sherpa_onnx": (".sherpa_onnx_engine", "SherpaOnnxParakeetEngine"),
    "sherpa_onnx_moonshine": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "sherpa_moonshine": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "moonshine_sherpa_onnx": (".sherpa_onnx_engine", "SherpaOnnxMoonshineEngine"),
    "kroko_onnx": (".kroko_onnx_engine", "KrokoOnnxEngine"),
    "kroko": (".kroko_onnx_engine", "KrokoOnnxEngine"),
    "banafo_kroko": (".kroko_onnx_engine", "KrokoOnnxEngine"),
    "funasr": (".funasr_engine", "FunASREngine"),
    "whistle": (".whistle_engine", "WhistleEngine"),
    "cactus_whistle": (".whistle_engine", "WhistleEngine"),
}


# Package each engine module imports at load time (None: nothing beyond the core install).
ENGINE_MODULE_PACKAGES = {
    ".faster_whisper_engine": "faster_whisper",
    ".whisper_cpp_engine": "pywhispercpp",
    ".openai_whisper_engine": "whisper",
    ".openai_api_engine": None,
    ".parakeet_engine": "nemo",
    ".cohere_transcribe_engine": "transformers",
    ".granite_speech_engine": "transformers",
    ".qwen3_asr_engine": "qwen_asr",
    ".omnilingual_asr_engine": "omnilingual_asr",
    ".moonshine_engine": "transformers",
    ".sherpa_onnx_engine": "sherpa_onnx",
    ".kroko_onnx_engine": "kroko_onnx",
    ".funasr_engine": "funasr",
    ".whistle_engine": "needle",
}


def normalize_engine_name(name):
    """
    Returns the registered spelling of an engine name (aliases keep their own entry).
    """
    return (name or "faster_whisper").strip().lower().replace("-", "_")


def engine_package(name):
    """
    Returns the package an engine needs, or None. Raises for unknown engines.
    """
    normalized_name = normalize_engine_name(name)
    if normalized_name not in ENGINE_CLASS_PATHS:
        raise UnsupportedTranscriptionEngineError(f"Unsupported transcription engine '{name}'")
    return ENGINE_MODULE_PACKAGES.get(ENGINE_CLASS_PATHS[normalized_name][0])


def engine_installed(name):
    """
    Reports whether an engine's package is importable here, without importing it.
    """
    package = engine_package(name)
    return package is None or importlib.util.find_spec(package) is not None


def _load_engine_class(name):
    """
    Loads the engine class registered for a normalized name.
    """
    module_name, class_name = ENGINE_CLASS_PATHS[name]
    module = import_module(module_name, package=__package__)
    return getattr(module, class_name)


def create_transcription_engine(name, config: TranscriptionEngineConfig):
    """
    Creates a transcription engine for a configured backend name.
    """
    normalized_name = normalize_engine_name(name)
    if normalized_name not in ENGINE_CLASS_PATHS:
        available_engines = ", ".join(sorted(ENGINE_CLASS_PATHS))
        raise UnsupportedTranscriptionEngineError(
            f"Unsupported transcription engine '{name}'. Available engines: {available_engines}"
        )
    engine_cls = _load_engine_class(normalized_name)
    return engine_cls(config)


def get_supported_transcription_engines():
    """
    Returns the sorted list of supported engine names.
    """
    return sorted(ENGINE_CLASS_PATHS)
