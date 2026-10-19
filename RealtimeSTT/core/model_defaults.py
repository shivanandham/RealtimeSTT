"""
Default engine, model and compute-type selection per device.

Shared by stt-server and stt-runner so both pick the same models. The values
come from tests/asr_engine_benchmark.py (see RealtimeSTT_server/model_benchmarks.json).
"""

import importlib.util


# (final model, realtime model, compute type) per device, for the faster_whisper engine.
# On CPU this is the fallback when sherpa-onnx is not installed or the language is not English.
DEVICE_DEFAULTS = {
    'cuda': ('large-v3-turbo', 'small.en', 'float16'),
    'cpu': ('small', 'tiny.en', 'int8'),
}


# Best measured on GPU (2.9% WER, ~65 ms per call), but English-only and needs NVIDIA NeMo.
PARAKEET_MODEL = 'nvidia/parakeet-tdt-0.6b-v2'

# Best measured on CPU (Apple M4): Parakeet v3 int8 has 3.9% WER at ~250 ms per final call, and
# Moonshine tiny int8 returns partials in ~20 ms. Both are downloaded on first use.
SHERPA_PARAKEET_MODEL = 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8'
SHERPA_MOONSHINE_MODEL = 'sherpa-onnx-moonshine-tiny-en-int8'


def apply_device_defaults(args):
    """
    Fills in device, engine, model and compute-type defaults on an argparse namespace.

    Expects the attributes device, engine, rt_engine, model, rt_model,
    compute_type, lang and use_main_model_for_realtime; None means "not given".
    """
    if args.device == 'auto':
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, rt_model, compute_type = DEVICE_DEFAULTS['cuda' if args.device == 'cuda' else 'cpu']

    if (args.engine is None and args.model is None and args.device == 'cuda'
            and (args.lang or 'en') == 'en' and importlib.util.find_spec('nemo') is not None):
        args.engine, args.model = 'parakeet', PARAKEET_MODEL
        if args.rt_engine is None and args.rt_model is None and args.use_main_model_for_realtime is None:
            # Parakeet is fast enough for realtime too; sharing one loaded model saves VRAM.
            args.use_main_model_for_realtime = True
    elif (args.engine is None and args.model is None and args.device != 'cuda'
            and (args.lang or 'en') == 'en' and importlib.util.find_spec('sherpa_onnx') is not None):
        args.engine, args.model = 'sherpa_onnx_parakeet', SHERPA_PARAKEET_MODEL
        if args.rt_engine is None and args.rt_model is None and args.use_main_model_for_realtime is None:
            # A separate tiny model keeps partials fast and never queues finals behind them.
            args.rt_engine, args.rt_model = 'sherpa_onnx_moonshine', SHERPA_MOONSHINE_MODEL
    args.engine = args.engine or 'faster_whisper'
    args.use_main_model_for_realtime = bool(args.use_main_model_for_realtime)
    if args.engine == 'faster_whisper':
        args.model = args.model or model

    # Whisper realtime model names only load in faster-whisper, so a non-Whisper main engine
    # gets a faster-whisper realtime engine unless it also does the realtime work.
    if args.rt_engine is None and args.engine != 'faster_whisper' and not args.use_main_model_for_realtime:
        args.rt_engine = 'faster_whisper'
    if (args.rt_engine or args.engine) == 'faster_whisper':
        args.rt_model = args.rt_model or rt_model
    args.compute_type = args.compute_type or compute_type
    return args
