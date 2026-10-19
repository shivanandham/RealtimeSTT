# Release Notes

## Unreleased

### Added

- Added multi-channel transcription: `RealtimeSTT.SharedEngine` loads one
  model on one inference thread and serves many recorders (finals before
  partials); `RealtimeSTT.MultiChannelRecorder` runs several inputs with their
  own VAD/utterance state on those shared engines and emits channel-tagged
  events with utterance ids and timestamps. See `docs/multichannel.md`.
- Added `stt-runner`, a JSON-lines child process for embedding programs:
  channels by device name/index or PulseAudio source, `ready`/`speech_start`/
  `realtime`/`speech_paused`/`speech_end`/`fullSentence` events, and
  `pause`/`resume`/`clear`/`set_channel`/`set_parameter`/`get_info`/`shutdown`
  commands. `--list-devices` lists PortAudio inputs and Pulse sources.
- `AudioToTextRecorder` accepts `pulse_source` (capture a PulseAudio/PipeWire
  source such as an output's monitor), accepts a device name for
  `input_device_index`, and reports the device actually opened through
  `input_device_info`.
- Added the `whistle` transcription engine (Cactus Whistle through
  `cactus-needle`, CPU only) and the `whistle` install extra.
- Added `tests/asr_engine_benchmark.py`, which measures realtime-style latency,
  final-call latency and WER per engine, and
  `RealtimeSTT_server/model_benchmarks.json` with results from an RTX 3070
  Laptop GPU and i7-11370H.
- Added the `get_info` control command to `stt-server`. It reports the active
  engine, models and device, each connected data client's sample rate and
  microphone, the startup configuration, and the benchmarked models.
  `AudioToTextRecorderClient` now sends its input device with each audio chunk.
- Added `--engine` and `--rt-engine` to `stt-server` to select any transcription
  engine.

### Changed

- `stt-server` now defaults to `--device auto`. On GPU it uses
  `nvidia/parakeet-tdt-0.6b-v2` for final and realtime transcription when
  NVIDIA NeMo is installed and the language is English, otherwise
  `large-v3-turbo` (final) and `small.en` (realtime) in float16. CPU defaults
  stay `small` / `tiny.en` / int8.
- On CPU with sherpa-onnx installed (`pip install "realtimestt[sherpa-onnx]"`)
  and language `en`, `stt-server` and `stt-runner` now default to sherpa-onnx
  Parakeet v3 int8 for final transcription and sherpa-onnx Moonshine tiny int8
  for realtime. On an Apple M4 that is 3.9% WER at about 250 ms per final
  (faster-whisper `small.en`: 6.0% at about 2.1 s) and about 20 ms per partial
  under 5 s (`tiny.en`: about 150 ms). Without sherpa-onnx, or for other
  languages, the CPU defaults stay faster-whisper `small` / `tiny.en` / int8.
- The sherpa-onnx engines download and extract the Parakeet v3 int8 and
  Moonshine tiny int8 bundles on first use (into `download_root` or
  `~/.cache/realtimestt/sherpa-onnx`; disable with `auto_download_model`).
  sherpa-onnx Parakeet now defaults to 4 threads instead of 1.
- `model_benchmarks.json` adds Apple M4 CPU measurements (entries carry a
  `hardware` field). On macOS the pywhispercpp wheel runs `whisper_cpp` on the
  Metal GPU even with `device="cpu"`; both are listed.
- faster-whisper loads cuBLAS/cuDNN from `nvidia-*` pip wheels on CUDA, so
  `LD_LIBRARY_PATH` is no longer needed for them.

### Fixed

- Several recorders in one process no longer crash PortAudio on start
  (`InitializeHostApis` assertion): capture workers now serialize PortAudio
  initialization, device probing and stream opens.
- Corrected the `early_transcription_on_silence` docstring: the value is in
  seconds, not milliseconds.
- Microphones that reject 16 kHz (for example USB mics that only support
  48 kHz) no longer silently fall back to the default input device; the stream
  opens at a supported rate and is resampled.
- `stt-server` no longer forces the legacy torch.hub Silero loader, which needed
  a download and an interactive trust prompt and blocked headless startup.

## 1.0.2 - 2026-05-31

### Changed

- Split the `AudioToTextRecorder` implementation behind the existing public
  facade into focused core modules for lifecycle, recording, realtime
  transcription, voice activity, wake-word handling, initialization, shutdown,
  and formatting.
- Refreshed recorder architecture and compatibility documentation so the public
  facade, threading boundaries, and regression checks are easier to audit.
- Normalized package docstrings and comments to use block-style summaries and
  focused runtime explanations.

### Removed

- Removed an unused internal console color helper from `audio_recorder.py`.

## 1.0.1 - 2026-05-20

### Added

- Added a generic streaming transcription session interface so engines can
  opt in to incremental realtime decoding while existing engines keep the
  full-buffer fallback behavior.
- Added `kroko_onnx` transcription engine for Kroko/Banafo `.data` streaming
  models.
- Added Kroko realtime preview support that feeds streaming engines only newly
  recorded audio frames through a persistent session.
- Added `stt-install-kroko`, exposed through the `kroko-builder` extra, to help
  build and install Kroko-ONNX for the active Python environment.
- Added focused Kroko and realtime streaming unit coverage plus a public manual
  `tests/realtimestt_kroko_test.py` smoke script.
- Added `omnilingual_asr` transcription engine for Meta
  Omnilingual ASR on Linux/WSL2 with Python 3.11.x, with support for the
  published CTC and LLM model cards. Native Windows is not supported because
  `fairseq2n` has no Windows wheel; Python 3.12.x is blocked by upstream
  `omnilingual-asr` package metadata.
- Added `docs/licenses.md` with engine and model-family license notes.

### Changed

- Kroko Community models with known public filenames can be auto-downloaded
  into the RealtimeSTT cache when `auto_download_model` is enabled.
- Kroko final transcription remains one-shot. The new streaming path is used
  for realtime previews only when the realtime engine advertises streaming
  support.
- Kroko model cadence is used to choose automatic finalization tail padding.
- Omnilingual ASR uses `omniASR_CTC_1B_v2` as the default when the
  recorder is still configured with a Whisper default model name.
- Omnilingual in-memory audio is passed to the backend as predecoded waveform
  dictionaries to avoid the upstream package treating raw float arrays as
  encoded audio bytes.

### Notes

- Install/build Kroko-ONNX separately with
  `pip install "RealtimeSTT[kroko-builder,silero-onnx-cpu]"` followed by
  `stt-install-kroko --build`, or install a compatible Kroko-ONNX wheel in the
  same Python environment. The `silero-onnx-cpu` extra provides the local VAD
  backend used by recorder-based Kroko smoke tests and live microphone use.
- Licensed Pro models require a Pro-capable Kroko wheel and a key supplied at
  runtime through configuration, CLI, or environment variables. Do not commit
  keys, Pro models, generated logs, local wheels, or local cache contents.
- `Pro-16-L` is the recommended realtime model for the fastest partials. Local
  private validation observed the expected low-latency partial behavior, but
  exact cadence depends on runtime, provider, hardware, and scheduling.
- `suppress_native_output=True` redirects Kroko native stdout/stderr during
  recognizer calls and sets `KROKO_ONNX_SUPPRESS_LICENSE_OUTPUT=1`. Reliable
  suppression of asynchronous Pro license refresh messages requires a Kroko
  wheel rebuilt with RealtimeSTT's native quiet-output patch; older Kroko wheels
  may still print background license status text.
- Omnilingual ASR support is optional and Linux/WSL2 Python 3.11.x-oriented.
  Native Windows installs are not supported by the upstream dependency stack at
  this time, and Python 3.12.x cannot resolve the current upstream package.
- `omniASR_CTC_1B_v2` is the recommended Omnilingual starting point for local
  realtime tests in this release. Smaller/larger CTC and LLM models are exposed
  through model-card plumbing but require their own quality and memory checks.
