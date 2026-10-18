# AGENTS.md

Orientation for coding agents working in this repository. Humans: start with
`README.md` and `docs/`.

## What this is

RealtimeSTT turns live audio into text with low latency. Audio (microphone or
`feed_audio()`) goes through wake-word detection and voice activity detection,
optional realtime transcription for live partial text, and a final
transcription when the speaker pauses. The internal audio format is 16 kHz
mono int16 PCM.

## Where things are

| Path | What it holds |
| --- | --- |
| `RealtimeSTT/audio_recorder.py` | `AudioToTextRecorder`, the public library API. Its constructor signature and defaults are pinned by `tests/unit/test_audio_recorder_public_api.py`; keep them backward compatible. |
| `RealtimeSTT/core/` | Recorder internals: initialization, microphone worker (`audio_input_worker.py`), VAD, realtime loop, final transcription. |
| `RealtimeSTT/transcription_engines/` | One adapter per ASR backend, registered by name in `factory.py`. Backends import lazily. |
| `RealtimeSTT/audio_recorder_client.py` | `AudioToTextRecorderClient`, which streams a microphone to `stt-server`. |
| `RealtimeSTT_server/stt_server.py` | `stt-server`: control WebSocket (default 8011) and data WebSocket (default 8012). |
| `RealtimeSTT/multichannel.py` | `SharedEngine` (one model, one inference thread, finals before partials) and `MultiChannelRecorder` (N inputs, one recorder each, shared engines, channel-tagged events). |
| `RealtimeSTT_server/stt_runner.py` | `stt-runner`: multi-channel JSON-lines child process for other programs (stdout events, stdin commands). Protocol in `docs/multichannel.md`. |
| `RealtimeSTT/core/model_defaults.py` | Device-based engine/model defaults shared by `stt-server` and `stt-runner`. |
| `RealtimeSTT_server/model_benchmarks.json` | Measured latency and WER per engine/model, served by `get_info`. |
| `example_fastapi_server/` | Multi-session FastAPI server (one recorder per session, shared models). |
| `tests/unit/` | Fast unit tests, no models or audio hardware needed. |
| `tests/asr_engine_benchmark.py` | Engine latency/WER benchmark that produces the benchmarks JSON. |
| `docs/` | `capabilities.md` (one-page summary for integrators), `module-map.md` (ownership and side effects), `api-compatibility.md`, `transcription-engines.md`, `engines/*.md`, `installation.md`, `testing.md`. |

## Running and testing

- Use the project virtualenv (`.venv/bin/python`). It is managed by `uv` and
  has no `pip`: install with `uv pip install --python .venv/bin/python <pkg>`.
- Unit tests: `.venv/bin/python -m unittest discover -s tests/unit -t .`
  (about 3 s; skipped tests are opt-in model tests).
- Run code from a checkout without installing: `PYTHONPATH=. .venv/bin/python ...`.
- Server: `PYTHONPATH=. .venv/bin/python -m RealtimeSTT_server.stt_server`.
- Multi-channel runner: `PYTHONPATH=. .venv/bin/python -m RealtimeSTT_server.stt_runner --channel name=me pulse_source=@DEFAULT_SOURCE@ --channel name=them pulse_source=@DEFAULT_MONITOR@`;
  `--list-devices` shows what can be opened.
  Use `-c`/`-d` to pick other ports when testing next to a running server.
- Benchmark: `PYTHONPATH=. .venv/bin/python tests/asr_engine_benchmark.py <dir of .wav + .txt> --only <label substrings>`.

## Talking to a running stt-server

Send `{"command": "get_info", "request_id": 1}` on the control WebSocket. The
reply describes the active engine, models and device, the connected audio
sources, the startup config, the benchmarked models, and the full protocol
(commands, allowed parameters and methods, audio packet layout, and the message
types sent on the data WebSocket). Prefer it over hardcoding assumptions. See
`RealtimeSTT_server/README.md` for the human-readable version.

## Adding a transcription engine

1. Add `RealtimeSTT/transcription_engines/<name>_engine.py` subclassing
   `BaseTranscriptionEngine` (`transcribe(audio, language, use_prompt)` takes
   float32 16 kHz mono and returns `TranscriptionResult`). Import the backend
   package lazily and raise `TranscriptionEngineError` with an install hint if
   it is missing.
2. Register the name and aliases in `factory.py`, and add an install extra in
   `setup.py` (and its docstring list).
3. Add `tests/unit/test_<name>_engine.py` with a fake backend.
4. Document it in `docs/engines/<name>.md`, `docs/transcription-engines.md` and
   `docs/installation.md`.
5. Add it to `ENGINES` in `tests/asr_engine_benchmark.py`, and record measured
   results in `RealtimeSTT_server/model_benchmarks.json`.

## Gotchas found the hard way

- `stt-server` has one recorder: every data client's audio is mixed into it.
  Run one server per audio source, or use `example_fastapi_server`.
- `stt-server` on GPU defaults to NVIDIA Parakeet (`nvidia/parakeet-tdt-0.6b-v2`)
  when NeMo is installed and the language is English; NeMo adds about 20 s to
  startup. Otherwise it uses faster-whisper `large-v3-turbo` / `small.en`.
- Some USB microphones only accept 48 kHz. The microphone worker must open the
  device at a rate it supports and resample, never fall back to another device.
- faster-whisper needs CUDA 12 cuBLAS/cuDNN. They are preloaded from the
  `nvidia-*` pip wheels, so do not rely on `LD_LIBRARY_PATH`.
- The legacy torch.hub Silero loader prompts for trust on stdin and blocks
  headless runs. Leave `silero_use_onnx` unset so the bundled ONNX model is used.
- PortAudio is not thread-safe to initialize: capture workers in one process
  (several recorders) must take `audio_input_worker._portaudio_lock` around
  `PyAudio()`, device probing and stream opens, never across retry sleeps.
- `PULSE_SOURCE` is process-wide and read when the ALSA `pulse` device opens;
  `_pulse_source_env` sets it under that lock and restores it. Capture workers
  are threads on Linux (`start_recorder_worker`), processes elsewhere.
- PortAudio device indices change between processes, and PipeWire hides a
  hardware mic while another client records from it. Select inputs by name or
  Pulse source and read `input_device_info` for what actually opened.
- Code style: block-style docstrings on public functions and classes, comment
  density like the surrounding code.
