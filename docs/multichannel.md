# Multi-Channel Transcription

Transcribe several audio sources at once with one loaded model set. Each
channel has its own input, voice activity detection and utterance state;
inference is shared, so two channels cost one model's memory plus VAD.

Typical use: a meeting or interview assistant that transcribes the local
microphone ("me") and the remote party heard through the speakers or
headphones ("them") as two labelled streams.

Two surfaces:

- `stt-runner`: a command for programs that embed RealtimeSTT as a child
  process. JSON lines on stdout, commands on stdin.
- `RealtimeSTT.MultiChannelRecorder` and `RealtimeSTT.SharedEngine`: the
  Python API behind it.

## stt-runner

```bash
stt-runner --channel name=me pulse_source=alsa_input.usb-0c76_USB_PnP_Audio_Device-00.mono-fallback \
           --channel name=them pulse_source=@DEFAULT_MONITOR@ \
           --silence_timing --early_transcription_on_silence 0.15
```

From a source checkout: `PYTHONPATH=. python -m RealtimeSTT_server.stt_runner ...`.

Each `--channel` takes `KEY=VALUE` tokens:

| Key | Meaning |
| --- | --- |
| `name` | Channel name carried on every event (required). |
| `device` | PortAudio input: an index, or a name substring such as `USB PnP` (case-insensitive). Prefer names; indices change between processes. |
| `pulse_source` | PulseAudio/PipeWire source: a microphone (`alsa_input....`), an output's monitor (`....monitor`), or `@DEFAULT_SOURCE@` / `@DEFAULT_MONITOR@`. Linux only. |
| anything else | A per-channel recorder setting, for example `silero_sensitivity=0.2`. |

`stt-runner --list-devices` prints the PortAudio inputs and the PulseAudio
sources this machine offers; the `ready` event carries the same lists.

Model selection follows `stt-server`: `--device auto` picks CUDA when
available, and on CUDA with NVIDIA NeMo installed and `--lang en` the final
and realtime work both run on `nvidia/parakeet-tdt-0.6b-v2` (one model, about
2.7 GB of VRAM). On CPU with sherpa-onnx installed and `--lang en`, final
and realtime work both run on sherpa-onnx Parakeet v2 int8 (about 480 MB,
downloaded to `~/.cache/realtimestt/sherpa-onnx` on first start); add
`--hotwords "JSON, SQL, Kubernetes"` (or a file with one term per line) to bias
it towards your vocabulary. Otherwise faster-whisper `large-v3-turbo` + `small.en` on GPU
or `small` + `tiny.en` on CPU. `--engine`, `--rt-engine`, `-m`, `-r`,
`--use_main_model_for_realtime` and `--no-realtime` override this.

Timing flags are forwarded to every channel's recorder when given:
`--post_speech_silence_duration`, `--early_transcription_on_silence` (seconds),
`--min_length_of_recording`, `--realtime_processing_pause`,
`--silero_sensitivity`, `--webrtc_sensitivity`, `--silero_deactivity_detection`.
`--silence_timing` shortens the end-of-utterance silence to
`--end_of_sentence_detection_pause` (0.45 s) when the partial text ends a
sentence, uses `--unknown_sentence_detection_pause` (0.7 s) otherwise and
`--mid_sentence_detection_pause` (2.0 s) after a trailing `...`.

### Events (stdout)

One JSON object per line. Lines that are not JSON are library noise and can
be ignored. Every channel event carries `channel`, `utterance_id` (per
channel; the partials and the final of one utterance share it) and `t_ms`
(wall clock, milliseconds).

| `type` | When | Extra fields |
| --- | --- | --- |
| `ready` | Listening. | `channels` (with `input_device` actually opened), `input_devices`, `pulse_sources`, `engines`, `active`, `benchmark`, `models`, `host`, `settings`, `protocol` |
| `speech_start` | Voice detected; new `utterance_id`. | `speech_start_ms` |
| `realtime` | Partial text, re-transcribed from the utterance start. | `text` |
| `realtime_stabilized` | Partial whose prefix is unlikely to change. | `text` |
| `speech_paused` | Silence began. The utterance ends if it lasts `post_speech_silence_duration`. | |
| `speech_resumed` | Speech continued before that. | |
| `speech_end` | Utterance audio complete; final transcription queued. | `speech_start_ms`, `speech_end_ms` |
| `transcription_start` | Final transcription running. | `audio_s` |
| `fullSentence` | Final text. | `text`, `speech_start_ms`, `speech_end_ms`, `latency_ms` (since `speech_end`) |
| `channel` | Reply to `set_channel`. | `input_device` |
| `engine` | Progress of `set_engine`, `set_hotwords`, `set_initial_prompt`: `status` `loading`, then `ready`. | `command`, `status`, `lanes` (lanes whose engine changed), `active` |
| `info` | Reply to `get_info`; same shape as `ready`. | |
| `ack` | Command accepted. | `command`, `channel` |
| `warning` | Non-fatal problem, e.g. a requested device could not be opened and the default was used. | `channel`, `message` |
| `error` | Failed command, or a fatal startup error. | `message` |

`speech_paused` arrives at the start of the pause, roughly
`post_speech_silence_duration` before `fullSentence`; a consumer that wants to
act early can start on it and confirm with the final text. With
`--early_transcription_on_silence`, the final text is usually ready the moment
the silence threshold passes (`latency_ms` of a few milliseconds).

### Commands (stdin)

One JSON object per line; `request_id` is echoed in the reply when given.
Bare words `pause`, `resume`, `clear`, `shutdown` apply to all channels.

| Command | Fields | Effect |
| --- | --- | --- |
| `pause` / `resume` | `channel` (optional) | Stop or resume reading audio. Models stay loaded. |
| `clear` | `channel` (optional) | Drop queued audio. |
| `set_channel` | `channel`, `device` and/or `pulse_source`, `recorder_kwargs` | Start or replace a channel. Only that channel's capture and VAD restart (well under a second); replies with a `channel` event. |
| `remove_channel` | `channel` | Stop a channel. |
| `set_parameter` | `parameter`, `value`, `channel` (optional) | Change a recorder setting at runtime; the allowed names are in `protocol.settable_parameters`. |
| `get_info` | | Reply with an `info` event. |
| `set_engine` | `lane` (`final` or `realtime`), then `engine` and/or `model`, or `default: true`, or `same_as_final: true` (realtime only) | Switch that lane's model without restarting channels; replies with `engine` events or an `error`. |
| `set_hotwords` | `hotwords` (comma-separated string, list or file path; `null` clears) | Replace the sherpa-onnx Parakeet hotwords; affected models reload. |
| `set_initial_prompt` | `initial_prompt` (string or `null`), `lane` (default `final`) | Replace the Whisper-style prompt; applied without a reload. |
| `shutdown` | | Stop everything and exit. Closing stdin does the same. |

### Models and switching them at runtime

`ready` and `info` describe the models:

- `active`: what is running. `transcription_engine`, `model`,
  `realtime_transcription_engine`, `realtime_model`,
  `use_main_model_for_realtime`, `enable_realtime_transcription`, `device`,
  `compute_type`, `language`, `hotwords`, `initial_prompt`,
  `initial_prompt_realtime`; `choices` (each lane's mode: `default`,
  `explicit` with `engine`/`model`, `same_as_final` or `off`); `defaults`
  (what `default` resolves to now, for example
  `{"final": {"engine": "sherpa_onnx_parakeet", "model": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"}, "realtime": {"same_as_final": true}}`);
  `loading` (lanes whose new model is still loading).
- `models`: the rows of `RealtimeSTT_server/model_benchmarks.json` (latency,
  WER, `default_for`, `notes`), each with `installed` (the engine's package is
  importable in the runner's Python) and `measured_on_this_host` (the row's
  hardware matches `host`). `benchmark` describes the measurement.
- `host`: `cpu` and `gpu` names of this machine.

`set_engine` loads the new model while the old one keeps transcribing, then
swaps it in between requests; the replaced model is unloaded once its queued
work is done. Requests run one at a time in order.

```
> {"command": "set_engine", "lane": "final", "engine": "faster_whisper", "model": "small.en", "request_id": 1}
< {"type": "engine", "status": "loading", "command": "set_engine", "lanes": ["final", "realtime"], "active": {..., "loading": {"final": {...}, "realtime": {...}}}, "request_id": 1}
< {"type": "engine", "status": "ready", "command": "set_engine", "lanes": ["final", "realtime"], "active": {...}, "request_id": 1}
> {"command": "set_engine", "lane": "realtime", "same_as_final": true, "request_id": 2}
> {"command": "set_engine", "lane": "final", "default": true, "request_id": 3}
```

A realtime lane left on `default` is re-resolved when the final model
changes: it shares a Parakeet final model and gets faster-whisper `tiny.en`
(CPU) or `small.en` (GPU) next to a Whisper one. An engine whose package is
not installed is rejected at once; a model that fails to load replies with an
`error` (with `command` and `active`) and the old models stay. Hotwords and
prompts carry over to the new models. `--engine_options` and
`--rt_engine_options` only apply while the lane runs the engine they were
given for.

## Python API

```python
from RealtimeSTT import ChannelSpec, MultiChannelRecorder, SharedEngine
from RealtimeSTT.transcription_engines import TranscriptionEngineConfig

engine = SharedEngine("parakeet", TranscriptionEngineConfig(
    model="nvidia/parakeet-tdt-0.6b-v2", device="cuda")).wait_ready()

recorder = MultiChannelRecorder(
    [ChannelSpec("me", input_device_index="USB PnP"),
     ChannelSpec("them", pulse_source="@DEFAULT_MONITOR@")],
    on_event=print,                 # dicts, same shape as the runner's events
    final_engine=engine,
    realtime_engine=engine,         # None disables partials; a second SharedEngine uses another model
    language="en",
    post_speech_silence_duration=0.5,
)
recorder.pause("them")
recorder.set_channel(ChannelSpec("me", pulse_source="@DEFAULT_SOURCE@"))
recorder.info()
recorder.shutdown()
```

`SharedEngine` loads one engine on its own thread and serializes requests,
serving final transcriptions before realtime partials. `engine.lane("final")`
and `engine.lane("realtime")` are executors for `AudioToTextRecorder`'s
`transcription_executor` / `realtime_transcription_executor`, which is how
`MultiChannelRecorder` wires the recorders; a recorder with both executors
loads no models of its own. Streaming engines (Kroko) fall back to
re-transcribing the growing utterance when used through a shared lane.

Keyword arguments other than `final_engine`, `realtime_engine` and
`recorder_factory` go to every `AudioToTextRecorder`; `ChannelSpec.recorder_kwargs`
overrides them for one channel.

## Audio input notes

- On PipeWire/PulseAudio systems, capture microphones through `pulse_source`
  rather than their ALSA `hw:` device. PipeWire holds a hardware device while
  any client records from it, which makes the direct device disappear from
  PortAudio's list and fail to open for a second program. Pulse sources are
  shared.
- PortAudio device indices differ between processes (devices that are busy or
  absent are left out). Select by name, and read `input_device` on the
  `ready` / `channel` events to see what was actually opened.
- `AudioToTextRecorder.input_device_info` exposes the same information for a
  single recorder, and `AudioToTextRecorder(pulse_source=...)` opens a Pulse
  source directly.
- Every channel adds a capture thread and two VADs; inference is shared and
  serialized, so channels that speak at the same time queue behind each other
  (finals first). The `engines` section of `info` reports queue delay and
  inference time per lane.
