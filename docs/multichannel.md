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
2.7 GB of VRAM). On CPU with sherpa-onnx installed and `--lang en`, finals
run on sherpa-onnx Parakeet v3 int8 and partials on sherpa-onnx Moonshine tiny
int8 (about 640 MB of models, downloaded to `~/.cache/realtimestt/sherpa-onnx`
on first start). Otherwise faster-whisper `large-v3-turbo` + `small.en` on GPU
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
| `ready` | Listening. | `channels` (with `input_device` actually opened), `input_devices`, `pulse_sources`, `engines`, `settings`, `protocol` |
| `speech_start` | Voice detected; new `utterance_id`. | `speech_start_ms` |
| `realtime` | Partial text, re-transcribed from the utterance start. | `text` |
| `realtime_stabilized` | Partial whose prefix is unlikely to change. | `text` |
| `speech_paused` | Silence began. The utterance ends if it lasts `post_speech_silence_duration`. | |
| `speech_resumed` | Speech continued before that. | |
| `speech_end` | Utterance audio complete; final transcription queued. | `speech_start_ms`, `speech_end_ms` |
| `transcription_start` | Final transcription running. | `audio_s` |
| `fullSentence` | Final text. | `text`, `speech_start_ms`, `speech_end_ms`, `latency_ms` (since `speech_end`) |
| `channel` | Reply to `set_channel`. | `input_device` |
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
| `shutdown` | | Stop everything and exit. Closing stdin does the same. |

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
