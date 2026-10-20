# RealtimeSTT Capabilities

A one-page summary of what RealtimeSTT offers, written for people and agents
integrating it from another program. Details live in the linked documents.

RealtimeSTT is a local, low-latency speech-to-text library (Python 3.11+,
Linux/macOS/Windows, GPU via CUDA). It turns live audio into realtime partial
text and a final transcript per utterance; voice activity detection decides
where utterances start and end.

## Ways to use it

1. **Python library**: `RealtimeSTT.AudioToTextRecorder` (one input) and
   `RealtimeSTT.MultiChannelRecorder` + `SharedEngine` (several inputs, one
   model set). See [configuration.md](configuration.md) and
   [multichannel.md](multichannel.md).
2. **`stt-runner`** (child process, JSON lines):
   `stt-runner --channel name=<n> [device=<index|name substring>] [pulse_source=<src>] [command=<cmdline>] ...`,
   or from a checkout `PYTHONPATH=<repo> <venv python> -m RealtimeSTT_server.stt_runner`.
   Runs any number of channels on one shared model set. See
   [multichannel.md](multichannel.md).
3. **`stt-server`** (WebSockets, ports 8011 control / 8012 data) with a
   `get_info` command that returns the active models, device, benchmarks and
   the full protocol. One recorder per server. See
   [RealtimeSTT_server/README.md](../RealtimeSTT_server/README.md).
4. **`example_fastapi_server`**: multi-session browser server. See
   [fastapi-server.md](fastapi-server.md).

## Multi-channel transcription

- Each channel is an independent audio input with its own VAD and utterance
  state: a microphone (PortAudio index or case-insensitive name substring) or
  a PulseAudio/PipeWire source (`pulse_source`: a microphone source, an
  output's monitor, `@DEFAULT_SOURCE@`, `@DEFAULT_MONITOR@`; Linux only), or
  the raw PCM stdout of a helper process the runner starts and owns
  (`command`, with `pcm_rate` / `pcm_channels` / `pcm_format` `s16le|f32le`;
  for example a macOS process-tap helper).
- All channels share one loaded model for final and realtime work, so N
  channels cost one model's memory plus VAD. Final requests are served before
  partials.
- Events (`stt-runner` stdout, one JSON object per line; library: dicts passed
  to `on_event`). All carry `channel`, `utterance_id` (per channel; the
  partials and the final of one utterance share it) and `t_ms` (wall clock ms):

  | Event | Meaning |
  | --- | --- |
  | `ready` | Listening. Channels with the input device actually opened, `input_devices`, `pulse_sources`, `engines`, `settings`, `protocol`. |
  | `speech_start` | Voice detected, new `utterance_id`; `speech_start_ms`. |
  | `realtime` | Partial text (`text`), re-transcribed from the utterance start. |
  | `realtime_stabilized` | Partial whose prefix is unlikely to change (`text`). |
  | `speech_paused` | Silence began; the utterance ends if it lasts `post_speech_silence_duration`. |
  | `speech_resumed` | Speech continued before that. |
  | `speech_end` | Utterance audio complete; `speech_start_ms`, `speech_end_ms`. |
  | `transcription_start` | Final transcription running; `audio_s`. |
  | `fullSentence` | Final text; `text`, `speech_start_ms`, `speech_end_ms`, `latency_ms` since `speech_end`. |
  | `channel`, `info`, `ack`, `warning`, `error` | Command replies and problems. |

  Non-JSON lines on stdout are library noise and can be ignored.
- Commands (`stt-runner` stdin, JSON per line, `request_id` echoed; bare
  words `pause`/`resume`/`clear`/`shutdown` apply to all channels):

  | Command | Effect |
  | --- | --- |
  | `pause`, `resume`, `clear` | Per channel or all; models stay loaded. |
  | `set_channel` | Start or replace a channel (`device` and/or `pulse_source`, or `command_line` + `pcm_*`, `recorder_kwargs`). Only that channel restarts (about 0.5 s); replies with `channel`. |
  | `remove_channel` | Stop a channel. |
  | `set_parameter` | Change a recorder setting at runtime (names in `protocol.settable_parameters`, e.g. `post_speech_silence_duration`, `language`). |
  | `get_info` | Same payload as `ready`. |
  | `shutdown` | Exit; closing stdin does the same. |
- `stt-runner --list-devices` prints the PortAudio inputs and the PulseAudio
  sources (`name`, `description`, `monitor` true/false).

## Models and defaults

- `--device auto` uses CUDA when available. On CUDA with NVIDIA NeMo installed
  and language `en`, final and realtime both run `nvidia/parakeet-tdt-0.6b-v2`
  (one model, about 2.7 GB of VRAM, about 65 ms per call standalone, 2.9% WER
  on a LibriSpeech sample). On CPU with sherpa-onnx installed and language
  `en`, final and realtime both run sherpa-onnx Parakeet v2 int8 (one model,
  3.1% WER, about 270 ms per final and 130 ms per partial under 5 s on an
  Apple M4); it downloads (about 480 MB) to `~/.cache/realtimestt/sherpa-onnx`
  on first start. Otherwise faster-whisper
  `large-v3-turbo` + `small.en` on GPU, or `small` + `tiny.en` on CPU.
  Overrides: `--engine`,
  `--rt-engine`, `-m`, `-r`, `--compute_type`, `--use_main_model_for_realtime`,
  `--no-realtime`.
- Other engines: `whisper_cpp`, `openai_whisper`, `moonshine`, `granite_speech`,
  `qwen3_asr`, `whistle`, `kroko_onnx`, `sherpa_onnx_*`. See
  [transcription-engines.md](transcription-engines.md). Measured latency and
  WER per model: [`RealtimeSTT_server/model_benchmarks.json`](../RealtimeSTT_server/model_benchmarks.json)
  (also returned by `stt-server`'s `get_info`).
- `--initial_prompt` / `--initial_prompt_realtime` bias Whisper engines toward a
  vocabulary; Parakeet has no prompt and ignores them. For sherpa-onnx
  Parakeet use `--hotwords "JSON, SQL, Kubernetes"` (or a file with one term
  per line): on accented, jargon-heavy speech a list of about 875 software
  terms cut WER from 24.5% to 18.8%.

## Timing controls

Flags on `stt-runner`, keyword arguments on the recorders:

- `post_speech_silence_duration` (default 0.6 s) ends an utterance and is the
  floor on final-text latency.
- `early_transcription_on_silence` (seconds) starts the final transcription
  during the pause, so `fullSentence` arrives within milliseconds of
  `speech_end`.
- `--silence_timing` shortens the pause to 0.45 s when the partial text ends a
  sentence, 0.7 s otherwise and 2.0 s after a trailing `...` (adjustable).
- `realtime_processing_pause`, `min_length_of_recording`, `silero_sensitivity`,
  `webrtc_sensitivity` and `silero_deactivity_detection` are also exposed.

## Audio input facts

- PortAudio device indices differ between processes; names are stable. Every
  channel reports the device it actually opened (`ready` / `channel` events,
  `AudioToTextRecorder.input_device_info`), and a `warning` event is sent when
  a requested device could not be opened and the default was used instead.
- On PipeWire/PulseAudio systems, a hardware microphone held by another
  recording client disappears from PortAudio and cannot be opened directly;
  Pulse sources are shared between programs.
- stderr carries library logs (NeMo, ALSA, PortAudio noise); stdout is the
  event stream only.
