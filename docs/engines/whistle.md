# Cactus Whistle

`whistle` uses Cactus Whistle through the optional `cactus-needle` package. It is a
very small (about 17 MB) CPU-only speech-to-text model for English, German,
French, Spanish, Italian, Dutch and Polish.

## Install

```bash
python -m pip install "RealtimeSTT[whistle]"
```

Then select the engine, typically as the realtime engine on CPU-only machines:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="small",
    device="cpu",
    enable_realtime_transcription=True,
    realtime_transcription_engine="whistle",
)
```

## Model Behavior

On first use `cactus-needle` downloads its native engine library and the base
Whistle weights from Hugging Face into `~/.cache/cactus-needle`. The `model`
value is ignored unless it ends in `.cact`, in which case it is loaded as tuned
weights.

The adapter uses the `needle.Whistle` class. Only `cactus-needle`'s module-level
`needle.transcribe()` helper sends anonymous usage counts, so RealtimeSTT does
not trigger them; set `NEEDLE_TELEMETRY=0` to disable them for other uses of
the package.

## Options

Pass options through `transcription_engine_options` or
`realtime_transcription_engine_options`:

| Option | Effect |
| --- | --- |
| `keywords` | List of words or phrases to bias recognition towards, such as names. |
| `weights` | Path to a tuned `.cact` weights file. |

Languages outside the seven supported ones fall back to Whistle's own language
detection.

## Performance

Measured with `tests/asr_engine_benchmark.py` on an Intel i7-11370H (see
`RealtimeSTT_server/model_benchmarks.json`): about 75 ms per realtime call while
an utterance is shorter than 5 s, at 10.0% WER on a LibriSpeech sample. Latency
grows with utterance length (about 750 ms at 5 s or more, up to about 3 s near
30 s), while Whisper pads every call to 30 s and stays flat. Whistle therefore
suits short utterances on CPU-only machines; on a GPU, faster-whisper or
Parakeet is faster.

## Troubleshooting

- If `needle` cannot be imported, install `cactus-needle` in the active
  environment.
- There is no GPU path; `device` is ignored.
- Audio longer than 30 s is split internally and may produce filler text on
  noisy recordings.
