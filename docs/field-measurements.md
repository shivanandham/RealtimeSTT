# Field measurements (from free-cluely)

Raw numbers measured while running `stt-runner` inside free-cluely, the
interview assistant that drives it (github.com/shivanandham/free-cluely). They
complement `RealtimeSTT_server/model_benchmarks.json`, which measures engines
on LibriSpeech; these measure the runner's settings on real use. Hardware:
Apple Silicon MacBook Air, CPU only, sherpa-onnx Parakeet v2 int8 unless a row
says otherwise. The scripts named are in the free-cluely repo.

## End-of-utterance latency vs `post_speech_silence_duration`

12 Nov 2026, `scripts/check-stt-silence.js`. 12 real interview questions made
with `say -o`, streamed at real time into a `command=` channel with free-cluely's
arguments, 1.5 s apart, a 0.5 s mid-sentence pause in four of them.
`early_transcription_on_silence` 0.15 s. 60 sentences per setting, on a machine
loaded by other runners. Time from the end of the audio to the final.

| silence (s) | median (s) | p75 (s) | p90 (s) | end detected after (s) |
|---|---|---|---|---|
| 0.6 | 0.93 | 1.26 | 1.75 | 0.88 |
| 0.45 | 0.79 | 1.19 | 1.48 | 0.72 |
| 0.35 | 0.88 | 1.29 | 1.56 | 0.62 |
| 0.25 | 2.57 | 3.70 | 4.71 | 0.53 |

- The recorder sees silence about 0.28 s late, so end detection is the setting + 0.28 s.
- The rest is the final model. It costs nothing while the early transcription finishes inside the silence window. At 0.35 it often doesn't under load (median +0.26 s). On a quiet machine: 0.35 → 0.72 s, 0.45 → 0.75 s. Starting the early transcription earlier (0.05 s) didn't help.
- A 0.5 s pause inside a sentence: no split at any setting ≥ 0.35 s. 0.25 split 4 of 24 and piled up the final model. Expect splits from ~0.6 s pauses at 0.35 and ~0.75 s at 0.45.
- A 0.5 s pause between two whole sentences came out as two finals: 0/60 at 0.6, 3/60 at 0.45, 5/60 at 0.35.
- Live interview (12 Nov 2026, interviewer channel at 0.6): last partial → final median 0.80 s, p90 3.0 s.
- free-cluely now runs the interviewer channel at 0.45 and the mic at 0.6.

## Vocabulary: initial prompt and hotwords

`scripts/check-vocab.js`, 6 technical interview sentences, words right per sentence.

| model | prompt / hotwords | sentences fully right |
|---|---|---|
| faster_whisper small.en | term sheet as `initial_prompt` | 6/6 |
| faster_whisper small (multilingual) | same | 4/6 |
| faster_whisper large-v3-turbo | same | 3/6 |
| Parakeet | none (no prompt support then) | 1/6 |

- Whisper keeps only the last ~223 prompt tokens. Term lists run ~2.8 characters per token, so free-cluely caps the prompt at 450 characters (≈160 tokens).
- The end of the prompt weighs most: "Celery" was misheard mid-list and right at the end.
- Tiny realtime models given the full prompt hallucinate listed words; they get 200 characters.
- Parakeet `hotwords`: free-cluely sends ~900 terms (sheet + interview-focus terms + its software list). The 12 Nov 2026 interview's misses with the generic sheet: "Glangra Farney" for LangGraph, "Fast Airflow" for FastAPI, "Taitan" for Python, "flows and sets" for frozenset, "Atta operator" for decorator.

## Echo cancellation on the mic channel (`command=` channel, WebRTC AEC3)

The mic is captured by free-cluely's `audiotap mic --aec` helper with the
interviewer's process tap as the echo reference. MacBook Air speakers + built-in mic.

| measure | raw mic | with AEC3 |
|---|---|---|
| speaker → mic delay | 32–43 ms | — |
| interviewer-only words heard in the mic (Whisper scoring) | 12/21 | 0/21 |
| candidate's words recognised (Whisper scoring) | 64% | 61–62% |
| replayed through the runner (Parakeet + vocabulary): interviewer words | — | 0 |
| same: candidate's words | — | 38/49 |

- A residual gate after the canceller cut the starts of the candidate's sentences (38/49 → 27/49) and was removed.
- The first seconds of the first echo, while the canceller adapts, can still yield a stray word.
- Bluetooth headphones drop to 16 kHz when a call opens their mic. A helper that keeps converting at the start rate delivers a third of the audio at 3× speed, unusable for STT. The helper must follow the device's real rate.

## Named POSIX semaphores (macOS)

`stt-runner` creates named semaphores through spawn-context multiprocessing
Locks/Queues. Only the resource tracker unlinks them, and it lives in the
runner's process group.

| how the runner was stopped | names leaked per runner |
|---|---|
| SIGKILL to the whole process group | 11 |
| SIGTERM to the group (the tracker cleans up on the runner's exit) | 0 |

- macOS can't list or free leaked names until a reboot. After a day of test runners killed by group SIGKILL (12 Nov 2026) the table had 22–33 free names, and every new runner failed with `OSError: [Errno 28] No space left on device` from `mp.Event()`.
- The default `kern.posix.sem.max` is 10000. Raising it with `sudo sysctl -w kern.posix.sem.max=20000` (root) works at once without a reboot.
- Stop a runner with SIGTERM to the group, then SIGKILL to the leader only.
