"""
Benchmarks transcription engines for realtime and final transcription.

Simulates the realtime loop (re-transcribing the growing utterance every --step
seconds) and reports per-call latency, the latency of the final full-clip call,
and WER on the full clips.

Usage:
    pip install jiwer  # plus each engine's own package
    python tests/asr_engine_benchmark.py [audio_dir] [--step 1.0] [--only whistle,tiny]

audio_dir holds <id>.wav files with matching <id>.txt transcripts
(default: tests/unit/audio).
"""

import argparse
import re
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from RealtimeSTT.transcription_engines import TranscriptionEngineConfig, create_transcription_engine

SAMPLE_RATE = 16000

def _fw(model, device="cuda"):
    compute_type = "float16" if device == "cuda" else "int8"
    return ("faster_whisper", dict(model=model, device=device, compute_type=compute_type, beam_size=3, batch_size=16))


def _gpu(engine, model):
    return (engine, dict(model=model, device="cuda"))


ENGINES = {
    # Mirrors the stt-server realtime defaults (tiny.en, int8, CPU, beam 3, batch 16).
    "faster_whisper tiny.en CPU": _fw("tiny.en", "cpu"),
    "faster_whisper base.en CPU": _fw("base.en", "cpu"),
    "faster_whisper tiny.en": _fw("tiny.en"),
    "faster_whisper base.en": _fw("base.en"),
    "faster_whisper small.en": _fw("small.en"),
    "faster_whisper medium.en": _fw("medium.en"),
    "faster_whisper distil-large-v3": _fw("distil-large-v3"),
    "faster_whisper large-v3-turbo": _fw("large-v3-turbo"),
    # Whistle's engine is CPU-only.
    "whistle CPU": ("whistle", dict(model="whistle")),
    "moonshine streaming-tiny": _gpu("moonshine", "UsefulSensors/moonshine-streaming-tiny"),
    "moonshine streaming-medium": _gpu("moonshine", "UsefulSensors/moonshine-streaming-medium"),
    "parakeet tdt-0.6b-v2": _gpu("parakeet", "nvidia/parakeet-tdt-0.6b-v2"),
    "parakeet tdt-0.6b-v3": _gpu("parakeet", "nvidia/parakeet-tdt-0.6b-v3"),
    "qwen3_asr 0.6B": _gpu("qwen3_asr", "Qwen/Qwen3-ASR-0.6B"),
    "qwen3_asr 1.7B": _gpu("qwen3_asr", "Qwen/Qwen3-ASR-1.7B"),
    "granite speech-4.1-2b": _gpu("granite_speech", "ibm-granite/granite-speech-4.1-2b"),
    "cohere transcribe": _gpu("cohere_transcribe", "CohereLabs/cohere-transcribe-03-2026"),
}


def load_clips(audio_dir):
    clips = []
    for wav in sorted(Path(audio_dir).glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        rate, data = wavfile.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data / 32768.0
        if rate != SAMPLE_RATE:
            data = resample_poly(data, SAMPLE_RATE, rate)
        clips.append((data.astype(np.float32), txt.read_text().strip()))
    return clips


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def bench(engine, clips, step):
    engine.transcribe(clips[0][0][:SAMPLE_RATE], language="en", use_prompt=False)  # warmup
    latencies, short, long, final, refs, hyps = [], [], [], [], [], []
    for audio, ref in clips:
        cut = int(step * SAMPLE_RATE)
        while True:
            cut = min(cut, len(audio))
            start = time.perf_counter()
            text = engine.transcribe(audio[:cut], language="en", use_prompt=False).text
            ms = (time.perf_counter() - start) * 1000
            latencies.append(ms)
            (short if cut < 5 * SAMPLE_RATE else long).append(ms)
            if cut == len(audio):
                final.append(ms)
                break
            cut += int(step * SAMPLE_RATE)
        hyps.append(normalize(text))
        refs.append(normalize(ref))
    latencies.sort()
    return {
        "short_p50_ms": statistics.median(short),
        "long_p50_ms": statistics.median(long),
        "final_p50_ms": statistics.median(final),
        "calls": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "wer": jiwer.wer(refs, hyps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", nargs="?", default=str(Path(__file__).parent / "unit" / "audio"))
    parser.add_argument("--step", type=float, default=1.0, help="Seconds of new audio between realtime calls.")
    parser.add_argument("--only", default="", help="Comma-separated substrings of engine labels to run.")
    args = parser.parse_args()
    wanted = [w.strip() for w in args.only.split(",") if w.strip()]

    clips = load_clips(args.audio_dir)
    seconds = sum(len(audio) for audio, _ in clips) / SAMPLE_RATE
    width = 32
    print(f"{len(clips)} clips, {seconds:.0f}s of audio, step {args.step}s\n")
    print(f"{'engine':<{width}}{'calls':>7}{'mean ms':>10}{'p50 ms':>9}{'p95 ms':>9}{'<5s p50':>9}{'>=5s p50':>10}{'final p50':>11}{'WER':>8}")
    for label, (name, options) in ENGINES.items():
        if wanted and not any(w in label for w in wanted):
            continue
        try:
            engine = create_transcription_engine(name, TranscriptionEngineConfig(**options))
            r = bench(engine, clips, args.step)
        except Exception as e:
            print(f"{label:<{width}}FAILED: {type(e).__name__}: {str(e)[:150]}")
            continue
        print(f"{label:<{width}}{r['calls']:>7}{r['mean_ms']:>10.1f}{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}{r['short_p50_ms']:>9.1f}{r['long_p50_ms']:>10.1f}{r['final_p50_ms']:>11.1f}{r['wer']:>8.2%}")


if __name__ == "__main__":
    main()
