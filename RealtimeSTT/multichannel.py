"""
Several audio channels transcribed by one shared model set.

Each channel is an AudioToTextRecorder with its own input (a microphone index,
a PulseAudio source such as an output's monitor, or the raw PCM stdout of a
helper process) and its own VAD and recording state. Final and realtime inference go through SharedEngine lanes,
so the models are loaded once per process no matter how many channels run.
"""

import itertools
import logging
import os
import queue
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .transcription_engines import TranscriptionEngineConfig, create_transcription_engine


logger = logging.getLogger("realtimestt")

WARMUP_AUDIO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "warmup_audio.wav"
)
SAMPLE_RATE = 16000

_FINAL_PRIORITY = 0
_REALTIME_PRIORITY = 1
_DRAIN_PRIORITY = 2
_STOP = object()

# Raw PCM sample formats a command channel can read: numpy dtype per format.
PCM_FORMATS = {"s16le": "<i2", "f32le": "<f4"}


def _now_ms():
    """
    Returns the wall-clock time in milliseconds.
    """
    return int(time.time() * 1000)


def _to_ms(seconds):
    """
    Converts a time.time() value to milliseconds, keeping None.
    """
    return None if not seconds else int(seconds * 1000)


class _LaneStats:
    """
    Tracks call count, queue wait and inference time for one lane.
    """

    def __init__(self):
        self.calls = 0
        self.queue_ms = 0.0
        self.run_ms = 0.0
        self.max_run_ms = 0.0

    def record(self, queue_seconds, run_seconds):
        """
        Adds one completed call.
        """
        self.calls += 1
        self.queue_ms += queue_seconds * 1000
        self.run_ms += run_seconds * 1000
        self.max_run_ms = max(self.max_run_ms, run_seconds * 1000)

    def snapshot(self):
        """
        Returns the averages as a dict.
        """
        calls = max(1, self.calls)
        return {
            "calls": self.calls,
            "mean_queue_ms": round(self.queue_ms / calls, 1),
            "mean_ms": round(self.run_ms / calls, 1),
            "max_ms": round(self.max_run_ms, 1),
        }


class SharedEngine:
    """
    One transcription engine on one inference thread, shared by many recorders.

    Requests are serialized; final-transcription requests are served before
    realtime ones. Plug lanes into AudioToTextRecorder with
    transcription_executor=engine.lane("final") and
    realtime_transcription_executor=engine.lane("realtime").
    """

    def __init__(self, engine_name, config: TranscriptionEngineConfig, warmup=True,
                 engine_factory=create_transcription_engine):
        """
        Starts loading the engine on its own thread.
        """
        self.engine_name = engine_name
        self.config = config
        self.engine = None
        self.load_error = None
        self.ready = threading.Event()
        self.stats = {"final": _LaneStats(), "realtime": _LaneStats()}
        self._queue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._thread = threading.Thread(
            target=self._run,
            args=(warmup, engine_factory),
            name=f"RealtimeSTT-SharedEngine-{engine_name}",
            daemon=True,
        )
        self._thread.start()

    def _run(self, warmup, engine_factory):
        """
        Loads the engine, then serves queued requests until closed.
        """
        try:
            self.engine = engine_factory(self.engine_name, self.config)
            if warmup and os.path.exists(WARMUP_AUDIO_PATH):
                import soundfile as sf

                audio, _ = sf.read(WARMUP_AUDIO_PATH, dtype="float32")
                self.engine.warmup(audio)
        except Exception as exc:
            logger.exception("Could not load %s engine", self.engine_name)
            self.load_error = exc
        finally:
            self.ready.set()

        while True:
            _, _, item = self._queue.get()
            if item is _STOP:
                break
            future, audio, language, use_prompt, kind, queued_at = item
            started = time.monotonic()
            try:
                if self.load_error is not None:
                    raise RuntimeError(f"{self.engine_name} engine failed to load: {self.load_error}")
                result = self.engine.transcribe(audio, language=language, use_prompt=use_prompt)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            self.stats[kind].record(started - queued_at, time.monotonic() - started)

        while not self._queue.empty():
            _, _, item = self._queue.get_nowait()
            if item is not _STOP:
                item[0].set_exception(RuntimeError("shared engine closed"))

    def wait_ready(self, timeout=None):
        """
        Blocks until the engine is loaded; raises its load error if it failed.
        """
        if not self.ready.wait(timeout):
            raise TimeoutError(f"{self.engine_name} engine did not load within {timeout}s")
        if self.load_error is not None:
            raise self.load_error
        return self

    def transcribe(self, audio, language=None, use_prompt=True, kind="final"):
        """
        Queues one request and blocks until its result is available.
        """
        future = Future()
        priority = _FINAL_PRIORITY if kind == "final" else _REALTIME_PRIORITY
        self._queue.put((priority, next(self._seq), (future, audio, language, use_prompt, kind, time.monotonic())))
        return future.result()

    def lane(self, kind):
        """
        Returns an executor for AudioToTextRecorder that tags requests as final or realtime.
        """
        return _Lane(self, kind)

    def info(self):
        """
        Describes the engine and its lane statistics.
        """
        return {
            "engine": self.engine_name,
            "model": self.config.model,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "ready": self.ready.is_set(),
            "error": str(self.load_error) if self.load_error else None,
            "queued": self._queue.qsize(),
            "lanes": {kind: stats.snapshot() for kind, stats in self.stats.items()},
        }

    def close(self, drain=False):
        """
        Stops the inference thread; pending requests fail unless drain is set.

        With drain=True, requests already queued are served first (used when
        an engine is swapped out while channels keep running).
        """
        self._queue.put((_DRAIN_PRIORITY if drain else -1, -1, _STOP))
        self._thread.join(timeout=60 if drain else 10)


class _Lane:
    """
    Routes one recorder's transcribe() calls into a SharedEngine as one kind of work.
    """

    def __init__(self, engine, kind):
        """
        Binds the lane to its engine and kind.
        """
        self.engine = engine
        self.kind = kind

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes through the shared engine.
        """
        return self.engine.transcribe(audio, language=language, use_prompt=use_prompt, kind=self.kind)


class _SwitchableLane:
    """
    Routes a recorder's calls to whichever engine currently serves the lane.

    Recorders keep one lane object for their lifetime, so
    MultiChannelRecorder.set_engines can swap models without restarting channels.
    """

    def __init__(self, owner, kind):
        """
        Binds the lane to its MultiChannelRecorder and kind.
        """
        self.owner = owner
        self.kind = kind

    @property
    def engine(self):
        """
        Returns the SharedEngine serving this lane now.
        """
        return self.owner.final_engine if self.kind == "final" else self.owner.realtime_engine

    def transcribe(self, audio, language=None, use_prompt=True):
        """
        Transcribes through the current engine.
        """
        return self.engine.transcribe(audio, language=language, use_prompt=use_prompt, kind=self.kind)


@dataclass
class ChannelSpec:
    """
    One audio source: a PortAudio input index and/or a PulseAudio source, or a command.

    command is a command line (split with shlex, run without a shell) whose
    stdout carries raw little-endian PCM in pcm_format (s16le or f32le) at
    pcm_rate Hz with pcm_channels interleaved channels; the channel owns the
    process. recorder_kwargs override the MultiChannelRecorder's shared
    recorder settings for this channel only (for example a different
    silero_sensitivity for a monitor source that has no room noise).
    """

    name: str
    input_device_index: Optional[int] = None
    pulse_source: Optional[str] = None
    recorder_kwargs: Dict[str, Any] = field(default_factory=dict)
    command: Optional[str] = None
    pcm_rate: int = SAMPLE_RATE
    pcm_channels: int = 1
    pcm_format: str = "s16le"

    def validate(self):
        """
        Raises ValueError for a spec that cannot start.
        """
        if self.command is None:
            return
        if self.input_device_index is not None or self.pulse_source is not None:
            raise ValueError(f"channel {self.name!r}: command cannot be combined with device or pulse_source")
        if self.pcm_format not in PCM_FORMATS:
            raise ValueError(f"channel {self.name!r}: pcm_format must be one of {', '.join(PCM_FORMATS)}")
        if not (isinstance(self.pcm_rate, int) and self.pcm_rate > 0
                and isinstance(self.pcm_channels, int) and self.pcm_channels > 0):
            raise ValueError(f"channel {self.name!r}: pcm_rate and pcm_channels must be positive integers")
        argv = shlex.split(self.command)
        if not argv:
            raise ValueError(f"channel {self.name!r}: empty command")
        if shutil.which(argv[0]) is None:
            raise ValueError(f"channel {self.name!r}: command not found: {argv[0]}")

    def to_dict(self):
        """
        Returns the spec as plain data.
        """
        data = {
            "name": self.name,
            "input_device_index": self.input_device_index,
            "pulse_source": self.pulse_source,
            "recorder_kwargs": dict(self.recorder_kwargs),
            "command": self.command,
        }
        if self.command is not None:
            data.update(pcm_rate=self.pcm_rate, pcm_channels=self.pcm_channels, pcm_format=self.pcm_format)
        return data


def pcm_to_int16(data, pcm_format="s16le", channels=1, rate=SAMPLE_RATE):
    """
    Converts whole frames of raw PCM to 16 kHz mono int16 bytes.
    """
    if pcm_format == "s16le" and channels == 1 and rate == SAMPLE_RATE:
        return bytes(data)
    import numpy as np

    samples = np.frombuffer(data, dtype=PCM_FORMATS[pcm_format]).astype(np.float32)
    if pcm_format == "f32le":
        samples *= 32767.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(SAMPLE_RATE, rate)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


class _CommandSource:
    """
    Runs a channel's audio command and feeds its stdout PCM to the recorder.

    stdout is always read, also while paused (the audio is dropped then), so
    the pipe never fills up and blocks the process. stderr lines are logged
    and passed to on_message("warning", ...); an exit that stop() did not
    cause is reported as on_message("error", ...) with the last stderr lines.
    """

    CHUNK_SECONDS = 0.05
    STDERR_LINES = 20

    def __init__(self, spec, feed, on_message):
        """
        Starts the process and its stdout and stderr reader threads.
        """
        self.spec = spec
        self.feed = feed
        self.on_message = on_message
        self.paused = False
        self.stopping = False
        self.stderr_tail = deque(maxlen=self.STDERR_LINES)
        self.process = subprocess.Popen(
            shlex.split(spec.command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True,
                                               name=f"RealtimeSTT-command-{spec.name}-stderr")
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True,
                                               name=f"RealtimeSTT-command-{spec.name}")
        self._stderr_thread.start()
        self._stdout_thread.start()

    def info(self):
        """
        Describes the source in the shape of input_device_info, plus source and command.
        """
        return {
            "source": "command",
            "command": self.spec.command,
            "name": self.spec.command,
            "pid": self.process.pid,
            "sample_rate": self.spec.pcm_rate,
            "channels": self.spec.pcm_channels,
            "format": self.spec.pcm_format,
            "exit_code": self.process.poll(),
        }

    def _read_stdout(self):
        """
        Feeds whole frames to the recorder until stdout closes, then reports the exit.
        """
        spec = self.spec
        frame = spec.pcm_channels * (2 if spec.pcm_format == "s16le" else 4)
        size = max(frame, int(spec.pcm_rate * self.CHUNK_SECONDS) * frame)
        fd = self.process.stdout.fileno()
        pending = b""
        while True:
            try:
                data = os.read(fd, size)
            except OSError:
                break
            if not data:
                break
            if self.paused:
                pending = b""
                continue
            data = pending + data
            whole = len(data) - len(data) % frame
            pending = data[whole:]
            if not whole:
                continue
            try:
                self.feed(pcm_to_int16(data[:whole], spec.pcm_format, spec.pcm_channels, spec.pcm_rate))
            except Exception:
                logger.exception("Channel %s: feeding command audio failed", spec.name)
        code = self.process.wait()
        if self.stopping:
            return
        self._stderr_thread.join(timeout=1)
        tail = list(self.stderr_tail)
        message = f"audio command exited with code {code}"
        if tail:
            message += f": {tail[-1]}"
        self.on_message("error", message=message, exit_code=code, stderr=tail)

    def _read_stderr(self):
        """
        Logs and forwards the process's stderr lines.
        """
        for raw in iter(self.process.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            self.stderr_tail.append(line)
            logger.info("Channel %s command: %s", self.spec.name, line)
            if not self.stopping:
                self.on_message("warning", message=line, stderr=True)

    def stop(self, timeout=2.0):
        """
        Terminates the process (kills it after timeout) and joins the readers.
        """
        self.stopping = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not threading.current_thread():
                thread.join(timeout)
        for pipe in (self.process.stdout, self.process.stderr):
            try:
                pipe.close()
            except OSError:
                pass


class _Channel:
    """
    Holds one running channel.
    """

    def __init__(self, spec):
        """
        Starts with no recorder and no utterances.
        """
        self.spec = spec
        self.recorder = None
        self.source = None
        self.thread = None
        self.utterance_id = 0
        self.closing = False


class MultiChannelRecorder:
    """
    Transcribes several channels with one shared engine set.

    on_event receives one dict per event with at least type, channel,
    utterance_id and t_ms (wall clock, ms). Event types, in the order they
    occur for one utterance: speech_start, realtime (repeated),
    realtime_stabilized (repeated), speech_paused / speech_resumed (silence
    began / speech continued), speech_end, transcription_start, fullSentence.
    Partials and the final of one utterance share its utterance_id.

    Keyword arguments other than the ones named here are passed to every
    AudioToTextRecorder (language, VAD and timing settings, prompts, ...).
    """

    EVENT_TYPES = (
        "speech_start", "realtime", "realtime_stabilized", "speech_paused",
        "speech_resumed", "speech_end", "transcription_start", "fullSentence",
    )

    def __init__(self, channels: List[ChannelSpec], on_event: Callable[[dict], None],
                 final_engine: SharedEngine, realtime_engine: Optional[SharedEngine] = None,
                 recorder_factory=None, **recorder_kwargs):
        """
        Starts every channel. realtime_engine=None disables realtime transcription.
        """
        self.on_event = on_event
        self.final_engine = final_engine
        self.realtime_engine = realtime_engine
        self.recorder_kwargs = recorder_kwargs
        self._recorder_factory = recorder_factory
        self._channels: Dict[str, _Channel] = {}
        self._lock = threading.RLock()
        for spec in channels:
            self.set_channel(spec)

    def _make_recorder(self, **kwargs):
        """
        Creates a recorder with the configured factory (AudioToTextRecorder by default).
        """
        factory = self._recorder_factory
        if factory is None:
            from .audio_recorder import AudioToTextRecorder

            factory = AudioToTextRecorder
        return factory(**kwargs)

    # --- channel lifecycle ---

    def channel_names(self):
        """
        Returns the running channel names.
        """
        with self._lock:
            return list(self._channels)

    def set_channel(self, spec: ChannelSpec):
        """
        Starts a channel, replacing a running one with the same name.

        Only that channel's capture and VAD restart; the shared engines keep
        their loaded models, so switching inputs takes about a second. An
        invalid spec raises ValueError and leaves the running channel alone.
        """
        spec.validate()
        with self._lock:
            old = self._channels.pop(spec.name, None)
        if old is not None:
            self._stop_channel(old)

        channel = _Channel(spec)
        kwargs = dict(self.recorder_kwargs)
        kwargs.update(spec.recorder_kwargs)
        kwargs.update(
            use_microphone=spec.command is None,
            spinner=False,
            input_device_index=spec.input_device_index,
            pulse_source=spec.pulse_source,
            transcription_executor=_SwitchableLane(self, "final"),
            use_main_model_for_realtime=False,
            on_recording_start=lambda: self._on_recording_start(channel),
            on_recording_stop=lambda: self._on_recording_stop(channel),
            on_turn_detection_start=lambda: self._emit(channel, "speech_paused"),
            on_turn_detection_stop=lambda: self._on_turn_detection_stop(channel),
            on_transcription_start=lambda audio: self._on_transcription_start(channel, audio),
        )
        if self.realtime_engine is not None:
            kwargs.update(
                enable_realtime_transcription=True,
                realtime_transcription_executor=_SwitchableLane(self, "realtime"),
                on_realtime_transcription_update=lambda text: self._emit(channel, "realtime", text=text),
                on_realtime_transcription_stabilized=lambda text: self._emit(
                    channel, "realtime_stabilized", text=text
                ),
            )
        else:
            kwargs["enable_realtime_transcription"] = False

        channel.recorder = self._make_recorder(**kwargs)
        if spec.command is not None:
            try:
                channel.source = _CommandSource(
                    spec, channel.recorder.feed_audio,
                    lambda type_, **payload: self._emit(channel, type_, **payload),
                )
            except Exception:
                channel.closing = True
                channel.recorder.shutdown()
                raise
        channel.thread = threading.Thread(
            target=self._text_loop,
            args=(channel,),
            name=f"RealtimeSTT-channel-{spec.name}",
            daemon=True,
        )
        with self._lock:
            self._channels[spec.name] = channel
        channel.thread.start()
        return channel.recorder

    def remove_channel(self, name):
        """
        Stops one channel; the engines stay loaded.
        """
        with self._lock:
            channel = self._channels.pop(name, None)
        if channel is not None:
            self._stop_channel(channel)
        return channel is not None

    def _stop_channel(self, channel):
        """
        Shuts a channel's recorder down and waits for its text loop.
        """
        channel.closing = True
        if channel.source is not None:
            channel.source.stop()
        try:
            channel.recorder.shutdown()
        except Exception:
            logger.exception("Channel %s shutdown failed", channel.spec.name)
        if channel.thread is not None and channel.thread is not threading.current_thread():
            channel.thread.join(timeout=10)

    def set_engines(self, final_engine=None, realtime_engine=None):
        """
        Switches the engines serving the final and/or realtime lane.

        Running channels use the new engines from their next request on.
        Engines no longer serving any lane are closed in the background once
        their queued requests finish. The realtime lane can only be switched
        when the channels were started with a realtime engine.
        """
        with self._lock:
            if realtime_engine is not None and self.realtime_engine is None:
                raise ValueError("realtime transcription is disabled")
            previous = [self.final_engine, self.realtime_engine]
            if final_engine is not None:
                self.final_engine = final_engine
            if realtime_engine is not None:
                self.realtime_engine = realtime_engine
            unused = []
            for engine in previous:
                if (engine is not None and engine is not self.final_engine
                        and engine is not self.realtime_engine and engine not in unused):
                    unused.append(engine)
        for engine in unused:
            threading.Thread(target=engine.close, kwargs={"drain": True},
                             name="RealtimeSTT-SharedEngine-close", daemon=True).start()
        return unused

    def shutdown(self):
        """
        Stops every channel, then the shared engines.
        """
        for name in self.channel_names():
            self.remove_channel(name)
        engines = [self.final_engine]
        if self.realtime_engine is not None and self.realtime_engine is not self.final_engine:
            engines.append(self.realtime_engine)
        for engine in engines:
            engine.close()

    def wait_for_inputs(self, timeout=10.0):
        """
        Waits until every channel has opened its input device (or the timeout passes).

        Returns {channel: input_device_info or None}.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                channels = list(self._channels.values())
            inputs = {ch.spec.name: self._input_info(ch) for ch in channels}
            if all(inputs.values()) or time.monotonic() >= deadline:
                return inputs
            time.sleep(0.05)

    @staticmethod
    def _input_info(channel):
        """
        Returns what the channel listens to: the opened device, or its command source.
        """
        if channel.source is not None:
            return channel.source.info()
        return channel.recorder.input_device_info

    # --- per-channel controls ---

    def _select(self, channel):
        """
        Returns the channels a control applies to (all when channel is None).
        """
        with self._lock:
            if channel is None:
                return list(self._channels.values())
            return [self._channels[channel]]

    def pause(self, channel=None):
        """
        Stops reading audio on the channel(s) without unloading anything.
        """
        for ch in self._select(channel):
            if ch.source is not None:
                ch.source.paused = True
            else:
                ch.recorder.set_microphone(False)

    def resume(self, channel=None):
        """
        Resumes reading audio on the channel(s).
        """
        for ch in self._select(channel):
            if ch.source is not None:
                ch.source.paused = False
            else:
                ch.recorder.set_microphone(True)

    def clear(self, channel=None):
        """
        Drops queued audio on the channel(s).
        """
        for ch in self._select(channel):
            ch.recorder.clear_audio_queue()

    def set_parameter(self, name, value, channel=None):
        """
        Sets a recorder attribute (for example post_speech_silence_duration) on the channel(s).
        """
        for ch in self._select(channel):
            setattr(ch.recorder, name, value)

    def info(self):
        """
        Describes the channels, what they actually listen to, and the engines.
        """
        with self._lock:
            channels = list(self._channels.values())
        return {
            "channels": {
                ch.spec.name: {
                    **ch.spec.to_dict(),
                    "input_device": self._input_info(ch),
                    "paused": ch.source.paused if ch.source is not None else not bool(ch.recorder.use_microphone.value),
                    "recording": bool(getattr(ch.recorder, "is_recording", False)),
                    "utterance_id": ch.utterance_id,
                }
                for ch in channels
            },
            "engines": {
                "final": self.final_engine.info(),
                "realtime": self.realtime_engine.info() if self.realtime_engine is not None else None,
                "realtime_shares_final": self.realtime_engine is self.final_engine,
            },
        }

    # --- events ---

    def _emit(self, channel, type_, **payload):
        """
        Delivers one event unless the channel is shutting down.
        """
        if channel.closing:
            return
        event = {
            "type": type_,
            "channel": channel.spec.name,
            "utterance_id": channel.utterance_id,
            "t_ms": _now_ms(),
        }
        event.update(payload)
        try:
            self.on_event(event)
        except Exception:
            logger.exception("on_event failed for %s", type_)

    def _on_recording_start(self, channel):
        """
        Starts a new utterance.
        """
        channel.utterance_id += 1
        self._emit(channel, "speech_start", speech_start_ms=_to_ms(channel.recorder.recording_start_time))

    def _on_turn_detection_stop(self, channel):
        """
        Reports speech continuing after a pause (the recorder is still recording).
        """
        if getattr(channel.recorder, "is_recording", False):
            self._emit(channel, "speech_resumed")

    def _on_recording_stop(self, channel):
        """
        Reports the end of the utterance's audio.
        """
        recorder = channel.recorder
        self._emit(
            channel,
            "speech_end",
            speech_start_ms=_to_ms(recorder.recording_start_time),
            speech_end_ms=_to_ms(recorder.recording_stop_time),
        )

    def _on_transcription_start(self, channel, audio):
        """
        Reports the final transcription starting; never aborts it.
        """
        self._emit(channel, "transcription_start", audio_s=round(len(audio) / SAMPLE_RATE, 3))
        return False

    def _text_loop(self, channel):
        """
        Turns each finished utterance into a fullSentence event.
        """
        recorder = channel.recorder
        while not channel.closing and getattr(recorder, "is_running", True) and not getattr(recorder, "is_shut_down", False):
            try:
                text = recorder.text()
            except Exception:
                logger.exception("Channel %s text() failed", channel.spec.name)
                break
            if channel.closing or getattr(recorder, "is_shut_down", False):
                break
            if not text:
                continue
            speech_end_ms = _to_ms(getattr(recorder, "last_recording_stop_time", None))
            self._emit(
                channel,
                "fullSentence",
                text=text,
                speech_start_ms=_to_ms(getattr(recorder, "last_recording_start_time", None)),
                speech_end_ms=speech_end_ms,
                latency_ms=(_now_ms() - speech_end_ms) if speech_end_ms else None,
            )


__all__ = ["ChannelSpec", "MultiChannelRecorder", "SharedEngine", "pcm_to_int16"]
