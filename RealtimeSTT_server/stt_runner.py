"""
stt-runner: transcribes one or more audio channels and prints JSON lines.

For programs that embed RealtimeSTT as a child process. Every line on stdout
is one JSON event; commands arrive on stdin as JSON (or a bare word such as
"pause"). Library output goes to stderr, so stdout stays machine-readable.

    stt-runner --channel name=me device=4 --channel name=them pulse_source=@DEFAULT_MONITOR@

All channels share one loaded model set (see RealtimeSTT.multichannel).
"""

import argparse
import json
import logging
import signal
import sys
import threading

from RealtimeSTT.core.audio_input_worker import list_input_devices, list_pulse_sources
from RealtimeSTT.core.model_defaults import apply_device_defaults
from RealtimeSTT.multichannel import ChannelSpec, MultiChannelRecorder, SharedEngine
from RealtimeSTT.transcription_engines import TranscriptionEngineConfig


# Recorder settings a client may change at runtime with set_parameter.
SETTABLE_PARAMETERS = (
    "language",
    "post_speech_silence_duration",
    "min_length_of_recording",
    "min_gap_between_recordings",
    "early_transcription_on_silence",
    "silero_sensitivity",
    "webrtc_sensitivity",
    "realtime_processing_pause",
    "init_realtime_after_seconds",
)

# Recorder constructor options exposed as flags; forwarded only when given.
RECORDER_FLAGS = (
    ("post_speech_silence_duration", float, "Silence (s) that ends an utterance. Recorder default 0.6."),
    ("early_transcription_on_silence", float, "Start the final transcription this many seconds into the silence (0 = off)."),
    ("min_length_of_recording", float, "Minimum utterance length in seconds."),
    ("min_gap_between_recordings", float, "Minimum gap between utterances in seconds."),
    ("pre_recording_buffer_duration", float, "Audio kept before speech starts, in seconds."),
    ("realtime_processing_pause", float, "Seconds between realtime partials. Recorder default 0.2."),
    ("init_realtime_after_seconds", float, "Delay before the first realtime partial of an utterance."),
    ("silero_sensitivity", float, "Silero VAD sensitivity (0-1)."),
    ("webrtc_sensitivity", int, "WebRTC VAD aggressiveness (0-3)."),
)

# How to talk to the runner; included in the ready and info events.
PROTOCOL = {
    "stdout": "One JSON object per line. Lines that are not JSON are library noise and can be ignored.",
    "common_fields": {"channel": "channel name", "utterance_id": "per channel; partials and the final of one utterance share it", "t_ms": "wall clock, ms"},
    "events": {
        "ready": "Runner is listening. Carries channels (with the input device actually opened), input_devices (PortAudio inputs; pick by name), pulse_sources (PulseAudio/PipeWire mics and monitors for pulse_source), engines, settings and this protocol.",
        "speech_start": "Voice detected; a new utterance_id begins. speech_start_ms.",
        "realtime": "Partial text of the current utterance (text). Re-transcribed from the utterance start on each update.",
        "realtime_stabilized": "Partial text whose prefix is unlikely to change (text).",
        "speech_paused": "Silence began; the utterance ends if it lasts post_speech_silence_duration.",
        "speech_resumed": "Speech continued before the silence threshold passed.",
        "speech_end": "Utterance audio complete; final transcription queued. speech_start_ms, speech_end_ms.",
        "transcription_start": "Final transcription running (audio_s = utterance length).",
        "fullSentence": "Final text (text, speech_start_ms, speech_end_ms, latency_ms since speech_end).",
        "channel": "Reply to set_channel: channel, input_device.",
        "info": "Reply to get_info.",
        "ack": "Command accepted (command, channel?).",
        "warning": "Non-fatal problem (channel?, message).",
        "error": "Failed command or fatal startup problem (message, request_id?).",
    },
    "stdin": {
        "format": "One JSON object per line: {\"command\": ..., \"channel\": optional, \"request_id\": optional, echoed back}. Bare words pause/resume/clear/shutdown apply to all channels.",
        "pause": "Stop reading audio (channel or all). Models stay loaded.",
        "resume": "Resume reading audio (channel or all).",
        "clear": "Drop queued audio (channel or all).",
        "set_channel": "Start or replace a channel: channel, device (input index or name substring) and/or pulse_source, recorder_kwargs {}. Only that channel restarts; replies with a channel event.",
        "remove_channel": "Stop a channel: channel.",
        "set_parameter": "Set a recorder setting: parameter (one of settable_parameters), value, channel (optional = all).",
        "get_info": "Reply with an info event.",
        "shutdown": "Stop everything and exit.",
    },
    "settable_parameters": list(SETTABLE_PARAMETERS),
}


def coerce(value):
    """
    Turns a KEY=VALUE string value into bool, int, float or str.
    """
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_channel(tokens):
    """
    Builds a ChannelSpec from --channel KEY=VALUE tokens.

    Keys: name (required), device (PortAudio input index), pulse_source
    (PulseAudio source such as @DEFAULT_MONITOR@); anything else is a
    per-channel recorder setting.
    """
    values = {}
    for token in tokens:
        if "=" not in token:
            raise argparse.ArgumentTypeError(f"--channel expects KEY=VALUE, got {token!r}")
        key, value = token.split("=", 1)
        values[key.strip()] = coerce(value.strip())
    if "name" not in values:
        raise argparse.ArgumentTypeError("--channel needs name=<channel name>")
    return ChannelSpec(
        name=str(values.pop("name")),
        input_device_index=values.pop("device", None),
        pulse_source=values.pop("pulse_source", None),
        recorder_kwargs=values,
    )


def channel_from_command(command):
    """
    Builds a ChannelSpec from a set_channel command.
    """
    if not command.get("channel"):
        raise ValueError("set_channel needs a channel name")
    return ChannelSpec(
        name=str(command["channel"]),
        input_device_index=command.get("device"),
        pulse_source=command.get("pulse_source"),
        recorder_kwargs=dict(command.get("recorder_kwargs") or {}),
    )


def build_parser():
    """
    Defines the command line.
    """
    parser = argparse.ArgumentParser(
        prog="stt-runner",
        description="Transcribe one or more audio channels with a shared model set, printing JSON lines.",
    )
    parser.add_argument("--channel", action="append", nargs="+", default=[], metavar="KEY=VALUE",
                        help="A channel: name=<name> plus device=<input index or name substring> and/or "
                             "pulse_source=<source>. Repeat for more channels. "
                             "Default: one channel 'default' on the default microphone.")
    parser.add_argument("--list-devices", dest="list_devices", action="store_true",
                        help="Print the PortAudio input devices and PulseAudio sources as JSON and exit.")
    parser.add_argument("--engine", "--transcription_engine", default=None,
                        help="Final transcription engine. Default: parakeet on GPU with NeMo installed and English, else faster_whisper.")
    parser.add_argument("--rt-engine", "--realtime_transcription_engine", dest="rt_engine", default=None,
                        help="Realtime engine. Default follows --engine.")
    parser.add_argument("-m", "--model", default=None, help="Final model. Default per device, see stt-server.")
    parser.add_argument("-r", "--rt-model", "--realtime_model_type", dest="rt_model", default=None,
                        help="Realtime model. Default small.en on GPU, tiny.en on CPU (unused when the final model serves realtime).")
    parser.add_argument("-l", "--lang", "--language", dest="lang", default="en", help="Language code. Default en.")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu. Default auto.")
    parser.add_argument("--compute_type", default=None, help="CTranslate2 compute type. Default float16 on GPU, int8 on CPU.")
    parser.add_argument("--gpu_device_index", type=int, default=0)
    parser.add_argument("--download_root", default=None)
    parser.add_argument("--use_main_model_for_realtime", action="store_true", default=None,
                        help="Serve realtime partials from the final model (one model in memory).")
    parser.add_argument("--no-realtime", dest="no_realtime", action="store_true", help="Disable realtime partials.")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--beam_size_realtime", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--realtime_batch_size", type=int, default=16)
    parser.add_argument("--initial_prompt", default=None, help="Prompt/vocabulary for the final model.")
    parser.add_argument("--initial_prompt_realtime", default=None, help="Prompt for a separate realtime model.")
    parser.add_argument("--engine_options", default=None, help="JSON object passed to the final engine.")
    parser.add_argument("--rt_engine_options", default=None, help="JSON object passed to a separate realtime engine.")
    parser.add_argument("--faster_whisper_vad_filter", action=argparse.BooleanOptionalAction, default=True)
    for name, cast, help_text in RECORDER_FLAGS:
        parser.add_argument(f"--{name}", type=cast, default=None, help=help_text)
    parser.add_argument("--silero_deactivity_detection", action="store_true", default=None)
    parser.add_argument("--silence_timing", action="store_true",
                        help="Shorten the end-of-utterance silence when the partial text ends a sentence "
                             "(same rule as stt-server).")
    parser.add_argument("--end_of_sentence_detection_pause", type=float, default=0.45)
    parser.add_argument("--unknown_sentence_detection_pause", type=float, default=0.7)
    parser.add_argument("--mid_sentence_detection_pause", type=float, default=2.0)
    parser.add_argument("--ready_timeout", type=float, default=15.0,
                        help="Seconds to wait for the input devices to open before sending ready.")
    parser.add_argument("-D", "--debug", action="store_true", help="Debug logging on stderr.")
    return parser


def engine_config(args, realtime=False):
    """
    Builds the engine configuration for the final or realtime model.
    """
    options = args.rt_engine_options if realtime else args.engine_options
    return TranscriptionEngineConfig(
        model=args.rt_model if realtime else args.model,
        download_root=args.download_root,
        compute_type=args.compute_type,
        gpu_device_index=args.gpu_device_index,
        device=args.device,
        beam_size=args.beam_size_realtime if realtime else args.beam_size,
        initial_prompt=(args.initial_prompt_realtime or args.initial_prompt) if realtime else args.initial_prompt,
        batch_size=args.realtime_batch_size if realtime else args.batch_size,
        vad_filter=args.faster_whisper_vad_filter,
        engine_options=json.loads(options) if options else None,
    )


def recorder_kwargs(args):
    """
    Collects the recorder settings shared by every channel.
    """
    kwargs = {
        "language": args.lang,
        "no_log_file": True,
        "level": logging.DEBUG if args.debug else logging.WARNING,
    }
    for name, _, _ in RECORDER_FLAGS:
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.silero_deactivity_detection:
        kwargs["silero_deactivity_detection"] = True
    return kwargs


class SilenceTiming:
    """
    Picks the end-of-utterance silence from the partial text, like stt-server --silence_timing.

    A partial ending a sentence (after a sentence-ending previous partial)
    gets the short pause; one ending with "..." the long mid-sentence pause;
    anything else the unknown pause.
    """

    SENTENCE_END = (".", "!", "?", "。")

    def __init__(self, end_of_sentence=0.45, unknown=0.7, mid_sentence=2.0):
        """
        Stores the three pauses in seconds.
        """
        self.end_of_sentence = end_of_sentence
        self.unknown = unknown
        self.mid_sentence = mid_sentence
        self._previous = {}

    @staticmethod
    def _ellipsis(text):
        """
        Reports a partial trailing off.
        """
        return text.endswith("...") or (len(text) > 1 and text[:-1].endswith("..."))

    def _sentence_end(self, text):
        """
        Reports a partial that ends a sentence.
        """
        return bool(text) and text[-1] in self.SENTENCE_END

    def pause_for(self, channel, text):
        """
        Returns the silence duration to use after this partial.
        """
        previous = self._previous.get(channel, "")
        self._previous[channel] = text
        if self._ellipsis(text):
            return self.mid_sentence
        if self._sentence_end(text) and self._sentence_end(previous) and not self._ellipsis(previous):
            return self.end_of_sentence
        return self.unknown


class Runner:
    """
    Routes recorder events to stdout and stdin commands to the recorder.
    """

    def __init__(self, recorder, write, silence_timing=None, settings=None, ready_timeout=15.0):
        """
        Binds the runner to a MultiChannelRecorder (may be set later) and an event writer.
        """
        self.recorder = recorder
        self.write = write
        self.silence_timing = silence_timing
        self.settings = settings or {}
        self.ready_timeout = ready_timeout

    def on_event(self, event):
        """
        Applies silence timing to partials and writes the event.
        """
        if self.silence_timing is not None and event.get("type") == "realtime":
            pause = self.silence_timing.pause_for(event["channel"], event.get("text", ""))
            try:
                self.recorder.set_parameter("post_speech_silence_duration", pause, event["channel"])
            except KeyError:
                pass
        self.write(event)

    def info(self):
        """
        Describes the running setup.
        """
        return {
            **self.recorder.info(),
            "input_devices": self.list_devices(),
            "pulse_sources": list_pulse_sources(),
            "settings": self.settings,
            "protocol": PROTOCOL,
        }

    @staticmethod
    def list_devices():
        """
        Enumerates input devices; empty when PortAudio is unavailable.
        """
        try:
            return list_input_devices()
        except Exception as exc:
            logging.getLogger("realtimestt").warning("Could not list input devices: %s", exc)
            return []

    def check_inputs(self, inputs):
        """
        Warns when a channel opened something other than the requested device.
        """
        channels = self.recorder.info()["channels"]
        for name, opened in inputs.items():
            requested = channels.get(name, {}).get("input_device_index")
            if opened is None:
                self.write({"type": "warning", "channel": name, "message": "input device not opened yet; see stderr"})
            elif isinstance(requested, int) and opened["index"] != requested:
                self.write({"type": "warning", "channel": name,
                            "message": f"requested device {requested} could not be opened; using {opened['index']} ({opened['name']})"})
            elif isinstance(requested, str) and requested.lower() not in opened["name"].lower():
                self.write({"type": "warning", "channel": name,
                            "message": f"requested device {requested!r} not found; using {opened['index']} ({opened['name']})"})

    def _reply(self, event, request_id):
        """
        Writes a reply, echoing the request id when given.
        """
        if request_id is not None:
            event["request_id"] = request_id
        self.write(event)

    def handle(self, line):
        """
        Executes one stdin line. Returns False when the runner should exit.
        """
        line = line.strip()
        if not line:
            return True
        if line.startswith("{"):
            try:
                command = json.loads(line)
            except ValueError:
                self.write({"type": "error", "message": f"invalid JSON command: {line[:200]}"})
                return True
        else:
            command = {"command": line}
        name = command.get("command")
        channel = command.get("channel")
        request_id = command.get("request_id")
        try:
            if name in ("pause", "resume", "clear"):
                getattr(self.recorder, name)(channel)
                self._reply({"type": "ack", "command": name, "channel": channel}, request_id)
            elif name == "set_channel":
                spec = channel_from_command(command)
                self.recorder.set_channel(spec)
                inputs = self.recorder.wait_for_inputs(self.ready_timeout)
                self.check_inputs({spec.name: inputs.get(spec.name)})
                self._reply({"type": "channel", "channel": spec.name, "input_device": inputs.get(spec.name)}, request_id)
            elif name == "remove_channel":
                if not self.recorder.remove_channel(channel):
                    raise KeyError(channel)
                self._reply({"type": "ack", "command": name, "channel": channel}, request_id)
            elif name == "set_parameter":
                parameter = command.get("parameter")
                if parameter not in SETTABLE_PARAMETERS:
                    raise ValueError(f"parameter {parameter!r} is not settable; see settable_parameters")
                self.recorder.set_parameter(parameter, command.get("value"), channel)
                self._reply({"type": "ack", "command": name, "parameter": parameter, "channel": channel}, request_id)
            elif name == "get_info":
                self._reply({"type": "info", **self.info()}, request_id)
            elif name == "shutdown":
                self._reply({"type": "ack", "command": name}, request_id)
                return False
            else:
                raise ValueError(f"unknown command {name!r}")
        except KeyError as exc:
            self._reply({"type": "error", "message": f"unknown channel {exc.args[0]!r}"}, request_id)
        except Exception as exc:
            self._reply({"type": "error", "message": str(exc)}, request_id)
        return True


def main(argv=None):
    """
    Runs the channels until stdin closes or a shutdown command arrives.
    """
    args = build_parser().parse_args(argv)
    if args.list_devices:
        devices = {"input_devices": Runner.list_devices(), "pulse_sources": list_pulse_sources()}
        sys.stdout.write(json.dumps(devices, indent=1) + "\n")
        return 0
    apply_device_defaults(args)

    # Library prints (shutdown notices, model timings) must not corrupt the event stream.
    out = sys.stdout
    sys.stdout = sys.stderr
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, stream=sys.stderr)
    write_lock = threading.Lock()

    def write(event):
        """
        Writes one event line.
        """
        with write_lock:
            out.write(json.dumps(event) + "\n")
            out.flush()

    try:
        channels = [parse_channel(tokens) for tokens in args.channel] or [ChannelSpec("default")]
    except argparse.ArgumentTypeError as exc:
        write({"type": "error", "message": str(exc)})
        return 2

    final = SharedEngine(args.engine, engine_config(args))
    if args.no_realtime:
        realtime = None
    elif args.use_main_model_for_realtime:
        realtime = final
    else:
        realtime = SharedEngine(args.rt_engine or args.engine, engine_config(args, realtime=True))
    try:
        final.wait_ready()
        if realtime is not None and realtime is not final:
            realtime.wait_ready()
    except Exception as exc:
        write({"type": "error", "message": f"engine failed to load: {exc}"})
        return 1

    settings = {
        "engine": args.engine, "model": args.model,
        "realtime_engine": None if realtime is None else (args.engine if realtime is final else (args.rt_engine or args.engine)),
        "realtime_model": None if realtime is None else (args.model if realtime is final else args.rt_model),
        "device": args.device, "compute_type": args.compute_type, "language": args.lang,
        "silence_timing": args.silence_timing,
        "recorder": {k: v for k, v in recorder_kwargs(args).items() if k != "level"},
    }
    timing = None
    if args.silence_timing:
        timing = SilenceTiming(args.end_of_sentence_detection_pause, args.unknown_sentence_detection_pause,
                               args.mid_sentence_detection_pause)
    runner = Runner(None, write, timing, settings, args.ready_timeout)
    recorder = MultiChannelRecorder(channels, runner.on_event, final, realtime, **recorder_kwargs(args))
    runner.recorder = recorder

    inputs = recorder.wait_for_inputs(args.ready_timeout)
    write({"type": "ready", **runner.info()})
    runner.check_inputs(inputs)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def read_commands():
        """
        Handles stdin until it closes or shutdown is requested.
        """
        for line in sys.stdin:
            if not runner.handle(line):
                break
        stop.set()

    threading.Thread(target=read_commands, name="stt-runner-stdin", daemon=True).start()
    while not stop.wait(0.5):
        pass
    recorder.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
