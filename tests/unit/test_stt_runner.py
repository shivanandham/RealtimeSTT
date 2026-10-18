import argparse
import json
import unittest
from unittest.mock import patch

from RealtimeSTT.core.model_defaults import apply_device_defaults
from RealtimeSTT.multichannel import ChannelSpec
from RealtimeSTT_server.stt_runner import (
    PROTOCOL,
    Runner,
    SilenceTiming,
    build_parser,
    channel_from_command,
    engine_config,
    parse_channel,
    recorder_kwargs,
)


class FakeMulti:
    def __init__(self):
        self.calls = []
        self.names = {"me", "them"}

    def _check(self, channel):
        if channel is not None and channel not in self.names:
            raise KeyError(channel)

    def pause(self, channel=None):
        self._check(channel)
        self.calls.append(("pause", channel))

    def resume(self, channel=None):
        self._check(channel)
        self.calls.append(("resume", channel))

    def clear(self, channel=None):
        self._check(channel)
        self.calls.append(("clear", channel))

    def set_parameter(self, name, value, channel=None):
        self._check(channel)
        self.calls.append(("set", name, value, channel))

    def set_channel(self, spec):
        self.calls.append(("set_channel", spec))
        self.names.add(spec.name)

    def remove_channel(self, name):
        if name in self.names:
            self.names.remove(name)
            return True
        return False

    def wait_for_inputs(self, timeout):
        return {name: {"index": 4, "name": "dev", "sample_rate": 48000} for name in self.names}

    def info(self):
        requested = {"me": 4, "them": "pulse"}
        return {"channels": {name: {"input_device_index": requested.get(name)} for name in self.names}, "engines": {}}


class ChannelParsingTests(unittest.TestCase):
    def test_parse_channel_tokens(self):
        spec = parse_channel(["name=them", "pulse_source=@DEFAULT_MONITOR@", "silero_sensitivity=0.2", "device=4"])
        self.assertEqual(spec, ChannelSpec("them", 4, "@DEFAULT_MONITOR@", {"silero_sensitivity": 0.2}))
        self.assertEqual(parse_channel(["name=me", "silero_deactivity_detection=true"]).recorder_kwargs,
                         {"silero_deactivity_detection": True})
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_channel(["device=4"])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_channel(["name=me", "device"])

    def test_channel_from_command(self):
        spec = channel_from_command({"command": "set_channel", "channel": "me", "device": 5, "recorder_kwargs": {"webrtc_sensitivity": 2}})
        self.assertEqual(spec, ChannelSpec("me", 5, None, {"webrtc_sensitivity": 2}))
        with self.assertRaises(ValueError):
            channel_from_command({"command": "set_channel"})


class SilenceTimingTests(unittest.TestCase):
    def test_pause_follows_punctuation(self):
        timing = SilenceTiming(0.45, 0.7, 2.0)
        self.assertEqual(timing.pause_for("me", "Hello"), 0.7)
        self.assertEqual(timing.pause_for("me", "Hello there."), 0.7)  # previous partial was mid-sentence
        self.assertEqual(timing.pause_for("me", "Hello there. Yes."), 0.45)
        self.assertEqual(timing.pause_for("me", "Well..."), 2.0)
        self.assertEqual(timing.pause_for("me", "Well... ok."), 0.7)  # previous trailed off
        self.assertEqual(timing.pause_for("them", "Other channel."), 0.7)  # channels are independent


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.multi = FakeMulti()
        self.out = []
        self.runner = Runner(self.multi, self.out.append, settings={"engine": "x"})

    def test_bare_words_apply_to_all_channels(self):
        self.assertTrue(self.runner.handle("pause\n"))
        self.assertEqual(self.multi.calls, [("pause", None)])
        self.assertEqual(self.out, [{"type": "ack", "command": "pause", "channel": None}])

    def test_json_commands_echo_request_id(self):
        self.runner.handle(json.dumps({"command": "resume", "channel": "me", "request_id": 7}))
        self.assertEqual(self.multi.calls, [("resume", "me")])
        self.assertEqual(self.out[-1], {"type": "ack", "command": "resume", "channel": "me", "request_id": 7})

    def test_errors(self):
        self.runner.handle(json.dumps({"command": "pause", "channel": "nope"}))
        self.assertEqual(self.out[-1]["type"], "error")
        self.assertIn("unknown channel", self.out[-1]["message"])
        self.runner.handle(json.dumps({"command": "set_parameter", "parameter": "model", "value": 1}))
        self.assertIn("not settable", self.out[-1]["message"])
        self.runner.handle("{not json")
        self.assertIn("invalid JSON", self.out[-1]["message"])
        self.runner.handle(json.dumps({"command": "dance"}))
        self.assertIn("unknown command", self.out[-1]["message"])
        self.runner.handle(json.dumps({"command": "remove_channel", "channel": "nope"}))
        self.assertIn("unknown channel", self.out[-1]["message"])
        self.assertTrue(self.runner.handle("   "))
        self.assertEqual(len(self.out), 5)

    def test_set_parameter_and_get_info(self):
        self.runner.handle(json.dumps({"command": "set_parameter", "parameter": "post_speech_silence_duration", "value": 0.4, "channel": "me"}))
        self.assertEqual(self.multi.calls, [("set", "post_speech_silence_duration", 0.4, "me")])
        self.runner.handle(json.dumps({"command": "get_info", "request_id": "a"}))
        info = self.out[-1]
        self.assertEqual(info["type"], "info")
        self.assertEqual(info["request_id"], "a")
        self.assertEqual(set(info["channels"]), {"me", "them"})
        self.assertEqual(info["settings"], {"engine": "x"})
        self.assertIs(info["protocol"], PROTOCOL)

    def test_set_and_remove_channel(self):
        self.runner.handle(json.dumps({"command": "set_channel", "channel": "guest", "device": 9}))
        self.assertEqual(self.multi.calls[-1], ("set_channel", ChannelSpec("guest", 9, None, {})))
        self.assertEqual(self.out[-1], {"type": "channel", "channel": "guest",
                                        "input_device": {"index": 4, "name": "dev", "sample_rate": 48000}})
        self.runner.handle(json.dumps({"command": "remove_channel", "channel": "guest"}))
        self.assertEqual(self.out[-1], {"type": "ack", "command": "remove_channel", "channel": "guest"})
        self.assertNotIn("guest", self.multi.names)

    def test_shutdown_returns_false(self):
        self.assertFalse(self.runner.handle("shutdown"))
        self.assertEqual(self.out[-1], {"type": "ack", "command": "shutdown"})

    def test_check_inputs_warns_on_fallbacks_only(self):
        self.runner.check_inputs({
            "me": {"index": 4, "name": "USB PnP Audio Device", "sample_rate": 48000},
            "them": {"index": 11, "name": "pulse", "sample_rate": 16000},
        })
        self.assertEqual(self.out, [])
        self.runner.check_inputs({
            "me": {"index": 12, "name": "default", "sample_rate": 16000},
            "them": None,
        })
        self.assertEqual([e["type"] for e in self.out], ["warning", "warning"])
        self.assertIn("requested device 4 could not be opened; using 12 (default)", self.out[0]["message"])
        self.assertIn("not opened yet", self.out[1]["message"])
        self.multi.names.add("guest")
        self.multi.info = lambda: {"channels": {"guest": {"input_device_index": "USB PnP"}}, "engines": {}}
        self.runner.check_inputs({"guest": {"index": 12, "name": "default", "sample_rate": 16000}})
        self.assertIn("'USB PnP' not found; using 12 (default)", self.out[-1]["message"])

    def test_silence_timing_adjusts_pause_on_partials(self):
        runner = Runner(self.multi, self.out.append, silence_timing=SilenceTiming(0.45, 0.7, 2.0))
        runner.on_event({"type": "realtime", "channel": "me", "text": "Hi..."})
        self.assertEqual(self.multi.calls, [("set", "post_speech_silence_duration", 2.0, "me")])
        self.assertEqual(self.out[-1]["type"], "realtime")
        runner.on_event({"type": "speech_start", "channel": "me"})
        self.assertEqual(len(self.multi.calls), 1)


class ArgumentTests(unittest.TestCase):
    def parse(self, *argv):
        args = build_parser().parse_args(list(argv))
        with patch("importlib.util.find_spec", return_value=None):
            return apply_device_defaults(args)

    def test_cpu_defaults_and_engine_configs(self):
        args = self.parse("--channel", "name=me", "device=4", "--device", "cpu", "--engine_options", '{"keywords": ["x"]}')
        self.assertEqual((args.engine, args.model, args.rt_model, args.compute_type), ("faster_whisper", "small", "tiny.en", "int8"))
        final, realtime = engine_config(args), engine_config(args, realtime=True)
        self.assertEqual((final.model, final.beam_size, final.engine_options), ("small", 5, {"keywords": ["x"]}))
        self.assertEqual((realtime.model, realtime.beam_size, realtime.engine_options), ("tiny.en", 3, None))
        self.assertEqual([parse_channel(t) for t in args.channel], [ChannelSpec("me", 4)])

    def test_recorder_kwargs_only_include_given_flags(self):
        args = self.parse("--device", "cpu")
        self.assertEqual(recorder_kwargs(args), {"language": "en", "no_log_file": True, "level": 30})
        args = self.parse("--device", "cpu", "--post_speech_silence_duration", "0.4", "--silero_deactivity_detection", "-l", "de")
        kwargs = recorder_kwargs(args)
        self.assertEqual(kwargs["post_speech_silence_duration"], 0.4)
        self.assertTrue(kwargs["silero_deactivity_detection"])
        self.assertEqual(kwargs["language"], "de")


if __name__ == "__main__":
    unittest.main()
